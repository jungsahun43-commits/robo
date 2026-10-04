import io
import unittest

import torch

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES
from safelog_ai.context_classifier import ARCH, ContextClassifier, load_auxiliary_initializer
from safelog_ai.presence_classifier import build_model


class _FixedBackbone(torch.nn.Module):
    def forward(self, image):
        return {'high': image.new_zeros((len(image), 960, 1, 1)),
                'low': image.new_zeros((len(image), 40, 20, 20))}


class _FixedMaps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The 32 strongest cells have mean 4; the 256 strongest mean 1.375.
        # Remaining cells distinguish top256 from averaging the whole map.
        maps = torch.full((1, 7, 20, 20), -2.)
        flat = maps.flatten(2)
        flat[:, :, :32] = 4.
        flat[:, :, 32:256] = 1.
        self.register_buffer('maps', maps)

    def forward(self, features):
        return self.maps.expand(len(features['high']), -1, -1, -1)


def _controlled_model():
    model = ContextClassifier(7, pretrained=False).eval()
    model.backbone = _FixedBackbone()
    model.segmentation_head = _FixedMaps()
    with torch.no_grad():
        model.photo_head.weight.zero_()
        model.photo_head.bias.zero_()
        model.mix.zero_()
    return model


class ContextClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_gate_transfers_exact_outputs_and_uses_one_backbone_pass(self):
        torch.manual_seed(111)
        baseline = AuxiliaryClassifier(7, pretrained=False).eval()
        context = ContextClassifier(7, pretrained=False).eval()
        proof = load_auxiliary_initializer(context, baseline.state_dict())
        self.assertTrue(proof['shared_state_tensors_equal'])
        self.assertEqual(proof['new_state_tensor_count'], 1)
        self.assertEqual(torch.count_nonzero(context.context_gate).item(), 0)
        # 64 source cells also exercise the declared min(256, cells) bound.
        image = torch.randn(2, 3, 64, 64)
        calls = []
        hook = context.backbone.register_forward_hook(lambda *args: calls.append(1))
        with torch.inference_mode():
            expected = baseline.forward_training(image)
            actual = context.forward_training(image)
        hook.remove()
        self.assertEqual(calls, [1])
        self.assertEqual([tuple(item.shape) for item in actual],
                         [(2, 7), (2, 7, 8, 8), (2, len(AUX_CLASSES))])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        with torch.inference_mode():
            self.assertTrue(torch.equal(context(image), actual[0]))

    def test_seven_new_parameters_preserve_constructor_rng_and_control_transfer(self):
        torch.manual_seed(112)
        baseline = AuxiliaryClassifier(7, pretrained=False)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(112)
        context = ContextClassifier(7, pretrained=False)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        baseline_state, context_state = baseline.state_dict(), context.state_dict()
        self.assertEqual(set(context_state) - set(baseline_state), {'context_gate'})
        self.assertEqual(tuple(context.context_gate.shape), (7,))
        self.assertEqual(sum(p.numel() for p in context.parameters())
                         - sum(p.numel() for p in baseline.parameters()), 7)
        self.assertTrue(all(torch.equal(context_state[key], value)
                            for key, value in baseline_state.items()))
        restored_control = AuxiliaryClassifier(7, pretrained=False)
        proof = load_auxiliary_initializer(restored_control, baseline_state)
        self.assertTrue(proof['shared_state_tensors_equal'])
        self.assertEqual(proof['new_state_tensor_count'], 0)
        self.assertTrue(all(torch.equal(restored_control.state_dict()[key], value)
                            for key, value in baseline_state.items()))

    def test_zero_gate_has_finite_nonzero_first_step_gradient(self):
        model = _controlled_model()
        photo, _, _ = model.forward_training(torch.zeros(1, 3, 64, 64))
        photo.sum().backward()
        gradient = model.context_gate.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        # At zero, tanh derivative is 1 and existing mix is 0.5.
        torch.testing.assert_close(gradient, torch.full((7,), -1.3125),
                                   rtol=0., atol=1e-7)

    def test_signed_gate_changes_photo_evidence_without_changing_maps_or_aux(self):
        model = _controlled_model()
        image = torch.zeros(1, 3, 64, 64)
        with torch.inference_mode():
            original = model.forward_training(image)
            signed = torch.tensor([.5, -.5, 0., 0., 0., 0., 0.])
            model.context_gate.copy_(torch.atanh(signed))
            actual = model.forward_training(image)
        expected = torch.full((1, 7), 2.)
        expected[0, 0] = 1.34375
        expected[0, 1] = 2.65625
        torch.testing.assert_close(actual[0], expected, rtol=0., atol=1e-7)
        self.assertLess(actual[0][0, 0].item(), original[0][0, 0].item())
        self.assertGreater(actual[0][0, 1].item(), original[0][0, 1].item())
        self.assertTrue(torch.equal(actual[0][:, 2:], original[0][:, 2:]))
        self.assertTrue(torch.equal(actual[1], original[1]))
        self.assertTrue(torch.equal(actual[2], original[2]))

    def test_initializer_rejects_missing_extra_shape_dtype_nonfinite_and_nonzero_gate(self):
        baseline = AuxiliaryClassifier(7, pretrained=False)
        state = baseline.state_dict()
        first = next(key for key, value in state.items() if value.is_floating_point())
        for kind in ('missing', 'extra', 'shape', 'dtype', 'nonfinite'):
            changed = dict(state)
            if kind == 'missing': changed.pop(first)
            if kind == 'extra': changed['context_gate'] = torch.zeros(7)
            if kind == 'shape': changed[first] = state[first].unsqueeze(0)
            if kind == 'dtype': changed[first] = state[first].double()
            if kind == 'nonfinite': changed[first] = torch.full_like(state[first], float('nan'))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_auxiliary_initializer(ContextClassifier(7, pretrained=False), changed)
        context = ContextClassifier(7, pretrained=False)
        with torch.no_grad():
            context.context_gate[0] = .1
        with self.assertRaises(ValueError):
            load_auxiliary_initializer(context, state)

    def test_factory_roundtrip_preserves_nonzero_gate_and_strict_state(self):
        model = build_model(7, pretrained=False, architecture=ARCH).eval()
        self.assertIsInstance(model, ContextClassifier)
        with torch.no_grad():
            model.context_gate.copy_(torch.linspace(-.6, .6, 7))
        contents = io.BytesIO()
        torch.save({'architecture': ARCH, 'classes': list(range(7)),
                    'state_dict': model.state_dict()}, contents)
        contents.seek(0)
        checkpoint = torch.load(contents, map_location='cpu', weights_only=True)
        restored = build_model(len(checkpoint['classes']), pretrained=False,
                               architecture=checkpoint['architecture']).eval()
        restored.load_state_dict(checkpoint['state_dict'], strict=True)
        self.assertTrue(torch.equal(restored.context_gate, model.context_gate))
        incomplete = dict(checkpoint['state_dict'])
        incomplete.pop('context_gate')
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(incomplete, strict=True)
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            expected, actual = model.forward_training(image), restored.forward_training(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))


if __name__ == '__main__':
    unittest.main()
