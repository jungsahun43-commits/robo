import io
import unittest

import torch

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES
from safelog_ai.detail_classifier import ARCH, DETAIL_RECIPE, DetailClassifier, load_auxiliary_initializer
from safelog_ai.presence_classifier import build_model


class DetailClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_outputs_are_exactly_equal_and_one_backbone_pass(self):
        torch.manual_seed(91)
        baseline = AuxiliaryClassifier(7, pretrained=False).eval()
        detail = DetailClassifier(7, pretrained=False).eval()
        proof = load_auxiliary_initializer(detail, baseline.state_dict())
        self.assertTrue(proof['shared_state_tensors_equal'])
        self.assertEqual(proof['new_state_tensor_count'], 8)
        self.assertTrue(proof['new_output_projection_zero'])
        self.assertEqual(sum(p.numel() for p in detail.detail_head.parameters()),
                         DETAIL_RECIPE['new_parameter_count_for_seven_classes'])
        inputs = torch.randn(2, 3, 64, 64)
        calls = []
        hook = detail.backbone.register_forward_hook(lambda *args: calls.append(1))
        with torch.inference_mode():
            expected = baseline.forward_training(inputs)
            actual = detail.forward_training(inputs)
        hook.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual([tuple(x.shape) for x in actual], [(2, 7), (2, 7, 8, 8), (2, len(AUX_CLASSES))])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        with torch.inference_mode():
            self.assertTrue(torch.equal(detail(inputs), actual[0]))

    def test_control_transfer_is_strict_and_extra_constructor_rng_is_isolated(self):
        torch.manual_seed(92)
        baseline = AuxiliaryClassifier(7, pretrained=False)
        after_base = torch.get_rng_state().clone()
        torch.manual_seed(92)
        detail = DetailClassifier(7, pretrained=False)
        self.assertTrue(torch.equal(after_base, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(detail.state_dict()[k], v) for k, v in baseline.state_dict().items()))
        copy = AuxiliaryClassifier(7, pretrained=False)
        proof = load_auxiliary_initializer(copy, baseline.state_dict())
        self.assertEqual(proof, {'shared_state_tensors_equal': True,
                         'shared_state_tensor_count': len(baseline.state_dict()),
                         'new_state_tensor_count': 0, 'new_output_projection_zero': None})

    def test_early_branch_receives_gradient_after_final_projection_is_nonzero(self):
        torch.manual_seed(93)
        model = DetailClassifier(7, pretrained=False).eval()
        inputs = torch.randn(2, 3, 64, 64)
        photo, maps, _ = model.forward_training(inputs)
        (photo.square().mean() + maps.square().mean()).backward()
        self.assertGreater(torch.count_nonzero(model.detail_head[-1].weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(model.detail_head[0].weight.grad).item(), 0)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.detail_head[-1].weight.normal_(0, .02)
        photo, maps, _ = model.forward_training(inputs)
        loss = photo.square().mean() + maps.square().mean()
        loss.backward()
        gradient = model.detail_head[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        self.assertTrue(torch.isfinite(model.detail_head[-1].weight.grad).all())

    def test_factory_checkpoint_roundtrip_loads_new_architecture_strictly(self):
        model = build_model(7, pretrained=False, architecture=ARCH).eval()
        self.assertIsInstance(model, DetailClassifier)
        contents = io.BytesIO()
        torch.save({'architecture': ARCH, 'state_dict': model.state_dict(), 'classes': list(range(7))}, contents)
        contents.seek(0)
        checkpoint = torch.load(contents, map_location='cpu', weights_only=True)
        restored = build_model(len(checkpoint['classes']), architecture=checkpoint['architecture']).eval()
        restored.load_state_dict(checkpoint['state_dict'], strict=True)
        inputs = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(inputs), restored(inputs)))

    def test_transfer_rejects_missing_legacy_extra_shape_dtype_and_nonzero_branch(self):
        baseline = AuxiliaryClassifier(7, pretrained=False)
        state = baseline.state_dict()
        first = next(k for k, v in state.items() if v.is_floating_point())
        for kind in ('missing', 'extra', 'shape', 'dtype', 'nonfinite'):
            changed = dict(state)
            if kind == 'missing': changed.pop(first)
            if kind == 'extra': changed['detail_head.undeclared'] = torch.zeros(1)
            if kind == 'shape': changed[first] = state[first].unsqueeze(0)
            if kind == 'dtype': changed[first] = state[first].double()
            if kind == 'nonfinite': changed[first] = torch.full_like(state[first], float('nan'))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_auxiliary_initializer(DetailClassifier(7, pretrained=False), changed)
        detail = DetailClassifier(7, pretrained=False)
        with torch.no_grad(): detail.detail_head[-1].bias.fill_(.1)
        with self.assertRaises(ValueError): load_auxiliary_initializer(detail, state)


if __name__ == '__main__':
    unittest.main()
