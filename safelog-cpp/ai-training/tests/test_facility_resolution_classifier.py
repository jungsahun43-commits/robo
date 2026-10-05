"""Synthetic CPU checks of resolution evidence; no source photos/checkpoints."""
import io
import unittest

import torch
from torch.nn import functional as F

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES
from safelog_ai.presence_classifier import build_model
from safelog_ai.resolution_classifier import ARCH, RECIPE, ResolutionClassifier, project_logits


class _FixedBackbone(torch.nn.Module):
    def forward(self, image):
        return {"high": image.new_zeros((len(image), 960, 1, 1)),
                "low": image.new_zeros((len(image), 40, 120, 120))}


class _NativeMaps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A narrow one-row spike distinguishes pooling a native 120 map from
        # pooling its declared 80-grid area projection.
        maps = torch.zeros((1, 7, 120, 120))
        maps.flatten(2)[:, :, :32] = 9.
        self.register_buffer("maps", maps)

    def forward(self, features):
        return self.maps.expand(len(features["high"]), -1, -1, -1)


class ResolutionClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_constructor_rng_and_strictly_identical_legacy_state(self):
        torch.manual_seed(901)
        baseline = AuxiliaryClassifier(7, pretrained=False)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(901)
        resolution = ResolutionClassifier(7, pretrained=False)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(set(baseline.state_dict()), set(resolution.state_dict()))
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),
                         sum(p.numel() for p in resolution.parameters()))
        self.assertEqual(RECIPE["additional_parameters"], 0)
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, resolution.state_dict()[key]), key)
        resolution.load_state_dict(baseline.state_dict(), strict=True)
        incomplete = dict(baseline.state_dict())
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(RuntimeError):
            resolution.load_state_dict(incomplete, strict=True)

    def test_actual_640_outputs_equal_legacy_tensor_for_tensor_and_one_backbone(self):
        torch.manual_seed(902)
        baseline = AuxiliaryClassifier(7, pretrained=False).eval()
        resolution = ResolutionClassifier(7, pretrained=False).eval()
        resolution.load_state_dict(baseline.state_dict(), strict=True)
        image = torch.randn(1, 3, 640, 640)
        calls = []
        hook = resolution.backbone.register_forward_hook(lambda *_: calls.append(1))
        with torch.inference_mode():
            expected = baseline.forward_training(image)
            actual = resolution.forward_training(image)
        hook.remove()
        self.assertEqual(calls, [1])
        self.assertEqual([tuple(value.shape) for value in actual],
                         [(1, 7), (1, 7, 80, 80), (1, len(AUX_CLASSES))])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    def test_native_120_logits_are_area_projected_before_photo_pooling(self):
        model = ResolutionClassifier(7, pretrained=False).eval()
        model.backbone = _FixedBackbone()
        model.segmentation_head = _NativeMaps()
        with torch.no_grad():
            model.photo_head.weight.zero_()
            model.photo_head.bias.zero_()
            model.mix.zero_()
        image = torch.zeros(1, 3, 960, 960)
        with torch.inference_mode():
            photo, maps, auxiliary = model.forward_training(image)
        native = model.segmentation_head.maps
        projected = F.interpolate(native, size=(80, 80), mode="area")
        pooled_projected = projected.flatten(2).topk(32, dim=-1).values.mean(-1)
        pooled_native = native.flatten(2).topk(32, dim=-1).values.mean(-1)
        self.assertTrue(torch.equal(maps, projected))
        torch.testing.assert_close(photo, .5 * pooled_projected, rtol=0., atol=0.)
        self.assertFalse(torch.equal(photo, .5 * pooled_native))
        self.assertTrue(torch.all(photo < .5 * pooled_native))
        self.assertEqual(tuple(auxiliary.shape), (1, len(AUX_CLASSES)))

    def test_projection_identity_downsample_and_gradient_conservation(self):
        same = torch.randn(2, 7, 80, 80, requires_grad=True)
        self.assertIs(project_logits(same), same)
        native = torch.randn(1, 7, 120, 120, requires_grad=True)
        actual = project_logits(native)
        expected = F.interpolate(native, size=(80, 80), mode="area")
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertIsNotNone(native.grad)
        self.assertTrue(torch.isfinite(native.grad).all())
        self.assertTrue(torch.all(native.grad > 0))
        torch.testing.assert_close(native.grad.sum(),
                                   torch.tensor(float(7 * 80 * 80)), rtol=0., atol=.01)

    def test_projection_refuses_label_like_inputs_and_upsampling(self):
        floating = torch.zeros(1, 7, 80, 80)
        for logits, shape in ((floating.long(), (80, 80)),
                              (floating[0], (80, 80)),
                              (floating, (120, 120)),
                              (floating, (40, 120)),
                              (floating, (0, 80)),
                              (floating, (True, 80)),
                              (floating, (80., 80)),
                              (floating, (80,))):
            with self.subTest(shape=shape, dimensions=logits.ndim), self.assertRaises(ValueError):
                project_logits(logits, shape)

    def test_factory_checkpoint_roundtrip_actual_960_contract(self):
        torch.manual_seed(903)
        model = build_model(7, pretrained=False, architecture=ARCH).eval()
        self.assertIsInstance(model, ResolutionClassifier)
        checkpoint_bytes = io.BytesIO()
        torch.save({"architecture": ARCH, "imgsz": 960,
                    "classes": list(range(7)), "state_dict": model.state_dict()}, checkpoint_bytes)
        checkpoint_bytes.seek(0)
        checkpoint = torch.load(checkpoint_bytes, map_location="cpu", weights_only=True)
        restored = build_model(len(checkpoint["classes"]),
                               architecture=checkpoint["architecture"]).eval()
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        image = torch.randn(1, 3, checkpoint["imgsz"], checkpoint["imgsz"])
        with torch.inference_mode():
            expected = model.forward_training(image)
            actual = restored.forward_training(image)
            public = restored(image)
        self.assertEqual([tuple(value.shape) for value in actual],
                         [(1, 7), (1, 7, 80, 80), (1, len(AUX_CLASSES))])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        self.assertTrue(torch.equal(public, actual[0]))
        self.assertTrue(all(torch.isfinite(value).all() for value in actual))

    def test_actual_960_backward_reaches_backbone_and_all_existing_heads(self):
        torch.manual_seed(904)
        model = ResolutionClassifier(7, pretrained=False).eval()
        photo, maps, auxiliary = model.forward_training(torch.randn(1, 3, 960, 960))
        (photo.square().mean() + maps.square().mean() + auxiliary.square().mean()).backward()
        selected = [next(model.backbone.parameters()),
                    model.segmentation_head.low_classifier.weight,
                    model.photo_head.weight, model.auxiliary_head.weight, model.mix]
        for parameter in selected:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
