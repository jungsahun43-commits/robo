import unittest
import torch
from safelog_ai.component_discriminator import DiscriminatorMain, synthetic_features
from tests.test_equipment_reference_photo import FakeBackbone


class ComponentDiscriminatorTests(unittest.TestCase):
    def test_synthetic_examples_preserve_norm_and_require_matching_noise(self):
        x = torch.nn.functional.normalize(torch.randn(9, 1000), dim=1)
        altered = synthetic_features(x, torch.randn_like(x))
        self.assertTrue(torch.allclose(altered.norm(dim=1), torch.ones(9), atol=1e-6))
        self.assertGreater(float((altered-x).abs().max()), .001)
        with self.assertRaises(ValueError): synthetic_features(x, torch.zeros(2, 1000))

    def test_only_added_discriminator_receives_training_gradients(self):
        model = DiscriminatorMain({}, ['ad_screw']).requires_grad_(False)
        model.component_discriminator.requires_grad_(True)
        output = model.component_discriminator(torch.randn(8, 1000), torch.zeros(8, dtype=torch.long))
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.component_discriminator.parameters()))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith('component_discriminator.')))

    def test_evidence_uses_one_backbone_and_preserves_location(self):
        model = DiscriminatorMain({}, ['ad_screw']).eval(); model.backbone = FakeBackbone()
        with torch.inference_mode():
            location, evidence = model.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'ad_screw')
            self.assertEqual(model.backbone.calls, 1)
            expected = model.forward_equipment(torch.zeros(1, 3, 256, 256), 'ad_screw')
        self.assertTrue(torch.equal(location, expected)); self.assertEqual(tuple(evidence.shape), (1, 256, 256))
        self.assertTrue(bool(((evidence >= 0) & (evidence <= 1)).all()))


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
