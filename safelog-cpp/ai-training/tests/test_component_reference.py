import unittest
import torch
from safelog_ai.component_reference import ComponentReferenceMain
from safelog_ai.equipment_reference_photo import ReferencePhotoMain
from tests.test_equipment_reference_photo import FakeBackbone
from safelog_ai.metal_reference import patch_features


class ComponentReferenceTests(unittest.TestCase):
    def test_existing_reference_buffers_and_scores_survive_strict_extension(self):
        prior = ReferencePhotoMain({'ad_cable': 8}).eval()
        torch.manual_seed(719); prior.normal_reference_ad_cable.copy_(torch.randn(8, 1000))
        extended = ComponentReferenceMain({'ad_cable': 8, 'ad_screw': 4}).eval()
        missing, extra = extended.load_state_dict(prior.state_dict(), strict=False)
        self.assertEqual(missing, ['normal_reference_ad_screw']); self.assertEqual(extra, [])
        prior.backbone = FakeBackbone(); extended.backbone = FakeBackbone()
        with torch.inference_mode():
            before = prior.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'ad_cable')
            after = extended.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'ad_cable')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_new_component_uses_one_backbone_and_seen_normal_has_zero_distance(self):
        model = ComponentReferenceMain({'loco_splicing_connectors': 256}).eval(); model.backbone = FakeBackbone()
        features = {'low': model.backbone.low, 'high': model.backbone.high}
        model.normal_reference_loco_splicing_connectors.copy_(patch_features(features).permute(0, 2, 3, 1).reshape(-1, 1000))
        with torch.inference_mode():
            location, presence = model.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'loco_splicing_connectors')
            expected_location = model.forward_equipment(torch.zeros(1, 3, 256, 256), 'loco_splicing_connectors')
        self.assertEqual(model.backbone.calls, 2)  # one evidence pass plus explicit comparison
        self.assertTrue(torch.equal(location, expected_location)); self.assertLess(float(presence.max()), 1e-5)

    def test_invalid_or_unsupported_reference_cannot_silently_be_normal(self):
        for value in ({'internal_motor_fault': 20}, {'ad_screw': 0}, {'ad_screw': True}):
            with self.assertRaises(ValueError): ComponentReferenceMain(value)


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
