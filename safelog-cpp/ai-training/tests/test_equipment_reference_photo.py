import unittest
import torch
from torch import nn
from safelog_ai.equipment_reference_photo import ReferencePhotoMain
from safelog_ai.equipment_expansion import features_for_equipment, anomaly_map, CATEGORIES
from safelog_ai.metal_reference import patch_features


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0
        generator = torch.Generator().manual_seed(613)
        self.register_buffer('low', torch.randn(1, 40, 32, 32, generator=generator))
        self.register_buffer('high', torch.randn(1, 960, 16, 16, generator=generator))

    def forward(self, x):
        self.calls += 1
        return {'low': self.low.expand(len(x), -1, -1, -1), 'high': self.high.expand(len(x), -1, -1, -1)}


class ReferencePhotoTests(unittest.TestCase):
    def test_one_backbone_call_preserves_location_and_separates_presence(self):
        model = ReferencePhotoMain({'ad_cable': 256}).eval(); model.backbone = FakeBackbone()
        features = {'low': model.backbone.low, 'high': model.backbone.high}
        patches = patch_features(features).permute(0, 2, 3, 1).reshape(-1, 1000)
        model.normal_reference_ad_cable.copy_(patches)
        with torch.inference_mode():
            location, evidence = model.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'ad_cable')
            target = features_for_equipment(features)
            expected = anomaly_map(model.equipment_head(target, torch.tensor([CATEGORIES.index('ad_cable')])), target)
        self.assertEqual(model.backbone.calls, 1)
        self.assertTrue(torch.equal(location, expected))
        self.assertLess(float(evidence.max()), 1e-5)
        self.assertGreater(float(location.mean()), .01)

    def test_unconfigured_category_retains_same_score_map(self):
        model = ReferencePhotoMain({}).eval(); model.backbone = FakeBackbone()
        with torch.inference_mode():
            location, evidence = model.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'ksdd2_surface')
        self.assertTrue(torch.equal(location, evidence)); self.assertEqual(model.backbone.calls, 1)

    def test_unknown_reference_or_task_fails_instead_of_healthy_default(self):
        for counts in ({'internal_motor_fault': 8}, {'ad_cable': 0}):
            with self.assertRaises(ValueError): ReferencePhotoMain(counts)
        model = ReferencePhotoMain({})
        with self.assertRaises(ValueError): model.forward_equipment_evidence(torch.zeros(1, 3, 256, 256), 'unknown')


if __name__ == '__main__':
    torch.set_num_threads(4); unittest.main()
