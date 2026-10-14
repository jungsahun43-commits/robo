import unittest
import torch
from safelog_ai.component_detail import fine_features, whole_template_distance, DetailLayoutMain
from safelog_ai.metal_reference import nearest_cosine
from tests.test_equipment_reference_photo import FakeBackbone


class ComponentDetailTests(unittest.TestCase):
    def test_whole_template_rejects_impossible_combinations_of_normal_parts(self):
        # Two individually normal patches cannot be independently chosen from
        # different normal photos to explain an abnormal arrangement.
        bank = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        unusual = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
        full_map, score = whole_template_distance(unusual, bank, 1)
        pooled = nearest_cosine(unusual.permute(0, 2, 3, 1).reshape(-1, 2), bank)
        self.assertEqual(float(pooled.max()), 0.)
        self.assertAlmostEqual(float(score[0]), 1.)
        self.assertEqual(tuple(full_map.shape), (1, 1, 2))
        normal = torch.tensor([[[[1., 1.]], [[0., 0.]]]])
        self.assertAlmostEqual(float(whole_template_distance(normal, bank, 9)[1][0]), 0.)

    def test_zero_activation_background_has_finite_nonzero_fine_descriptor(self):
        descriptor = fine_features({'low': torch.zeros(1, 40, 32, 32)})
        self.assertEqual(tuple(descriptor.shape), (1, 41, 32, 32))
        self.assertTrue(torch.isfinite(descriptor).all()); self.assertTrue(torch.allclose(descriptor.norm(dim=1), torch.ones(1, 32, 32)))

    def test_fine_model_has_one_backbone_call_and_preserves_original_location(self):
        model = DetailLayoutMain({'ad_screw': 1024}, {'ad_screw': 'fine_low41'}).eval(); model.backbone = FakeBackbone()
        features = {'low': model.backbone.low, 'high': model.backbone.high}
        model.normal_reference_ad_screw.copy_(fine_features(features).permute(0, 2, 3, 1).reshape(-1, 41))
        with torch.inference_mode():
            location, reference, score = model.forward_component_outputs(torch.zeros(1, 3, 256, 256), 'ad_screw')
            self.assertEqual(model.backbone.calls, 1)
            expected = model.forward_equipment(torch.zeros(1, 3, 256, 256), 'ad_screw')
        self.assertTrue(torch.equal(location, expected)); self.assertLess(float(reference.max()), 1e-5); self.assertLess(float(score[0]), 1e-5)


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
