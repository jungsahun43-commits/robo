import unittest
import torch
from safelog_ai.native_component_server import map_native_heatmap
from safelog_ai.native_component_views import NativeComponentMain
from tests.test_equipment_reference_photo import FakeBackbone


class NativeServerGeometryTests(unittest.TestCase):
    def test_reference_is_mapped_to_original_coordinates_with_validity_mask(self):
        heat = torch.tensor([[2., 0.], [0., 0.]])
        mapped, valid = map_native_heatmap(heat, (9, 8), {'source_from_view_affine': [1., 0., 4., 0., 1., 3.]})
        self.assertEqual(mapped.getpixel((4, 3)), 255); self.assertEqual(mapped.getpixel((0, 0)), 0)
        self.assertEqual(valid.getpixel((4, 3)), 255); self.assertEqual(valid.getpixel((0, 0)), 0)

    def test_native_inference_has_two_passes_of_same_backbone_and_exact_location(self):
        model = NativeComponentMain({}, {'ad_screw': {'method': 'aligned_hilo', 'patches': 2}}).eval(); model.backbone = FakeBackbone()
        model.native_reference_ad_screw.copy_(torch.randn(2, 1000))
        with torch.inference_mode():
            location, reference, score, _ = model.forward_native(torch.zeros(1, 3, 256, 256), torch.zeros(1, 3, 256, 768), 'ad_screw')
            self.assertEqual(model.backbone.calls, 2)
            expected = model.forward_equipment(torch.zeros(1, 3, 256, 256), 'ad_screw')
        self.assertTrue(torch.equal(location, expected)); self.assertEqual(tuple(reference.shape), (1, 256, 768)); self.assertTrue(torch.isfinite(score).all())


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
