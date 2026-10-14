import unittest
from PIL import Image
import torch
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain, surface_view
from tests.test_equipment_reference_photo import FakeBackbone


class IndustrialSurfaceTests(unittest.TestCase):
    def test_non_square_view_is_letterboxed_without_stretch(self):
        image, record = surface_view(Image.new('RGB', (800, 400), 'red'), 'aspect512')
        self.assertEqual(image.size, (512, 512)); self.assertEqual(record['offset'], [0, 128])
        self.assertEqual(image.getpixel((0, 0)), (127, 127, 127)); self.assertEqual(image.getpixel((0, 128)), (255, 0, 0))

    def test_original_location_is_exact_for_both_one_and_two_pass_methods(self):
        for method, calls in (('pooled256', 1), ('aspect512', 2)):
            model = IndustrialSurfaceMain({}, {}, 'linear', {'mpdd_tubes': {'method': method, 'patches': 2}}).eval()
            model.backbone = FakeBackbone(); model.surface_reference_mpdd_tubes.copy_(torch.randn(2, 1000))
            original = torch.zeros(1, 3, 256, 256); view = torch.zeros(1, 3, 512 if calls == 2 else 256, 512 if calls == 2 else 256)
            with torch.inference_mode():
                location, heat, score = model.forward_surface(original, view, 'mpdd_tubes')
                self.assertEqual(model.backbone.calls, calls)
                expected = model.forward_equipment(original, 'mpdd_tubes')
            self.assertTrue(torch.equal(location, expected)); self.assertEqual(tuple(heat.shape[-2:]), tuple(view.shape[-2:])); self.assertTrue(torch.isfinite(score).all())

    def test_unknown_or_unfitted_surface_is_rejected(self):
        with self.assertRaises(ValueError): IndustrialSurfaceMain({}, {}, 'linear', {'internal_bearing_failure': {'method': 'pooled256', 'patches': 2}})


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
