import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.industrial_foreground import foreground_view, foreground_score, ForegroundMain
from tests.test_equipment_reference_photo import FakeBackbone


class ForegroundTests(unittest.TestCase):
    def test_uniform_or_small_input_requests_review(self):
        for rgb in (Image.new('RGB', (512, 512), 'green'), Image.new('RGB', (32, 32))):
            with self.assertRaises(ValueError): foreground_view(rgb)

    def test_letterbox_padding_is_not_foreground(self):
        a = np.full((200, 400, 3), (60, 100, 40), dtype=np.uint8)
        a[50:150, 100:300] = (10, 10, 10)
        view, mask, meta = foreground_view(Image.fromarray(a))
        self.assertEqual(view.size, (512, 512)); self.assertEqual(meta['offset'], [0, 128])
        self.assertFalse(mask[:128].any()); self.assertFalse(mask[384:].any())
        self.assertTrue(mask[220:290, 200:300].all())

    def test_background_activations_cannot_decide_the_photo(self):
        heat = torch.ones(1, 16, 16)*100.; mask = torch.zeros_like(heat, dtype=torch.bool)
        mask[:, 4:12, 4:12] = True; heat[mask] = .1
        self.assertAlmostEqual(float(foreground_score(heat, mask)[0]), .1, places=6)
        with self.assertRaises(ValueError): foreground_score(heat, torch.zeros_like(mask))

    def test_same_backbone_two_passes_and_original_location_exact(self):
        model = ForegroundMain({}, {}, 'linear', {}, {'mpdd_tubes': {'patches': 4}}).eval()
        model.backbone = FakeBackbone()
        model.foreground_reference_mpdd_tubes.copy_(torch.randn(4, 1000))
        x = torch.zeros(1, 3, 256, 256); native = torch.zeros(1, 3, 512, 512)
        mask = torch.ones(1, 512, 512, dtype=torch.bool)
        with torch.inference_mode():
            location, _, score = model.forward_foreground(x, native, mask, 'mpdd_tubes')
            self.assertEqual(model.backbone.calls, 2)
            expected = model.forward_equipment(x, 'mpdd_tubes')
        self.assertTrue(torch.equal(location, expected)); self.assertTrue(torch.isfinite(score).all())


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
