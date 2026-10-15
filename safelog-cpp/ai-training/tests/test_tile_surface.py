import inspect
import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.tile_surface import tile_boxes, tile_view, per_tile_loss, group_photo_scores, DetailSurfaceMain
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
from safelog_ai.magnetic_surface import RealMaskMain


class TileSurfaceTests(unittest.TestCase):
    def test_grid_covers_odd_source_edges_without_GT(self):
        size = (641, 481); boxes = tile_boxes(size)
        covered = np.zeros((size[1], size[0]), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            covered[y0:y1, x0:x1] = True
        self.assertEqual(len(set(boxes)), 9)
        self.assertTrue(covered.all())
        self.assertEqual(boxes[-1][2:], size)

    def test_native_mask_crop_coordinates_and_valid(self):
        image = Image.new('RGB', (640, 480), 'gray')
        values = np.zeros((480, 640), dtype='uint8'); values[200:210, 300:310] = 255
        truth = Image.fromarray(values)
        positive = 0
        for box in tile_boxes(image.size):
            view, mask, valid, geom = tile_view(image, box, truth)
            x0, y0, x1, y1 = box
            expected = bool(values[y0:y1, x0:x1].any())
            self.assertEqual(bool(mask.any()), expected)
            self.assertFalse(bool(mask[~valid].any()))
            self.assertEqual(view.size, (512, 512))
            positive += expected
        self.assertGreater(positive, 0)

    def test_loss_vector_and_original_photo_aggregation(self):
        x = torch.zeros(2, 1, 16, 16, requires_grad=True)
        mask = torch.zeros_like(x); mask[0, 0, 3, 4] = 1
        valid = torch.ones_like(x)
        losses = per_tile_loss(x, mask, valid)
        self.assertEqual(losses.shape, (2,)); losses.mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLess(float(x.grad[0, 0, 3, 4]), 0)
        self.assertEqual(group_photo_scores([.2, .8, .1], ['a', 'a', 'b']), {'a': .8, 'b': .1})

    def test_original_dispatch_preserved(self):
        self.assertIs(DetailSurfaceMain.forward_surface, IndustrialSurfaceMain.forward_surface)
        self.assertIs(DetailSurfaceMain.forward_real_mask, RealMaskMain.forward_surface)
        inspect.signature(DetailSurfaceMain.forward_surface).bind(object(), object(), object(), 'mpdd_bracket_white')
        self.assertNotIn('category', inspect.signature(DetailSurfaceMain.forward_detail_view).parameters)


if __name__ == '__main__':
    unittest.main()
