"""Native coverage, target alignment, stitching and conservative selection."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from safelog_ai.belt_tiles import uniform_boxes, raster_view, stitch, policy, select_tiles


class TileTests(unittest.TestCase):
    def test_landscape_and_portrait_geometry_covers_all_native_pixels_without_GT(self):
        for width, height in [(1920, 1080), (3840, 2160), (2160, 3840)]:
            boxes = uniform_boxes(width, height); self.assertLessEqual(len(boxes), 6)
            covered = np.zeros((height, width), bool)
            for left, top, right, bottom in boxes:
                self.assertGreaterEqual(left, 0); self.assertGreaterEqual(top, 0)
                self.assertLessEqual(right, width); self.assertLessEqual(bottom, height)
                self.assertEqual(right-left, bottom-top); covered[top:bottom, left:right] = True
            self.assertTrue(covered.all())

    def test_tiny_and_unsupported_panorama_fall_back_without_uncovered_tiles(self):
        self.assertEqual(uniform_boxes(640, 360), []); self.assertEqual(uniform_boxes(8000, 1000), [])

    def test_native_marker_and_publisher_target_have_same_coordinates(self):
        image = Image.new('RGB', (1920, 1080)); ImageDraw.Draw(image).rectangle((900, 500, 920, 520), fill='white')
        annotation = {'imageWidth': 1920, 'imageHeight': 1080, 'shapes': [{'label': 'crack', 'shape_type': 'rectangle', 'points': [[900, 500], [920, 520]]}]}
        for box in uniform_boxes(*image.size):
            rgb, mask, valid = raster_view(image, annotation, box)
            left, top, right, bottom = box
            if left+20 < 910 < right-20 and top+20 < 510 < bottom-20:
                x, y = round((910-left)*640/(right-left)), round((510-top)*640/(bottom-top))
                self.assertGreater(rgb.getpixel((x, y))[0], 250); self.assertEqual(mask.getpixel((x, y)), 255)
            self.assertTrue(np.all(np.asarray(valid) == 255))
        _, empty, _ = raster_view(image, annotation, (0, 0, 640, 640))
        self.assertFalse(np.any(empty))  # Local author mask empty; no whole-photo health label.

    def test_stitch_preserves_constant_and_native_coordinate_ramp(self):
        width, height = 1920, 1080; boxes = uniform_boxes(width, height)
        maps = torch.full((len(boxes), 1, 640, 640), .3)
        self.assertTrue(torch.allclose(stitch(maps, boxes, width, height), torch.full((1, 1, height, width), .3), atol=1e-6))
        ramp = torch.linspace(0, 1, width).view(1, 1, 1, width).expand(1, 1, height, width)
        maps = torch.cat([F.interpolate(ramp[:, :, t:b, l:r], size=(640, 640), mode='bilinear', align_corners=False) for l, t, r, b in boxes])
        self.assertLess(float((stitch(maps, boxes, width, height)-ramp).abs().max()), .002)

    def test_policy_identity_changes_with_mode_and_never_accepts_unknown_mode(self):
        self.assertNotEqual(policy('global_only')['sha256'], policy('global_tile_mean')['sha256'])
        self.assertFalse(policy('global_tile_mean')['annotations_at_inference'])
        with self.assertRaises(ValueError): policy('guess')

    def test_selection_rejects_more_misses_or_background_even_with_higher_IoU(self):
        base = {'pixel_iou': .405, 'pixel_fpr': .009, 'pixel_fnr': .38, 'photo_rates': {'fnr': .0085}, 'positive_photos_without_any_gt_overlap': 37}
        results = {n: dict(base, pixel_iou=.43, pixel_fpr=.01) for n in ['accepted_tiles', 'trained_global', 'trained_tiles']}
        self.assertEqual(select_tiles(results, base)['selected'], 'prior_native_mix')


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
