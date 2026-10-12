"""Real native annotation contract, paired geometry and all-other-task retention."""
import base64
import io
import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.belt_detail import native_crop, paired_flip, configure_belt_only, non_belt_state, select_detail
from safelog_ai.unified_main import UnifiedMain
from safelog_ai.retention_distillation import state_sha256
from scripts.prepare_beltcrack_starter import rasterize
from scripts.unified_main_data import ROOT, DATA, read


class DetailTests(unittest.TestCase):
    def test_native_crop_image_marker_and_raster_target_have_same_coordinates(self):
        from PIL import ImageDraw
        image = Image.new('RGB', (1920, 1080))
        ImageDraw.Draw(image).rectangle((950, 500, 970, 520), fill=(255, 255, 255))
        annotation = {'imageWidth': 1920, 'imageHeight': 1080, 'shapes': [
            {'label': 'crack', 'shape_type': 'rectangle', 'points': [[950, 500], [970, 520]]}]}
        rgb, mask, valid, box = native_crop(image, annotation)
        x = round((960-box[0])*640/(box[2]-box[0])); y = round((510-box[1])*640/(box[3]-box[1]))
        self.assertGreater(rgb.getpixel((x, y))[0], 250); self.assertEqual(mask.getpixel((x, y)), 255)
        self.assertEqual(mask.getpixel((x-60, y)), 0); self.assertEqual(valid.getpixel((x, y)), 255)

    def test_real_native_photo_matches_fullframe_and_crop_uses_author_geometry(self):
        row = read(DATA/'manifest.json')['fit']['belt_damage'][0]
        path = ROOT/'data/equipment-beltcrack/expanded-cached-v2/original-annotations'/row['id']
        annotation = json.loads(path.read_text(encoding='utf-8'))
        image = Image.open(io.BytesIO(base64.b64decode(annotation['imageData'], validate=True))).convert('RGB')
        for kind, actual in zip(('image', 'mask', 'valid'), rasterize(annotation, image)):
            with Image.open(ROOT/row[kind]) as previous:
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(previous))
        rgb, mask, valid, box = native_crop(image, annotation)
        self.assertEqual(rgb.size, (640, 640)); self.assertGreater(np.count_nonzero(mask), 0)
        self.assertTrue(np.all(np.asarray(valid) == 255)); self.assertLessEqual(box[2], image.width)
        self.assertLessEqual(box[3], image.height)
        self.assertGreater((box[2]-box[0]), 0)

    def test_paired_flip_keeps_native_pixel_target_and_valid_aligned_without_mutation(self):
        image = torch.zeros(2, 3, 8, 8); mask = torch.zeros(2, 1, 8, 8); valid = torch.zeros_like(mask)
        image[0, :, 3, 1] = 7; mask[0, :, 3, 1] = 1; valid[0, :, :, :5] = 1
        before = tuple(v.clone() for v in (image, mask, valid))
        x, y, v = paired_flip(image, mask, valid, torch.tensor([True, False]))
        self.assertEqual(float(x[0, 0, 3, 6]), 7); self.assertEqual(float(y[0, 0, 3, 6]), 1)
        self.assertEqual(float(v[0, 0, 3, 6]), 1); self.assertEqual(float(v[0, 0, 3, 0]), 0)
        for original, stored in zip((image, mask, valid), before): self.assertTrue(torch.equal(original, stored))

    def test_only_belt_changes_including_no_other_bn_running_stats(self):
        torch.manual_seed(427); model = UnifiedMain(); before = state_sha256(non_belt_state(model))
        belt = state_sha256(model.belt_head); opt = torch.optim.AdamW(configure_belt_only(model), lr=.001)
        value = model.forward_task(torch.randn(2, 3, 128, 128), 'belt_damage').square().mean()
        value.backward(); opt.step()
        self.assertEqual(before, state_sha256(non_belt_state(model)))
        self.assertNotEqual(belt, state_sha256(model.belt_head))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith('belt_head.')))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))

    def test_selection_requires_location_and_miss_guards(self):
        base = {'pixel_iou': .38, 'pixel_fnr': .48, 'photo_rates': {'fnr': .02}, 'positive_photos_without_any_gt_overlap': 41}
        worse = dict(base, pixel_iou=.39, pixel_fnr=.49)
        result = select_detail({'full_control': worse, 'native_mix': dict(base)}, base)
        self.assertEqual(result['selected'], 'prior_v1'); self.assertFalse(result['normal_belt_photo_FPR_available'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
