import unittest
import torch
from PIL import Image
from safelog_ai.ksdd_supervised import ksdd_view, restore_map, ksdd_photo_score, mask_loss, SurfaceMaskHead
from scripts.prepare_ksdd_supervised import assign_groups


class KSDDSupervisedTests(unittest.TestCase):
    def test_view_preserves_aspect_mask_and_original_coordinates(self):
        rgb = Image.new('RGB', (100, 300), 'red'); mask = Image.new('L', rgb.size, 255)
        view, truth, valid, geometry = ksdd_view(rgb, mask)
        self.assertEqual(view.size, (256, 704)); self.assertTrue(torch.equal(truth.bool(), valid))
        restored = restore_map(valid.float(), geometry)
        self.assertEqual(tuple(restored.shape), (300, 100)); self.assertTrue(torch.all(restored > .99))

    def test_padding_cannot_set_photo_score(self):
        p = torch.zeros(1, 64, 64); valid = torch.ones_like(p, dtype=torch.bool); valid[:, :10] = False; p[:, :10] = 1
        self.assertEqual(float(ksdd_photo_score(p, valid)[0]), 0.)

    def test_real_mask_loss_trains_head_and_ignores_unknown_padding(self):
        head = SurfaceMaskHead(); low = torch.randn(2, 40, 4, 4); high = torch.randn(2, 960, 2, 2)
        logits = head(low, high); truth = torch.zeros_like(logits); truth[0, :, 1:3, 1:3] = 1
        logits = torch.nn.functional.interpolate(logits, size=(8, 8), mode='bilinear', align_corners=False)
        truth = torch.nn.functional.interpolate(truth, size=(8, 8), mode='nearest'); valid = torch.ones_like(truth)
        loss = mask_loss(logits, truth, valid); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))

    def test_TRAIN_duplicate_of_TEST_is_quarantined_even_when_labelled_defect(self):
        rows = [{'id': 'train/10000.png', 'publisher_split': 'train', 'phase': 'reserved_training_defect', 'label': 1, 'rgb_sha256': 'x', 'orientation_phashes': ['0000000000000000']},
                {'id': 'test/20000.png', 'publisher_split': 'test', 'phase': 'test', 'label': 1, 'rgb_sha256': 'y', 'orientation_phashes': ['0000000000000001']}]
        assign_groups(rows); self.assertEqual(rows[0]['phase'], 'excluded_TEST_link'); self.assertEqual(rows[1]['phase'], 'test')


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
