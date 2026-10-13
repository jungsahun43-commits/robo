"""Test supervised negative selection, boundary/padding exclusion and recall guards."""
import unittest
import torch
from safelog_ai.belt_background import background_loss, positive_anchor, training_loss, select_background
from scripts.unified_main_data import Cache, DATA, read


class BackgroundTests(unittest.TestCase):
    def test_padding_polygon_and_boundary_band_have_zero_hard_negative_gradient(self):
        x = torch.zeros(1, 1, 32, 32, requires_grad=True); mask = torch.zeros_like(x); valid = torch.ones_like(x)
        mask[0, 0, 10, 10] = 1; valid[:, :, :4] = 0
        with torch.no_grad():
            x[0, 0, 0, 0] = 12; x[0, 0, 10, 10] = 11; x[0, 0, 12, 12] = 10; x[0, 0, 24, 24] = 9
        background_loss(x, mask, valid).backward()
        self.assertEqual(float(x.grad[0, 0, 0, 0]), 0); self.assertEqual(float(x.grad[0, 0, 10, 10]), 0)
        self.assertEqual(float(x.grad[0, 0, 12, 12]), 0); self.assertGreater(float(x.grad[0, 0, 24, 24]), 0)
        self.assertLessEqual(int(torch.count_nonzero(x.grad)), 9)

    def test_anchor_uses_only_publisher_positive_and_never_trains_reference(self):
        x = torch.zeros(1, 1, 8, 8, requires_grad=True); ref = torch.ones_like(x, requires_grad=True)
        mask = torch.zeros_like(x); valid = torch.ones_like(x); mask[0, 0, 4, 4] = 1
        positive_anchor(x, ref, mask, valid).backward()
        self.assertLess(float(x.grad[0, 0, 4, 4]), 0); self.assertEqual(int(torch.count_nonzero(x.grad)), 1)
        self.assertIsNone(ref.grad)

    def test_no_eligible_background_returns_finite_zero_gradient(self):
        x = torch.zeros(1, 1, 8, 8, requires_grad=True); mask = torch.ones_like(x); valid = torch.ones_like(x)
        value = background_loss(x, mask, valid); self.assertEqual(float(value.detach()), 0)
        value.backward(); self.assertEqual(int(torch.count_nonzero(x.grad)), 0)
        with self.assertRaises(ValueError): background_loss(x, mask*.5, valid)

    def test_real_full_and_native_truth_accept_same_spatial_contract_without_mutation(self):
        for manifest in [DATA/'manifest.json', DATA.parent/'unified-belt-native-detail-v1/manifest.json']:
            data = read(manifest); rows = data['fit']['belt_damage'] if 'fit' in data else data['rows']
            cache = Cache(rows, {0}); _, truth = cache.batch([0], device='cpu'); before = {k: v.clone() for k, v in truth.items()}
            x = torch.zeros_like(truth['mask'], requires_grad=True); reference = torch.zeros_like(x)
            value = training_loss('hard_background', x, reference, truth); self.assertTrue(torch.isfinite(value))
            value.backward(); self.assertTrue(torch.isfinite(x.grad).all())
            for key in truth: self.assertTrue(torch.equal(truth[key], before[key]))

    def test_lower_background_FPR_cannot_hide_recall_or_location_regression(self):
        base = {'pixel_fpr': .009, 'pixel_iou': .4, 'pixel_fnr': .38, 'photo_rates': {'fnr': .01}, 'positive_photos_without_any_gt_overlap': 37}
        worse = dict(base, pixel_fpr=.005, pixel_fnr=.39)
        result = select_background({'anchor_control': dict(base), 'hard_background': worse}, base)
        self.assertEqual(result['selected'], 'prior_native_mix'); self.assertFalse(result['normal_belt_photo_FPR_available'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
