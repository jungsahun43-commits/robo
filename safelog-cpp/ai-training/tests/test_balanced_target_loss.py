import unittest
import torch
from safelog_ai.balanced_target_loss import balanced_pixel_bce, balanced_target_loss


class BalancedTargetTests(unittest.TestCase):
    def test_positive_negative_gradient_mass_equal_independent_of_area(self):
        x = torch.zeros(1, 1, 16, 16, requires_grad=True)
        target = torch.zeros_like(x); target[..., 0, 0] = 1
        valid = torch.ones_like(x)
        balanced_pixel_bce(x, target, valid).backward()
        self.assertAlmostEqual(float(x.grad[target.bool()].abs().sum()), .25, places=6)
        self.assertAlmostEqual(float(x.grad[~target.bool()].abs().sum()), .25, places=6)

    def test_normal_and_padding_have_valid_gradients_only(self):
        x = torch.zeros(1, 1, 8, 8, requires_grad=True)
        valid = torch.ones_like(x); valid[..., :2, :] = 0
        balanced_pixel_bce(x, torch.zeros_like(x), valid).backward()
        self.assertEqual(float(x.grad[~valid.bool()].abs().sum()), 0)
        self.assertAlmostEqual(float(x.grad.abs().sum()), .5, places=6)

    def test_full_loss_finite_and_old_weight_has_no_effect(self):
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.zeros_like(x); target[0, 0, 5, 5] = 1
        valid = torch.ones_like(x)
        a = balanced_target_loss(x, target, valid, 10)
        self.assertTrue(torch.equal(a, balanced_target_loss(x, target, valid, 100)))
        a.backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
