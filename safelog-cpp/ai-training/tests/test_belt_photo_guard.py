import unittest
import torch
from safelog_ai.belt_photo_guard import positive_photo_losses, project_photo_step


class PhotoGuardTests(unittest.TestCase):
    def test_small_and_large_masks_get_separate_normalized_losses(self):
        logits = torch.tensor([[[[-2., 4., 4.]]], [[[-2., -2., -2.]]]], requires_grad=True)
        mask = torch.tensor([[[[1., 0., 0.]]], [[[1., 1., 1.]]]])
        losses = positive_photo_losses(logits, mask, torch.ones_like(mask))
        self.assertTrue(torch.allclose(losses[0], losses[1]))
        losses.mean().backward()
        self.assertAlmostEqual(float(logits.grad[0].sum()), float(logits.grad[1].sum()), places=6)
        self.assertEqual(float(logits.grad[0, 0, 0, 1]), 0)

    def test_validity_and_every_positive_photo_are_required(self):
        x = torch.tensor([[[[-2., -8.]]]], requires_grad=True)
        losses = positive_photo_losses(x, torch.ones_like(x), torch.tensor([[[[1., 0.]]]]))
        losses.sum().backward()
        self.assertEqual(float(x.grad[0, 0, 0, 1]), 0)
        with self.assertRaises(ValueError):
            positive_photo_losses(x, torch.zeros_like(x), torch.ones_like(x))

    def test_two_constraints_remove_two_harmful_axes_keep_orthogonal_motion(self):
        old = torch.tensor([.1, .1, .1])
        p = torch.nn.Parameter(torch.tensor([.3, .2, -.2]))
        r = project_photo_step([p], [old], [[torch.tensor([1., 0., 0.])], [torch.tensor([0., 1., 0.])]])
        self.assertTrue(r['projected']); self.assertFalse(r['reverted'])
        self.assertTrue(all(v <= r['tolerance'] for v in r['dot_after']))
        self.assertAlmostEqual(float(p.detach()[2]), -.2, places=6)

    def test_safe_step_is_byte_identical_including_zero_gradient(self):
        p = torch.nn.Parameter(torch.tensor([-.2, -.1]))
        proposed = p.detach().clone()
        r = project_photo_step([p], [torch.zeros(2)], [[torch.tensor([1., 0.])], [torch.zeros(2)]])
        self.assertFalse(r['projected']); self.assertTrue(torch.equal(p, proposed))

    def test_opposing_constraints_restore_exact_old_parameters(self):
        old = torch.tensor([.1, .2])
        p = torch.nn.Parameter(torch.tensor([.3, -.2]))
        r = project_photo_step([p], [old], [[torch.tensor([1., 0.])], [torch.tensor([-1., 0.])]])
        self.assertTrue(r['reverted']); self.assertTrue(torch.equal(p, old))
        self.assertTrue(all(v == 0 for v in r['dot_after']))

    def test_real_adam_momentum_is_guarded_without_changing_optimizer_moments(self):
        p = torch.nn.Parameter(torch.tensor([.1, .1]))
        opt = torch.optim.AdamW([p], lr=.02, weight_decay=.001)
        for _ in range(2):
            old = p.detach().clone(); opt.zero_grad(); p.grad = torch.tensor([-1., 1.]); opt.step()
        moment = opt.state[p]['exp_avg'].clone()
        r = project_photo_step([p], [old], [[torch.tensor([1., 0.])]])
        self.assertTrue(r['projected']); self.assertLess(float(p.detach()[1]), float(old[1]))
        self.assertTrue(torch.equal(moment, opt.state[p]['exp_avg']))

    def test_dense_float32_rounding_checks_all_photos(self):
        rng = torch.Generator().manual_seed(449)
        old = torch.randn(160000, generator=rng) * .03
        shared = torch.randn(160000, generator=rng)
        gradients = [shared + .2 * torch.randn(160000, generator=rng) for _ in range(6)]
        p = torch.nn.Parameter(old + shared * 2e-5)
        r = project_photo_step([p], [old], [[g] for g in gradients])
        self.assertEqual(r['photo_constraints'], 6)
        self.assertTrue(all(v <= r['tolerance'] for v in r['dot_after']))
        self.assertFalse(r['reverted'])


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
