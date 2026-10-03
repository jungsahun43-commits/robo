import math
import unittest

import torch
import torch.nn.functional as F

from scripts.facility_target_ranking import target_ranking_loss


class TargetRankingTests(unittest.TestCase):
    def test_wrong_pair_has_correct_gradients_and_improved_order_reduces_loss(self):
        logits = torch.tensor([[-1., 0.], [1., 0.]], requires_grad=True)
        labels = torch.tensor([[1., -1.], [0., -1.]])
        known = labels >= 0
        loss, stats = target_ranking_loss(logits, labels, known, [0, 0], [1, 1])
        self.assertAlmostEqual(loss.item(), F.softplus(torch.tensor(2.)).item())
        self.assertEqual(stats, {'pair_count': 1, 'contributing_groups': 1,
                                'groups': [{'domain': 0, 'target_index': 0,
                                            'positive_rows': 1, 'negative_rows': 1, 'pair_count': 1}]})
        loss.backward()
        self.assertLess(logits.grad[0, 0].item(), 0.)
        self.assertGreater(logits.grad[1, 0].item(), 0.)
        self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros(2)))
        improved, _ = target_ranking_loss(-logits.detach(), labels, known, [0, 0], [1, 1])
        self.assertLess(improved.item(), loss.item())

    def test_unknown_crop_and_cross_source_pairs_are_excluded(self):
        logits = torch.tensor([[0., 0.], [1., 0.], [99., 0.], [-99., 0.], [3., 0.]], requires_grad=True)
        labels = torch.tensor([[1., -1.], [0., -1.], [-1., -1.], [0., -1.], [1., -1.]])
        known = labels >= 0
        loss, stats = target_ranking_loss(logits, labels, known, [0, 0, 0, 0, 1], [1, 1, 1, 0, 1])
        self.assertEqual(stats['pair_count'], 1)
        self.assertEqual(stats['contributing_groups'], 1)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[2:], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros(5)))
        # Cross-source positive/negative rows alone do not create a pair.
        cross = logits.detach()[:2].requires_grad_()
        zero, info = target_ranking_loss(cross, labels[:2], known[:2], [0, 1], [1, 1])
        self.assertEqual(zero.item(), 0.)
        self.assertEqual(info['pair_count'], 0)
        zero.backward()
        self.assertTrue(torch.equal(cross.grad, torch.zeros_like(cross)))

    def test_each_contributing_group_mean_has_equal_weight(self):
        logits = torch.tensor([[0., 0.], [2., 0.], [2., 0.], [4., 0.], [1., 0.]], requires_grad=True)
        labels = torch.tensor([[1., -1.], [0., -1.], [0., -1.], [1., -1.], [0., -1.]])
        loss, stats = target_ranking_loss(logits, labels, labels >= 0, [0, 0, 0, 1, 1], [1]*5)
        expected = (F.softplus(torch.tensor(2.)) + F.softplus(torch.tensor(-3.))) / 2
        self.assertAlmostEqual(loss.item(), expected.item())
        self.assertEqual(stats['pair_count'], 3)
        self.assertEqual(stats['contributing_groups'], 2)
        shifted = logits.detach().clone()
        shifted[:3, 0] += 20.; shifted[3:, 0] -= 20.
        unchanged, _ = target_ranking_loss(shifted, labels, labels >= 0, [0, 0, 0, 1, 1], [1]*5)
        self.assertEqual(unchanged.item(), loss.item())

    def test_both_targets_work_with_seven_columns_and_clamped_unknown_labels(self):
        logits = torch.zeros((4, 7), requires_grad=True)
        # Trainer clamps unknown labels to zero and supplies a separate mask.
        labels = torch.tensor([[1., 0., 1., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0.],
                               [1., 1., 1., 0., 0., 0., 0.]])
        known = torch.ones_like(labels); known[2] = 0
        loss, stats = target_ranking_loss(logits, labels, known, [0]*4, [1, 1, 1, 0])
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(stats['pair_count'], 2)
        self.assertEqual(stats['contributing_groups'], 2)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[2:], torch.zeros(2, 7)))
        self.assertTrue(torch.equal(logits.grad[:, 2:], torch.zeros(4, 5)))
        self.assertLess(logits.grad[0, 0].item(), 0.)
        self.assertGreater(logits.grad[1, 0].item(), 0.)
        self.assertGreater(logits.grad[0, 1].item(), 0.)
        self.assertLess(logits.grad[1, 1].item(), 0.)

    def test_amp_originated_extreme_logits_and_empty_batch_are_stable(self):
        logits = torch.tensor([[-60000., 0.], [60000., 0.]], dtype=torch.float16, requires_grad=True)
        labels = torch.tensor([[1., -1.], [0., -1.]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, _ = target_ranking_loss(logits, labels, labels >= 0, [0, 0], [1, 1])
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 0].item(), -1.)
        self.assertEqual(logits.grad[1, 0].item(), 1.)
        empty = torch.empty((0, 2), requires_grad=True)
        zero, stats = target_ranking_loss(empty, torch.empty((0, 2)), torch.empty((0, 2)), [], [])
        self.assertEqual(zero.item(), 0.)
        self.assertEqual(stats['pair_count'], 0)
        zero.backward()

    def test_invalid_numeric_shapes_and_scopes_are_rejected(self):
        valid = {'logits': torch.zeros((2, 2)), 'labels': [[1, 0], [0, 1]],
                 'known': [[1, 1], [1, 1]], 'domains': [0, 0], 'full_mask': [1, 1]}
        cases = [dict(logits=torch.tensor([[math.inf, 0.], [0., 0.]])),
                 dict(logits=torch.zeros((2, 2), dtype=torch.long)),
                 dict(labels=[[1, -1], [0, 1]]), dict(labels=[[2, 0], [0, 1]]),
                 dict(known=[[1, .5], [1, 1]]), dict(known=[[1, math.nan], [1, 1]]),
                 dict(domains=[0, 3]), dict(domains=[0, .5]), dict(domains=[True, False]),
                 dict(full_mask=[1, .5]), dict(full_mask=[1]),
                 dict(target_indices=(0, 0)), dict(target_indices=(0, 2)), dict(target_indices=(True,)),
                 dict(target_indices=())]
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                target_ranking_loss(**{**valid, **changed})


if __name__ == '__main__':
    unittest.main()
