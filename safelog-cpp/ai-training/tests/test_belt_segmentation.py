import unittest

import torch

from safelog_ai.belt_segmentation import loss, pixel_counts


class BeltTests(unittest.TestCase):
    def test_padding_never_counts_as_false_positive(self):
        probability = torch.tensor([[[[0.9, 0.1], [0.9, 0.9]]]])
        target = torch.tensor([[[[1, 1], [0, 0]]]])
        valid = torch.tensor([[[[1, 1], [0, 0]]]])
        self.assertEqual(pixel_counts(probability, target, valid), {'tp': 1, 'fn': 1, 'fp': 0, 'tn': 0})

    def test_padding_gradient_is_zero(self):
        logits = torch.randn(1, 1, 2, 2, requires_grad=True)
        mask = torch.tensor([[[[1., 0.], [0., 0.]]]])
        valid = torch.tensor([[[[1., 1.], [0., 0.]]]])
        value = loss(logits, mask, valid, 20)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(int(torch.count_nonzero(logits.grad[:, :, 1])), 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[:, :, 0])), 0)

    def test_shapes_cannot_broadcast_silently(self):
        with self.assertRaises(ValueError):
            loss(torch.zeros(2, 1, 2, 2), torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2), 20)


if __name__ == '__main__':
    unittest.main()
