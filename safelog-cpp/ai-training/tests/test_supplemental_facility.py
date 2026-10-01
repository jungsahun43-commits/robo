import unittest
import tempfile
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from scripts.train_facility_presence import partial_label_loss
from scripts.refine_synthcavity import choose, acceptable
from scripts.prepare_synthcavity import mask_union


class SupplementalLabelTests(unittest.TestCase):
    def test_both_cavity_masks_contribute_to_the_target(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.png", Path(directory) / "b.png"
            Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(a)
            Image.fromarray(np.array([[0, 0], [0, 255]], dtype=np.uint8)).save(b)
            np.testing.assert_array_equal(mask_union([a, b], (2, 2)), [[True, False], [False, True]])

    def test_mask_size_mismatch_stops_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.png"
            Image.new("L", (2, 3)).save(a)
            with self.assertRaises(ValueError): mask_union([a], (2, 2))

    def test_validation_rejects_recall_improvement_with_more_false_alarms(self):
        target = np.array([1, 1, 0, 0], dtype=bool)
        choice = choose(target, np.zeros(4, dtype=bool), np.array([.9, .2, .1, .1]), .8,
                        np.array([.9, .7, .95, .1]))
        self.assertFalse(choice["eligible"])

    def test_release_rejects_one_class_regression_despite_aggregate_gain(self):
        self.assertFalse(acceptable([
            {"round1": {"fp": 5, "fn": 5}, "round3": {"fp": 1, "fn": 1}},
            {"round1": {"fp": 5, "fn": 5}, "round3": {"fp": 6, "fn": 5}}]))

    def test_unannotated_labels_do_not_change_the_gradient(self):
        logits = torch.tensor([[1., 2., 3.]], requires_grad=True)
        loss = partial_label_loss(logits, torch.tensor([[0., 1., 0.]]),
                                  torch.tensor([[0., 1., 0.]]), torch.tensor([.2]), torch.tensor([6., 6., 6.]))
        loss.backward()
        self.assertEqual(logits.grad[0, 0].item(), 0)
        self.assertEqual(logits.grad[0, 2].item(), 0)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(2.), torch.tensor(1.)) * .2
        self.assertAlmostEqual(loss.item(), expected.item())

    def test_original_labels_keep_the_weighted_bce(self):
        logits = torch.tensor([[1., 2., 3.], [-1., -2., -3.]])
        labels = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
        weights = torch.tensor([2., 3., 4.])
        loss = partial_label_loss(logits, labels, torch.ones_like(labels), torch.ones(2), weights)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights)
        self.assertAlmostEqual(loss.item(), expected.item())


if __name__ == "__main__": unittest.main()
