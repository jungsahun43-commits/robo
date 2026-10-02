import unittest
import numpy as np
import torch
from scripts.facility_error_target import rates, under_target, operating_point, wilson_interval
from scripts.train_facility_target import masked_focal


class ErrorTargetTests(unittest.TestCase):
    def test_zero_observed_misses_is_not_proof_of_subfive_population_error(self):
        self.assertIsNone(wilson_interval(0, 0))
        interval = wilson_interval(0, 58)
        self.assertGreater(interval[1], .05)
        self.assertLess(wilson_interval(0, 100)[1], .05)
        self.assertAlmostEqual(wilson_interval(10, 100)[0], .055229, places=5)
    def test_exactly_five_percent_is_not_below_the_target(self):
        self.assertFalse(under_target([{"fnr": .05, "fpr": 0.}]))
        self.assertTrue(under_target([{"fnr": .0499, "fpr": .0499}]))
        self.assertFalse(under_target([{"fnr": None, "fpr": 0.}]))
        self.assertFalse(under_target([]))

    def test_rare_damage_cannot_hide_missed_positives_in_overall_error(self):
        metric = rates([True] + [False] * 99, [False] * 100)
        self.assertEqual(metric["photo_error_fraction"], .01)
        self.assertEqual(metric["fnr"], 1.)
        self.assertFalse(under_target([metric]))

    def test_a_single_threshold_must_work_in_both_validation_domains(self):
        result = operating_point({"a": (np.array([1, 0]), np.array([.3, .2])),
                                  "b": (np.array([1, 0]), np.array([.9, .8]))})
        self.assertFalse(result["target_passed"])
        self.assertEqual(result["worst_error"], 1.)
        with self.assertRaises(ValueError): operating_point({"a": (np.array([1, 1]), np.array([.3, .2]))})

    def test_focal_training_does_not_turn_unknown_classes_into_negatives(self):
        logits = torch.tensor([[.1, -.2, .3]], requires_grad=True)
        loss = masked_focal(logits, torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 1., 0.]]), torch.tensor([2., 3., 4.]))
        loss.backward()
        self.assertEqual(logits.grad[0, 2].item(), 0.)
        self.assertLess(logits.grad[0, 0].item(), 0.)
        self.assertGreater(logits.grad[0, 1].item(), 0.)


if __name__ == "__main__": unittest.main()
