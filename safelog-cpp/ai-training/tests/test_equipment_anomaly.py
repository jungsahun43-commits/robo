import unittest

import numpy as np
import torch

from safelog_ai.equipment_anomaly import (normal_calibration_threshold, classification_rates,
    rank_auroc, matching_loss, inspection_result)


class EquipmentTests(unittest.TestCase):
    def test_normal_threshold_finite_sample_rank(self):
        values = np.arange(90, dtype=float)
        threshold = normal_calibration_threshold(values)
        self.assertEqual(threshold, 86)
        self.assertEqual(int((values > threshold).sum()), 3)

    def test_threshold_rejects_nan_and_tiny_holdout(self):
        with self.assertRaises(ValueError):
            normal_calibration_threshold([float('nan')] * 90)
        with self.assertRaises(ValueError):
            normal_calibration_threshold([0] * 5)

    def test_threshold_ties_are_not_anomalies(self):
        result = classification_rates([0, 0, 1, 1], [1, 2, 2, 3], 2)
        self.assertEqual((result['fp'], result['tn'], result['tp'], result['fn']), (0, 2, 1, 1))
        self.assertEqual(result['fnr'], 0.5)

    def test_missing_denominator_not_zero(self):
        self.assertIsNone(classification_rates([1, 1], [1, 2], 0)['fpr'])
        self.assertIsNone(rank_auroc([1, 1], [1, 2]))

    def test_auroc_handles_ties(self):
        self.assertEqual(rank_auroc([0, 1], [1, 1]), 0.5)
        self.assertEqual(rank_auroc([0, 1], [0, 1]), 1.0)
        self.assertEqual(rank_auroc([0, 1], [1, 0]), 0.0)

    def test_unknown_machinery_refused(self):
        with self.assertRaises(ValueError):
            inspection_result('conveyor_belt', 0.1, 0.2, 'abc')

    def test_normal_visual_result_not_mechanical_safe(self):
        result = inspection_result('pcb1', 0.1, 0.2, 'abc')
        self.assertEqual(result['mechanical_failure_state'], 'not_assessed')
        self.assertTrue(result['human_confirmation_required'])
        self.assertFalse(result['score_is_probability'])

    def test_teacher_detached(self):
        teacher = [torch.randn(2, 4, 3, 3, requires_grad=True) for _ in range(3)]
        student = [torch.randn(2, 4, 3, 3, requires_grad=True) for _ in range(3)]
        matching_loss(student, teacher).backward()
        self.assertTrue(all(t.grad is None for t in teacher))
        self.assertTrue(all(s.grad is not None for s in student))


if __name__ == '__main__':
    unittest.main()
