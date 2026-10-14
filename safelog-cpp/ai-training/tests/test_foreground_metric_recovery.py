import unittest
from scripts.recover_foreground_metrics import worst_case_rates
from safelog_ai.equipment_anomaly import classification_rates


class MetricRecoveryTests(unittest.TestCase):
    def test_finite_predictions_remain_same_as_original_helper(self):
        labels = [0, 0, 1, 1]; scores = [.1, .8, .2, .9]
        self.assertEqual(worst_case_rates(labels, scores, .5), classification_rates(labels, scores, .5))

    def test_missing_photo_is_error_in_its_true_class_not_healthy(self):
        rates = worst_case_rates([0, 0, 1, 1], [float('inf'), .1, -float('inf'), .9], .5)
        self.assertEqual((rates['fp'], rates['tn'], rates['fn'], rates['tp']), (1, 1, 1, 1))
        with self.assertRaises(ValueError): worst_case_rates([0, 1], [float('nan'), .9], .5)


if __name__ == '__main__': unittest.main()
