import unittest
from unittest.mock import patch
import numpy as np
from scripts.evaluate_facility_ensemble import combine, test as run_test


class EnsembleTests(unittest.TestCase):
    def test_probability_mean_retains_multi_label_shape(self):
        np.testing.assert_allclose(combine([[[.2,.9]],[[.4,.1]]]), [[.3,.5]])
        for predictions in ([[[1.1]],[[.1]]], [[[float('nan')]],[[.1]]], [[[.2,.9]],[[.4]]]):
            with self.assertRaises(ValueError):combine(predictions)

    def test_failed_validation_blocks_test_inference(self):
        with patch('scripts.evaluate_facility_ensemble.read',return_value={'target_passed':False}), \
             patch('scripts.evaluate_facility_ensemble.frozen_members') as members:
            with self.assertRaisesRegex(ValueError,'Validation target not achieved'):run_test('cpu')
            members.assert_not_called()
