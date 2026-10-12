import unittest
from collections import Counter
from safelog_ai.pcb_multisource import CATEGORIES, balanced_indices


class MixedTrainingTests(unittest.TestCase):
    def rows(self):
        return [{'category': c, 'source': 'RobustAD' if c == 'robustad_pcb' else 'VisA',
                 'phase': 'fit', 'split': 'train', 'label': 0} for c in CATEGORIES for _ in range(3)]

    def test_minority_source_contributes_without_counting_draws_as_unique_photos(self):
        rows = self.rows(); indices = balanced_indices(rows, 241, 7).tolist()
        self.assertEqual(Counter(rows[i]['category'] for i in indices), dict.fromkeys(CATEGORIES, 7))
        self.assertEqual(len(indices), 35); self.assertEqual(len(rows), 15)
        self.assertEqual(indices, balanced_indices(rows, 241, 7).tolist())

    def test_test_calibration_anomaly_and_missing_source_cannot_enter_optimizer(self):
        for field, value in [('phase', 'test'), ('phase', 'calibration'), ('split', 'test0'), ('label', 1)]:
            rows = self.rows(); rows[0][field] = value
            with self.assertRaises(ValueError):
                balanced_indices(rows, 1)
        with self.assertRaises(ValueError):
            balanced_indices([r for r in self.rows() if r['source'] == 'VisA'], 1)


if __name__ == '__main__':
    unittest.main()
