import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from safelog_ai.foreign_calibration import fit_threshold
from prepare_foreign_calibration_check import choose, PRIOR, SOURCE, read


class CalibrationTests(unittest.TestCase):
    def test_test_rows_never_fit_threshold(self):
        with self.assertRaises(ValueError):
            fit_threshold([{'split':'test','label':0,'score':.1},{'split':'calibration','label':1,'score':.2}])

    def test_low_separated_scores_can_have_valid_decision_threshold(self):
        rows = [{'split':'calibration','label':y,'score':s} for y,s in [(0,.001),(0,.002),(1,.01),(1,.02)]]
        result = fit_threshold(rows)
        self.assertTrue(.002<result['threshold']<.01)
        self.assertEqual(result['calibration_max_error_rate'],0)

    def test_missing_class_or_nonfinite_scores_are_rejected(self):
        for rows in [[{'split':'calibration','label':0,'score':.1}],
                     [{'split':'calibration','label':0,'score':.1},{'split':'calibration','label':1,'score':float('nan')}]]:
            with self.assertRaises(ValueError):
                fit_threshold(rows)

    def test_float32_sigmoid_saturation_is_supported(self):
        result = fit_threshold([{'split':'calibration','label':0,'score':0},
                                {'split':'calibration','label':1,'score':1}])
        self.assertEqual(result['calibration_max_error_rate'],0)
        self.assertTrue(0<result['threshold']<1)

    def test_new_check_frames_are_not_training_calibration_or_previously_scored(self):
        prior = read(PRIOR/'manifest.json')
        rows = choose(read(SOURCE/'curated-label-inventory.json')['rows'],prior)
        used = {r['id'] for r in prior['rows']}
        self.assertTrue(rows)
        self.assertTrue(all(r['id'] not in used and prior['group_splits'][r['recording_group']]=='test' for r in rows))
        self.assertTrue(all(r['mechanical_health_label']=='unknown' for r in rows))


if __name__=='__main__':
    unittest.main()
