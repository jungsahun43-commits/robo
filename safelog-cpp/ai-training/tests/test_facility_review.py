import unittest
import numpy as np
from scripts.analyze_facility_review import choose_tail,tail_metrics,joint_metrics


class ReviewDiagnosticTests(unittest.TestCase):
    def test_joint_photo_error_counts_wrong_photo_once_and_preserves_review(self):
        targets=[[1,0],[1,1],[0,1],[1,0],[-1,0]]
        scores=[[.9,.1],[.9,.9],[.9,.1],[.5,.1],[.9,.1]]
        result=joint_metrics(targets,scores,[(.2,.8),(.2,.8)])
        self.assertEqual(result['accepted'],3)
        self.assertEqual(result['review_required'],2)
        self.assertEqual(result['accepted_wrong_photos'],1)
        self.assertEqual(result['coverage'],.6)

    def test_abstained_and_unknown_cases_are_not_counted_correct(self):
        result=tail_metrics([1,0,1,-1],[.9,.1,.5,.99],.8,True)
        self.assertEqual(result['accepted'],1)
        self.assertEqual(result['known_photos'],3)
        self.assertEqual(result['errors'],0)

    def test_small_zero_error_sample_does_not_support_wilson_target(self):
        domains={'a':(np.ones(20),np.full(20,.99)),'b':(np.ones(20),np.full(20,.99))}
        self.assertTrue(choose_tail(domains,.5,True,False)['supported'])
        self.assertFalse(choose_tail(domains,.5,True,True)['supported'])

    def test_common_tail_rule_must_work_in_each_domain(self):
        domains={'a':(np.ones(100),np.full(100,.99)),'b':(np.zeros(100),np.full(100,.99))}
        result=choose_tail(domains,.5,True,False)
        self.assertFalse(result['supported'])
        self.assertTrue(all(m['accepted']==0 for m in result['domains'].values()))
