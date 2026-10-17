import unittest
import numpy as np
from safelog_ai.corrosion_fit_diagnosis import summary,balanced_difference


class FitDiagnosisTest(unittest.TestCase):
    def test_unknown_does_not_become_negative_or_change_balanced_loss(self):
        gt=np.array([[0,1,255]],np.uint8);a=np.array([[.1,.8,.01]],np.float32);b=a.copy();b[0,2]=.99
        self.assertEqual(summary(a,gt),summary(b,gt))
        self.assertEqual(balanced_difference(a,b,gt),0.)
        self.assertAlmostEqual(summary(a,gt)['balanced_known_BCE'],(-np.log(.9)-np.log(.8))/2,places=6)

    def test_full_negative_uses_its_only_present_class_and_empty_gt_rejected(self):
        gt=np.zeros((2,3),np.uint8);p=np.full((2,3),.2,np.float32)
        self.assertAlmostEqual(summary(p,gt)['balanced_known_BCE'],-np.log(.8),places=6)
        self.assertEqual(set(summary(p,gt)['classes']),{'0'})
        with self.assertRaises(ValueError):summary(p,np.full_like(gt,255))
