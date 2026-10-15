import unittest
import torch
from safelog_ai.corrosion_review_gate import mask_photo_score,corroborated,passes_guard


class ReviewGateTests(unittest.TestCase):
    def test_strict_conjunction_preserves_parent_negatives(self):
        parent=torch.tensor([.5,.6,.6,.2]);mask=torch.tensor([.9,.7,.71,1.])
        self.assertEqual(corroborated(parent,.5,mask,.7).tolist(),[False,False,True,False])

    def test_discrete_guard_rejects_small_fp_change_and_excess_misses(self):
        base={'fpr':6/31,'fnr':0.}
        make=lambda fp,fn:{'fpr':fp/31,'fnr':fn/33,'positive_photos':33,'negative_photos':31}
        self.assertTrue(passes_guard(base,make(4,1),{'0':31,'1':33},12))
        self.assertFalse(passes_guard(base,make(5,0),{'0':31,'1':33},12))
        self.assertFalse(passes_guard(base,make(4,2),{'0':31,'1':33},12))
        self.assertFalse(passes_guard(base,make(4,1),{'0':31,'1':33},0))

    def test_top_two_percent_rounding(self):
        value=torch.zeros(1,512,512);value.flatten(1)[:,:4000]=1
        self.assertAlmostEqual(float(mask_photo_score(value)),4000/5243,places=6)
        with self.assertRaises(ValueError):mask_photo_score(torch.zeros(1,256,256))


if __name__=='__main__':unittest.main()
