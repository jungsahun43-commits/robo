import unittest
import torch
from safelog_ai.metal_reference import nearest_cosine,normal_budget_threshold


class MetalReferenceTests(unittest.TestCase):
    def test_seen_normal_and_unseen_defect_have_different_distances(self):
        normal=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        query=torch.tensor([[1.,0.,0.],[0.,0.,1.]])
        scores=nearest_cosine(query,normal)
        self.assertAlmostEqual(float(scores[0]),0,places=6);self.assertAlmostEqual(float(scores[1]),1,places=6)

    def test_chunked_search_equals_full_matrix_and_rescaling_is_safe(self):
        torch.manual_seed(13);q=torch.randn(17,40);r=torch.randn(31,40)
        a=nearest_cosine(q,r,3,7);b=nearest_cosine(q*4,r*3,100,100)
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_empty_nonfinite_or_zero_bank_is_rejected(self):
        for r in [torch.empty(0,3),torch.zeros(2,3),torch.full((2,3),float('nan'))]:
            with self.assertRaises(ValueError):nearest_cosine(torch.ones(1,3),r)

    def test_twenty_percent_budget_uses_normal_only_order_statistic(self):
        threshold,rank=normal_budget_threshold(list(range(20)),.20)
        self.assertEqual(rank,17);self.assertEqual(threshold,16)
        with self.assertRaises(ValueError):normal_budget_threshold(list(range(19)),.20)


if __name__=='__main__':torch.set_num_threads(4);unittest.main()
