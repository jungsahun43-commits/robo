import unittest
import torch
from PIL import Image
from safelog_ai.neighbor_surface import NeighborHead,neighbor_view


class NeighborSurfaceTests(unittest.TestCase):
    def test_center_is_absent_in_feature_ring_not_claimed_in_RGB(self):
        torch.set_num_threads(4);head=NeighborHead()
        x=torch.zeros(1,1000,5,5,requires_grad=True)
        y=head.surrounding(x);y[0,0,2,2].backward()
        self.assertEqual(float(x.grad[0,0,2,2]),0)
        self.assertAlmostEqual(float(x.grad[0,0,1,2]),1/8)
        self.assertEqual(float(x.grad[0,1].abs().sum()),0)

    def test_evidence_consistency_and_finite_normalization(self):
        torch.manual_seed(15);head=NeighborHead().eval();x=torch.rand(2,1000,5,5)
        with torch.inference_mode():m,s=head.evidence(x)
        self.assertEqual(tuple(m.shape),(2,5,5));self.assertTrue(torch.isfinite(m).all())
        self.assertTrue(torch.equal(s,m.flatten(1).amax(1)))
        with self.assertRaises(ValueError):head.standardize(torch.zeros(1,999,4,4))

    def test_aspect_letterbox_and_manual_small_photo(self):
        v,g=neighbor_view(Image.new('RGB',(200,100),'white'))
        self.assertEqual(v.size,(512,512));self.assertEqual(g['offset'],[0,128])
        with self.assertRaises(ValueError):neighbor_view(Image.new('RGB',(63,100)))


if __name__=='__main__':unittest.main()
