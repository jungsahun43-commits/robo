import unittest
import torch
from safelog_ai.belt_normal_learning import normal_roi_loss


class NormalLearningTests(unittest.TestCase):
    def test_highest_roi_candidates_are_penalized_without_dense_truth(self):
        values=torch.full((2,1,10,10),-2.); values[:,:,0:3]=2.; values.requires_grad_(True)
        loss=normal_roi_loss(values); loss.backward()
        self.assertGreater(float(loss.detach()),float(normal_roi_loss(torch.full_like(values,-4.)).detach()))
        self.assertTrue((values.grad>=0).all()); self.assertGreater(float(values.grad[:,:,0:3].sum()),0)
        self.assertLessEqual(int((values.grad[0]>0).sum()),64)

    def test_invalid_output_is_rejected(self):
        for shape in [(2,10,10),(0,1,10,10),(2,7,10,10)]:
            with self.assertRaises(ValueError): normal_roi_loss(torch.zeros(shape))


if __name__=='__main__': unittest.main()
