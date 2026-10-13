import unittest
import torch
from safelog_ai.belt_positive_retention import pixel_retention


class PositiveRetentionTests(unittest.TestCase):
    def test_teacher_detached_and_valid_pixels_only(self):
        student=torch.tensor([[[[1.,-1.,4.]]]],requires_grad=True)
        teacher=torch.tensor([[[[2.,-2.,-4.]]]],requires_grad=True)
        valid=torch.tensor([[[[1.,1.,0.]]]])
        loss=pixel_retention(student,teacher,valid); self.assertGreater(float(loss.detach()),0)
        loss.backward(); self.assertIsNone(teacher.grad); self.assertEqual(float(student.grad[0,0,0,2]),0)
        self.assertLess(float(student.grad[0,0,0,0]),0); self.assertGreater(float(student.grad[0,0,0,1]),0)

    def test_equal_or_unknown_scores_have_zero_loss(self):
        s=torch.tensor([[[[2.,-3.]]]],requires_grad=True)
        self.assertEqual(float(pixel_retention(s,s.clone(),torch.ones_like(s)).detach()),0)
        loss=pixel_retention(s,-s,torch.zeros_like(s)); loss.backward()
        self.assertTrue(torch.equal(s.grad,torch.zeros_like(s)))
        with self.assertRaises(ValueError): pixel_retention(s,s,torch.full_like(s,.5))


if __name__=='__main__': unittest.main()
