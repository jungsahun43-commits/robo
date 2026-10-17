import unittest
import numpy as np
import torch
from safelog_ai.corrosion_native_mask import coverage_map
from safelog_ai.native_corrosion_grid import grid
from safelog_ai.corrosion_teacher_retention import retained_weights,teacher_tile,teacher_kl


class TeacherRetentionTest(unittest.TestCase):
    def test_matching_teacher_has_zero_divergence_without_teacher_gradient(self):
        teacher=torch.tensor([[[.2,.8]]],requires_grad=True);logits=torch.logit(teacher.detach()).requires_grad_()
        loss=teacher_kl(logits,teacher,torch.ones_like(teacher)/2);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),0.,places=6);self.assertIsNone(teacher.grad)
        self.assertLess(float(logits.grad.abs().max()),1e-6)

    def test_overlap_class_budgets_and_fullnegative_are_preserved(self):
        gt=np.zeros((617,1033),np.uint8);gt[380:400,380:400]=1;cov=coverage_map((1033,617));budget=np.zeros(2)
        for b in grid((1033,617)):
            t,w=retained_weights(gt,cov,b);budget += [w[t==k].sum() for k in (0,1)]
        np.testing.assert_allclose(budget,[.5,.5],atol=1e-6)
        gt[:]=0
        self.assertAlmostEqual(sum(float(retained_weights(gt,cov,b)[1].sum()) for b in grid((1033,617))),1.,places=6)

    def test_padding_unknown_do_not_generate_distillation_gradient(self):
        gt=np.full((100,200),255,np.uint8);gt[1:3,1:3]=0;box=grid((200,100))[0]
        target,weight=retained_weights(gt,np.ones_like(gt),box);teacher=teacher_tile(np.full_like(gt,.9,dtype=np.float32),box)
        logits=torch.zeros((1,512,512),requires_grad=True)
        teacher_kl(logits,torch.from_numpy(teacher)[None],torch.from_numpy(weight)[None]).backward()
        self.assertEqual(float(logits.grad[0][torch.from_numpy(target)==255].abs().sum()),0.)
        self.assertGreater(float(logits.grad.abs().sum()),0.)
