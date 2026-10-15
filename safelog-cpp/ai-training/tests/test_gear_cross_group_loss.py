import math
import unittest
import torch
from safelog_ai.gear_cross_group_loss import cross_group_loss


class GearCrossGroupLossTests(unittest.TestCase):
    def test_separated_classes_have_lower_loss_and_finite_gradient(self):
        good=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True)
        bad=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        labels=torch.tensor([0,0,1,1]);groups=['a','b','c','d']
        loss=cross_group_loss(good,labels,groups)
        self.assertLess(float(loss.detach()),float(cross_group_loss(bad,labels,groups)))
        loss.backward();self.assertTrue(torch.isfinite(good.grad).all())

    def test_same_group_negative_excluded(self):
        features=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        actual=float(cross_group_loss(features,torch.tensor([0,0,1,1]),['a','b','a','b']))
        self.assertAlmostEqual(actual,math.log1p(math.exp(-5)),places=6)

    def test_missing_cross_group_positive_rejected(self):
        with self.assertRaisesRegex(ValueError,'another group'):
            cross_group_loss(torch.ones(2,2),torch.tensor([0,1]),['a','b'])


if __name__=='__main__':unittest.main()
