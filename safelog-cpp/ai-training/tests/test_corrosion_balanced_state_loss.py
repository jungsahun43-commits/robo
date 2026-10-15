import unittest
import numpy as np
import torch
from safelog_ai.corrosion_state_mask import state_loss
from safelog_ai.corrosion_balanced_state_loss import coefficient_weights,weighted_state_loss


class BalancedStateLossTests(unittest.TestCase):
    def test_unit_weights_reproduce_original_objective(self):
        torch.manual_seed(0);x=torch.randn(2,4,3,4);y=torch.randint(0,4,(2,3,4));y[0,0,0]=255
        torch.testing.assert_close(weighted_state_loss(x,y,[1,1,1,1]),state_loss(x,y))

    def test_fit_budget_formula_and_absent_state_rejection(self):
        presence=np.array([[1,1,0,0],[1,1,1,0],[1,1,1,1],[1,0,0,0]],dtype=float)
        w=coefficient_weights(presence);budget=(presence/presence.sum(1,keepdims=True)).mean(0)
        np.testing.assert_allclose(w*budget,np.ones(4)/4)
        with self.assertRaises(ValueError):coefficient_weights(presence[:2])

    def test_rare_class_scale_and_ignore_gradient(self):
        x=torch.zeros(1,4,2,2,requires_grad=True);y=torch.tensor([[[3,3],[3,255]]])
        expected=4*state_loss(x,y);actual=weighted_state_loss(x,y,[1,1,1,4]);torch.testing.assert_close(actual,expected)
        actual.sum().backward();self.assertTrue(torch.equal(x.grad[0,:,1,1],torch.zeros(4)))


if __name__=='__main__':unittest.main()
