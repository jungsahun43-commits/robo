import unittest
import torch
from torch.nn import functional as F


class LinearRegularizationTests(unittest.TestCase):
    def test_explicit_penalty_gradient_does_not_penalize_bias(self):
        head=torch.nn.Linear(3072,1)
        with torch.no_grad():head.weight.fill_(.02);head.bias.zero_()
        x=torch.zeros(2,3072);y=torch.tensor([0.,1.])
        loss=F.binary_cross_entropy_with_logits(head(x).flatten(),y)+head.weight.square().sum();loss.backward()
        self.assertTrue(torch.allclose(head.weight.grad,2*head.weight))
        self.assertAlmostEqual(float(head.bias.grad),0.)

    def test_zero_initialization_is_finite_half_probability(self):
        head=torch.nn.Linear(3072,1);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        self.assertTrue(torch.equal(head(torch.randn(4,3072)).sigmoid(),torch.full((4,1),.5)))


if __name__=='__main__':unittest.main()
