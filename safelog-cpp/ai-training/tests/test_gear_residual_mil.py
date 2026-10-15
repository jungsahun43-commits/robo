import unittest
import torch
from safelog_ai.gear_residual_mil import bounded_correction
from safelog_ai.gear_spatial_mil import GearSpatialHead,mil_logit


class GearResidualTests(unittest.TestCase):
    def test_zero_output_preserves_global_logit_exactly(self):
        head=GearSpatialHead();global_logits=torch.tensor([-.8,.4,.9])
        result=global_logits+bounded_correction(mil_logit(head(torch.randn(3,24,8,8))))
        self.assertTrue(torch.equal(result,global_logits))

    def test_correction_bound_and_gradient(self):
        values=torch.tensor([-100.,0.,100.],requires_grad=True)
        result=bounded_correction(values);self.assertLessEqual(float(result.detach().abs().max()),.5)
        result.sum().backward();self.assertTrue(torch.isfinite(values.grad).all())
        self.assertAlmostEqual(float(values.grad[1]),.5)


if __name__=='__main__':unittest.main()
