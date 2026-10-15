import unittest
import torch
from torch.nn import functional as F
from safelog_ai.gear_spatial_mil import mil_logit, GearSpatialHead


class GearSpatialMILTests(unittest.TestCase):
    def test_top_fraction_does_not_depend_on_location_order(self):
        value=torch.arange(100,dtype=torch.float32).reshape(1,1,10,10)
        self.assertAlmostEqual(float(mil_logit(value)),94.5)
        self.assertTrue(torch.equal(mil_logit(value),mil_logit(value.flip((-1,-2)))))

    def test_patch_loss_backpropagates_without_any_pixel_target(self):
        head=GearSpatialHead();x=torch.randn(3,24,8,8)
        loss=F.binary_cross_entropy_with_logits(mil_logit(head(x)),torch.tensor([0.,0.,1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(head[-1].weight.grad).all())
        self.assertGreater(float(head[-1].weight.grad.abs().sum()),0)

    def test_bad_or_nonfinite_map_rejected(self):
        for value in (torch.zeros(1,2,4,4),torch.full((1,1,4,4),float('nan'))):
            with self.assertRaises(ValueError):mil_logit(value)


if __name__=='__main__':unittest.main()
