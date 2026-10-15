import unittest
import torch
from torch import nn
from safelog_ai.gear_adapter_probe import ZeroFineAdapterProbe,tail_from_fine


class Trap(nn.Module):
    def forward(self,value):raise RuntimeError('Prefix must not be rerun for cachedfine')


class GearAdapterProbeTests(unittest.TestCase):
    def backbone(self):
        value=nn.ModuleDict({'0':Trap(),'3':Trap(),'4':nn.Conv2d(24,40,1),'5':nn.Conv2d(40,960,1)})
        value.return_layers={'4':'low','5':'high'}
        return value.eval().requires_grad_(False)

    def test_zero_adapter_reuses_original_modules_and_is_exact(self):
        backbone=self.backbone();fine=torch.randn(2,24,8,8);adapter=ZeroFineAdapterProbe()
        direct=tail_from_fine(backbone,fine);adapted=tail_from_fine(backbone,adapter(fine))
        self.assertTrue(all(torch.equal(direct[k],adapted[k]) for k in ('low','high','fine')))
        self.assertEqual(backbone.return_layers,{'4':'low','5':'high'})

    def test_gradient_reaches_adapter_through_frozen_tail(self):
        backbone=self.backbone();adapter=ZeroFineAdapterProbe();out=tail_from_fine(backbone,adapter(torch.randn(2,24,8,8)))
        out['high'].square().mean().backward()
        self.assertGreater(float(adapter[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))


if __name__=='__main__':unittest.main()
