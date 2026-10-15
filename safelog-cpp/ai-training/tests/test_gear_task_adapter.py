import unittest
import torch
from torch import nn
from safelog_ai.gear_task_adapter import prefix_to_fine,adapted_descriptor
from safelog_ai.gear_adapter_probe import ZeroFineAdapterProbe,tail_from_fine
from safelog_ai.gearbox_supervised import gearbox_descriptor


class Trap(nn.Module):
    def forward(self,value):raise RuntimeError('Tail should not run in prefix')


class GearTaskAdapterTests(unittest.TestCase):
    def test_prefix_stops_at_stage3(self):
        backbone=nn.ModuleDict({'0':nn.Conv2d(3,24,1),'3':nn.Identity(),'4':Trap()})
        self.assertEqual(tuple(prefix_to_fine(backbone,torch.ones(1,3,8,8)).shape),(1,24,8,8))

    def test_zero_adapter_matches_declared_half_precision_boundary(self):
        backbone=nn.ModuleDict({'3':Trap(),'4':nn.Conv2d(24,40,1),'5':nn.Conv2d(40,960,1)}).eval().requires_grad_(False)
        backbone.return_layers={'4':'low','5':'high'}
        fine=torch.randn(2,24,8,8)
        actual=adapted_descriptor(backbone,ZeroFineAdapterProbe(),fine)
        expected=gearbox_descriptor(tail_from_fine(backbone,fine.half().float()))
        self.assertTrue(torch.equal(actual,expected))

    def test_trainable_adapter_receives_gradient_from_pooled_frozen_tail(self):
        backbone=nn.ModuleDict({'3':Trap(),'4':nn.Conv2d(24,40,1),'5':nn.Conv2d(40,960,1)}).eval().requires_grad_(False)
        backbone.return_layers={'4':'low','5':'high'};adapter=ZeroFineAdapterProbe()
        adapted_descriptor(backbone,adapter,torch.randn(2,24,8,8)).square().mean().backward()
        self.assertGreater(float(adapter[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(v.grad is None for v in backbone.parameters()))


if __name__=='__main__':unittest.main()
