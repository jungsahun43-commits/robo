import inspect
import unittest
import torch
from torch.nn import functional as F
from safelog_ai.conditioned_surface import ConditionedMaskHead
from safelog_ai.fine_target_surface import FineTargetHead, FineTargetMain, initialize_fine_head, trainable_fine_head
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
from safelog_ai.magnetic_surface import RealMaskMain


class FineTargetTests(unittest.TestCase):
    def test_zero_fine_inputs_then_learning_preserves_high_projection(self):
        torch.manual_seed(2020)
        old = ConditionedMaskHead().eval()
        head = FineTargetHead().eval()
        initialize_fine_head(head, old.state_dict())
        low = torch.randn(2, 40, 8, 8).half()
        high = torch.randn(2, 960, 4, 4).half()
        residual = torch.rand(2, 1, 4, 4).half()
        fine = torch.randn(2, 24, 16, 16).half()
        up = lambda value: F.interpolate(value, (16, 16), mode='bilinear', align_corners=False)
        expected = old.fusion(torch.cat((up(low.float()), up(old.high_projection(high.float())), up(residual.float())), 1))
        self.assertTrue(torch.allclose(head(low, high, residual, fine), expected, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(head(low, high, residual, fine), head(low, high, residual, fine * 2)))
        fixed = {name: value.clone() for name, value in head.high_projection.state_dict().items()}
        original_fine = head.fine_projection[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(trainable_fine_head(head), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            head(low, high, residual, fine).square().mean().backward()
            optimizer.step()
        self.assertTrue(all(torch.equal(value, head.high_projection.state_dict()[name]) for name, value in fixed.items()))
        self.assertGreater(float(head.fusion[0].weight[:, 73:].detach().abs().sum()), 0)
        self.assertFalse(torch.equal(original_fine, head.fine_projection[0].weight))

    def test_misaligned_fine_rejected(self):
        with self.assertRaises(ValueError):
            FineTargetHead()(torch.zeros(1, 40, 8, 8), torch.zeros(1, 960, 4, 4),
                             torch.zeros(1, 1, 4, 4), torch.zeros(1, 24, 8, 8))

    def test_original_runtime_dispatch_preserved(self):
        self.assertIs(FineTargetMain.forward_surface, IndustrialSurfaceMain.forward_surface)
        self.assertIs(FineTargetMain.forward_real_mask, RealMaskMain.forward_surface)
        inspect.signature(FineTargetMain.forward_surface).bind(object(), object(), object(), 'mpdd_bracket_white')
        inspect.signature(FineTargetMain.forward_fine_target_mask).bind(object(), object(), 'mpdd_tubes')


if __name__ == '__main__':
    unittest.main()
