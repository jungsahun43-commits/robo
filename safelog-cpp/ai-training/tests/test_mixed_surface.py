import inspect
import unittest
import torch
from safelog_ai.mixed_surface import MixedSurfaceMain, mixed_logits, freeze_projection
from safelog_ai.ksdd_supervised import SurfaceMaskHead, mask_loss
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
from safelog_ai.magnetic_surface import RealMaskMain


class MixedSurfaceTests(unittest.TestCase):
    def test_both_source_coordinates_and_valid_padding(self):
        torch.manual_seed(17)
        head = SurfaceMaskHead().eval()
        for size, low_size, high_size in [((512,512),(64,64),(32,32)), ((704,256),(88,32),(44,16))]:
            low = torch.randn(1,40,*low_size).half()
            high = torch.randn(1,960,*high_size).half()
            output = mixed_logits(head,low,high,size)
            self.assertEqual(tuple(output.shape),(1,1,*size))
            valid = torch.ones_like(output,dtype=torch.bool); valid[..., :8, :] = False
            truth = torch.zeros_like(output); truth[...,100:110,100:110]=1
            other = truth.clone(); other[..., :8, :] = 1
            # Unknown padding must not become a training target or a photo label.
            self.assertTrue(torch.isfinite(mask_loss(output,truth,valid)))
            self.assertAlmostEqual(float(mask_loss(output,truth,valid).detach()),float(mask_loss(output,other,valid).detach()),places=6)

    def test_high_projection_frozen_fusion_updates(self):
        torch.manual_seed(17); head = SurfaceMaskHead()
        params = freeze_projection(head)
        before = {n:v.clone() for n,v in head.state_dict().items()}
        opt = torch.optim.AdamW(params,lr=1e-4)
        out = mixed_logits(head,torch.randn(2,40,8,8),torch.randn(2,960,4,4),(64,64))
        out.square().mean().backward(); opt.step()
        self.assertTrue(all(torch.equal(v,head.state_dict()[n]) for n,v in before.items() if n.startswith('high_projection.')))
        self.assertTrue(any(not torch.equal(v,head.state_dict()[n]) for n,v in before.items() if n.startswith('fusion.')))

    def test_existing_dispatch_and_new_distinct_entry(self):
        self.assertIs(MixedSurfaceMain.forward_surface,IndustrialSurfaceMain.forward_surface)
        self.assertIs(MixedSurfaceMain.forward_real_mask,RealMaskMain.forward_surface)
        inspect.signature(MixedSurfaceMain.forward_surface).bind(object(),object(),object(),'mpdd_bracket_brown')
        inspect.signature(MixedSurfaceMain.forward_mixed_mask).bind(object(),object())


if __name__ == '__main__': unittest.main()
