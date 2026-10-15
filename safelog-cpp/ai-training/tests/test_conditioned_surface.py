import inspect
import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.conditioned_surface import ConditionedMaskHead,ConditionedSurfaceMain,warm_start
from safelog_ai.ksdd_supervised import SurfaceMaskHead
from safelog_ai.mixed_surface import freeze_projection
from safelog_ai.target_synthetic_r1 import synthesize,prepare_background
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
from safelog_ai.magnetic_surface import RealMaskMain


class ConditionedTests(unittest.TestCase):
    def test_zero_channel_initialization_and_frozen_projection(self):
        torch.manual_seed(19);old=SurfaceMaskHead().eval();head=ConditionedMaskHead().eval();warm_start(head,old.state_dict())
        low=torch.randn(2,40,8,8).half();high=torch.randn(2,960,4,4).half();residual=torch.randn(2,1,4,4).half()
        self.assertTrue(torch.allclose(head(low,high,residual),old(low.float(),high.float()),atol=1e-6,rtol=1e-5))
        before={n:t.clone() for n,t in head.state_dict().items()};params=freeze_projection(head)
        opt=torch.optim.AdamW(params,lr=1e-4);head(low,high,residual).square().mean().backward();opt.step()
        self.assertTrue(all(torch.equal(t,head.state_dict()[n]) for n,t in before.items() if n.startswith('high_projection.')))
        self.assertTrue(bool(head.fusion[0].weight[:,72:].abs().sum()>0))

    def test_procedural_label_exact_and_repeatable(self):
        a=np.full((512,512,3),(30,140,30),dtype='uint8');a[120:390,120:390]=180
        rgb=Image.fromarray(a);s=np.full((256,256),160,dtype='uint8');s[120:125,40:210]=20
        m=np.zeros((256,256),dtype='uint8');m[120:125,40:210]=255
        result,mask,meta=synthesize(rgb,Image.fromarray(s).convert('RGB'),Image.fromarray(m),1919)
        again,_,_=synthesize(rgb,Image.fromarray(s).convert('RGB'),Image.fromarray(m),1919)
        base=prepare_background(rgb)[0];changed=np.any(np.asarray(result)!=base,axis=2)
        self.assertTrue(np.array_equal(changed,np.asarray(mask)>0));self.assertTrue(np.array_equal(result,again))
        self.assertGreaterEqual(int(changed.sum()),32);self.assertGreaterEqual(meta['mean_absolute_RGB_change'],12)
        self.assertEqual(meta['new_natural_fault_labels'],0)

    def test_old_dispatch_signature_preserved(self):
        self.assertIs(ConditionedSurfaceMain.forward_surface,IndustrialSurfaceMain.forward_surface)
        self.assertIs(ConditionedSurfaceMain.forward_real_mask,RealMaskMain.forward_surface)
        inspect.signature(ConditionedSurfaceMain.forward_surface).bind(object(),object(),object(),'mpdd_bracket_white')
        inspect.signature(ConditionedSurfaceMain.forward_conditioned_mask).bind(object(),object(),'mpdd_tubes')


if __name__=='__main__':unittest.main()
