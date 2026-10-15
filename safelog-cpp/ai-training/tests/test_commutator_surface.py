import inspect
import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.commutator_surface import commutator_view,fine_mask_loss,CommutatorSurfaceMain,restore_map
from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
from safelog_ai.magnetic_surface import RealMaskMain


class CommutatorSurfaceTests(unittest.TestCase):
    def test_native_view_and_original_coordinates(self):
        rgb=Image.new('RGB',(500,1260),(80,80,80));a=np.zeros((1260,500),dtype=np.uint8);a[600:603,150:450]=255
        view,mask,valid,geometry=commutator_view(rgb,Image.fromarray(a))
        self.assertEqual(view.size,(512,1408));self.assertEqual(tuple(mask.shape),(1408,512));self.assertTrue(mask.any())
        restored=restore_map(mask[None].float(),geometry)[0]>.5
        self.assertGreater(float((restored & torch.from_numpy(a>0)).sum())/int((a>0).sum()),.90)
        self.assertFalse(bool(mask[~valid].any()))

    def test_padding_does_not_set_labels_or_affect_loss(self):
        logits=torch.zeros(2,1,64,64);valid=torch.ones_like(logits,dtype=torch.bool);valid[...,:10,:]=False
        mask=torch.zeros_like(logits);mask[0,0,30:32,20:30]=1
        unknown=mask.clone();unknown[...,:10,:]=1
        self.assertTrue(torch.equal(fine_mask_loss(logits,mask,valid,100),fine_mask_loss(logits,unknown,valid,100)))

    def test_old_dispatch_preserved(self):
        self.assertIs(CommutatorSurfaceMain.forward_surface,IndustrialSurfaceMain.forward_surface)
        self.assertIs(CommutatorSurfaceMain.forward_real_mask,RealMaskMain.forward_surface)
        inspect.signature(CommutatorSurfaceMain.forward_surface).bind(object(),object(),object(),'mpdd_bracket_brown')
        inspect.signature(CommutatorSurfaceMain.forward_commutator_mask).bind(object(),object())


if __name__=='__main__':unittest.main()
