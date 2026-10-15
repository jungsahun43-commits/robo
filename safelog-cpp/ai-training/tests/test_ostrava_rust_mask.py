import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.ostrava_rust_mask import source_view,native_probability,pixel_loss,confusion_at,RustMaskHead,OstravaRustMain
from safelog_ai.insplad_suspension import InspladSuspensionMain


class RustMaskTests(unittest.TestCase):
    def test_actual169_mask_and_original_coordinate_roundtrip(self):
        truth=np.zeros((16,16),dtype=np.uint8);truth[:,:8]=169
        rgb=Image.new('RGB',(16,16));mask=Image.fromarray(truth)
        _,target=source_view(rgb,mask)
        self.assertTrue(target[:,:256].all() and not target[:,256:].any())
        restored=native_probability(target.float(),(16,16))>.5
        self.assertTrue(torch.equal(restored,torch.from_numpy(truth>0)))
        with self.assertRaises(ValueError):source_view(rgb,Image.new('L',(15,16)))

    def test_pixel_loss_gradients_and_strict_threshold_ties(self):
        logits=torch.zeros(2,4,4,requires_grad=True);target=torch.zeros(2,4,4,dtype=torch.bool);target[0,:,:2]=True
        loss=pixel_loss(logits,target);loss.mean().backward()
        self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[0,:,:2]<0).all() and (logits.grad[1]>0).all())
        values=torch.tensor([[.5,.6],[.4,.5]]);truth=torch.tensor([[True,True],[False,False]])
        self.assertEqual(confusion_at(values,truth,[.5]).tolist(),[[1,1,0,2]])
        with self.assertRaises(ValueError):pixel_loss(logits,target[0])

    def test_existing_methods_inherited_and_fresh_parameter_scope(self):
        for name in ('forward_surface','forward_real_mask','forward_insplad_upper_shackle','forward_insplad_glass','forward_insplad_suspension'):
            self.assertIs(getattr(OstravaRustMain,name),getattr(InspladSuspensionMain,name))
        head=RustMaskHead();self.assertEqual(sum(p.numel() for p in head.parameters()),100593)
        self.assertTrue(torch.equal(head.fusion[-1].weight,torch.zeros_like(head.fusion[-1].weight)))
        self.assertTrue(all(not value.requires_grad for value in head.buffers()))


if __name__=='__main__':unittest.main()
