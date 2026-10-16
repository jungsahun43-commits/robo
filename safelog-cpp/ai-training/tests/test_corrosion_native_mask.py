import unittest
import numpy as np
import torch
from safelog_ai.native_corrosion_grid import grid
from safelog_ai.corrosion_native_mask import coverage_map,supervised_weights,native_loss,rgb_tile,add_probability


class NativeSparseTest(unittest.TestCase):
    def test_overlap_preserves_one_photo_class_budgets_and_loss(self):
        shape=(617,1033);truth=np.full(shape,255,dtype=np.uint8)
        truth[380:400,380:400]=1;truth[590:610,1000:1020]=0
        coverage=coverage_map((1033,617));counts=[int((truth==k).sum()) for k in (0,1)]
        budget=np.zeros(2);loss=0.
        for b in grid((1033,617)):
            t,w=supervised_weights(truth,coverage,b,counts)
            budget+=[w[t==k].sum() for k in (0,1)]
            loss+=float(native_loss(torch.zeros((1,512,512)),torch.from_numpy(t)[None],torch.from_numpy(w)[None]))
        np.testing.assert_allclose(budget,[.5,.5],atol=1e-6)
        self.assertAlmostEqual(loss,np.log(2),places=6)

    def test_unknown_padding_has_no_gradient_and_single_class_tile_is_valid(self):
        truth=np.full((100,200),255,dtype=np.uint8);truth[1:3,1:3]=1
        t,w=supervised_weights(truth,np.ones_like(truth),grid((200,100))[0],[4,4])
        logits=torch.zeros((1,512,512),requires_grad=True)
        native_loss(logits,torch.from_numpy(t)[None],torch.from_numpy(w)[None]).backward()
        self.assertEqual(float(logits.grad[0][torch.from_numpy(t)==255].abs().sum()),0.)
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        self.assertTrue(np.all(w[100:]==0))

    def test_native_rgb_and_probability_coordinates_survive_overlap_and_thin_axis(self):
        h,w=132,1348;rgb=np.zeros((h,w,3),dtype=np.uint8)
        rgb[:,:,0]=np.arange(w,dtype=np.uint16)[None,:]%256
        expected=np.tile(np.linspace(0,1,w,dtype=np.float32),(h,1));total=np.zeros((h,w),np.float32)
        for b in reversed(grid((w,h))):
            tile=rgb_tile(rgb,b);x,y=b['left'],b['top']
            np.testing.assert_array_equal(tile[:b['height'],:b['width']],rgb[y:y+b['height'],x:x+b['width']])
            np.testing.assert_array_equal(tile[h,0],[124,116,104])
            p=np.zeros((512,512),np.float32);p[:h,:b['width']]=expected[:,x:x+b['width']]
            add_probability(total,p,b)
        np.testing.assert_allclose(total/coverage_map((w,h)),expected,atol=1e-7)


if __name__=='__main__':unittest.main()
