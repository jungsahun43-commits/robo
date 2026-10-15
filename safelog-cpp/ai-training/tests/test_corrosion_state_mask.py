import unittest
import numpy as np
from PIL import Image
import torch
from safelog_ai.corrosion_state_mask import state_view,native_probabilities,state_loss,rates


class CorrosionStateTests(unittest.TestCase):
    def test_letterbox_ignore_and_native_coordinates(self):
        y=np.full((10,20),2,dtype=np.uint8);rgb=Image.new('RGB',(20,10))
        view,target,g=state_view(rgb,y)
        self.assertEqual(g,{'native_size':[20,10],'resized_size':[512,256],'left':0,'top':128})
        self.assertTrue(np.all(target[:128]==255));self.assertTrue(np.all(target[128:384]==2))
        p=torch.zeros(4,512,512);p[2]=1
        self.assertEqual(tuple(native_probabilities(p,g).shape),(4,10,20))

    def test_ignore_pixels_have_zero_gradient_and_no_fake_class(self):
        x=torch.zeros(1,4,2,2,requires_grad=True);target=torch.tensor([[[0,1],[2,255]]])
        state_loss(x,target).sum().backward();self.assertTrue(torch.equal(x.grad[0,:,1,1],torch.zeros(4)))
        with self.assertRaises(ValueError):state_loss(x,torch.tensor([[[0,1],[2,4]]]))

    def test_state_metrics_do_not_call_minimal_corrosion_zero_rust(self):
        r=rates([[8,1,1,0],[2,4,0,0],[0,1,4,0],[1,0,0,2]])
        self.assertAlmostEqual(r['at_least_Fair_pixel_FPR'],.2)
        self.assertAlmostEqual(r['at_least_Fair_pixel_FNR'],3/14)
        self.assertEqual(len(r['per_class_IoU']),4)


if __name__=='__main__':unittest.main()
