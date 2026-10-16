import unittest
import torch
from safelog_ai.corrosion_partial_mask import partial_loss,known_confusion,native_probability


class PartialMaskTest(unittest.TestCase):
    def test_unknown_has_no_gradient_or_background_count(self):
        z=torch.zeros(2,2,2,requires_grad=True)
        y=torch.tensor([[[0,1],[0,1]],[[1,255],[255,255]]])
        partial_loss(z,y,[True,False]).backward()
        self.assertTrue(torch.equal(z.grad[1][y[1]==255],torch.zeros(3)))
        self.assertEqual(known_confusion(torch.ones(2,2),y[1],.5).tolist(),[1,0,0,0])

    def test_fixed_foreground_background_budget_and_fullmask_dice(self):
        z=torch.zeros(2,2,2);y=torch.tensor([[[0,1],[0,1]],[[1,255],[255,255]]])
        expected=torch.log(torch.tensor(2.))+1-3/5
        self.assertAlmostEqual(float(partial_loss(z,y,[True,False])),float(expected),places=6)
        y[1,0,1]=0
        with self.assertRaises(ValueError):partial_loss(z,y,[True,False])

    def test_native_crop_discards_padding_without_bilinear_truth(self):
        p=torch.zeros(512,512);p[128:384]=1
        value=native_probability(p,{'native_size':[4,2],'resized_size':[512,256],'left':0,'top':128})
        self.assertTrue(torch.equal(value,torch.ones(2,4)))


if __name__=='__main__':unittest.main()
