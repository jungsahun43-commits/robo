import unittest,torch
from safelog_ai.corrosion_background_mask import background_loss


class BackgroundMaskTest(unittest.TestCase):
    def test_both_classes_balanced_independently_of_region_size(self):
        z=torch.zeros(1,4,4);t=torch.zeros(1,4,4,dtype=torch.long);t[0,0,0]=1
        self.assertAlmostEqual(float(background_loss(z,t)),float(torch.log(torch.tensor(2.))),places=6)

    def test_unknown_has_no_loss_gradient(self):
        z=torch.zeros(1,2,2,requires_grad=True);t=torch.tensor([[[1,0],[255,255]]])
        background_loss(z,t).mean().backward();self.assertTrue(torch.equal(z.grad[0,1],torch.zeros(2)))
        self.assertAlmostEqual(float(z.grad[0,0,0]),-.25,places=6);self.assertAlmostEqual(float(z.grad[0,0,1]),.25,places=6)

    def test_missing_class_is_not_filled_or_treated_as_normal(self):
        with self.assertRaises(ValueError):background_loss(torch.zeros(1,2,2),torch.full((1,2,2),255))


if __name__=='__main__':unittest.main()
