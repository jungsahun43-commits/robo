"""Scientific loss contracts: unknowns, easy negatives and retained gradients."""
import unittest
import torch
from torch.nn import functional as F
from safelog_ai.primary_asymmetric_loss import primary_asymmetric_photo_loss as loss
from scripts.train_facility_target import masked_focal

class PrimaryAsymmetricTests(unittest.TestCase):
    def values(self):
        x=torch.tensor([[-2.,-1.,0.,1.,2.,-3.,3.],[2.,1.,-1.,0.,3.,-2.,-3.]],requires_grad=True)
        y=torch.tensor([[1.,0.,1.,0.,1.,0.,1.],[0.,1.,0.,1.,0.,1.,0.]])
        return x,y,torch.ones_like(y),torch.tensor([1.,2.,3.,4.,5.,6.,.5])
    def test_unknown_has_no_value_or_gradient(self):
        x,y,k,w=self.values();k[:,0]=0;k[:,4]=0;y[:,0]=-1.;y[:,4]=-99.
        a=loss(x,y,k,w);a.backward();self.assertTrue(torch.equal(x.grad[:,[0,4]],torch.zeros(2,2)))
        alternate=y.clone();alternate[:,[0,4]]=37.;self.assertEqual(float(a.detach()),float(loss(x,alternate,k,w).detach()))
    def test_all_unknown_zero_connected_loss(self):
        x,y,k,w=self.values();k.zero_();y.fill_(-1);a=loss(x,y,k,w);a.backward()
        self.assertEqual(float(a.detach()),0.);self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))
    def test_other_five_equal_original_loss_and_gradient(self):
        x,y,k,w=self.values();k[:,:2]=0
        a=loss(x,y,k,w);ga=torch.autograd.grad(a,x)[0]
        b=masked_focal(x,y,k,w,1.,torch.tensor([2.,2.,1.,1.,1.,1.,1.]));gb=torch.autograd.grad(b,x)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    def test_easy_primary_negative_zero_gradient(self):
        x,y,k,w=self.values();x=x.detach().clone();x[:,:2]=-10.;x.requires_grad_();y[:,:2]=0
        loss(x,y,k,w).backward();self.assertTrue(torch.equal(x.grad[:,:2],torch.zeros(2,2)))
    def test_primary_hard_negative_gradient_positive(self):
        x,y,k,w=self.values();x=x.detach().clone();x[:,:2]=2.;x.requires_grad_();y[:,:2]=0
        loss(x,y,k,w).backward();self.assertTrue((x.grad[:,:2]>0).all())
    def test_primary_positive_gradient_negative(self):
        x,y,k,w=self.values();y[:,:2]=1.;loss(x,y,k,w).backward();self.assertTrue((x.grad[:,:2]<0).all())
    def test_extreme_wrong_positive_finite_nonzero_gradient(self):
        x,y,k,w=self.values();x=x.detach().clone();x[:,:2]=-1000.;x.requires_grad_();y[:,:2]=1.
        a=loss(x,y,k,w);a.backward();self.assertTrue(torch.isfinite(a));self.assertTrue((x.grad[:,:2]<0).all())
    def test_source_weights_support_per_photo(self):
        x,y,k,w=self.values();torch.testing.assert_close(loss(x,y,k,w),loss(x,y,k,w.expand(2,7)),rtol=0,atol=0)
    def test_known_count_and_primary_emphasis(self):
        x,y,k,w=self.values();k.zero_();k[:,0]=1;y[:,0]=1.
        expected=(-F.logsigmoid(x[:,0].float())*w[0]*2.).mean();torch.testing.assert_close(loss(x,y,k,w),expected)
    def test_no_grad_mode_preserved(self):
        x,y,k,w=self.values()
        with torch.no_grad():
            a=loss(x,y,k,w);self.assertFalse(torch.is_grad_enabled());self.assertFalse(a.requires_grad)
        self.assertTrue(torch.is_grad_enabled())
    def test_nonbinary_known_target_rejected(self):
        x,y,k,w=self.values();y[0,0]=.5
        with self.assertRaises(ValueError):loss(x,y,k,w)
    def test_nonfinite_rejected(self):
        x,y,k,w=self.values();x=x.detach();x[0,0]=float('nan')
        with self.assertRaises(ValueError):loss(x,y,k,w)
    def test_zero_positive_weight_rejected(self):
        x,y,k,w=self.values();w[0]=0
        with self.assertRaises(ValueError):loss(x,y,k,w)

if __name__=="__main__":unittest.main()
