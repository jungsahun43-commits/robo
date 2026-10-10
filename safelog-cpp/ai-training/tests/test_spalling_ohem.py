"""Real autograd checks for the one changed spalling background objective."""
import unittest
import torch
from safelog_ai.spatial_classifier import spatial_loss
from safelog_ai.spalling_ohem import spalling_ohem_spatial_loss

class OhemTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(67)
        self.x=torch.randn(3,7,8,8);self.y=(torch.rand_like(self.x)>.7).float()
        self.k=torch.ones(3,7);self.w=torch.linspace(1.,3.,7)
    def compare_gradients(self):
        a=self.x.clone().requires_grad_();b=self.x.clone().requires_grad_()
        spatial_loss(a,self.y,self.k,self.w).backward();spalling_ohem_spatial_loss(b,self.y,self.k,self.w).backward()
        return a.grad,b.grad
    def test_full_fraction_recovers_original_value_and_gradient(self):
        a=self.x.clone().requires_grad_();b=self.x.clone().requires_grad_()
        old=spatial_loss(a,self.y,self.k,self.w);new=spalling_ohem_spatial_loss(b,self.y,self.k,self.w,1.)
        old.backward();new.backward();self.assertTrue(torch.allclose(old,new,atol=1e-7,rtol=1e-6));self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-8,rtol=1e-6))
    def test_other_six_class_gradients_are_exact(self):
        a,b=self.compare_gradients();indices=[0,2,3,4,5,6];self.assertTrue(torch.equal(a[:,indices],b[:,indices]))
    def test_all_spalling_foreground_gradients_are_exact(self):
        a,b=self.compare_gradients();mask=self.y[:,1]>0;self.assertTrue(torch.equal(a[:,1][mask],b[:,1][mask]))
    def test_soft_foreground_is_not_mined_as_background(self):
        self.y[:,1]=.1;a,b=self.compare_gradients();self.assertTrue(torch.equal(a[:,1],b[:,1]))
    def test_unavailable_spalling_gradient_is_zero(self):
        self.k[:,1]=0;a,b=self.compare_gradients();self.assertEqual(int(b[:,1].count_nonzero()),0);self.assertTrue(torch.equal(a,b))
    def test_every_unknown_channel_has_zero_gradient(self):
        self.k.zero_();x=self.x.requires_grad_();loss=spalling_ohem_spatial_loss(x,self.y,self.k,self.w);loss.backward()
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(int(x.grad.count_nonzero()),0)
    def test_selected_count_is_per_photo_ceiling(self):
        _,d=spalling_ohem_spatial_loss(self.x,self.y,self.k,self.w,return_diagnostics=True)
        count=(self.y[:,1]==0).flatten(1).sum(1);self.assertTrue(torch.equal(d['known_background_cells'],count))
        self.assertTrue(torch.equal(d['selected_background_cells'],(count.double()*.1).ceil().long()))
    def test_hard_negative_gradient_increases_and_easy_focal_is_removed(self):
        self.x.zero_();self.x[:,1]=torch.linspace(-3.,3.,64).reshape(8,8);self.y.zero_();a,b=self.compare_gradients()
        # Original Dice is inactive for all-negative samples.
        self.assertEqual(int(b[:,1,:7].count_nonzero()),0);self.assertTrue((b[:,1,-1,-7:]>a[:,1,-1,-7:]).all())
    def test_foreground_only_and_unknown_rows_are_finite(self):
        self.y[:,1]=1.;self.k[1,1]=0.;loss,d=spalling_ohem_spatial_loss(self.x,self.y,self.k,self.w,return_diagnostics=True)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(int(d['selected_background_cells'].sum()),0)
    def test_extreme_logits_finite_loss_and_gradients(self):
        x=(self.x.sign()*100.).requires_grad_();loss=spalling_ohem_spatial_loss(x,self.y,self.k,self.w);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())
    def test_invalid_supervision_and_fraction_rejected(self):
        for fraction in(0.,-1.,1.01,float('nan'),True):
            with self.assertRaises(ValueError):spalling_ohem_spatial_loss(self.x,self.y,self.k,self.w,fraction)
        self.k[0,1]=.5
        with self.assertRaises(ValueError):spalling_ohem_spatial_loss(self.x,self.y,self.k,self.w)
if __name__=='__main__':unittest.main()
