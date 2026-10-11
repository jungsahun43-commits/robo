import unittest
import torch
from safelog_ai.primary_preservation import primary_preservation_loss as loss
from scripts.audit_facility_idea_transitions import transitions

class PrimaryPreservationTests(unittest.TestCase):
    def tensors(self):
        a=torch.zeros(3,7,requires_grad=True);b=torch.full((3,7),2.,requires_grad=True);known=torch.ones(3,7);rows=torch.tensor([True,True,False]);return a,b,known,rows
    def test_gradient_primary_original_only(self):
        a,b,k,r=self.tensors();value=loss(a,b,k,r);self.assertGreater(value.item(),0);value.backward()
        self.assertIsNone(b.grad);self.assertTrue((a.grad[:2,:2]<0).all());self.assertEqual(int(a.grad[:,2:].count_nonzero()),0);self.assertEqual(int(a.grad[2].count_nonzero()),0)
    def test_unknown_primary_excluded(self):
        a,b,k,r=self.tensors();k[0,0]=0;loss(a,b,k,r).backward();self.assertEqual(a.grad[0,0],0);self.assertNotEqual(a.grad[0,1],0)
    def test_only_supplement_returns_connected_zero(self):
        a,b,k,r=self.tensors();r[:]=False;v=loss(a,b,k,r);self.assertEqual(v.item(),0);v.backward();self.assertEqual(int(a.grad.count_nonzero()),0)
    def test_all_unknown_returns_zero(self):
        a,b,k,r=self.tensors();k[:]=0;v=loss(a,b,k,r);v.backward();self.assertEqual(v.item(),0);self.assertEqual(int(a.grad.count_nonzero()),0)
    def test_equal_teacher_zero(self):
        a,b,k,r=self.tensors();self.assertAlmostEqual(loss(a,a,k,r).item(),0.,places=6)
    def test_teacher_supplement_changes_have_no_effect(self):
        a,b,k,r=self.tensors();first=loss(a,b,k,r).item();b=b.detach().clone();b[2]=99;self.assertEqual(first,loss(a,b,k,r).item())
    def test_other5_changes_have_no_effect(self):
        a,b,k,r=self.tensors();first=loss(a,b,k,r).item();b=b.detach().clone();b[:,2:]=99;self.assertEqual(first,loss(a,b,k,r).item())
    def test_normalized_known_count(self):
        a,b,k,r=self.tensors();self.assertAlmostEqual(loss(a,b,k,r).item(),loss(a[:1],b[:1],k[:1],r[:1]).item(),places=6)
    def test_extreme_finite_logits(self):
        a,b,k,r=self.tensors();a=torch.full_like(a,80.,requires_grad=True);b=torch.full_like(b,-80.);v=loss(a,b,k,r);v.backward();self.assertTrue(torch.isfinite(v));self.assertTrue(torch.isfinite(a.grad).all())
    def test_fp16_loss_fp32(self):
        a,b,k,r=self.tensors();self.assertEqual(loss(a.half(),b.half(),k,r).dtype,torch.float32)
    def test_invalid_temperature(self):
        for t in(0,-1,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):loss(*self.tensors(),temperature=t)
    def test_invalid_known_mask(self):
        a,b,k,r=self.tensors();k[0,0]=.5
        with self.assertRaises(ValueError):loss(a,b,k,r)
    def test_invalid_rows(self):
        a,b,k,r=self.tensors()
        with self.assertRaises(ValueError):loss(a,b,k,r.float())
    def test_invalid_shape(self):
        a,b,k,r=self.tensors()
        with self.assertRaises(ValueError):loss(a[:,:6],b,k,r)
    def test_transition_accounting(self):
        truth=torch.tensor([1,1,0,0],dtype=torch.bool).numpy();old=torch.tensor([1,0,0,1],dtype=torch.bool).numpy();new=~old
        m=transitions(truth,old,new);self.assertEqual((m['newly_wrong'],m['corrected'],m['both_wrong'],m['both_correct']),(2,2,0,0));self.assertEqual(m['old_wrong']+m['newly_wrong']-m['corrected'],m['new_wrong'])

if __name__=='__main__':unittest.main()
