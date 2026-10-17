import unittest
import torch
from safelog_ai.ostrava_rust_mask import RustMaskHead
from safelog_ai.corrosion_context_head import ContextRustHead


class ContextHeadTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def heads(self):
        torch.manual_seed(5);start=torch.get_rng_state();old=RustMaskHead();rng=torch.get_rng_state()
        torch.set_rng_state(start);new=ContextRustHead();self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with torch.no_grad():old.fusion[-1].weight.normal_(0,.1)
        missing,extra=new.load_state_dict(old.state_dict(),strict=False)
        self.assertEqual(missing,['context_projection.weight']);self.assertFalse(extra)
        return old.eval(),new.eval()

    def features(self):return [torch.randn(1,c,s,s).half() for c,s in ((24,128),(40,64),(960,32))]

    def test_zero_context_path_preserves_original_logits_exactly(self):
        old,new=self.heads();f=self.features();c=torch.randn(1,1,128,128).clamp(-1,1)
        with torch.no_grad():self.assertTrue(torch.equal(old(*f),new(*f,c)))
        self.assertEqual(sum(p.numel() for p in new.parameters()),101169)

    def test_new_projection_can_learn_without_gradient_to_frozen_response(self):
        _,new=self.heads();c=torch.randn(1,1,128,128).clamp(-1,1).requires_grad_()
        new(*self.features(),c).square().mean().backward()
        self.assertIsNone(c.grad);self.assertGreater(float(new.context_projection.weight.grad.abs().sum()),0.)

    def test_misaligned_or_unbounded_context_is_rejected(self):
        _,new=self.heads();f=self.features()
        with self.assertRaises(ValueError):new(*f,torch.zeros(1,1,64,64))
        with self.assertRaises(ValueError):new(*f,torch.full((1,1,128,128),2.))
