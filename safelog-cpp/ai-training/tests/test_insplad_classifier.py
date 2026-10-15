import unittest
import torch
from safelog_ai.insplad_classifier import CorrosionClassifier,InspladMain
from safelog_ai.real_mask_server import RuntimeRealMaskMain


class CorrosionTests(unittest.TestCase):
    def test_initial_and_input_contract(self):
        head=CorrosionClassifier().eval()
        self.assertTrue(torch.equal(head(torch.randn(4,3072)),torch.zeros(4)))
        with self.assertRaises(ValueError):head(torch.zeros(4,3071))
        with self.assertRaises(ValueError):head(torch.full((1,3072),float('nan')))

    def test_training_does_not_update_normalization(self):
        head=CorrosionClassifier();before={n:t.clone() for n,t in head.named_buffers()}
        loss=torch.nn.functional.binary_cross_entropy_with_logits(head(torch.randn(4,3072)),torch.tensor([0.,1.,1.,0.]))
        loss.backward();optimizer=torch.optim.AdamW(head.head.parameters(),lr=3e-4);optimizer.step()
        self.assertTrue(head.head[-1].weight.grad.abs().sum()>0)
        self.assertTrue(all(torch.equal(before[n],t) for n,t in head.named_buffers()))

    def test_runtime_contracts_preserved(self):
        self.assertIs(InspladMain.forward_surface,RuntimeRealMaskMain.forward_surface)
        self.assertIs(InspladMain.forward_real_mask,RuntimeRealMaskMain.forward_real_mask)
