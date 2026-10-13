import unittest
import torch
from safelog_ai.belt_recall_step_v2 import project_step,positive_recall_loss


class RecallStepTests(unittest.TestCase):
    def test_harmful_direction_is_removed_but_orthogonal_progress_remains(self):
        p=torch.nn.Parameter(torch.tensor([.3,-.1])); before=torch.tensor([.1,.1]); g=torch.tensor([1.,0.])
        result=project_step([p],[before],[g])
        self.assertTrue(result['projected']); self.assertLessEqual(result['dot_after'],result['tolerance'])
        self.assertLessEqual(float(p.detach()[0]),float(before[0])); self.assertAlmostEqual(float(p.detach()[1]),-.1,places=6)

    def test_nonconflicting_step_and_zero_gradient_are_unchanged(self):
        for gradient in [torch.tensor([-1.,0.]),torch.zeros(2)]:
            p=torch.nn.Parameter(torch.tensor([.3,-.1])); proposed=p.detach().clone()
            result=project_step([p],[torch.tensor([.1,.1])],[gradient])
            self.assertFalse(result['projected']); self.assertTrue(torch.equal(p,proposed))

    def test_actual_adam_momentum_displacement_is_checked(self):
        p=torch.nn.Parameter(torch.tensor([.1,.1])); opt=torch.optim.AdamW([p],lr=.02,weight_decay=.001)
        opt.zero_grad(); p.grad=torch.tensor([-1.,1.]); opt.step()
        before=p.detach().clone(); opt.zero_grad(); p.grad=torch.tensor([-1.,1.]); opt.step()
        # Protection opposes the first coordinate, even though Adam has momentum.
        result=project_step([p],[before],[torch.tensor([1.,0.])])
        self.assertTrue(result['projected']); self.assertLessEqual(result['dot_after'],1e-9)
        self.assertLess(float(p.detach()[1]),float(before[1]))

    def test_recall_loss_uses_only_known_positive_pixels(self):
        x=torch.tensor([[[[-2.,4.,-5.]]]],requires_grad=True); mask=torch.tensor([[[[1.,0.,1.]]]]); valid=torch.tensor([[[[1.,1.,0.]]]])
        loss=positive_recall_loss(x,mask,valid); loss.backward()
        self.assertLess(float(x.grad[0,0,0,0]),0); self.assertEqual(float(x.grad[0,0,0,1]),0); self.assertEqual(float(x.grad[0,0,0,2]),0)
        with self.assertRaises(ValueError): positive_recall_loss(x,torch.zeros_like(mask),valid)


    def test_dense_float32_writes_still_meet_signed_dot_guard(self):
        generator=torch.Generator().manual_seed(449)
        before=torch.randn(160000,generator=generator)*.03
        gradient=torch.randn(160000,generator=generator)
        p=torch.nn.Parameter(before+gradient*2e-5)
        result=project_step([p],[before],[gradient])
        self.assertTrue(result['projected'])
        self.assertLessEqual(result['dot_after'],result['tolerance'])
        self.assertGreater(result['write_margin'],0)

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
