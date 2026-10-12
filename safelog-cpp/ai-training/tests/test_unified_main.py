import unittest
import torch
from safelog_ai.unified_main import UnifiedMain, masked_photo, freeze_bn_statistics


class SharedMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_all_three_tasks_use_the_same_backbone_and_shapes(self):
        model = UnifiedMain().eval(); x = torch.randn(2, 3, 128, 128)
        calls = []; handle = model.backbone.register_forward_hook(lambda *args: calls.append(id(model.backbone)))
        with torch.no_grad():
            a = model.forward_task(x, 'facility_surface'); b = model.forward_task(x, 'belt_damage'); c = model.forward_task(x, 'conveyor_foreign')
        handle.remove(); self.assertEqual(len(set(calls)), 1); self.assertEqual(len(calls), 3)
        self.assertEqual(tuple(a[0].shape), (2, 7)); self.assertEqual(tuple(b.shape), (2, 1, 128, 128)); self.assertEqual(tuple(c.shape), (2,))

    def test_unknown_targets_have_zero_gradient(self):
        x = torch.tensor([[1., 2., 3.]], requires_grad=True)
        value = masked_photo(x, torch.tensor([[1., -1., -1.]]), torch.tensor([[1., 0., 0.]]), torch.ones(3))
        value.backward(); self.assertNotEqual(float(x.grad[0, 0]), 0)
        self.assertEqual(float(x.grad[0, 1:].abs().sum()), 0)

    def test_inactive_heads_have_no_gradient_and_running_bn_is_frozen(self):
        model = UnifiedMain().train(); freeze_bn_statistics(model)
        output = model.forward_task(torch.randn(2, 3, 128, 128), 'conveyor_foreign')
        output.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() for p in model.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.belt_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.photo_head.parameters()))
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))
        with self.assertRaises(ValueError): model.forward_task(torch.randn(2, 3, 128, 128), 'unknown')


if __name__ == '__main__': unittest.main()
