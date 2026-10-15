import unittest
import numpy as np
import torch
from torch import nn
from safelog_ai.gear_adapter_probe import ZeroFineAdapterProbe
from safelog_ai.gear_task_adapter import adapted_descriptor
from safelog_ai.gear_cross_group import sample_cross_group_batch, cross_group_forward
from safelog_ai.gear_cross_group_loss import cross_group_loss


class CountConv(nn.Conv2d):
    def forward(self, value):
        self.calls = getattr(self, 'calls', 0) + 1
        return super().forward(value)


def toy_backbone():
    model = nn.ModuleDict({'3': nn.Identity(), '4': CountConv(24, 40, 1),
                          '5': CountConv(40, 960, 1)}).eval().requires_grad_(False)
    model.return_layers = {'4': 'low', '5': 'high'}
    return model


class CrossGroupTests(unittest.TestCase):
    def test_sampler_distinct_photos_and_groups_and_positive_for_each_anchor(self):
        rows = [{'label': label, 'group_id': group} for label in (0, 1)
                for group in range(6) for _ in range(3)]
        rng = np.random.default_rng(3333)
        for _ in range(20):
            ids = sample_cross_group_batch(rows, rng)
            self.assertEqual(len(ids), len(set(ids.tolist())))
            for label in (0, 1):
                selected = [rows[i] for i in ids if rows[i]['label'] == label]
                self.assertEqual(len(selected), 8)
                self.assertEqual(len({row['group_id'] for row in selected}), 4)
            groups = [rows[i]['group_id'] for i in ids]
            labels = torch.tensor([rows[i]['label'] for i in ids])
            self.assertTrue(torch.isfinite(cross_group_loss(torch.randn(16, 8), labels, groups)))
        with self.assertRaises(ValueError):
            sample_cross_group_batch(rows[:9], rng)

    def test_one_tail_forward_serves_both_outputs_and_matches_existing_zero_adapter(self):
        backbone = toy_backbone(); adapter = ZeroFineAdapterProbe(); head = nn.Linear(3072, 1)
        fine = torch.randn(4, 24, 8, 8); mean = torch.randn(3072); std = torch.rand(3072) + .1
        logits, embedding = cross_group_forward(backbone, adapter, head, mean, std, fine)
        self.assertEqual([backbone[k].calls for k in ('4', '5')], [1, 1])
        reference = ((adapted_descriptor(backbone, adapter, fine) - mean) / std).clamp(-8, 8)
        self.assertTrue(torch.equal(embedding, reference))
        self.assertTrue(torch.equal(logits, head(reference).flatten()))

    def test_auxiliary_gradient_reaches_adapter_and_leaves_shared_encoder_frozen(self):
        backbone = toy_backbone(); adapter = ZeroFineAdapterProbe(); head = nn.Linear(3072, 1)
        logits, embedding = cross_group_forward(backbone, adapter, head, torch.zeros(3072),
                                               torch.ones(3072), torch.randn(4, 24, 8, 8))
        labels = torch.tensor([0., 0., 1., 1.])
        auxiliary = cross_group_loss(embedding, labels, ['a', 'b', 'c', 'd'])
        auxiliary.backward(retain_graph=True)
        self.assertGreater(float(adapter[-1].weight.grad.abs().sum()), 0)
        self.assertIsNone(head.weight.grad)
        self.assertTrue(all(value.grad is None for value in backbone.parameters()))
        nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
        self.assertGreater(float(head.weight.grad.abs().sum()), 0)


if __name__ == '__main__': unittest.main()
