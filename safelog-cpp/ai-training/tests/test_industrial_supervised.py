import unittest
import torch
from safelog_ai.industrial_supervised import IndustrialSupervisedMain, wear_descriptor
from tests.test_equipment_reference_photo import FakeBackbone
from scripts.study_bsclass_supervised import role, calibrated_threshold


class IndustrialSupervisedTests(unittest.TestCase):
    def test_descriptor_includes_both_branches_without_nan_for_one_pixel(self):
        value = wear_descriptor({'low': torch.ones(2, 40, 1, 1)*2, 'high': torch.ones(2, 960, 1, 1)*3})
        self.assertEqual(tuple(value.shape), (2, 3000))
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.equal(value[:, :40], torch.full((2, 40), 2.)))
        self.assertTrue(torch.equal(value[:, 40:80], torch.zeros(2, 40)))

    def test_new_head_gradients_do_not_change_parent_and_use_one_backbone(self):
        model = IndustrialSupervisedMain({}, {}, 'mlp').eval().requires_grad_(False)
        model.backbone = FakeBackbone(); model.wear_head.requires_grad_(True)
        parent = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('wear_')}
        out = model.forward_wear(torch.zeros(2, 3, 256, 256)); out.sum().backward()
        self.assertEqual(model.backbone.calls, 1)
        self.assertTrue(any(p.grad is not None for p in model.wear_head.parameters()))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith('wear_head.')))
        self.assertTrue(all(torch.equal(v, model.state_dict()[n]) for n, v in parent.items()))

    def test_unknown_head_is_not_silently_selected(self):
        with self.assertRaises(ValueError): IndustrialSupervisedMain({}, {}, 'unknown')

    def test_validation_and_calibration_roles_keep_whole_groups_together(self):
        rows = [{'phase': 'cal', 'group_id': str(i//3)} for i in range(60)]
        self.assertEqual({role(r) for r in rows}, {'validation', 'calibration'})
        for start in range(0, len(rows), 3): self.assertEqual(len({role(r) for r in rows[start:start+3]}), 1)
        self.assertEqual(role({'phase': 'reserved_test'}), 'reserved_test')

    def test_calibration_minimax_threshold_uses_strict_greater_with_ties(self):
        labels, scores = [0, 0, 1, 1], [.1, .2, .2, .9]
        threshold = calibrated_threshold(labels, scores)
        self.assertEqual(threshold, .5)
        # At least one error is unavoidable because opposite labels share.2.
        self.assertEqual(sum((s > threshold) != bool(y) for y, s in zip(labels, scores)), 1)


if __name__ == '__main__': torch.set_num_threads(4); unittest.main()
