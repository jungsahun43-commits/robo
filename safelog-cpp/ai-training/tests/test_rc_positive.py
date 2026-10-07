"""The new loss may learn asserted positives, never unannotated negatives."""
import unittest

import numpy as np
import torch

from safelog_ai.positive_region_loss import positive_region_loss
from scripts.facility_rc_positive import (DRAW_COUNT, EPOCHS, REPLACEMENTS, fixed_draws,
                                          select_representatives, validate_positive_targets)


class PositiveRegionTests(unittest.TestCase):
    def test_unknown_cells_and_other_classes_have_zero_gradient(self):
        logits = torch.zeros(2, 7, 4, 4, requires_grad=True)
        asserted = torch.zeros_like(logits); asserted[0, 1, 1, 2] = 1
        loss = positive_region_loss(logits, asserted, torch.tensor([True, False]))
        loss.backward()
        expected = torch.zeros_like(logits.grad, dtype=torch.bool); expected[0, 1, 1, 2] = True
        self.assertTrue(torch.equal(logits.grad != 0, expected))
        self.assertLess(logits.grad[0, 1, 1, 2].item(), 0)

    def test_empty_unknown_supervision_produces_zero_loss(self):
        logits = torch.randn(2, 7, 4, 4, requires_grad=True)
        loss = positive_region_loss(logits, torch.zeros_like(logits), torch.tensor([True, True]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_extreme_scores_keep_finite_loss_and_gradient(self):
        logits = torch.full((1, 7, 4, 4), -100., requires_grad=True)
        asserted = torch.zeros_like(logits); asserted[0, 0, 0, 0] = 1
        loss = positive_region_loss(logits, asserted, torch.tensor([True])); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unsupported_other_class_foreground_rejected(self):
        logits = torch.zeros(1, 7, 4, 4); asserted = torch.zeros_like(logits); asserted[0, 3, 0, 0] = 1
        with self.assertRaises(ValueError): positive_region_loss(logits, asserted, torch.tensor([True]))
        with self.assertRaises(ValueError): positive_region_loss(logits, torch.zeros_like(logits), torch.tensor([1.]))

    def test_targets_reject_zero_bool_and_all_unknown(self):
        validate_positive_targets([1, -1, -1, -1, -1, -1, -1])
        for targets in ([1, 0, -1, -1, -1, -1, -1], [True, -1, -1, -1, -1, -1, -1], [-1] * 7):
            with self.assertRaises(ValueError): validate_positive_targets(targets)

    def test_exact_bounded_replacements_preserve_other_positions(self):
        original = np.tile(np.arange(DRAW_COUNT, dtype=np.int64), (EPOCHS, 1))
        altered = fixed_draws(original, 26289, 200)
        for epoch in range(EPOCHS):
            changed = altered[epoch] != original[epoch]
            self.assertEqual(int(changed.sum()), REPLACEMENTS)
            self.assertTrue(np.array_equal(original[epoch, ~changed], altered[epoch, ~changed]))
            self.assertTrue(((altered[epoch, changed] >= 26289) & (altered[epoch, changed] < 26489)).all())
        self.assertTrue(np.array_equal(altered, fixed_draws(original, 26289, 200)))

    def test_bad_sampler_budget_rejected(self):
        with self.assertRaises(ValueError): fixed_draws(np.zeros((1, 10), np.int64), 100, 10)
        with self.assertRaises(ValueError): fixed_draws(np.zeros((EPOCHS, DRAW_COUNT), np.int64), 100, 201)

    def test_group_selection_is_source_only_and_excludes_flagged_rows(self):
        rows = [{"image": "spall1", "group": "a", "native_instances": {"Concrete spalling": 1}, "excluded_reasons": []},
                {"image": "spall2", "group": "a", "native_instances": {"Concrete spalling": 1, "Crack": 1}, "excluded_reasons": []},
                {"image": "crack1", "group": "b", "native_instances": {"Crack": 1}, "excluded_reasons": []},
                {"image": "bad", "group": "c", "native_instances": {"Concrete spalling": 1}, "excluded_reasons": ["existing"]}]
        chosen = select_representatives(rows, set(), 100)
        self.assertEqual({r["image"] for r in chosen}, {"spall1", "crack1"})
        self.assertEqual(chosen, select_representatives(list(reversed(rows)), set(), 100))
        self.assertEqual(len({r["group"] for r in chosen}), len(chosen))


if __name__ == "__main__":
    unittest.main()
