"""Loss masking, immutable teacher state, fixed protocol and original draws."""
from collections import OrderedDict
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from safelog_ai.retention_distillation import masked_bernoulli_kl, state_sha256
from safelog_ai.spalling_sampler import draw_sha256
from scripts import facility_retention_study as study


class RetentionLossTests(TestCase):
    def test_identical_logits_have_exact_zero_kl_and_temperature_scaled_closed_form(self):
        student = torch.zeros(1, 7, requires_grad=True)
        teacher = torch.full((1, 7), 2.)
        known = torch.ones(1, 7)
        self.assertEqual(masked_bernoulli_kl(teacher, teacher, known).item(), 0.)
        actual = masked_bernoulli_kl(student, teacher, known, temperature=2.)
        p = 1 / (1 + math.exp(-1))
        expected = 4 * (p * math.log(p / .5) + (1 - p) * math.log((1 - p) / .5))
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_only_known_other_five_get_gradient_teacher_and_mask_are_detached(self):
        student = torch.zeros(2, 7, requires_grad=True)
        teacher = torch.full((2, 7), 2., requires_grad=True)
        known = torch.tensor([[1., 1., 1., 0., 0., 0., 0.], [1., 1., 0., 1., 0., 0., 0.]], requires_grad=True)
        masked_bernoulli_kl(student, teacher, known).backward()
        self.assertIsNone(teacher.grad); self.assertIsNone(known.grad)
        expected = torch.zeros_like(student, dtype=torch.bool); expected[0, 2] = True; expected[1, 3] = True
        self.assertTrue(torch.equal(student.grad != 0, expected))
        self.assertTrue((student.grad[expected] < 0).all())

    def test_no_known_retained_entries_return_differentiable_finite_zero(self):
        student = torch.randn(2, 7, requires_grad=True); teacher = torch.randn(2, 7, requires_grad=True)
        mask = torch.zeros(2, 7); mask[:, :2] = 1
        loss = masked_bernoulli_kl(student, teacher, mask)
        self.assertEqual(loss.item(), 0.); loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student))); self.assertIsNone(teacher.grad)

    def test_extreme_half_logits_are_finite_fp32_and_boolean_mask_is_equivalent(self):
        teacher = torch.full((2, 7), 60000., dtype=torch.float16)
        student = torch.full((2, 7), -60000., dtype=torch.float16, requires_grad=True)
        known = torch.ones(2, 7, dtype=torch.bool)
        loss = masked_bernoulli_kl(student, teacher, known)
        self.assertEqual(loss.dtype, torch.float32); self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), masked_bernoulli_kl(student, teacher, known.float()).item())
        loss.backward(); self.assertTrue(torch.isfinite(student.grad).all())

    def test_invalid_temperature_shape_dtype_nonfinite_and_unknown_mask_rejected(self):
        s = torch.zeros(1, 7); t = torch.ones(1, 7); k = torch.ones(1, 7)
        for temperature in (True, 0, -1., float("nan"), float("inf"), "2", torch.tensor(2.)):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                masked_bernoulli_kl(s, t, k, temperature)
        cases = [(s[:, :6], t, k), (s, t.repeat(2, 1), k), (s.int(), t, k),
                 (s, t, k.int()), (s, t, k * .5), (s, t, k * -1),
                 (s + float("nan"), t, k), (s, t + float("inf"), k)]
        for values in cases:
            with self.subTest(shapes=[tuple(v.shape) for v in values]), self.assertRaises(ValueError):
                masked_bernoulli_kl(*values)

    def test_original_logits_and_known_entries_are_never_modified(self):
        s = torch.randn(3, 7); t = torch.randn(3, 7); k = torch.tensor([[1., 0., 1., 0., 1., 0., 1.]] * 3)
        before = [v.clone() for v in (s, t, k)]
        masked_bernoulli_kl(s, t, k)
        for old, new in zip(before, (s, t, k)): self.assertTrue(torch.equal(old, new))

    def test_teacher_state_hash_is_order_independent_and_binds_scalar_dtype_shape_values(self):
        state = {"weights": torch.tensor([[1., 2.]]), "counter": torch.tensor(3, dtype=torch.int64)}
        digest = state_sha256(state)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, state_sha256(OrderedDict(reversed(list(state.items())))))
        for change in ("value", "dtype", "shape", "name"):
            value = deepcopy(state)
            if change == "value": value["counter"] += 1
            elif change == "dtype": value["weights"] = value["weights"].double()
            elif change == "shape": value["weights"] = value["weights"].reshape(2)
            else: value["renamed"] = value.pop("weights")
            with self.subTest(change=change): self.assertNotEqual(digest, state_sha256(value))
        model = torch.nn.Linear(2, 1)
        self.assertEqual(state_sha256(model), state_sha256(model.state_dict()))
        self.assertEqual(len(state_sha256({"bf16": torch.ones(2, dtype=torch.bfloat16)})), 64)
        for bad in ({}, {"x": "tensor"}, {"x": torch.tensor(float("nan"))}):
            with self.assertRaises(ValueError): state_sha256(bad)


class RetentionProtocolTests(TestCase):
    def protocol(self):
        p = study.fixed_values(); p["source_sha256"] = {name: "a" * 64 for name in study.SOURCE_FILES}
        p["protected_file_sha256"] = {"data/original.json": "a" * 64}
        return p

    def validate(self, p, args=None):
        with patch.object(study, "local_path", side_effect=lambda root, name: Path(root) / name), \
             patch.object(study, "sha", return_value="a" * 64), \
             patch.object(study, "expected_protected_hashes", return_value={"data/original.json": "a" * 64}):
            return study.validate_protocol(p, Path("/retention-test"), args)

    def test_fixed_teacher_recipe_reuses_identical_original_control_sampling(self):
        p = self.protocol(); result = self.validate(p)
        self.assertIsNot(result, p)
        self.assertEqual(len(study.SOURCE_FILES), 61)
        self.assertEqual(result["paired_draws_keys"], {"control": "control", "distill": "control"})
        self.assertEqual(result["distillation_weight_by_variant"], {"control": 0., "distill": 1.})
        self.assertEqual(result["distillation_recipe"]["retained_indices"], [2, 3, 4, 5, 6])
        self.assertTrue(result["distillation_recipe"]["teacher_forward_both_arms"])

    def test_temperature_weight_gate_teacher_bool_or_holdout_drift_is_rejected(self):
        for key, value in (("seed", 57), ("requested_epochs", 7), ("declared_before_training", 1),
                           ("source_test_inference_executed", 0), ("teacher_state_sha256", "b" * 64)):
            p = self.protocol(); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(p)
        for field, key, value in (("distillation_recipe", "temperature", 3.),
             ("distillation_recipe", "teacher_eval", 1), ("distillation_recipe", "teacher_no_grad", False),
             ("distillation_weight_by_variant", "distill", .5),
             ("retention_candidate_gate", "maximum_known_other_class_ap_drop_vs_reference", .05),
             ("paired_draws_keys", "distill", "treatment")):
            p = self.protocol(); p[field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError): self.validate(p)

    def test_previous_49_and_new_sources_must_keep_every_hash(self):
        p = self.protocol(); p["source_sha256"].pop(study.SOURCE_FILES[0])
        with self.assertRaisesRegex(ValueError, "inventory"): self.validate(p)
        for name in (study.SOURCE_FILES[0], study.NEW_SOURCES[0]):
            p = self.protocol(); p["source_sha256"][name] = "b" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "runtime source"): self.validate(p)

    def test_previous_protected_evidence_cannot_change_or_be_omitted(self):
        p = self.protocol(); p["protected_file_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "Protected previous/original"): self.validate(p)
        p = self.protocol(); p["previous_verification_sha256"] = "b" * 64
        with self.assertRaises(ValueError): self.validate(p)

    def test_cli_initializer_budget_temperature_and_weight_are_fixed(self):
        args = SimpleNamespace(variant="distill", name=study.NAMES["distill"], seed=56, epochs=6, patience=6,
            batch=8, draws_per_epoch=14248, backbone_lr=.00004, head_lr=.00025, auxiliary_weight=.5,
            temperature=2., distillation_weight=1., initial=Path("/retention-test/runs") / study.REFERENCE / "best.pt",
            auxiliary_manifest=Path("/retention-test/data/facility-auxiliary-training/train.json"))
        self.assertEqual(self.validate(self.protocol(), args)["imgsz"], 640)
        for field, value in (("temperature", True), ("temperature", 3.), ("distillation_weight", True),
             ("distillation_weight", 0.), ("variant", "negative"), ("epochs", 7),
             ("initial", Path("/retention-test/runs/new/best.pt"))):
            bad = deepcopy(args); setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(self.protocol(), bad)


class RetentionDrawTests(TestCase):
    def fixture(self, seed=56):
        weights = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed)
        draws = np.stack([torch.multinomial(weights, 100, replacement=True, generator=generator).numpy() for _ in range(6)])
        records = [{"epoch": k + 1, "control_order_sha256": draw_sha256(indices)} for k, indices in enumerate(draws)]
        return draws, weights, records, len(weights)

    def validate(self, values): return study.validate_control_draws(*values, expected_shape=(6, 100))

    def test_six_original_control_arrays_replay_exactly_for_both_arms(self):
        values = self.fixture(); before = values[0].copy(); records = self.validate(values)
        self.assertEqual(len(records), 6)
        self.assertTrue(all(row["control_order_sha256"] == row["distill_order_sha256"] for row in records))
        np.testing.assert_array_equal(values[0], before)

    def test_control_hash_shape_bounds_dtype_or_self_rehashed_other_seed_fail(self):
        for kind in ("shape", "bounds", "dtype", "hash", "other_seed"):
            values = list(self.fixture())
            if kind == "shape": values[0] = values[0][:5]
            elif kind == "bounds": values[0][0, 0] = 4
            elif kind == "dtype": values[0] = values[0].astype(float)
            elif kind == "hash": values[2][0]["control_order_sha256"] = "b" * 64
            else: values = self.fixture(seed=59)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validate(values)

    def test_frozen_epoch_sampler_keeps_one_based_order_and_consumes_no_rng(self):
        values = self.fixture(); sampler = study.FixedEpochSampler(values[0]); rng = torch.get_rng_state().clone()
        with self.assertRaises(ValueError): list(sampler)
        sampler.set_epoch(6)
        self.assertEqual(list(sampler), values[0][5].tolist())
        self.assertEqual(len(sampler), 100)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
