"""Reject protocol drift and private paired-draw substitutions before training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.spalling_sampler import coupling_strata, eligible_rows, paired_epoch, draw_sha256
from scripts import facility_subtype_study as study


class SubtypeProtocolTests(TestCase):
    def protocol(self):
        p = study.fixed_values()
        p["source_sha256"] = {path: "a" * 64 for path in study.SOURCE_FILES}
        p["protected_file_sha256"] = {"data/original.json": "a" * 64}
        return p

    def validate(self, p, args=None):
        def digest(path):
            text = Path(path).as_posix()
            if text.endswith("/best.pt"): return study.INITIAL_SHA
            if text.endswith("data/facility-spatial-training/train.json"): return study.CORE_SHA
            if text.endswith("data/facility-auxiliary-training/train.json"): return study.AUX_SHA
            if text.endswith("reports/facility-inference-profile.json"): return study.APP_SHA
            return "a" * 64
        with patch.object(study, "local_path", side_effect=lambda root, name: Path(root) / name), \
             patch.object(study, "sha", side_effect=digest), \
             patch.object(study, "expected_protected_hashes", return_value={"data/original.json": "a" * 64}):
            return study.validate_protocol(p, Path("/temporary-study"), args)

    def test_fixed_recipe_preserves_original_640_auxiliary_contract(self):
        p = self.protocol(); actual = self.validate(p)
        self.assertIsNot(actual, p)
        self.assertEqual(actual["imgsz_by_variant"], {"control": 640, "negative": 640})
        self.assertEqual(len(set(actual["architecture_by_variant"].values())), 1)
        self.assertEqual(actual["paired_draws_sha256"], actual["private_draw_archive_sha256"])
        self.assertEqual(actual["subtype_sampling_recipe"]["eligible_factor"], 1.5)

    def test_epoch_seed_factor_gate_and_holdout_changes_are_rejected(self):
        changes = [("requested_epochs", 7), ("seed", 57), ("source_test_inference_executed", True),
                   ("app_model_promoted", True), ("declared_before_training", False),
                   ("paired_draws_sha256", "b" * 64), ("source_test_inference_executed", 0),
                   ("declared_before_training", 1)]
        for key, value in changes:
            with self.subTest(key=key):
                p = self.protocol(); p[key] = value
                with self.assertRaises(ValueError): self.validate(p)
        for field, key in (("subtype_sampling_recipe", "eligible_factor"),
                           ("research_candidate_gate", "minimum_maximum_error_improvement_vs_reference_and_control")):
            p = self.protocol(); p[field][key] = 2
            with self.assertRaises(ValueError): self.validate(p)

    def test_runtime_inventory_and_each_runtime_hash_are_required(self):
        p = self.protocol(); p["source_sha256"].pop(study.SOURCE_FILES[-1])
        with self.assertRaisesRegex(ValueError, "inventory"): self.validate(p)
        p = self.protocol(); p["source_sha256"][study.SOURCE_FILES[0]] = "b" * 64
        with self.assertRaisesRegex(ValueError, "runtime source"): self.validate(p)

    def test_protected_input_inventory_cannot_change(self):
        p = self.protocol(); p["protected_file_sha256"]["data/original.json"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "Protected original/sampler"): self.validate(p)
        p = self.protocol(); p["auxiliary_manifest_sha256"] = "b" * 64
        with self.assertRaises(ValueError): self.validate(p)

    def test_cli_budget_variant_and_initializer_path_are_checked(self):
        args = SimpleNamespace(variant="negative", name=study.NAMES["negative"], seed=56, epochs=6,
            patience=6, batch=8, draws_per_epoch=14248, backbone_lr=.00004, head_lr=.00025,
            auxiliary_weight=.5, initial=Path("/temporary-study/runs") / study.REFERENCE / "best.pt",
            auxiliary_manifest=Path("/temporary-study/data/facility-auxiliary-training/train.json"))
        self.assertEqual(self.validate(self.protocol(), args)["imgsz"], 640)
        for field, value in (("epochs", 7), ("variant", "native"), ("name", study.NAMES["control"]),
                             ("initial", Path("/temporary-study/runs/other/best.pt"))):
            bad = deepcopy(args); setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(self.protocol(), bad)


class PairedDrawBoundaryTests(TestCase):
    def fixture(self):
        items = []
        for index, (domain, crack, spall) in enumerate((
            ("dacl", 0, 0), ("dacl", 0, 0), ("dacl", 1, 0), ("dacl", 1, 0),
            ("dacl", 0, 1), ("damsegment", 0, 0), ("codebrim", -1, 1))):
            items.append({"image": f"full{index}", "domain": domain, "targets": [crack, spall, 0, 0, 0, 0, 0]})
        items.append({"image": "crop", "domain": "dacl", "parent_image": "full0", "parent_split": "train",
                      "targets": [0] * 7})
        aux = {}
        for item in items[:5]:
            tag = [0] * len(AUX_CLASSES)
            tag[AUX_CLASSES.index("Spalling")] = item["targets"][1]
            if item["image"] in ("full0", "full2"):
                tag[AUX_CLASSES.index("Rockpocket")] = 1
            aux[item["image"]] = tag
        weights = np.array([1., 3., 2., 4., 1., 1., 1., .5])
        generator = torch.Generator().manual_seed(56); rng = np.random.default_rng(59)
        eligible = eligible_rows(items, 7, aux); strata = coupling_strata(items, 7)
        control = np.stack([torch.multinomial(torch.from_numpy(weights), 400, replacement=True,
            generator=generator).numpy() for _ in range(6)])
        treatment = np.stack([paired_epoch(row, weights, eligible, strata, rng) for row in control])
        records = [{"epoch": k + 1, "control_order_sha256": draw_sha256(a),
            "treatment_order_sha256": draw_sha256(b), "changed_positions": int((a != b).sum()),
            "eligible_draws_control": int(eligible[a].sum()), "eligible_draws_treatment": int(eligible[b].sum())}
            for k, (a, b) in enumerate(zip(control, treatment))]
        return control, treatment, items, 7, aux, weights, records

    def validate(self, fixture, replay=True):
        return study.validate_paired_draws(*fixture, expected_shape=(6, 400), replay=replay)

    def test_paired_replay_proves_original_strata_and_eligible_retention(self):
        fixture = self.fixture(); proof = self.validate(fixture)
        self.assertEqual(len(proof["epochs"]), 6)
        self.assertGreater(proof["total_changed_positions"], 0)
        self.assertGreater(proof["total_eligible_draws_treatment"], proof["total_eligible_draws_control"])
        self.assertTrue(all(row["every_position_domain_full_crop_and_two_targets_equal"] for row in proof["epochs"]))

    def test_private_array_shape_dtype_bounds_and_record_hash_tamper_fail(self):
        for kind in ("shape", "float", "bounds", "hash", "exposure"):
            values = list(self.fixture())
            if kind == "shape": values[0] = values[0][:5]
            elif kind == "float": values[0] = values[0].astype(float)
            elif kind == "bounds": values[0][0, 0] = len(values[2])
            elif kind == "hash": values[-1][0]["control_order_sha256"] = "b" * 64
            else: values[-1][0]["eligible_draws_treatment"] += 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validate(values)

    def test_even_self_rehashed_draw_cannot_change_source_or_original_eligible_position(self):
        for kind in ("source", "retained"):
            values = list(self.fixture()); control, treatment = values[:2]; eligible = eligible_rows(values[2], 7, values[4])
            position = np.flatnonzero(eligible[control[0]])[0]
            treatment[0, position] = 5 if kind == "source" else 1
            values[-1][0]["treatment_order_sha256"] = draw_sha256(treatment[0])
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validate(values, replay=False)

    def test_same_stratum_self_rehashed_other_seed_cannot_replace_fixed_rng_replay(self):
        values = list(self.fixture()); control, treatment = values[:2]
        eligible = eligible_rows(values[2], 7, values[4]); strata = coupling_strata(values[2], 7)
        rng = np.random.default_rng(60)
        treatment[:] = np.stack([paired_epoch(row, values[5], eligible, strata, rng) for row in control])
        values[-1] = [{"epoch": k + 1, "control_order_sha256": draw_sha256(a),
            "treatment_order_sha256": draw_sha256(b), "changed_positions": int((a != b).sum()),
            "eligible_draws_control": int(eligible[a].sum()), "eligible_draws_treatment": int(eligible[b].sum())}
            for k, (a, b) in enumerate(zip(control, treatment))]
        with self.assertRaisesRegex(ValueError, "seed59"): self.validate(values)

    def test_fixed_epoch_sampler_is_one_based_immutable_and_rng_neutral(self):
        source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        torch_state = torch.random.get_rng_state().clone(); numpy_state = deepcopy(np.random.get_state())
        sampler = study.FixedEpochSampler(source)
        with self.assertRaises(ValueError): list(sampler)
        for epoch in (0, 3, True, 1.0):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError): sampler.set_epoch(epoch)
        source[:] = 99
        sampler.set_epoch(2); self.assertEqual(list(sampler), [4, 5, 6]); self.assertEqual(len(sampler), 3)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
