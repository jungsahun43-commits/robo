"""Fixed stronger loss, historical control reuse, and unchanged training recipe."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from scripts import facility_retention_strength_study as study


class StrengthProtocolTests(TestCase):
    def protocol(self):
        p = study.fixed_values()
        p["source_sha256"] = {name: "a" * 64 for name in study.SOURCE_FILES}
        p["protected_file_sha256"] = {"data/original.json": "a" * 64}
        p["reused_control_artifact_sha256"] = {f"runs/{study.CONTROL}/history.json": "a" * 64}
        return p

    def validate(self, p, args=None):
        with patch.object(study, "local_path", side_effect=lambda root, name: Path(root) / name), \
             patch.object(study, "sha", return_value="a" * 64), \
             patch.object(study, "expected_protected_hashes", return_value={"data/original.json": "a" * 64}), \
             patch.object(study, "completed_records", return_value={f"runs/{study.CONTROL}/history.json": "a" * 64}):
            return study.validate_protocol(p, Path("/strength-test"), args)

    def test_only_one_weight_four_candidate_spends_six_new_epochs(self):
        p = self.protocol(); actual = self.validate(p)
        self.assertIsNot(actual, p)
        self.assertEqual(actual["distillation_weight_by_variant"], {"strong": 4.0})
        self.assertEqual(actual["new_training_epochs"], 6)
        self.assertEqual(actual["reused_control_training_epochs"], 6)
        self.assertFalse(actual["reused_control_retrained"])
        self.assertEqual(actual["control"], study.frozen.NAMES["control"])
        self.assertEqual(actual["previous_treatment"], study.frozen.NAMES["distill"])
        self.assertEqual(study.VARIANTS, ("strong",))
        self.assertEqual(len(study.SOURCE_FILES), 74)

    def test_teacher_temperature_mask_and_existing_gates_remain_unchanged(self):
        self.assertEqual(study.RETENTION_GATE, study.frozen.RETENTION_GATE)
        self.assertEqual(study.fixed_values()["research_candidate_gate"], study.frozen.fixed_values()["research_candidate_gate"])
        for key in ("temperature", "teacher_run", "teacher_weights_sha256", "teacher_state_sha256", "teacher_eval",
             "teacher_no_grad", "retained_classes", "retained_indices", "excluded_primary_indices", "mask", "direction",
             "reduction", "temperature_squared_scaling", "loss_compute_dtype", "teacher_forward_both_arms"):
            with self.subTest(key=key): self.assertEqual(study.RECIPE[key], study.frozen.RECIPE[key])

    def test_old_sixty_one_source_bytes_and_new_source_bytes_are_required(self):
        p = self.protocol(); p["source_sha256"].pop(study.frozen.SOURCE_FILES[-1])
        with self.assertRaisesRegex(ValueError, "inventory"): self.validate(p)
        for name in (study.frozen.SOURCE_FILES[0], "safelog_ai/retention_distillation.py", study.NEW_SOURCES[0]):
            p = self.protocol(); p["source_sha256"][name] = "b" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "runtime source"): self.validate(p)

    def test_weight_budget_reuse_boolean_or_gate_tuning_is_rejected(self):
        for key, value in (("new_training_epochs", 12), ("reused_control_training_epochs", 0),
             ("reused_control_retrained", 0), ("declared_before_training", 1), ("source_test_inference_executed", True),
             ("previous_comparison_sha256", "b" * 64)):
            p = self.protocol(); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(p)
        for field, key, value in (("distillation_weight_by_variant", "strong", 1.0),
             ("distillation_recipe", "temperature", 3.), ("distillation_recipe", "teacher_no_grad", 1),
             ("retention_candidate_gate", "maximum_known_other_class_ap_drop_vs_reference", .03)):
            p = self.protocol(); p[field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError): self.validate(p)

    def test_cli_cannot_train_a_fresh_control_or_initialize_from_previous_student(self):
        args = SimpleNamespace(variant="strong", name=study.RUN_NAME, seed=56, epochs=6, patience=6,
            batch=8, draws_per_epoch=14248, backbone_lr=.00004, head_lr=.00025, auxiliary_weight=.5,
            initial=Path("/strength-test/runs") / study.REFERENCE / "best.pt",
            auxiliary_manifest=Path("/strength-test/data/facility-auxiliary-training/train.json"))
        self.assertEqual(self.validate(self.protocol(), args)["imgsz"], 640)
        for field, value in (("variant", "control"), ("variant", "distill"), ("name", study.CONTROL),
             ("epochs", 12), ("initial", Path("/strength-test/runs") / study.PREVIOUS / "best.pt")):
            bad = deepcopy(args); setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(self.protocol(), bad)

    def test_reused_control_artifact_hashes_and_protected_inventory_are_bound(self):
        for field in ("reused_control_artifact_sha256", "protected_file_sha256"):
            p = self.protocol(); p[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(p)


class StrengthReuseAndTrainerTests(TestCase):
    def test_prepare_data_wraps_original_control_and_preserves_base_pair_proof(self):
        draw = np.arange(12, dtype=np.int64).reshape(2, 6)
        base = {"epoch_draws": draw, "pair_proof": {"photo_targets_equal": True, "all_loss_weights_equal": True}}
        original_proof = deepcopy(base["pair_proof"])
        with patch.object(study.frozen, "prepare_data", return_value=base) as prepare:
            result = study.prepare_data(Path("/strength-test"), self.protocol_stub(), "strong", "auxiliary")
            prepare.assert_called_once_with(Path("/strength-test"), self.protocol_stub(), "control", "auxiliary")
        self.assertIs(result["epoch_draws"], draw)
        self.assertEqual(result["pair_proof"], original_proof)
        self.assertFalse(result["control_reuse_proof"]["reused_control_retrained"])
        with patch.object(study.frozen, "prepare_data") as prepare:
            with self.assertRaises(ValueError): study.prepare_data(Path("/strength-test"), {}, "control")
            prepare.assert_not_called()

    @staticmethod
    def protocol_stub(): return {"sampler_plan_sha256": "a" * 64}

    def completed(self, changes=None):
        def record(path):
            is_control = study.CONTROL in Path(path).as_posix()
            value = {"status": "complete", "actual_epochs": 6, "model_variant": "control" if is_control else "distill",
                "distillation_weight": 0.0 if is_control else 1.0, "study_protocol_sha256": study.PRIOR_PROTOCOL_SHA,
                "weights_sha256": study.CONTROL_WEIGHTS_SHA if is_control else study.PREVIOUS_WEIGHTS_SHA}
            if is_control and changes: value.update(changes)
            return value
        def digest(path):
            text = Path(path).as_posix()
            if text.endswith("/best.pt"):
                return study.CONTROL_WEIGHTS_SHA if study.CONTROL in text else study.PREVIOUS_WEIGHTS_SHA
            return "a" * 64
        with patch.object(study, "read", side_effect=record), \
             patch.object(study, "local_path", side_effect=lambda root, name: Path(root) / name), \
             patch.object(study, "sha", side_effect=digest):
            return study.completed_records(Path("/strength-test"))

    def test_reuse_requires_completed_six_epoch_zero_control_and_verified_weights(self):
        result = self.completed()
        self.assertEqual(len(result), 2 * len(study.CONTROL_ARTIFACTS))
        for changes in ({"status": "running"}, {"actual_epochs": 5}, {"actual_epochs": True},
             {"distillation_weight": 4.0}, {"model_variant": "strong"}, {"weights_sha256": "b" * 64},
             {"study_protocol_sha256": "b" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.completed(changes)

    def test_trainer_preserves_masked_counts_initial_state_optimizer_loader_and_teacher_loss(self):
        root = study.ROOT
        old = ast.parse((root / "scripts/train_facility_retention.py").read_text(encoding="utf-8"))
        new = ast.parse((root / "scripts/train_facility_retention_strength.py").read_text(encoding="utf-8"))
        for name in ("photo_target_counts", "load_initial_state"):
            left = next(x for x in old.body if isinstance(x, ast.FunctionDef) and x.name == name)
            right = next(x for x in new.body if isinstance(x, ast.FunctionDef) and x.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right))
        def calls(tree, function):
            return [ast.dump(x) for x in ast.walk(tree) if isinstance(x, ast.Call) and ast.unparse(x.func) == function]
        for function in ("torch.manual_seed", "np.random.seed", "random.seed", "DataLoader", "torch.optim.AdamW",
             "torch.optim.lr_scheduler.CosineAnnealingLR", "teacher", "masked_bernoulli_kl", "masked_focal", "spatial_loss"):
            with self.subTest(function=function): self.assertEqual(calls(old, function), calls(new, function))

    def test_loader_unknown_counts_remain_unknown_after_clamping(self):
        from scripts.train_facility_retention_strength import photo_target_counts
        labels = torch.tensor([[1.] * 7, [0.] * 7, [0.] * 7])
        known = torch.tensor([[1.] * 7, [1.] * 7, [0.] * 7])
        counts = photo_target_counts(labels, known)
        self.assertTrue(torch.equal(counts, torch.tensor([[1, 1, 1]] * 7)))
