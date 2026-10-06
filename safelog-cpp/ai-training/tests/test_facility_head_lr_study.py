"""Fixed lower head LR, observed cosine schedule, and unchanged BN/teacher/data."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from scripts import facility_head_lr_study as study
from scripts import run_facility_head_lr as runner


class HeadLrStudyTests(TestCase):
    def protocol(self):
        p = study.fixed_values(); p["source_sha256"] = {name: "a" * 64 for name in study.SOURCE_FILES}
        p["protected_file_sha256"] = {"data/original.json": "a" * 64}
        p["reused_control_artifact_sha256"] = {f"runs/{study.CONTROL}/history.json": "a" * 64}
        return p

    def validate(self, p, args=None):
        with patch.object(study, "local_path", side_effect=lambda root, name: Path(root) / name), \
             patch.object(study, "sha", return_value="a" * 64), \
             patch.object(study, "expected_protected_hashes", return_value={"data/original.json": "a" * 64}), \
             patch.object(study, "completed_records", return_value={f"runs/{study.CONTROL}/history.json": "a" * 64}):
            return study.validate_protocol(p, Path("/head-lr-test"), args)

    def test_real_disposable_cosine_scheduler_matches_complete_curves_with_shared_floor(self):
        curves = []
        for head_lr in (study.CONTROL_HEAD_LR, study.CANDIDATE_HEAD_LR):
            backbone = torch.nn.Parameter(torch.zeros(1)); head = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.AdamW([{"params": [backbone], "lr": study.BACKBONE_LR},
                {"params": [head], "lr": head_lr}], weight_decay=.0002)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=.000005)
            actual = [tuple(group["lr"] for group in optimizer.param_groups)]
            for _ in range(6):
                optimizer.step(); scheduler.step()
                actual.append(tuple(group["lr"] for group in optimizer.param_groups))
            for step, values in enumerate(actual):
                self.assertTrue(math.isclose(values[0], study.cosine_curve(study.BACKBONE_LR)[step], rel_tol=1e-12, abs_tol=1e-15))
                self.assertTrue(math.isclose(values[1], study.cosine_curve(head_lr)[step], rel_tol=1e-12, abs_tol=1e-15))
            curves.append(actual)
        self.assertAlmostEqual(curves[1][0][1] / curves[0][0][1], .4)
        self.assertAlmostEqual(curves[1][-1][1] / curves[0][-1][1], 1.)
        self.assertNotAlmostEqual(curves[1][3][1] / curves[0][3][1], .4)
        for lr, epochs, floor in ((True, 6, .000005), (.0001, True, .000005), (.0001, 0, .000005),
             (.0001, 6, 0), (float("nan"), 6, .000005), (.0001, 6, .001)):
            with self.assertRaises(ValueError): study.cosine_curve(lr, epochs, floor)

    def test_single_candidate_reuses_bn_control_and_keeps_losses_teacher_policy_and_gates(self):
        p = self.validate(self.protocol())
        self.assertEqual(len(study.SOURCE_FILES), 97)
        self.assertEqual(p["head_lr"], .0001); self.assertEqual(p["backbone_lr"], .00004)
        self.assertEqual(p["control"], study.previous.RUN_NAME)
        self.assertEqual(p["new_training_epochs"], 6); self.assertFalse(p["reused_control_retrained"])
        self.assertEqual(p["distillation_weight_by_variant"], {"low_lr": 4.0})
        self.assertEqual(p["batchnorm_policy"], study.previous.BATCHNORM_POLICY)
        self.assertEqual(p["retention_candidate_gate"], study.previous.RETENTION_GATE)
        self.assertEqual(p["distillation_recipe"]["temperature"], 2.)
        self.assertFalse(p["head_lr_policy"]["uniform_ratio_across_curve"])

    def test_learning_rates_scheduler_floor_bn_gate_and_boolean_mutations_fail(self):
        for key, value in (("head_lr", .00025), ("backbone_lr", .00001), ("new_training_epochs", 12),
                           ("declared_before_training", 1), ("source_test_inference_executed", 0)):
            p = self.protocol(); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(p)
        for field, key, value in (("optimizer_schedule", "eta_min", 0), ("optimizer_schedule", "T_max", 5),
             ("head_lr_policy", "uniform_ratio_across_curve", True),
             ("batchnorm_policy", "affine_parameters_trainable", False),
             ("retention_candidate_gate", "maximum_known_other_class_ap_drop_vs_reference", .03)):
            p = self.protocol(); p[field][key] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(p)

    def test_all_old_eighty_six_sources_and_reused_control_artifacts_remain_bound(self):
        for name in (study.previous.SOURCE_FILES[0], "safelog_ai/frozen_batchnorm.py", study.NEW_SOURCES[0]):
            p = self.protocol(); p["source_sha256"][name] = "b" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "runtime source"): self.validate(p)
        p = self.protocol(); p["source_sha256"].pop(study.NEW_SOURCES[-1])
        with self.assertRaisesRegex(ValueError, "inventory"): self.validate(p)
        p = self.protocol(); p["reused_control_artifact_sha256"] = {}
        with self.assertRaises(ValueError): self.validate(p)

    def test_cli_requires_actual_lower_head_lr_and_fresh_original_initializer(self):
        args = SimpleNamespace(variant="low_lr", name=study.RUN_NAME, seed=56, epochs=6, patience=6, batch=8,
            draws_per_epoch=14248, backbone_lr=.00004, head_lr=.0001, auxiliary_weight=.5,
            initial=Path("/head-lr-test/runs") / study.REFERENCE / "best.pt",
            auxiliary_manifest=Path("/head-lr-test/data/facility-auxiliary-training/train.json"))
        self.assertEqual(self.validate(self.protocol(), args)["imgsz"], 640)
        for field, value in (("head_lr", .00025), ("head_lr", True), ("backbone_lr", .00001),
             ("variant", "frozen"), ("name", study.CONTROL),
             ("initial", Path("/head-lr-test/runs") / study.CONTROL / "best.pt")):
            bad = deepcopy(args); setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(self.protocol(), bad)

    def test_data_wrapper_retains_original_draw_array_and_pair_proof(self):
        draws = np.arange(12, dtype=np.int64).reshape(2, 6)
        base = {"epoch_draws": draws, "pair_proof": {"all_loss_weights_equal": True}}
        with patch.object(study.previous, "prepare_data", return_value=base) as prepare:
            result = study.prepare_data(Path("/head-lr-test"), {}, "low_lr", "auxiliary")
            prepare.assert_called_once_with(Path("/head-lr-test"), {}, "frozen", "auxiliary")
        self.assertIs(result["epoch_draws"], draws)
        self.assertEqual(result["pair_proof"], {"all_loss_weights_equal": True})
        self.assertEqual(result["control_reuse_proof"]["reused_control_head_lr"], .00025)

    def test_trainer_reuses_frozen_contract_functions_and_unchanged_algorithm_calls(self):
        from scripts import train_facility_head_lr as new_module, train_facility_batchnorm as old_module
        self.assertIs(new_module.load_initial_state, old_module.load_initial_state)
        self.assertIs(new_module.photo_target_counts, old_module.photo_target_counts)
        root = study.ROOT
        old = ast.parse((root / "scripts/train_facility_batchnorm.py").read_text(encoding="utf-8"))
        new = ast.parse((root / "scripts/train_facility_head_lr.py").read_text(encoding="utf-8"))
        def calls(tree, name):
            return [ast.dump(x) for x in ast.walk(tree) if isinstance(x, ast.Call) and ast.unparse(x.func) == name]
        for name in ("torch.manual_seed", "np.random.seed", "random.seed", "DataLoader", "torch.optim.AdamW",
             "torch.optim.lr_scheduler.CosineAnnealingLR", "teacher", "masked_bernoulli_kl", "masked_focal",
             "spatial_loss", "apply_frozen_batchnorm"):
            with self.subTest(name=name): self.assertEqual(calls(old, name), calls(new, name))

    def test_runner_trains_only_candidate_with_explicit_lr_and_finishes_source_val_pipeline(self):
        command = runner.training_command("python")
        self.assertEqual(command[command.index("--name") + 1], study.RUN_NAME)
        self.assertEqual(command[command.index("--head-lr") + 1], "0.0001")
        self.assertEqual(command[command.index("--epochs") + 1], "6")
        stages = runner.completion_commands("python")
        self.assertEqual([name for name, _ in stages], ["source-val", "small-region-audit", "technical-verification", "result-report", "result-plot"])
        selection = stages[0][1]
        self.assertEqual(selection[selection.index("--grids") + 1], "1")
        plot = stages[-1][1]
        self.assertEqual(plot[plot.index("--output") + 1], "reports/facility-head-lr-study-comparison.png")
        self.assertIn("-m", plot); self.assertNotIn(study.CONTROL, command)
