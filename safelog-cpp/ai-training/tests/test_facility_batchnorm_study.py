"""Tracked BN buffers stay fixed while affine and other student weights learn."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from safelog_ai.frozen_batchnorm import (apply_frozen_batchnorm, batchnorm_configuration,
    batchnorm_inventory, batchnorm_modules, batchnorm_state_sha256)
from scripts import facility_batchnorm_study as study


class FrozenBatchnormTests(TestCase):
    def fixture(self):
        return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 3),
                             nn.BatchNorm1d(3, eps=.003, momentum=.02), nn.Dropout(.2), nn.Linear(3, 1))

    def test_inventory_has_every_tracked_mean_variance_counter_and_affine_tensor(self):
        model = self.fixture(); inventory = batchnorm_inventory(model)
        self.assertEqual(inventory["layer_count"], 2); self.assertEqual(inventory["channel_count"], 7)
        self.assertEqual(inventory["buffer_tensor_count"], 6); self.assertEqual(inventory["affine_parameter_tensor_count"], 4)
        self.assertEqual(list(inventory["configuration"]), ["1", "4"])
        self.assertEqual(inventory["configuration"]["4"], {"channels": 3, "eps": .003, "momentum": .02})

    def test_policy_sets_only_batchnorm_eval_and_keeps_affine_and_other_gradients_enabled(self):
        model = self.fixture(); model.train(); before = batchnorm_configuration(model)
        inventory = apply_frozen_batchnorm(model)
        self.assertTrue(model.training); self.assertTrue(model[5].training)
        self.assertTrue(inventory["all_batchnorm_eval"]); self.assertTrue(inventory["affine_parameters_trainable"])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(batchnorm_configuration(model), before)

    def test_model_train_resets_bn_policy_and_each_epoch_must_reapply(self):
        model = self.fixture(); apply_frozen_batchnorm(model)
        for _ in range(3):
            model.train(); self.assertTrue(all(module.training for _, module in batchnorm_modules(model)))
            apply_frozen_batchnorm(model); self.assertTrue(all(not module.training for _, module in batchnorm_modules(model)))

    def test_training_forward_uses_existing_running_statistics_and_never_batch_statistics(self):
        module = nn.BatchNorm1d(2, eps=.001)
        with torch.no_grad():
            module.running_mean.copy_(torch.tensor([1., 2.])); module.running_var.copy_(torch.tensor([4., 9.]))
            module.weight.copy_(torch.tensor([2., 3.])); module.bias.copy_(torch.tensor([.2, .3]))
        before = batchnorm_state_sha256(module); module.train(); apply_frozen_batchnorm(module)
        value = torch.tensor([[5., 8.], [9., 11.]])
        actual = module(value)
        expected = (value - torch.tensor([1., 2.])) / torch.sqrt(torch.tensor([4., 9.]) + .001)
        expected = expected * torch.tensor([2., 3.]) + torch.tensor([.2, .3])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(batchnorm_state_sha256(module), before)

    def test_cpu_backward_and_optimizer_learn_affine_without_changing_any_bn_buffer(self):
        torch.manual_seed(5); model = self.fixture(); model.train(); apply_frozen_batchnorm(model)
        before = batchnorm_state_sha256(model)
        affine = [p for _, module in batchnorm_modules(model) for p in (module.weight, module.bias)]
        initial = [p.detach().clone() for p in affine]
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        loss = model(torch.randn(8, 4)).square().mean(); loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in affine))
        self.assertTrue(any((p.grad != 0).any() for p in affine))
        self.assertIsNotNone(model[0].weight.grad); self.assertIsNotNone(model[-1].weight.grad)
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(initial, affine)))
        self.assertEqual(batchnorm_state_sha256(model), before)

    def test_buffer_hash_binds_mean_variance_counter_but_excludes_trainable_affine(self):
        model = self.fixture(); before = batchnorm_state_sha256(model)
        with torch.no_grad(): model[1].weight.add_(.1)
        self.assertEqual(before, batchnorm_state_sha256(model))
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            current = deepcopy(model)
            with torch.no_grad(): getattr(current[1], name).add_(1)
            with self.subTest(name=name): self.assertNotEqual(before, batchnorm_state_sha256(current))

    def test_invalid_untracked_nonaffine_nan_or_frozen_teacher_is_rejected(self):
        for model in (nn.Linear(2, 2), nn.BatchNorm1d(2, track_running_stats=False), nn.BatchNorm1d(2, affine=False)):
            with self.subTest(model=model), self.assertRaises(ValueError): apply_frozen_batchnorm(model)
        model = self.fixture(); model.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "affine weights"): apply_frozen_batchnorm(model)
        self.assertTrue(all(module.training for _, module in batchnorm_modules(model)))
        model = self.fixture()
        with torch.no_grad(): model[1].running_var[0] = -1
        with self.assertRaises(ValueError): batchnorm_state_sha256(model)
        model = self.fixture()
        with torch.no_grad(): model[1].running_mean[0] = float("nan")
        with self.assertRaises(ValueError): batchnorm_inventory(model)

    def test_policy_and_buffer_hash_do_not_consume_training_rng(self):
        model = self.fixture(); before = torch.get_rng_state().clone()
        apply_frozen_batchnorm(model); batchnorm_state_sha256(model); batchnorm_configuration(model)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


class BatchnormProtocolTests(TestCase):
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
            return study.validate_protocol(p, Path("/bn-test"), args)

    def test_single_new_six_epoch_bn_candidate_reuses_done_normal_bn_lambda_four_control(self):
        result = self.validate(self.protocol())
        self.assertEqual(len(study.SOURCE_FILES), 86)
        self.assertEqual(result["control"], study.previous.RUN_NAME)
        self.assertEqual(result["distillation_weight_by_variant"], {"frozen": 4.0})
        self.assertEqual(result["new_training_epochs"], 6); self.assertFalse(result["reused_control_retrained"])
        self.assertEqual(result["batchnorm_policy"]["buffer_tensor_count"], 141)
        self.assertEqual(result["retention_candidate_gate"], study.previous.RETENTION_GATE)
        self.assertEqual(result["distillation_recipe"]["temperature"], 2.)

    def test_bn_policy_inventory_bool_initial_hash_or_existing_gate_changes_are_rejected(self):
        for field, key, value in (("batchnorm_policy", "layer_count", 46),
             ("batchnorm_policy", "affine_parameters_trainable", 1),
             ("batchnorm_policy", "training_forward_uses_current_batch_statistics", True),
             ("distillation_weight_by_variant", "frozen", 1.0),
             ("retention_candidate_gate", "maximum_known_other_class_ap_drop_vs_reference", .04)):
            p = self.protocol(); p[field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError): self.validate(p)
        p = self.protocol(); p["initial_batchnorm_buffers_sha256"] = "b" * 64
        with self.assertRaises(ValueError): self.validate(p)

    def test_all_old_seventy_four_and_new_runtime_sources_and_control_outputs_are_bound(self):
        for name in (study.previous.SOURCE_FILES[0], "safelog_ai/retention_distillation.py", study.NEW_SOURCES[0]):
            p = self.protocol(); p["source_sha256"][name] = "b" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "runtime source"): self.validate(p)
        p = self.protocol(); p["source_sha256"].pop(study.NEW_SOURCES[-1])
        with self.assertRaisesRegex(ValueError, "inventory"): self.validate(p)
        p = self.protocol(); p["reused_control_artifact_sha256"] = {}
        with self.assertRaises(ValueError): self.validate(p)

    def test_cli_never_retrains_control_or_initializes_student_from_trained_lambda_four(self):
        args = SimpleNamespace(variant="frozen", name=study.RUN_NAME, seed=56, epochs=6, patience=6,
            batch=8, draws_per_epoch=14248, backbone_lr=.00004, head_lr=.00025, auxiliary_weight=.5,
            initial=Path("/bn-test/runs") / study.REFERENCE / "best.pt",
            auxiliary_manifest=Path("/bn-test/data/facility-auxiliary-training/train.json"))
        self.assertEqual(self.validate(self.protocol(), args)["imgsz"], 640)
        for field, value in (("variant", "strong"), ("variant", "control"), ("name", study.CONTROL),
             ("initial", Path("/bn-test/runs") / study.CONTROL / "best.pt")):
            bad = deepcopy(args); setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(self.protocol(), bad)

    def test_prepare_data_uses_original_arrays_and_keeps_base_pair_proof_unchanged(self):
        draw = np.arange(12, dtype=np.int64).reshape(2, 6)
        base = {"epoch_draws": draw, "pair_proof": {"all_loss_weights_equal": True}}
        with patch.object(study.original, "prepare_data", return_value=base) as prepare:
            result = study.prepare_data(Path("/bn-test"), {}, "frozen", "auxiliary")
            prepare.assert_called_once_with(Path("/bn-test"), {}, "control", "auxiliary")
        self.assertIs(result["epoch_draws"], draw)
        self.assertEqual(result["pair_proof"], {"all_loss_weights_equal": True})
        self.assertEqual(result["control_reuse_proof"]["reused_control_weight"], 4.0)

    def test_trainer_preserves_optimizer_loader_rng_losses_and_applies_bn_after_model_train(self):
        root = study.ROOT
        old = ast.parse((root / "scripts/train_facility_retention_strength.py").read_text(encoding="utf-8"))
        new = ast.parse((root / "scripts/train_facility_batchnorm.py").read_text(encoding="utf-8"))
        def calls(tree, name):
            return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == name]
        for name in ("torch.manual_seed", "np.random.seed", "random.seed", "DataLoader", "torch.optim.AdamW",
             "torch.optim.lr_scheduler.CosineAnnealingLR", "teacher", "masked_bernoulli_kl", "masked_focal", "spatial_loss"):
            with self.subTest(name=name): self.assertEqual(calls(old, name), calls(new, name))
        main = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        epoch_loop = next(node for node in ast.walk(main) if isinstance(node, ast.For) and ast.unparse(node.target) == "epoch")
        train_at = next(k for k, node in enumerate(epoch_loop.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == "model.train")
        after = epoch_loop.body[train_at + 1]
        self.assertIsInstance(after, ast.Assign)
        self.assertEqual(ast.unparse(after.value.func), "apply_frozen_batchnorm")
