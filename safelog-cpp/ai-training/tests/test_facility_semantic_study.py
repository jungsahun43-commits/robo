"""Zero residual equivalence, frozen encoder, offline reload and fixed protocol."""
from copy import deepcopy
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import tempfile

import torch
from torch import nn

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as ORIGINAL_ARCH
from safelog_ai.frozen_batchnorm import apply_frozen_batchnorm, batchnorm_state_sha256
from safelog_ai import research_presence
from safelog_ai.semantic_residual_classifier import (SemanticResidualClassifier, ARCH, load_original_state,
    load_semantic_pretrained, semantic_pretrained_subset, semantic_encoder_state_sha256, model_inventory, original_state)
from scripts import facility_semantic_study as study


CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar", "wet_surface", "efflorescence", "surface_cavity"]


class SemanticModelTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4); torch.manual_seed(42)
        cls.base = AuxiliaryClassifier(7, pretrained=False)
        cls.model = SemanticResidualClassifier(7, pretrained=False)
        cls.checkpoint = {"architecture": ORIGINAL_ARCH, "classes": CLASSES, "auxiliary_classes": list(AUX_CLASSES),
            "imgsz": 640, "split_sha256": "a" * 64,
            "state_dict": {name: value.detach().clone() for name, value in cls.base.state_dict().items()}}

    def setUp(self):
        for head in (self.model.semantic_map_head, self.model.semantic_photo_head, self.model.semantic_auxiliary_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.base.load_state_dict(self.checkpoint["state_dict"], strict=True)
        load_original_state(self.model, self.checkpoint, CLASSES, "a" * 64)
        self.base.train(); self.model.train(); apply_frozen_batchnorm(self.base); apply_frozen_batchnorm(self.model)

    def test_all_original_324_tensors_and_zero_heads_preserve_exact_train_outputs(self):
        torch.manual_seed(8); image = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            old = self.base.forward_training(image); new = self.model.forward_training(image)
        for left, right in zip(old, new): self.assertTrue(torch.equal(left, right))
        self.assertTrue(all(torch.equal(value, self.checkpoint["state_dict"][name]) for name, value in original_state(self.model).items()))
        inventory = model_inventory(self.model)
        self.assertEqual(inventory["original_state_tensor_count"], 324)
        self.assertEqual(inventory["new_head_state_tensor_count"], 6)
        self.assertEqual(inventory["new_semantic_head_parameter_count"], 21345)
        self.assertEqual(inventory["parameter_count"], 31085624)
        self.assertTrue(inventory["new_semantic_heads_zero"])

    def test_encoder_remains_eval_no_grad_and_consumes_no_rng_after_outer_train(self):
        self.model.train()
        self.assertTrue(all(not module.training for module in self.model.semantic_encoder.modules()))
        self.assertTrue(all(not p.requires_grad for p in self.model.semantic_encoder.parameters()))
        image = torch.randn(2, 3, 64, 64, requires_grad=True); before = torch.get_rng_state().clone()
        low, pooled = self.model.semantic_encoder(image)
        self.assertFalse(low.requires_grad); self.assertFalse(pooled.requires_grad)
        self.assertEqual(tuple(low.shape), (2, 192, 8, 8)); self.assertEqual(tuple(pooled.shape), (2, 768))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cpu_step_learns_all_new_heads_with_encoder_and_original_bn_buffers_unchanged(self):
        encoder_before = semantic_encoder_state_sha256(self.model); bn_before = batchnorm_state_sha256(self.model)
        heads = [self.model.semantic_map_head, self.model.semantic_photo_head, self.model.semantic_auxiliary_head]
        initial = [head.weight.detach().clone() for head in heads]
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.assertFalse({id(p) for p in trainable} & {id(p) for p in self.model.semantic_encoder.parameters()})
        optimizer = torch.optim.SGD(trainable, lr=.0001)
        photo, maps, aux = self.model.forward_training(torch.randn(2, 3, 64, 64))
        (photo.sum() + maps.mean() + aux.sum()).backward()
        for head in heads:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            self.assertTrue((head.weight.grad != 0).any())
        self.assertTrue(all(p.grad is None for p in self.model.semantic_encoder.parameters()))
        optimizer.step()
        self.assertTrue(all(not torch.equal(old, head.weight) for old, head in zip(initial, heads)))
        self.assertEqual(encoder_before, semantic_encoder_state_sha256(self.model))
        self.assertEqual(bn_before, batchnorm_state_sha256(self.model))

    def test_strict_original_and_official_inventory_reject_missing_unknown_or_nonzero_heads(self):
        broken = dict(self.checkpoint); broken["state_dict"] = dict(self.checkpoint["state_dict"]); broken["state_dict"].pop("mix")
        with self.assertRaises(ValueError): load_original_state(self.model, broken, CLASSES, "a" * 64)
        with torch.no_grad(): self.model.semantic_map_head.bias[0] = 1
        with self.assertRaisesRegex(ValueError, "exactly zero"): load_original_state(self.model, self.checkpoint, CLASSES, "a" * 64)
        with torch.no_grad(): self.model.semantic_map_head.bias.zero_()
        official = {"classifier.0." + name[len("pool_norm."):] if name.startswith("pool_norm.") else name: tensor
            for name, tensor in self.model.semantic_encoder.state_dict().items()}
        official.update({"classifier.2.weight": torch.zeros(1000, 768), "classifier.2.bias": torch.zeros(1000)})
        mapped = semantic_pretrained_subset(self.model, official)
        self.assertEqual(len(mapped), 180)
        self.assertTrue(torch.equal(mapped["pool_norm.weight"], official["classifier.0.weight"]))
        for kind in ("poolnorm", "unknown", "shape"):
            values = dict(official)
            if kind == "poolnorm": values.pop("classifier.0.bias")
            elif kind == "unknown": values["foreign.weight"] = torch.zeros(1)
            else: values["classifier.0.weight"] = torch.zeros(1)
            with self.subTest(kind=kind), self.assertRaises(ValueError): semantic_pretrained_subset(self.model, values)
        with tempfile.TemporaryDirectory(dir=study.ROOT / "runs") as directory:
            path = Path(directory) / "incorrect.pth"; path.write_bytes(b"incorrect official weight bytes")
            with patch("safelog_ai.semantic_residual_classifier.torch.load") as load:
                with self.assertRaisesRegex(ValueError, "hash prefix"): load_semantic_pretrained(self.model, path)
                load.assert_not_called()

    def test_research_factory_strict_reload_is_offline_and_keeps_public_interface(self):
        self.model.eval()
        checkpoint = {"architecture": ARCH, "classes": CLASSES, "auxiliary_classes": list(AUX_CLASSES),
            "imgsz": 640, "state_dict": self.model.state_dict()}
        with patch.object(research_presence.torch, "load", return_value=checkpoint), \
             patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("No network allowed")) as download:
            classifier = research_presence.ResearchPresenceClassifier(Path("offline.pth"), device="cpu")
            download.assert_not_called()
        self.assertEqual(classifier.classes, CLASSES); self.assertEqual(classifier.imgsz, 640)
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad(): self.assertTrue(torch.equal(classifier.model(image), self.model(image)))
        self.assertTrue(all(not p.requires_grad for p in classifier.model.semantic_encoder.parameters()))


class SemanticProtocolTests(TestCase):
    def metadata(self): return json.loads((study.ROOT / study.METADATA_PATH).read_text(encoding="utf-8"))

    def protocol(self):
        with patch.object(study, "pretrained_metadata", return_value=self.metadata()), patch.object(study, "sha", return_value="a" * 64):
            p = study.fixed_values()
        p["source_sha256"] = {name: "a" * 64 for name in study.SOURCE_FILES}
        p["protected_file_sha256"] = {"data/original.json": "a" * 64}
        p["reused_control_artifact_sha256"] = {f"runs/{study.CONTROL}/history.json": "a" * 64}
        return p

    def validate(self, p):
        with patch.object(study, "pretrained_metadata", return_value=self.metadata()), \
             patch.object(study, "local_path", side_effect=lambda root, name: Path(root) / name), \
             patch.object(study, "sha", return_value="a" * 64), \
             patch.object(study, "expected_protected_hashes", return_value={"data/original.json": "a" * 64}), \
             patch.object(study, "completed_records", return_value={f"runs/{study.CONTROL}/history.json": "a" * 64}):
            return study.validate_protocol(p, study.ROOT)

    def test_policy_retains_entire_base_and_same_lr_bn_teacher_data_and_one_six_epoch_budget(self):
        p = self.validate(self.protocol())
        self.assertEqual(len(study.SOURCE_FILES), 112); self.assertNotIn("head_lr_policy", p)
        self.assertEqual(p["semantic_model_inventory"]["original_state_tensor_count"], 324)
        self.assertEqual(p["semantic_model_inventory"]["new_head_state_tensor_count"], 6)
        self.assertEqual(p["optimizer_policy"]["control_head_lr"], p["optimizer_policy"]["candidate_head_lr"])
        self.assertFalse(p["optimizer_policy"]["learning_rates_changed"])
        self.assertEqual(p["batchnorm_policy"], study.previous.BATCHNORM_POLICY)
        self.assertEqual(p["new_training_epochs"], 6); self.assertFalse(p["reused_control_retrained"])
        self.assertEqual(p["retention_candidate_gate"], study.previous.RETENTION_GATE)

    def test_frozen_encoder_zero_heads_actual_weights_and_previous_sources_are_immutable(self):
        for field, key, value in (("semantic_features_policy", "encoder_parameters_frozen", 1),
             ("semantic_features_policy", "new_semantic_heads_initialized_zero", False),
             ("optimizer_policy", "candidate_head_lr", .00025),
             ("retention_candidate_gate", "maximum_known_other_class_ap_drop_vs_reference", .04)):
            p = self.protocol(); p[field][key] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.validate(p)
        p = self.protocol(); p["semantic_encoder_state_sha256"] = "b" * 64
        with self.assertRaises(ValueError): self.validate(p)
        p = self.protocol(); p["semantic_pretrained_weights_sha256"] = "983f1562" + "b" * 56
        with self.assertRaises(ValueError): self.validate(p)
        p = self.protocol(); p["source_sha256"][study.previous.SOURCE_FILES[0]] = "b" * 64
        with self.assertRaisesRegex(ValueError, "runtime source"): self.validate(p)

    def test_original_control_array_wrapper_does_not_create_labels_or_change_pair_proof(self):
        source = {"epoch_draws": object(), "pair_proof": {"all_loss_weights_equal": True}}
        with patch.object(study.previous, "prepare_data", return_value=source) as prepare:
            data = study.prepare_data(study.ROOT, {}, "semantic", "auxiliary")
            prepare.assert_called_once_with(study.ROOT, {}, "low_lr", "auxiliary")
        self.assertEqual(data["pair_proof"], {"all_loss_weights_equal": True})
        self.assertEqual(data["control_reuse_proof"]["reused_control_head_lr"], .0001)
        with patch.object(study.previous, "prepare_data") as prepare:
            with self.assertRaises(ValueError): study.prepare_data(study.ROOT, {}, "low_lr")
            prepare.assert_not_called()
