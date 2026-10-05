"""Synthetic-only resolution study guards; never read real VAL/TEST photos."""
import argparse
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from safelog_ai.auxiliary_classifier import AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.resolution_classifier import ARCH as RESOLUTION_ARCH
from safelog_ai.spatial_classifier import spatial_loss
from scripts.facility_resolution_study import (
    CLASSES, DOMAINS, ResolutionPhotos, build_sampling, loss_grid,
    supervision_weights, validate_items, validate_protocol,
)
import scripts.facility_resolution_study as study


class ResolutionStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def manifest(self):
        full = []
        for domain in DOMAINS:
            for index, targets in enumerate(([1, 0], [0, 1], [0, 0])):
                full.append({"image": f"data/synthetic/{domain}-{index}.png",
                    "pixel_target": f"data/synthetic/{domain}-{index}.npz",
                    "domain": domain, "targets": list(targets) + [-1] * 5,
                    "split": "train"})
        crops = [{"image": f"data/synthetic/{domain}-crop.png",
                  "pixel_target": f"data/synthetic/{domain}-crop.npz",
                  "domain": domain, "targets": [-1] * 7,
                  "parent_image": f"data/synthetic/{domain}-0.png",
                  "parent_split": "train", "split": "train", "box_in_parent": [0, 0, 32, 32]}
                 for domain in DOMAINS]
        return {"split": "train", "classes": list(CLASSES), "full_count": len(full),
                "items": full + crops,
                "audit": {"per_label_pixel_cells": {label: {"positive": 10, "negative": 30}
                                                    for label in CLASSES}}}

    def test_loss_grid_identity_and_native_area_projection_preserve_target(self):
        original = torch.randn(2, 7, 80, 80, requires_grad=True)
        self.assertIs(loss_grid(original, (80, 80)), original)
        self.assertIs(loss_grid(original, (2, 7, 80, 80)), original)
        native = torch.randn(2, 7, 120, 120, requires_grad=True)
        target = torch.randint(0, 2, (2, 7, 80, 80)).float()
        snapshot = target.clone()
        projected = loss_grid(native, target.shape)
        self.assertTrue(torch.equal(projected, F.interpolate(native, size=(80, 80), mode="area")))
        self.assertTrue(torch.equal(target, snapshot))
        projected.square().mean().backward()
        self.assertTrue(torch.isfinite(native.grad).all())
        self.assertGreater(torch.count_nonzero(native.grad).item(), 0)

    def test_loss_grid_rejects_upsampling_mixed_axes_and_wrong_batch_class(self):
        logits = torch.zeros(2, 7, 80, 80)
        for shape in ((120, 120), (40, 120), (120, 40), (0, 80),
                      (True, 80), (80., 80), (80,), (3, 7, 80, 80), (2, 8, 80, 80)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                loss_grid(logits, shape)
        for invalid in (logits[0], logits.long()):
            with self.subTest(dimensions=invalid.ndim), self.assertRaises(ValueError):
                loss_grid(invalid, (80, 80))

    def test_unknown_pixel_classes_are_not_reinterpreted_as_negative_targets(self):
        native = torch.randn(1, 7, 120, 120, requires_grad=True)
        target = torch.zeros(1, 7, 80, 80)
        known = torch.zeros(1, 7)
        weights = torch.ones(7)
        zero = spatial_loss(loss_grid(native, target.shape), target, known, weights)
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertTrue(torch.equal(native.grad, torch.zeros_like(native)))
        known[0, 0] = 1
        target[0, 0, 20:30, 20:30] = 1
        maps = loss_grid(native.detach(), target.shape)
        changed = maps.clone()
        changed[:, 1:] = 50.
        self.assertTrue(torch.equal(spatial_loss(maps, target, known, weights),
                                    spatial_loss(changed, target, known, weights)))

    def test_train_item_validation_preserves_full_labels_and_crop_membership(self):
        manifest = self.manifest()
        before = copy.deepcopy(manifest)
        validate_items(manifest)
        self.assertEqual(manifest, before)

    def test_train_item_validation_rejects_scope_targets_paths_and_parent_changes(self):
        for mutation in ("val", "classes", "duplicate_image", "duplicate_pixel", "absolute",
                         "traversal", "outside_data", "bool_target", "float_target", "unknown_domain",
                         "short_targets", "unlisted_parent", "heldout_parent", "different_parent_domain",
                         "full_parent", "empty", "bool_full_count"):
            manifest = self.manifest()
            row = manifest["items"][0]
            if mutation == "val": manifest["split"] = "val"
            elif mutation == "classes": manifest["classes"][0] = "all_safe"
            elif mutation == "duplicate_image": manifest["items"][1]["image"] = row["image"]
            elif mutation == "duplicate_pixel": manifest["items"][1]["pixel_target"] = row["pixel_target"]
            elif mutation == "absolute": row["image"] = "C:/outside.png"
            elif mutation == "traversal": row["image"] = "data/../outside.png"
            elif mutation == "outside_data": row["pixel_target"] = "runs/private.npz"
            elif mutation == "bool_target": row["targets"][0] = True
            elif mutation == "float_target": row["targets"][0] = 1.
            elif mutation == "unknown_domain": row["domain"] = "new_untested_source"
            elif mutation == "short_targets": row["targets"].pop()
            elif mutation == "unlisted_parent": manifest["items"][-1]["parent_image"] = "data/unknown.png"
            elif mutation == "heldout_parent": manifest["items"][-1]["parent_split"] = "val"
            elif mutation == "different_parent_domain": manifest["items"][-1]["parent_image"] = "data/synthetic/dacl-0.png"
            elif mutation == "full_parent": row["parent_image"] = "data/synthetic/dacl-1.png"
            elif mutation == "empty": manifest["items"] = []
            elif mutation == "bool_full_count": manifest["full_count"] = True
            before = copy.deepcopy(manifest)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_items(manifest)
            self.assertEqual(manifest, before)

    def test_sampler_retains_source_mass_and_expected_known_class_balance(self):
        manifest = self.manifest()
        before = copy.deepcopy(manifest)
        result = build_sampling(manifest["items"], manifest["full_count"])
        self.assertEqual(manifest, before)
        for domain, index in DOMAINS.items():
            selected = result["domains"] == index
            expected_mass = (.7, .1, .2)[index]
            self.assertAlmostEqual(result["sampling"][selected].sum().item(), expected_mass, places=12)
        # Three positive rows per class among all twelve rows give a
        # 2.0 multiplier; each source has full weights [2,2,1] and one
        # unknown crop of weight1. This is the fixed original sampler recipe.
        for index in DOMAINS.values():
            weight = (.7, .1, .2)[index] / 6.
            full_start = index * 3
            torch.testing.assert_close(result["sampling"][full_start:full_start + 3],
                                       torch.tensor([2., 2., 1.], dtype=torch.float64) * weight,
                                       rtol=0., atol=1e-12)
            self.assertAlmostEqual(result["sampling"][manifest["full_count"] + index].item(), weight, places=12)
        expected_targets = torch.tensor([row["targets"] for row in manifest["items"]])
        self.assertTrue(torch.equal(result["labels"], expected_targets))
        self.assertTrue((result["labels"][:, 2:] == -1).all())
        torch.testing.assert_close(result["photo_weights"][:, :2],
                                   torch.full((3, 2), 1.5), rtol=1e-6, atol=1e-7)

    def test_sampling_rejects_missing_sources_invalid_proportions_and_label_changes(self):
        manifest = self.manifest()
        for proportions in ((.7, .1, .3), (.7, .3), (.7, .3, 0.),
                            (True, .1, .2), (.7, float("nan"), .2)):
            with self.subTest(proportions=proportions), self.assertRaises(ValueError):
                build_sampling(manifest["items"], manifest["full_count"], proportions=proportions)
        for mutation in ("missing_source", "changed_target", "full_boundary", "undeclared_source"):
            changed = copy.deepcopy(manifest)
            if mutation == "missing_source":
                changed["items"] = [row for row in changed["items"] if row["domain"] != "codebrim"]
                changed["full_count"] = 6
            elif mutation == "changed_target": changed["items"][0]["targets"][0] = 2
            elif mutation == "full_boundary": changed["full_count"] = 0
            elif mutation == "undeclared_source": changed["items"][0]["domain"] = "factory"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_sampling(changed["items"], changed["full_count"])

    def test_original_pixel_and_auxiliary_weight_recipe_uses_only_known_native_tags(self):
        manifest = self.manifest()
        manifest["audit"]["per_label_pixel_cells"][CLASSES[0]] = {"positive": 1, "negative": 200}
        manifest["audit"]["per_label_pixel_cells"][CLASSES[1]] = {"positive": 20, "negative": 1}
        auxiliary = {"items": [{"targets": [1] + [0] * 18},
                                {"targets": [0, 1] + [0] * 17},
                                {"targets": [0] * 19}]}
        before = copy.deepcopy((manifest, auxiliary))
        weights = supervision_weights(manifest, auxiliary)
        torch.testing.assert_close(weights["pixel_weights"],
                                   torch.tensor([20., 1.] + [3.] * 5), rtol=0., atol=0.)
        torch.testing.assert_close(weights["auxiliary_weights"],
                                   torch.tensor([2., 2.] + [3.] * 17), rtol=0., atol=0.)
        self.assertEqual((manifest, auxiliary), before)
        for mutation in ("unknown", "wrong_vocabulary_shape"):
            changed = copy.deepcopy(auxiliary)
            if mutation == "unknown": changed["items"][0]["targets"][0] = -1
            else:
                for row in changed["items"]: row["targets"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                supervision_weights(manifest, changed)

    def test_manifest_resolved_files_stay_in_data_and_missing_files_fail(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in manifest["items"]:
                for field in ("image", "pixel_target"):
                    path = root / row[field]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic provenance-only fixture")
            validate_items(manifest, root=root)
            (root / manifest["items"][0]["pixel_target"]).unlink()
            with self.assertRaises(ValueError):
                validate_items(manifest, root=root)

    def test_resolution_dataset_changes_only_rgb_size_with_same_randomness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            pixel_path = root / "pixel.npz"
            Image.new("RGB", (96, 64), (71, 133, 202)).save(image_path)
            target = np.zeros((7, 80, 80), dtype=np.uint8)
            target[0, 20:30, 20:30] = 1
            np.savez(pixel_path, mask=target, known=np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.uint8))
            # The dataset uses its module ROOT for paths, so these absolute
            # fixture paths never rely on or read real training material.
            row = {"image": str(image_path), "pixel_target": str(pixel_path),
                   "targets": [1, 0, -1, -1, -1, -1, -1], "domain": "dacl"}
            auxiliary = {str(image_path): [1] + [-1] * (len(AUX_CLASSES) - 1)}
            datasets = [ResolutionPhotos([row], auxiliary, full_count=1, imgsz=size)
                        for size in (640, 960)]
            outputs = []
            for dataset in datasets:
                random.seed(912)
                torch.manual_seed(912)
                outputs.append(dataset[0])
            self.assertEqual(tuple(outputs[0][0].shape), (3, 640, 640))
            self.assertEqual(tuple(outputs[1][0].shape), (3, 960, 960))
            for small, large in zip(outputs[0][1:], outputs[1][1:]):
                if isinstance(small, torch.Tensor): self.assertTrue(torch.equal(small, large))
                else: self.assertEqual(small, large)
            self.assertEqual(row["targets"], [1, 0, -1, -1, -1, -1, -1])
            with np.load(pixel_path) as saved:
                self.assertTrue(np.array_equal(saved["mask"], target))
            self.assertTrue(torch.equal(outputs[0][2], torch.tensor([1., 1., 0., 0., 0., 0., 0.])))

    def test_resolution_dataset_refuses_undeclared_size(self):
        for size in (0, 320, 800, 1280, 640., True):
            with self.subTest(imgsz=size), self.assertRaises(ValueError):
                ResolutionPhotos([], imgsz=size)

    def protocol_fixture(self, root, variant="control"):
        def fixture_file(relative, contents):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            return path, hashlib.sha256(contents).hexdigest()
        initial, initial_sha = fixture_file("runs/synthetic-initial.pt", b"synthetic weights bytes")
        auxiliary, auxiliary_sha = fixture_file("data/facility-auxiliary-training/train.json", b'{"fixture": "auxiliary"}')
        _, core_sha = fixture_file("data/facility-spatial-training/train.json", b'{"fixture": "core"}')
        sources = {relative: fixture_file(relative, (relative + " synthetic bytes").encode())[1]
                   for relative in study.SOURCE_FILES}
        protocol = {
            "schema": "facility_resolution_study_protocol_v1",
            "declared_before_training": True,
            "reference": "facility-presence-target-roi-control",
            "control": "facility-presence-target-resolution-control",
            "treatment": "facility-presence-target-resolution-highres",
            "architecture_by_variant": {"control": AUX_ARCH, "highres": RESOLUTION_ARCH},
            "imgsz_by_variant": {"control": 640, "highres": 960},
            "classes": list(CLASSES), "auxiliary_classes": list(AUX_CLASSES),
            "seed": 56, "requested_epochs": 6, "patience": 6, "batch_size": 8,
            "draws_per_epoch": 14248, "backbone_lr": .00004, "head_lr": .00025,
            "auxiliary_weight": .5, "target_ranking_weight": 0., "domain_proportions": [.7, .1, .2],
            "loader_randomness": {"sampler_seed": 56, "training_worker_seed": 57,
                                  "post_model_seed": 58,
                                  "validation_worker_seeds": {domain: 156 + index for domain, index in DOMAINS.items()}},
            "initial_weights_sha256": initial_sha, "core_spatial_manifest_sha256": core_sha,
            "auxiliary_manifest_sha256": auxiliary_sha, "source_sha256": sources,
        }
        args = argparse.Namespace(name=protocol["control" if variant == "control" else "treatment"],
            variant=variant, initial=initial, auxiliary_manifest=auxiliary, seed=56, epochs=6,
            patience=6, draws_per_epoch=14248, backbone_lr=.00004, head_lr=.00025,
            auxiliary_weight=.5, batch=8)
        return protocol, args

    def test_protocol_accepts_only_declared_640_and_960_variants_and_fixed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variant, size, architecture in (("control", 640, AUX_ARCH),
                                                 ("highres", 960, RESOLUTION_ARCH)):
                protocol, args = self.protocol_fixture(root, variant)
                before = copy.deepcopy(protocol)
                with mock.patch.object(study, "INITIAL_SHA", protocol["initial_weights_sha256"]):
                    result = validate_protocol(protocol, args, root)
                self.assertEqual(result, {"imgsz": size, "architecture": architecture})
                self.assertEqual(protocol, before)

    def test_protocol_rejects_recipe_cli_source_hash_and_path_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mutation in ("schema", "false_declaration", "changed_reference", "imgsz", "architecture", "class_order", "aux_class_order", "seed",
                             "epochs", "loader_seed", "ranking", "proportions", "learning_rate", "cli_batch",
                             "cli_name", "variant", "source_omission", "source_bytes", "source_traversal",
                             "source_abs", "malformed_hash", "weights_bytes", "aux_bytes"):
                protocol, args = self.protocol_fixture(root)
                expected_initial = protocol["initial_weights_sha256"]
                first_source = next(iter(protocol["source_sha256"]))
                if mutation == "schema": protocol["schema"] = "anything_else"
                elif mutation == "false_declaration": protocol["declared_before_training"] = False
                elif mutation == "changed_reference": protocol["reference"] = "another_initial_run"
                elif mutation == "imgsz": protocol["imgsz_by_variant"]["highres"] = 1280
                elif mutation == "architecture": protocol["architecture_by_variant"]["highres"] = AUX_ARCH
                elif mutation == "class_order": protocol["classes"].reverse()
                elif mutation == "aux_class_order": protocol["auxiliary_classes"].reverse()
                elif mutation == "seed": protocol["seed"] += 1
                elif mutation == "epochs": args.epochs += 1
                elif mutation == "loader_seed": protocol["loader_randomness"]["training_worker_seed"] += 1
                elif mutation == "ranking": protocol["target_ranking_weight"] = .25
                elif mutation == "proportions": protocol["domain_proportions"] = [.6, .2, .2]
                elif mutation == "learning_rate": args.head_lr *= 2
                elif mutation == "cli_batch": args.batch = 4
                elif mutation == "cli_name": args.name = "uncontrolled_run"
                elif mutation == "variant": args.variant = "other"
                elif mutation == "source_omission": protocol["source_sha256"].pop(first_source)
                elif mutation == "source_bytes": (root / first_source).write_bytes(b"modified after freeze")
                elif mutation == "source_traversal": protocol["source_sha256"]["../outside.py"] = "a" * 64
                elif mutation == "source_abs": protocol["source_sha256"]["C:/outside.py"] = "a" * 64
                elif mutation == "malformed_hash": protocol["core_spatial_manifest_sha256"] = "not-a-sha"
                elif mutation == "weights_bytes": args.initial.write_bytes(b"different initializer")
                elif mutation == "aux_bytes": args.auxiliary_manifest.write_bytes(b"different native tags")
                before = copy.deepcopy(protocol)
                with self.subTest(mutation=mutation), mock.patch.object(study, "INITIAL_SHA", expected_initial), self.assertRaises(ValueError):
                    validate_protocol(protocol, args, root)
                self.assertEqual(protocol, before)


if __name__ == "__main__":
    unittest.main()
