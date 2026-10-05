"""Completion proof guards, synthetic data only; no actual training or CUDA."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from safelog_ai.auxiliary_classifier import ARCH as AUX_ARCH, AUX_CLASSES
from safelog_ai.presence_classifier import MEAN, STD
from safelog_ai.resolution_classifier import ARCH as RES_ARCH, RECIPE
from scripts.facility_resolution_study import CLASSES
from scripts.report_facility_resolution import validate_histories
from scripts import verify_facility_resolution as proof


def digest(value):
    return hashlib.sha256(value).hexdigest()


class ResolutionVerificationTests(unittest.TestCase):
    def protocol(self):
        return {"schema": "facility_resolution_study_protocol_v1", "declared_before_training": True,
                "reference": "facility-presence-target-roi-control",
                "architecture_by_variant": {"control": AUX_ARCH, "highres": RES_ARCH},
                "imgsz_by_variant": {"control": 640, "highres": 960},
                "classes": list(CLASSES), "auxiliary_classes": list(AUX_CLASSES),
                "seed": 56, "requested_epochs": 6, "patience": 6, "batch_size": 8,
                "draws_per_epoch": 14248, "backbone_lr": .00004, "head_lr": .00025,
                "auxiliary_weight": .5, "domain_proportions": [.7, .1, .2],
                "loader_randomness": {"sampler_seed": 56},
                "initial_weights_sha256": "1" * 64,
                "core_spatial_manifest_sha256": "2" * 64,
                "auxiliary_manifest_sha256": "3" * 64,
                "resolution_architecture": copy.deepcopy(RECIPE)}

    def preflight(self, protocol):
        protected = {"initial": "1" * 64, "core": "2" * 64, "auxiliary": "3" * 64,
                     "app_profile": "4" * 64, "protocol": "5" * 64}
        result = {"schema": "facility_resolution_preflight_v1", "status": "passed",
                  "training_epochs": 0, "accuracy_measured": False,
                  "actual_train_source_cases": 8, "bounded_replay_batches_per_variant": 3,
                  "constructor_rng_equal": True, "640_initial_outputs_identical": True,
                  "mask_label_known_index_replay_equal": True,
                  "resized_image_tensors_expected_to_differ": True,
                  "protected_files_unchanged": True, "protected_file_sha256": protected,
                  "protocol_sha256": "5" * 64, "variants": {}, "replay": {}}
        for variant, size in protocol["imgsz_by_variant"].items():
            native = 80 if size == 640 else 120
            result["variants"][variant] = {
                "imgsz": size, "raw_map_shape": [8, 7, native, native],
                "loss_and_pool_grid": [80, 80], "photo_shape": [8, 7], "auxiliary_shape": [8, 19],
                "actual_optimizer_steps": 1, "validation_like_train_repeated_batch_size": 16,
                "validation_size_forward_finite": True, "all_gradients_finite": True,
                "disposable_serialization_and_factory_cpu_reload_verified": True,
                "parameter_count": 7, "peak_cuda_allocated_bytes": 1024,
                "loss": 1.2, "disposable_serialized_checkpoint_sha256": "6" * 64}
            result["replay"][variant] = [{
                "non_image_batch_sha256": digest(f"labels{i}".encode()),
                "indices_sha256": digest(f"indices{i}".encode()),
                "image_tensor_sha256": digest(f"{variant}{i}".encode())} for i in range(3)]
        return result, protected

    def history(self):
        return [{"epoch": epoch, "sampled_row_indices_sha256": digest(str(epoch).encode()),
                 "sampled_domain_counts": {"dacl": 10000, "damsegment": 1400, "codebrim": 2848},
                 "sampled_row_type_counts": {"full": 10848, "crop": 3400},
                 "sampled_full_target_joint_counts": {
                     "dacl": {"00": 1700, "10": 1700, "01": 1700, "11": 1700, "unknown": 0},
                     "damsegment": {"00": 300, "10": 300, "01": 300, "11": 300, "unknown": 0},
                     "codebrim": {"00": 712, "10": 712, "01": 712, "11": 712, "unknown": 0}},
                 "optimizer_step_diagnostics": {"attempted_batches": 1781, "actual_optimizer_steps": 1780,
                                                "amp_skipped_steps": 1},
                 "worst_target_error": .2} for epoch in range(1, 7)]

    def completed(self, protocol, history, variant="highres"):
        state = {"weight": torch.ones(7), "counter": torch.tensor(1, dtype=torch.int64)}
        training = {"status": "complete", "actual_epochs": 6,
                    "architecture": protocol["architecture_by_variant"][variant],
                    "imgsz": protocol["imgsz_by_variant"][variant], "model_variant": variant,
                    "classes": list(CLASSES), "auxiliary_classes": list(AUX_CLASSES),
                    "auxiliary_source_class_count": 19, "validation_batch_size": 16,
                    "initial_weights_sha256": "1" * 64, "core_spatial_manifest_sha256": "2" * 64,
                    "spatial_manifest_sha256": "2" * 64, "auxiliary_manifest_sha256": "3" * 64,
                    "source_sha256": protocol["source_sha256"], "study_protocol_sha256": "5" * 64,
                    "resolution_architecture": copy.deepcopy(RECIPE), "photo_pooling_grid": [80, 80],
                    "source_pixel_target_grid": [80, 80], "raw_spatial_grid": [120, 120],
                    "new_photo_targets": 0, "new_pixel_targets": 0, "app_model_promoted": False,
                    "target_ranking_weight": 0., "weights_sha256": "7" * 64, "split_sha256": "8" * 64,
                    "initial_state_transfer": {"shared_state_tensors_equal": True,
                        "shared_state_tensor_count": len(state), "new_state_tensor_count": 0,
                        "new_output_projection_zero": None, "strict_state_load": True, "additional_parameters": 0},
                    "target_ranking": {"weight": 0., "classes": ["concrete_crack", "concrete_spalling"],
                        "sampling_changed": False, "new_labels_asserted": 0, "public_outputs_changed": False},
                    "optimizer_step_diagnostics": {k: sum(row["optimizer_step_diagnostics"][k] for row in history)
                        for k in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")},
                    "elapsed_training_minutes": 12.5, "peak_cuda_allocated_bytes": 1024,
                    "best_worst_target_error": .2}
        for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
                    "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions"):
            training[key] = copy.deepcopy(protocol[key])
        checkpoint = {"state_dict": copy.deepcopy(state), "architecture": RES_ARCH,
                      "classes": list(CLASSES), "auxiliary_classes": list(AUX_CLASSES), "imgsz": 960,
                      "mean": MEAN, "std": STD, "selection_split": "val", "split_sha256": "8" * 64,
                      "epoch": 3, "worst_target_error": .2}
        return training, checkpoint, state

    def test_snapshot_matches_dynamic_commit_raw_git_and_current_eighteen_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); blobs = {}
            for index, relative in enumerate(proof.RUNTIME_SOURCES):
                blobs[relative] = f"synthetic source {index}\r\n".encode()
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blobs[relative])
            sources = {p: digest(b) for p, b in blobs.items()}
            protocol = self.protocol(); protocol["source_sha256"] = sources
            preflight = {"source_sha256": sources, "protocol_sha256": "5" * 64}
            snapshot = {"schema": "facility_resolution_source_before_training_v1",
                        "declared_before_training": True, "git_blob_bytes_match_protocol_sources": True,
                        "new_completed_training_epochs_at_snapshot": 0, "source_git_commit": "a" * 40,
                        "source_sha256": sources, "protocol_sha256": "5" * 64,
                        "preflight_sha256": "9" * 64, "initial_weights_sha256": "1" * 64}
            seen = []
            def reader(commit, relative):
                seen.append(commit); return blobs[relative]
            result = proof.verify_frozen_sources(snapshot, protocol, preflight, root, "5" * 64, "9" * 64, reader)
            self.assertEqual(result["runtime_source_count"], 18)
            self.assertEqual(set(seen), {"a" * 40})
            # A different valid recorded commit is accepted: no stale commit constant.
            snapshot["source_git_commit"] = "b" * 40
            self.assertEqual(proof.verify_frozen_sources(snapshot, protocol, preflight, root,
                "5" * 64, "9" * 64, reader)["source_git_commit"], "b" * 40)
            for mutation in ("uncommitted", "text_decoded_blob", "missing_source", "after_training",
                             "preflight_hash", "protocol_hash", "commit", "path_escape"):
                broken = copy.deepcopy(snapshot); p = copy.deepcopy(protocol); f = copy.deepcopy(preflight)
                changed = root / proof.RUNTIME_SOURCES[0]; previous = changed.read_bytes()
                bad_reader = reader
                if mutation == "uncommitted": changed.write_bytes(previous + b"changed")
                elif mutation == "text_decoded_blob": bad_reader = lambda c, r: blobs[r].decode().replace("\r\n", "\n").encode()
                elif mutation == "missing_source": broken["source_sha256"].pop(proof.RUNTIME_SOURCES[0])
                elif mutation == "after_training": broken["new_completed_training_epochs_at_snapshot"] = 1
                elif mutation == "preflight_hash": broken["preflight_sha256"] = "0" * 64
                elif mutation == "protocol_hash": f["protocol_sha256"] = "0" * 64
                elif mutation == "commit": broken["source_git_commit"] = "main"
                elif mutation == "path_escape": p["source_sha256"]["../secret"] = "0" * 64
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    proof.verify_frozen_sources(broken, p, f, root, "5" * 64, "9" * 64, bad_reader)
                changed.write_bytes(previous)

    def test_preflight_requires_actual_bounded_replay_and_unchanged_protected_bytes(self):
        protocol = self.protocol(); valid, protected = self.preflight(protocol)
        before = copy.deepcopy(valid)
        result = proof.validate_preflight(valid, protocol, "5" * 64, protected)
        self.assertTrue(result["passed"]); self.assertEqual(valid, before)
        for mutation in ("status", "zero_updates", "nan_loss", "gradient", "reload", "grid",
                         "new_parameters", "replay_order", "replay_labels", "same_pixels", "short_replay",
                         "app_changed", "protected_flag", "wrong_protocol", "boolean_cases"):
            value = copy.deepcopy(valid); current = copy.deepcopy(protected); row = value["variants"]["highres"]
            if mutation == "status": value["status"] = "running"
            elif mutation == "zero_updates": row["actual_optimizer_steps"] = 0
            elif mutation == "nan_loss": row["loss"] = float("nan")
            elif mutation == "gradient": row["all_gradients_finite"] = False
            elif mutation == "reload": row["disposable_serialization_and_factory_cpu_reload_verified"] = False
            elif mutation == "grid": row["loss_and_pool_grid"] = [120, 120]
            elif mutation == "new_parameters": row["parameter_count"] += 1
            elif mutation == "replay_order": value["replay"]["highres"][0]["indices_sha256"] = "0" * 64
            elif mutation == "replay_labels": value["replay"]["highres"][0]["non_image_batch_sha256"] = "0" * 64
            elif mutation == "same_pixels": value["replay"]["highres"][0] = copy.deepcopy(value["replay"]["control"][0])
            elif mutation == "short_replay": value["replay"]["highres"].pop()
            elif mutation == "app_changed": current["app_profile"] = "0" * 64
            elif mutation == "protected_flag": value["protected_files_unchanged"] = False
            elif mutation == "wrong_protocol": value["protocol_sha256"] = "0" * 64
            elif mutation == "boolean_cases": value["actual_train_source_cases"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                proof.validate_preflight(value, protocol, "5" * 64, current)

    def test_actual_tests_require_thirty_seven_passes_zero_skips_and_exact_tested_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in proof.TEST_SOURCES + (proof.VERIFIER_SOURCE,):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(relative.encode())
            result = {"schema": "facility_resolution_test_results_v1", "status": "passed", "tests_run": 37,
                      "failures": 0, "errors": 0, "skipped": 0, "before_after_sources_equal": True,
                      "test_source_sha256": {p: proof.sha(root / p) for p in proof.TEST_SOURCES},
                      "source_sha256": {proof.VERIFIER_SOURCE: proof.sha(root / proof.VERIFIER_SOURCE)}}
            self.assertEqual(proof.validate_test_results(result, root)["tests_run"], 37)
            for mutation in ("not_enough", "failure", "error", "skip", "false_zero", "changed_source",
                             "missing_test", "wrong_test_sha", "wrong_verifier_sha", "new_verifier_bytes"):
                value = copy.deepcopy(result); path = root / proof.VERIFIER_SOURCE; previous = path.read_bytes()
                if mutation == "not_enough": value["tests_run"] = 36
                elif mutation == "failure": value["failures"] = 1
                elif mutation == "error": value["errors"] = 1
                elif mutation == "skip": value["skipped"] = 1
                elif mutation == "false_zero": value["errors"] = False
                elif mutation == "changed_source": value["before_after_sources_equal"] = False
                elif mutation == "missing_test": value["test_source_sha256"].pop(proof.TEST_SOURCES[0])
                elif mutation == "wrong_test_sha": value["test_source_sha256"][proof.TEST_SOURCES[0]] = "0" * 64
                elif mutation == "wrong_verifier_sha": value["source_sha256"][proof.VERIFIER_SOURCE] = "0" * 64
                elif mutation == "new_verifier_bytes": path.write_bytes(previous + b"changed")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    proof.validate_test_results(value, root)
                path.write_bytes(previous)

    def test_completed_recipe_state_updates_and_cpu_seven_eighty_nineteen_contract(self):
        protocol = self.protocol(); protocol["source_sha256"] = {p: "0" * 64 for p in proof.RUNTIME_SOURCES}
        history = self.history(); validate_histories(history, copy.deepcopy(history), 6, 14248)
        training, checkpoint, initial = self.completed(protocol, history)
        before = copy.deepcopy(training)
        self.assertEqual(proof.validate_run_metadata(training, checkpoint, history, "highres", protocol,
                         "5" * 64, initial, "7" * 64, "8" * 64)["new_state_tensor_count"], 0)
        self.assertEqual(training, before)
        for mutation in ("running", "epoch", "recipe", "pixel_grid", "labels", "promoted", "source",
                         "amp_total", "new_state", "nonfinite", "shape", "dtype", "metadata", "test_selection"):
            t, c = copy.deepcopy(training), copy.deepcopy(checkpoint)
            if mutation == "running": t["status"] = "running"
            elif mutation == "epoch": t["actual_epochs"] = 5
            elif mutation == "recipe": t["resolution_architecture"]["photo_topk"] = 72
            elif mutation == "pixel_grid": t["source_pixel_target_grid"] = [120, 120]
            elif mutation == "labels": t["new_pixel_targets"] = 1
            elif mutation == "promoted": t["app_model_promoted"] = True
            elif mutation == "source": t["source_sha256"][proof.RUNTIME_SOURCES[0]] = "1" * 64
            elif mutation == "amp_total": t["optimizer_step_diagnostics"]["actual_optimizer_steps"] += 1
            elif mutation == "new_state": c["state_dict"]["new"] = torch.zeros(1)
            elif mutation == "nonfinite": c["state_dict"]["weight"][0] = float("nan")
            elif mutation == "shape": c["state_dict"]["weight"] = torch.zeros(8)
            elif mutation == "dtype": c["state_dict"]["weight"] = c["state_dict"]["weight"].double()
            elif mutation == "metadata": c["imgsz"] = 640
            elif mutation == "test_selection": c["selection_split"] = "test"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                proof.validate_run_metadata(t, c, history, "highres", protocol, "5" * 64,
                                            initial, "7" * 64, "8" * 64)
        class SyntheticModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.segmentation_head = torch.nn.Identity()
                self.weight = torch.nn.Parameter(torch.zeros(7)); self.bad = False
            def forward_training(self, batch):
                self.segmentation_head(torch.zeros(1, 7, 120, 120))
                photo = self.weight[None].clone()
                if self.bad: photo[0, 0] = float("nan")
                return photo, torch.zeros(1, 7, 80, 80), torch.zeros(1, 19)
            def forward(self, batch):
                return self.forward_training(batch)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "data").mkdir()
            Image.new("RGB", (8, 8)).save(root / "data/train.png")
            provider = SimpleNamespace(device=torch.device("cpu"), model=SyntheticModel(),
                                       transform=lambda image: torch.zeros(3, 960, 960))
            with patch.object(proof, "PresenceClassifier", return_value=provider) as factory:
                out = proof.verify_cpu_reload(root / "synthetic.pt", {"image": "data/train.png"}, root, 7, "highres")
                factory.assert_called_once_with(root / "synthetic.pt", device="cpu")
                self.assertEqual(out["photo_shape"], [1, 7]); self.assertEqual(out["loss_and_pool_map_shape"], [1, 7, 80, 80])
                self.assertEqual(out["auxiliary_shape"], [1, 19]); self.assertNotIn("image", out)
                provider.model.bad = True
                with self.assertRaises(ValueError):
                    proof.verify_cpu_reload(root / "synthetic.pt", {"image": "data/train.png"}, root, 7, "highres")


if __name__ == "__main__":
    unittest.main()
