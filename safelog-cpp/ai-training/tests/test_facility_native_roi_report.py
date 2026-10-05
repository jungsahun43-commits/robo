"""Native ROI completion and reporting contracts, using synthetic fixtures only."""
import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import report_facility_native_roi as report
from scripts import verify_facility_native_roi as verify
from scripts.facility_native_roi_study import fixed_values
from scripts.report_facility_discrimination import KNOWN_INDICES
from scripts.report_facility_small_region import CLASSES, COUNTS, DOMAINS, TARGETS
from safelog_ai.presence_classifier import MEAN, STD


def digest(value):
    return hashlib.sha256(value).hexdigest()


class NativeROIReportTests(unittest.TestCase):
    def protocol(self):
        protocol = copy.deepcopy(fixed_values())
        protocol.update(
            source_sha256={"runtime.py": "a" * 64},
            core_spatial_manifest_sha256="b" * 64,
            auxiliary_manifest_sha256="c" * 64,
            paired_manifest_sha256={"control": "d" * 64, "native": "e" * 64},
            paired_data_audit_sha256="f" * 64,
            app_profile_sha256="1" * 64,
            source_records_sha256="2" * 64,
            source_preparation_sha256="3" * 64,
        )
        return protocol

    def entries(self):
        protocol = self.protocol()
        entries = []
        for index, (name, errors, small) in enumerate((
            (protocol["reference"], 20, (15, 18)),
            (protocol["control"], 18, (14, 17)),
            (protocol["treatment"], 12, (10, 12)),
        )):
            per_class = {}
            for label in TARGETS:
                per_class[label] = {"threshold": .5, "domains": {}}
                for domain in DOMAINS:
                    positive, negative = 200, COUNTS[domain] - 200
                    per_class[label]["domains"][domain] = {
                        "tp": positive - errors, "fn": errors,
                        "fp": errors, "tn": negative - errors,
                        "positive_photos": positive, "negative_photos": negative,
                        "fnr": errors / positive, "fpr": errors / negative,
                    }
            ranking = {}
            for domain in DOMAINS:
                ranking[domain] = {}
                for k, label in enumerate(CLASSES):
                    known = k in KNOWN_INDICES[domain]
                    ranking[domain][label] = {
                        "ap": .80 + .01 * index if known else None,
                        "known_photos": COUNTS[domain] if known else 0,
                        "positive_photos": (200 if label in TARGETS else 100) if known else 0,
                        "status": "measured_on_asserted_source_labels" if known
                            else "not_measured_source_labels_unknown",
                    }
            entry = {
                "run": name, "title": ("추가 학습 전 모델", "processed 대조군", "native 보강군")[index],
                "imgsz": 640, "weights_sha256": str(index + 4) * 64,
                "actual_epochs": 6, "best_epoch": index + 1,
                "configured_grids": [1], "selected_grid": 1,
                "per_class": per_class, "ranking_ap": ranking,
                "worst_error": max(point[rate] for task in per_class.values()
                    for point in task["domains"].values() for rate in ("fnr", "fpr")),
                "target_passed": False, "test_executed": False, "test_result": None,
                "small_dacl_polygon_area_below_one_percent": {
                    label: {"positive_photos": support, "false_negatives": fn, "fnr": fn / support}
                    for label, support, fn in zip(TARGETS, (93, 105), small)
                },
            }
            if index:
                entry["resources"] = {
                    "elapsed_training_minutes": 12.5 + index,
                    "peak_cuda_allocated_bytes": 2 * 1024 ** 3,
                    "optimizer_step_diagnostics": {"attempted_batches": 10686,
                        "actual_optimizer_steps": 10680, "amp_skipped_steps": 6},
                }
            entries.append(entry)
        return entries

    def refresh_error(self, entry):
        entry["worst_error"] = max(point[rate] for task in entry["per_class"].values()
            for point in task["domains"].values() for rate in ("fnr", "fpr"))
        entry["target_passed"] = entry["worst_error"] < .05

    def set_errors(self, entry, count):
        for task in entry["per_class"].values():
            for point in task["domains"].values():
                point.update(fn=count, tp=point["positive_photos"] - count,
                    fp=count, tn=point["negative_photos"] - count,
                    fnr=count / point["positive_photos"], fpr=count / point["negative_photos"])
        self.refresh_error(entry)

    def set_small(self, entry, counts):
        for label, fn in zip(TARGETS, counts):
            point = entry["small_dacl_polygon_area_below_one_percent"][label]
            point.update(false_negatives=fn, fnr=fn / point["positive_photos"])

    def technical_proof(self):
        protocol = self.protocol()
        module_counts = dict(verify.MIN_TEST_COUNTS)
        test_count = sum(module_counts.values())
        trainings = [{"weights_sha256": protocol["initial_weights_sha256"]},
                     {"weights_sha256": "4" * 64}, {"weights_sha256": "5" * 64}]
        protected = {"reports/app-profile.json": "1" * 64}
        proof = {
            "schema": "facility_native_roi_study_verification_v1", "status": "passed",
            "protocol_sha256": "6" * 64, "source_sha256": protocol["source_sha256"],
            "source_git_commit": "7" * 40, "runtime_source_count": 1,
            "git_blob_bytes_verified": True, "working_runtime_sources_unchanged": True,
            "protected_files_unchanged": True, "protected_file_sha256": protected,
            "actual_completed_training_epochs": 12, "verification_training_epochs": 0,
            "source_test_inference_executed": False, "app_model_promoted": False,
            "deployed": False, "accuracy_measured_by_verifier": False,
            "additional_expert_confirmed_labels": 0, "label_changes": 0,
            "new_photo_targets": 0, "new_pixel_targets": 0,
            "tests": {"tests_run": test_count, "expected_tests_collected": test_count,
                "tests_by_module": module_counts, "failures": 0, "errors": 0, "skipped": 0,
                "source_sha256": protocol["source_sha256"],
                "test_source_sha256": {path: "9" * 64 for path in verify.TEST_SOURCES}},
            "prepared_data_integrity": {"status": "passed", "full_rows_unchanged": True,
                "labels_and_masks_unchanged": True, "row_order_unchanged": True,
                "derived_png_hashes_verified": True, "changed_rows": 2, "derived_pngs": 4},
            "experiments": [],
        }
        for variant, training in zip(("control", "native"), trainings[1:]):
            proof["experiments"].append({"variant": variant,
                "weights_sha256": training["weights_sha256"], "actual_epochs": 6,
                "imgsz": 640, "architecture": protocol["architecture_by_variant"][variant],
                "strict_state_inventory_verified": True, "new_state_tensor_count": 0,
                "cpu_reload": {"strict_factory_reload_verified": True, "all_outputs_finite": True,
                    "public_output_equals_training_photo_output": True}})
        return protocol, trainings, protected, proof

    def test_report_requires_matching_completed_checkpoints_app_sources_and_test_proof(self):
        protocol, trainings, protected, proof = self.technical_proof()
        before = copy.deepcopy(proof)
        self.assertEqual(report.validate_technical_proof(proof, protocol, "6" * 64,
            trainings, protected)["tests_run"], sum(verify.MIN_TEST_COUNTS.values()))
        self.assertEqual(proof, before)
        for mutation in ("status", "protocol", "source", "git", "epoch", "new_training",
                         "app", "labels", "test_failure", "test_skip", "boolean_error",
                         "no_tests", "subset_one", "missing_module", "module_low", "module_bool",
                         "count_sum", "collection_different", "tested_runtime", "missing_test_provenance",
                         "checkpoint", "weights", "architecture", "size", "reload",
                         "png_bytes", "labels_masks", "png_budget", "row_order"):
            changed = copy.deepcopy(proof)
            if mutation == "status": changed["status"] = "running"
            elif mutation == "protocol": changed["protocol_sha256"] = "0" * 64
            elif mutation == "source": changed["source_sha256"] = {}
            elif mutation == "git": changed["git_blob_bytes_verified"] = False
            elif mutation == "epoch": changed["actual_completed_training_epochs"] = 11
            elif mutation == "new_training": changed["verification_training_epochs"] = 1
            elif mutation == "app": changed["protected_file_sha256"]["reports/app-profile.json"] = "0" * 64
            elif mutation == "labels": changed["label_changes"] = 1
            elif mutation == "test_failure": changed["tests"]["failures"] = 1
            elif mutation == "test_skip": changed["tests"]["skipped"] = 1
            elif mutation == "boolean_error": changed["tests"]["errors"] = False
            elif mutation == "no_tests": changed["tests"]["tests_run"] = 0
            elif mutation in ("subset_one", "missing_module", "module_low", "module_bool",
                              "count_sum", "collection_different"):
                self.corrupt_test_collection(changed["tests"], mutation)
            elif mutation == "tested_runtime": changed["tests"]["source_sha256"] = {}
            elif mutation == "missing_test_provenance": changed["tests"]["test_source_sha256"].pop(verify.TEST_SOURCES[0])
            elif mutation == "checkpoint": changed["experiments"].pop()
            elif mutation == "weights": changed["experiments"][1]["weights_sha256"] = "0" * 64
            elif mutation == "architecture": changed["experiments"][1]["architecture"] = "other"
            elif mutation == "size": changed["experiments"][1]["imgsz"] = 960
            elif mutation == "reload": changed["experiments"][1]["cpu_reload"]["all_outputs_finite"] = False
            elif mutation == "png_bytes": changed["prepared_data_integrity"]["derived_png_hashes_verified"] = False
            elif mutation == "labels_masks": changed["prepared_data_integrity"]["labels_and_masks_unchanged"] = False
            elif mutation == "png_budget": changed["prepared_data_integrity"]["derived_pngs"] = 3
            else: changed["prepared_data_integrity"]["row_order_unchanged"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_technical_proof(changed, protocol, "6" * 64, trainings, protected)

    def corrupt_test_collection(self, record, mutation):
        module = next(iter(verify.MIN_TEST_COUNTS))
        if mutation == "subset_one":
            record.update(tests_run=1, expected_tests_collected=1,
                tests_by_module={key: int(key == module) for key in verify.MIN_TEST_COUNTS})
        elif mutation == "missing_module": record["tests_by_module"].pop(module)
        elif mutation == "module_low":
            record["tests_by_module"][module] = verify.MIN_TEST_COUNTS[module] - 1
            record["tests_run"] = record["expected_tests_collected"] = sum(record["tests_by_module"].values())
        elif mutation == "module_bool": record["tests_by_module"][module] = True
        elif mutation == "count_sum":
            record["tests_run"] += 1
            record["expected_tests_collected"] = record["tests_run"]
        else: record["expected_tests_collected"] += 1

    def test_source_snapshot_checks_dynamic_commit_raw_git_and_all_current_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = {}
            for index, relative in enumerate(verify.RUNTIME_SOURCES):
                blobs[relative] = f"synthetic native source {index}\r\n".encode()
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blobs[relative])
            sources = {relative: digest(payload) for relative, payload in blobs.items()}
            protocol = self.protocol()
            protocol["source_sha256"] = sources
            preflight = {"source_sha256": sources, "protocol_sha256": "6" * 64}
            snapshot = {"schema": "facility_native_roi_source_before_training_v1",
                "declared_before_training": True, "git_blob_bytes_match_protocol_sources": True,
                "new_completed_training_epochs_at_snapshot": 0, "source_git_commit": "7" * 40,
                "source_sha256": sources, "protocol_sha256": "6" * 64, "preflight_sha256": "8" * 64}
            seen = []
            def reader(commit, relative):
                seen.append(commit)
                return blobs[relative]
            result = verify.verify_frozen_sources(snapshot, protocol, preflight, root,
                "6" * 64, "8" * 64, reader)
            self.assertEqual(result["runtime_source_count"], len(verify.RUNTIME_SOURCES))
            self.assertEqual(set(seen), {"7" * 40})
            snapshot["source_git_commit"] = "9" * 40
            self.assertEqual(verify.verify_frozen_sources(snapshot, protocol, preflight, root,
                "6" * 64, "8" * 64, reader)["source_git_commit"], "9" * 40)
            for mutation in ("runtime_bytes", "decoded_blob", "missing_source", "after_training",
                             "preflight", "protocol", "commit", "path_escape"):
                broken, p, f = copy.deepcopy(snapshot), copy.deepcopy(protocol), copy.deepcopy(preflight)
                path = root / verify.RUNTIME_SOURCES[0]
                original = path.read_bytes()
                blob_reader = reader
                if mutation == "runtime_bytes": path.write_bytes(original + b"changed")
                elif mutation == "decoded_blob":
                    blob_reader = lambda commit, relative: blobs[relative].replace(b"\r\n", b"\n")
                elif mutation == "missing_source": broken["source_sha256"].pop(verify.RUNTIME_SOURCES[0])
                elif mutation == "after_training": broken["new_completed_training_epochs_at_snapshot"] = 1
                elif mutation == "preflight": broken["preflight_sha256"] = "0" * 64
                elif mutation == "protocol": f["protocol_sha256"] = "0" * 64
                elif mutation == "commit": broken["source_git_commit"] = "main"
                else: p["source_sha256"]["../outside"] = "0" * 64
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify.verify_frozen_sources(broken, p, f, root, "6" * 64, "8" * 64, blob_reader)
                path.write_bytes(original)

    def completed_metadata(self, variant):
        protocol = self.protocol()
        state = {"weight": torch.ones(7), "counter": torch.tensor(1, dtype=torch.int64)}
        history = [{"epoch": epoch, "worst_target_error": .2,
            "optimizer_step_diagnostics": {"attempted_batches": 1781,
                "actual_optimizer_steps": 1780, "amp_skipped_steps": 1}} for epoch in range(1, 7)]
        training = {
            "status": "complete", "actual_epochs": 6,
            "architecture": protocol["architecture_by_variant"][variant], "imgsz": 640,
            "model_variant": variant, "classes": protocol["classes"],
            "auxiliary_classes": protocol["auxiliary_classes"], "auxiliary_source_class_count": 19,
            "validation_batch_size": 16, "validation_domains": list(DOMAINS),
            "train_dacl": 6225, "train_damsegment": 1585, "train_codebrim": 6438,
            "val_dacl": 710, "val_damsegment": 424,
            "initial_weights_sha256": protocol["initial_weights_sha256"],
            "core_spatial_manifest_sha256": protocol["core_spatial_manifest_sha256"],
            "spatial_manifest_sha256": protocol["paired_manifest_sha256"][variant],
            "spatial_manifest_path": protocol["paired_manifest_paths"][variant],
            "auxiliary_manifest_sha256": protocol["auxiliary_manifest_sha256"],
            "source_sha256": protocol["source_sha256"], "study_protocol_sha256": "6" * 64,
            "study_protocol_path": verify.PROTOCOL_PATH,
            "native_roi_recipe": protocol["native_roi_recipe"],
            "paired_data_audit_sha256": protocol["paired_data_audit_sha256"],
            "photo_pooling_grid": [80, 80], "source_pixel_target_grid": [80, 80],
            "raw_spatial_grid": [80, 80], "new_photo_targets": 0,
            "new_pixel_targets": 0, "app_model_promoted": False, "target_ranking_weight": 0.,
            "weights_sha256": "7" * 64, "split_sha256": "8" * 64,
            "initial_state_transfer": {"shared_state_tensors_equal": True,
                "shared_state_tensor_count": len(state), "new_state_tensor_count": 0,
                "new_output_projection_zero": None, "strict_state_load": True, "additional_parameters": 0},
            "target_ranking": {"weight": 0., "classes": list(TARGETS), "sampling_changed": False,
                "new_labels_asserted": 0, "public_outputs_changed": False},
            "paired_label_preservation": {"full_rows_equal": True, "photo_targets_equal": True,
                "pixel_target_paths_equal": True, "sample_weights_equal": True,
                "auxiliary_crop_targets_remain_unknown": True, "row_count": 26289,
                "changed_dacl_detail_rows": 5928},
            "optimizer_step_diagnostics": {"attempted_batches": 10686,
                "actual_optimizer_steps": 10680, "amp_skipped_steps": 6},
            "elapsed_training_minutes": 12.5, "peak_cuda_allocated_bytes": 1024,
            "best_worst_target_error": .2,
        }
        for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
                    "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions"):
            training[key] = copy.deepcopy(protocol[key])
        checkpoint = {"architecture": training["architecture"], "classes": protocol["classes"],
            "auxiliary_classes": protocol["auxiliary_classes"], "imgsz": 640,
            "mean": MEAN, "std": STD, "selection_split": "val", "split_sha256": "8" * 64,
            "epoch": 3, "worst_target_error": .2, "state_dict": copy.deepcopy(state)}
        return protocol, training, checkpoint, history, state

    def test_completed_metadata_preserves_640_state_labels_epochs_and_selected_checkpoint(self):
        for variant in ("control", "native"):
            protocol, training, checkpoint, history, initial = self.completed_metadata(variant)
            before = copy.deepcopy(training)
            result = verify.validate_run_metadata(training, checkpoint, history, variant,
                protocol, "6" * 64, initial, "7" * 64, "8" * 64)
            self.assertEqual(result["new_state_tensor_count"], 0)
            self.assertEqual(result["actual_epochs"], 6)
            self.assertEqual(training, before)
        for mutation in ("running", "epoch", "short_history", "recipe", "paired_manifest", "protocol",
                         "source", "app", "new_labels", "pixel_grid", "unknown_auxiliary", "row_count",
                         "no_changed_pixels", "sampling", "optimizer_total", "new_state", "nonfinite",
                         "shape", "dtype", "test_selection", "checkpoint_epoch", "selected_measurement"):
            protocol, training, checkpoint, history, initial = self.completed_metadata("native")
            if mutation == "running": training["status"] = "running"
            elif mutation == "epoch": training["actual_epochs"] = 5
            elif mutation == "short_history": history.pop()
            elif mutation == "recipe": training["native_roi_recipe"] = {}
            elif mutation == "paired_manifest": training["spatial_manifest_sha256"] = "0" * 64
            elif mutation == "protocol": training["study_protocol_sha256"] = "0" * 64
            elif mutation == "source": training["source_sha256"] = {}
            elif mutation == "app": training["app_model_promoted"] = True
            elif mutation == "new_labels": training["new_pixel_targets"] = 1
            elif mutation == "pixel_grid": training["source_pixel_target_grid"] = [160, 160]
            elif mutation == "unknown_auxiliary": training["paired_label_preservation"]["auxiliary_crop_targets_remain_unknown"] = False
            elif mutation == "row_count": training["paired_label_preservation"]["row_count"] += 1
            elif mutation == "no_changed_pixels": training["paired_label_preservation"]["changed_dacl_detail_rows"] = 0
            elif mutation == "sampling": training["paired_label_preservation"]["sample_weights_equal"] = False
            elif mutation == "optimizer_total": training["optimizer_step_diagnostics"]["actual_optimizer_steps"] += 1
            elif mutation == "new_state": checkpoint["state_dict"]["new"] = torch.zeros(1)
            elif mutation == "nonfinite": checkpoint["state_dict"]["weight"][0] = float("nan")
            elif mutation == "shape": checkpoint["state_dict"]["weight"] = torch.zeros(8)
            elif mutation == "dtype": checkpoint["state_dict"]["weight"] = torch.zeros(7, dtype=torch.float64)
            elif mutation == "test_selection": checkpoint["selection_split"] = "test"
            elif mutation == "checkpoint_epoch": checkpoint["epoch"] = True
            else: checkpoint["worst_target_error"] = .3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.validate_run_metadata(training, checkpoint, history, "native",
                    protocol, "6" * 64, initial, "7" * 64, "8" * 64)

    def test_executed_tests_bind_all_test_and_runtime_bytes_and_allow_no_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in set(verify.TEST_SOURCES) | set(verify.RUNTIME_SOURCES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic {relative}\r\n".encode())
            test_count = sum(verify.MIN_TEST_COUNTS.values())
            record = {"schema": "facility_native_roi_test_results_v1", "status": "passed",
                "tests_run": test_count, "expected_tests_collected": test_count,
                "tests_by_module": dict(verify.MIN_TEST_COUNTS), "failures": 0, "errors": 0, "skipped": 0,
                "before_after_sources_equal": True,
                "test_source_sha256": {p: digest((root / p).read_bytes()) for p in verify.TEST_SOURCES},
                "source_sha256": {p: digest((root / p).read_bytes()) for p in verify.RUNTIME_SOURCES}}
            self.assertEqual(verify.validate_test_results(record, root)["tests_run"], test_count)
            for mutation in ("zero_tests", "failure", "error", "skip", "boolean_zero", "source_flag",
                             "subset_one", "missing_module", "module_low", "module_bool", "count_sum",
                             "collection_different", "missing_test", "missing_runtime", "wrong_test_hash",
                             "wrong_runtime_hash", "changed_bytes"):
                value = copy.deepcopy(record)
                path = root / verify.RUNTIME_SOURCES[0]
                original = path.read_bytes()
                if mutation == "zero_tests": value["tests_run"] = 0
                elif mutation == "failure": value["failures"] = 1
                elif mutation == "error": value["errors"] = 1
                elif mutation == "skip": value["skipped"] = 1
                elif mutation == "boolean_zero": value["errors"] = False
                elif mutation == "source_flag": value["before_after_sources_equal"] = False
                elif mutation in ("subset_one", "missing_module", "module_low", "module_bool",
                                  "count_sum", "collection_different"):
                    self.corrupt_test_collection(value, mutation)
                elif mutation == "missing_test": value["test_source_sha256"].pop(verify.TEST_SOURCES[0])
                elif mutation == "missing_runtime": value["source_sha256"].pop(verify.RUNTIME_SOURCES[0])
                elif mutation == "wrong_test_hash": value["test_source_sha256"][verify.TEST_SOURCES[0]] = "0" * 64
                elif mutation == "wrong_runtime_hash": value["source_sha256"][verify.RUNTIME_SOURCES[0]] = "0" * 64
                else: path.write_bytes(original + b"changed")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify.validate_test_results(value, root)
                path.write_bytes(original)

    def test_consistent_aggregate_measurements_have_42_known_ap_points_and_do_not_mutate(self):
        entries, protocol = self.entries(), self.protocol()
        before = copy.deepcopy(entries)
        result = report.validate_measurements(entries, protocol)
        self.assertEqual(entries, before)
        self.assertEqual(result["verified_known_class_ap_measurements"], 42)
        self.assertEqual(len(result["error_rows"]), 18)
        self.assertTrue(result["research_gate"]["research_candidate_nominated"])
        self.assertAlmostEqual(result["comparisons"]["maximum_error_treatment_minus_initializer_pp"], -4.)
        self.assertAlmostEqual(result["comparisons"]["maximum_error_treatment_minus_control_pp"], -3.)

    def test_count_rate_headline_and_fixed_small_support_corruption_is_rejected(self):
        for mutation in ("negative_count", "boolean_count", "float_count", "missing_count",
                         "rate", "support", "zero_support", "count_over_support", "worst",
                         "strict_pass", "missing_domain", "missing_target", "small_support",
                         "small_boolean_support", "small_fn", "small_rate"):
            entries = self.entries()
            native = entries[2]
            point = native["per_class"][TARGETS[0]]["domains"]["dacl"]
            small = native["small_dacl_polygon_area_below_one_percent"][TARGETS[0]]
            if mutation == "negative_count": point["fn"] = -1
            elif mutation == "boolean_count": point["tp"] = True
            elif mutation == "float_count": point["fp"] = float(point["fp"])
            elif mutation == "missing_count": point.pop("tn")
            elif mutation == "rate": point["fnr"] += .001
            elif mutation == "support": point["positive_photos"] += 1
            elif mutation == "zero_support": point.update(tp=0, fn=0, positive_photos=0)
            elif mutation == "count_over_support": point["fn"] = 201
            elif mutation == "worst": native["worst_error"] += .001
            elif mutation == "strict_pass": native["target_passed"] = True
            elif mutation == "missing_domain": native["per_class"][TARGETS[0]]["domains"].pop("codebrim")
            elif mutation == "missing_target": native["per_class"].pop(TARGETS[1])
            elif mutation == "small_support": small["positive_photos"] = 94
            elif mutation == "small_boolean_support": small["positive_photos"] = True
            elif mutation == "small_fn": small["false_negatives"] = 94
            else: small["fnr"] += .001
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_measurements(entries, self.protocol())

    def test_unknown_ap_and_known_ap_support_corruption_is_rejected(self):
        for mutation in ("unknown_ap", "unknown_known", "unknown_positive", "known_missing_ap",
                         "known_count", "positive_count", "boolean_support", "nonfinite_ap", "missing_class"):
            entries = self.entries()
            unknown = entries[2]["ranking_ap"]["damsegment"][CLASSES[2]]
            known = entries[2]["ranking_ap"]["dacl"][CLASSES[2]]
            if mutation == "unknown_ap": unknown["ap"] = 0.
            elif mutation == "unknown_known": unknown["known_photos"] = COUNTS["damsegment"]
            elif mutation == "unknown_positive": unknown["positive_photos"] = 1
            elif mutation == "known_missing_ap": known["ap"] = None
            elif mutation == "known_count": known["known_photos"] -= 1
            elif mutation == "positive_count": known["positive_photos"] += 1
            elif mutation == "boolean_support": known["positive_photos"] = True
            elif mutation == "nonfinite_ap": known["ap"] = float("nan")
            else: entries[2]["ranking_ap"]["codebrim"].pop(CLASSES[2])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_measurements(entries, self.protocol())

    def test_measurements_reject_test_results_changed_views_and_changed_run_order(self):
        for mutation in ("test_executed", "test_result", "grid", "boolean_grid", "view_selection", "run_order"):
            entries = self.entries()
            if mutation == "test_executed": entries[2]["test_executed"] = True
            elif mutation == "test_result": entries[2]["test_result"] = {"target_passed": True}
            elif mutation == "grid": entries[2]["selected_grid"] = 2
            elif mutation == "boolean_grid": entries[2]["selected_grid"] = True
            elif mutation == "view_selection": entries[2]["configured_grids"] = [1, 2]
            else: entries[0], entries[1] = entries[1], entries[0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_measurements(entries, self.protocol())

    def test_exactly_five_percent_is_not_a_strict_target_pass(self):
        entries = self.entries()
        native = entries[2]
        for task in native["per_class"].values():
            for point in task["domains"].values():
                point.update(tp=190, fn=10, fnr=.05, fp=0, tn=point["negative_photos"], fpr=0.)
        self.set_small(native, (5, 5))
        self.refresh_error(native)
        self.assertFalse(native["target_passed"])
        report.validate_measurements(entries, self.protocol())
        native["target_passed"] = True
        with self.assertRaises(ValueError):
            report.validate_measurements(entries, self.protocol())

    def test_research_nomination_checks_both_comparators_and_each_small_and_ap_guard(self):
        for comparator in (0, 1):
            for guard in ("maximum_error", "target_rate", "small_total", "small_class", "other_ap"):
                entries = self.entries()
                base, native = entries[comparator], entries[2]
                if guard == "maximum_error":
                    self.set_errors(base, 12)
                    self.set_small(base, (12, 12))
                elif guard == "target_rate":
                    point = base["per_class"][TARGETS[0]]["domains"]["dacl"]
                    point.update(fp=0, tn=point["negative_photos"], fpr=0.)
                elif guard == "small_total":
                    self.set_small(base, (10, 12))
                elif guard == "small_class":
                    self.set_small(base, (5, 18))
                    self.set_small(native, (8, 8))
                else:
                    base["ranking_ap"]["dacl"][CLASSES[2]]["ap"] = .85
                with self.subTest(comparator=comparator, guard=guard):
                    measured = report.validate_measurements(entries, self.protocol())
                    self.assertFalse(measured["research_gate"]["research_candidate_nominated"])

    def test_render_reports_measured_counts_and_honest_preprocessing_scope(self):
        entries = self.entries()
        measured = report.validate_measurements(entries, self.protocol())
        result = {"experiments": entries, **measured,
            "prepared_data_integrity": {"changed_dacl_parents": 1, "changed_rows": 2, "derived_pngs": 4},
            "technical_verification": {"runtime_source_count": 27, "tests_run": 11},
            "actual_new_completed_training_epochs": 12}
        before = copy.deepcopy(result)
        document = report.render(result)
        self.assertEqual(result, before)
        self.assertIn("| native 보강군 | 640 | 6 | 3 | 6.00% | 미달 |", document)
        self.assertIn("연구 후보 기준: **통과**", document)
        self.assertIn("-4.00pp", document)
        self.assertIn("-3.00pp", document)
        self.assertIn("12/200 | 6.00%", document)
        self.assertIn("| 균열 | 15/93 | 14/93 | 10/93 |", document)
        self.assertIn("| 박락 | 18/105 | 17/105 | 12/105 |", document)
        z, p, n = 1.959963984540054, .06, 200
        denominator = 1 + z*z/n
        center = (p + z*z/(2*n)) / denominator
        radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
        self.assertIn(f"{100*(center-radius):.2f}%–{100*(center+radius):.2f}%", document)
        self.assertIn("14.50분 | 2.000 GiB | 10680 | 6", document)
        for phrase in ("두 신규 군의 640 PNG는 과거 최대512 JPEG crop과 다르다",
                       "native 정보 효과의 직접 대조는 새 대조군과 보강군 사이",
                       "같은 VAL에서 반복한 epoch·임계값 선택", "고유 사진198장",
                       "엄격한5%·현장 검증·배포 승인과 별도", "버려지는 AMP2회",
                       "AP42개", "보류 TEST 추론은 실행하지 않았다",
                       "전문가 확정 라벨·원본 정답 변경·새 독립 사진은0개",
                       "프로필 SHA를 유지", "미래 현장의 오차율5% 미만을 보장하지 않는다"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, document)

    def test_preparation_resources_match_verified_png_bytes_and_do_not_count_as_training_time(self):
        audit = {"preparation_elapsed_minutes": 3.5, "worker_count": 4,
            "derived_png_bytes_by_variant": {"control": 100, "native": 200},
            "derived_png_bytes_total": 300, "control_parent_resizes": 5}
        data_proof = {"output_png_bytes": 300,
            "output_png_bytes_by_variant": {"control": 100, "native": 200},
            "changed_dacl_parents": 5}
        before = copy.deepcopy((audit, data_proof))
        result = report.validate_preparation_resources(audit, data_proof)
        self.assertEqual((audit, data_proof), before)
        self.assertEqual(result["derived_png_bytes_total"], 300)
        self.assertEqual(result["control_parent_resizes"], 5)
        self.assertIs(result["included_in_training_elapsed_minutes"], False)
        for mutation in ("elapsed_nan", "elapsed_bool", "elapsed_zero", "workers_zero",
                         "workers_bool", "workers_above_bound", "total_plus", "total_minus",
                         "variant_swapped", "parents_plus", "parents_minus"):
            value = copy.deepcopy(audit)
            if mutation == "elapsed_nan": value["preparation_elapsed_minutes"] = float("nan")
            elif mutation == "elapsed_bool": value["preparation_elapsed_minutes"] = True
            elif mutation == "elapsed_zero": value["preparation_elapsed_minutes"] = 0.
            elif mutation == "workers_zero": value["worker_count"] = 0
            elif mutation == "workers_bool": value["worker_count"] = True
            elif mutation == "workers_above_bound": value["worker_count"] = 5
            elif mutation == "total_plus": value["derived_png_bytes_total"] += 1
            elif mutation == "total_minus": value["derived_png_bytes_total"] -= 1
            elif mutation == "variant_swapped": value["derived_png_bytes_by_variant"] = {"control": 200, "native": 100}
            elif mutation == "parents_plus": value["control_parent_resizes"] += 1
            else: value["control_parent_resizes"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_preparation_resources(value, data_proof)


if __name__ == "__main__":
    unittest.main()
