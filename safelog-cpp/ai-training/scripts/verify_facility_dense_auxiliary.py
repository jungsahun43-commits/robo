"""Prove preserved inputs, native19 exposure and the derived public checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_dense_auxiliary import (
    CONTROL, DRAW_COUNT, EPOCHS, INITIAL, MANIFEST, NAME, NEW_SOURCES, PROTOCOL,
    SEED, dense_fields, load_data, read, sha, validate_protocol,
)
from scripts.fetch_rc2119 import write_new
from scripts.facility_rc_positive import NAME as PREVIOUS_RUN
from scripts.verify_facility_rc_positive import git, snapshots, verify_git_sources
from scripts.train_facility_target import DOMAINS
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.dense_auxiliary_classifier import (
    ARCH, DenseAuxiliaryClassifier, original_state, zero_dense_projection,
)
from safelog_ai.frozen_batchnorm import batchnorm_state_sha256
from safelog_ai.presence_classifier import image_transform
from safelog_ai.retention_distillation import state_sha256

BEFORE = "runs/facility-dense-auxiliary-before.json"
LEDGER = "runs/facility-dense-auxiliary-input-ledger.json"
TESTS = "runs/facility-dense-auxiliary-tests.json"
PREFLIGHT = "runs/facility-dense-auxiliary-preflight/preflight.json"
DISPOSABLE = "runs/facility-dense-auxiliary-preflight/disposable.pt"
PREVIOUS_PROOF = "reports/facility-rc-positive-study-verification.json"
PREVIOUS_LEDGER = "runs/facility-rc-positive-input-ledger.json"
PREPARATION_LEDGER = "data/facility-dense-auxiliary-training/input-ledger.json"
DATA_AUDIT = "reports/facility-dense-auxiliary-data-audit.json"
PREVIOUS_COMPARISON = "reports/facility-rc-positive-study-comparison.json"
PREVIOUS_MARKDOWN = "reports/FACILITY_RC_POSITIVE_STUDY_RESULTS_KO.md"
DRAWS = "runs/facility-spalling-sampler-plan/paired-draws.npz"
APP_PROFILE = "reports/facility-inference-profile.json"
OUTPUT = "reports/facility-dense-auxiliary-study-verification.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, message, *, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), message)
    return value


def finite(value, message):
    require(type(value) in (int, float) and math.isfinite(value), message)
    return float(value)


def verify_tests(protocol):
    tests = read(ROOT / TESTS)
    integer(tests.get("tests_run"), "Actually completed focused tests are required", positive=True)
    for key in ("failures", "errors", "skipped"):
        require(type(tests.get(key)) is int and tests[key] == 0, "Focused tests did not all pass")
    require(tests.get("source_sha256") == protocol["source_sha256"],
            "Focused tests must bind every declared runtime source")
    if "protocol_sha256" in tests:
        require(tests["protocol_sha256"] == sha(ROOT / PROTOCOL), "Focused test protocol binding changed")
    return tests


def verify_preflight():
    preflight = read(ROOT / PREFLIGHT)
    require(preflight.get("status") == "passed"
            and preflight.get("protocol_sha256") == sha(ROOT / PROTOCOL)
            and type(preflight.get("actual_disposable_fp32_updates")) is int
            and preflight["actual_disposable_fp32_updates"] == 1
            and type(preflight.get("actual_training_epochs")) is int
            and preflight["actual_training_epochs"] == 0,
            "One actual TRAIN-only disposable preflight update is required")
    for key in ("strict_cpu_328_state_reload", "dense_projection_actual_gradient",
                "unknown_dense_supervision_preserved", "frozen_teacher_and_bn_preserved",
                "strict_fp32_diagnostic_only_flags_restored"):
        require(preflight.get(key) is True, "Actual dense preflight contract failed: " + key)
    require(preflight.get("output_shapes") == [[2, 7], [2, 7, 80, 80], [2, 19], [2, 19, 80, 80]],
            "Actual dense preflight shape contract differs")
    require(finite(preflight.get("loss"), "Preflight loss must be finite") > 0,
            "Preflight loss must be positive")
    return {key: preflight[key] for key in (
        "status", "actual_disposable_fp32_updates", "actual_training_epochs",
        "strict_cpu_328_state_reload", "dense_projection_actual_gradient",
        "unknown_dense_supervision_preserved", "frozen_teacher_and_bn_preserved",
        "strict_fp32_diagnostic_only_flags_restored", "output_shapes")}


def verify_source_inventory(protocol, previous):
    old = previous["source_sha256"]
    require(previous.get("status") == "passed" and previous.get("runtime_source_count") == len(old)
            and len(old) == 123, "Established original123 runtime sources are required")
    require(set(protocol["source_sha256"]) == set(old) | set(NEW_SOURCES)
            and all(protocol["source_sha256"][name] == digest for name, digest in old.items()),
            "Every prior source byte must remain in the new frozen source inventory")


def verify_prepared_data(data, protocol):
    manifest = read(ROOT / MANIFEST)
    audit = read(ROOT / DATA_AUDIT)
    require(audit.get("status") == "prepared" and audit.get("manifest_sha256") == sha(ROOT / MANIFEST)
            and audit.get("full_train_photos") == len(data["dense"]) == 6225
            and audit.get("derived_mask_files") == 6225
            and audit.get("classes") == list(AUX_CLASSES)
            and audit.get("per_label_pixel_cells") == data["dense_counts"]
            and protocol["dense_train_pixel_counts"] == data["dense_counts"]
            and protocol["dense_positive_weights"] == data["dense_weights"].tolist(),
            "Prepared native19 aggregate/weight arithmetic differs from the actual masks")
    require(all(audit.get(key) is False for key in (
        "heldout_images_or_annotations_opened", "model_inference_executed",
        "original_photo_targets_changed", "original_seven_channel_targets_changed")),
            "Dense preparation changed original truth or used heldouts/inference")
    preparation = read(ROOT / PREPARATION_LEDGER)
    require(audit.get("private_input_ledger_sha256") == sha(ROOT / PREPARATION_LEDGER)
            and preparation.get("all_inputs_preserved") is True
            and preparation["snapshots_before"] == preparation["snapshots_after"],
            "Dense preparation input preservation proof changed")
    require(manifest["audit"]["per_label_pixel_cells"] == data["dense_counts"],
            "Private manifest and public aggregate disagree")
    return manifest, preparation


def before_training(protocol):
    require(not (ROOT / BEFORE).exists() and not (ROOT / LEDGER).exists()
            and not (ROOT / "runs" / NAME).exists(), "Preserve an existing before-training proof or run")
    previous = read(ROOT / PREVIOUS_PROOF)
    verify_source_inventory(protocol, previous)
    require(sha(ROOT / PREVIOUS_LEDGER) == previous["input_ledger_sha256"],
            "Established prior protected input ledger changed")
    old_ledger = read(ROOT / PREVIOUS_LEDGER)
    old = old_ledger["snapshots_before"]
    require(len(old) == previous["protected_file_count"], "Prior protected file count differs from its ledger")
    data = load_data()
    manifest, preparation = verify_prepared_data(data, protocol)
    tests = verify_tests(protocol)
    verify_preflight()
    paths = set(old) | set(protocol["source_sha256"]) | set(protocol["input_sha256"])
    paths.update((PROTOCOL, TESTS, PREFLIGHT, DISPOSABLE, PREVIOUS_LEDGER,
                  PREPARATION_LEDGER, DATA_AUDIT, APP_PROFILE, PREVIOUS_COMPARISON, PREVIOUS_MARKDOWN))
    previous_run = ROOT / "runs" / PREVIOUS_RUN
    require(all((previous_run / name).is_file() for name in ("best.pt", "TRAINING.json", "history.json")),
            "Completed previous RC run must be preserved")
    paths.update(path.relative_to(ROOT).as_posix() for path in previous_run.rglob("*") if path.is_file())
    paths.update(preparation["snapshots_before"])
    for row in manifest["items"]:
        paths.update(row[key] for key in ("image", "annotation", "source", "mask"))
    current = snapshots(paths)
    require(all(current[path] == expected for path, expected in old.items()),
            "Prior original input SHA/size/mtime changed")
    require(all(current[path] == expected for path, expected in preparation["snapshots_before"].items()),
            "Prepared native19 source SHA/size/mtime changed")
    commit = git(["rev-parse", "HEAD"]).decode().strip()
    verify_git_sources({**protocol["source_sha256"], PROTOCOL: sha(ROOT / PROTOCOL)}, commit)
    write_new(ROOT / LEDGER, {
        "schema": "facility_dense_auxiliary_input_ledger_v1", "local_only": True,
        "snapshots_before": current, "prior_protected_file_count": len(old),
        "original_input_files": old_ledger["original_input_files"],
        "previous_ledger_sha256": sha(ROOT / PREVIOUS_LEDGER),
        "new_dense_mask_files": len(manifest["items"]),
    })
    write_new(ROOT / BEFORE, {
        "status": "passed", "protocol_sha256": sha(ROOT / PROTOCOL), "source_git_commit": commit,
        "source_sha256": protocol["source_sha256"], "git_blob_bytes_verified": True,
        "input_ledger_sha256": sha(ROOT / LEDGER), "protected_files": len(current),
        "prior_protected_files_preserved": len(old), "new_dense_mask_files": len(manifest["items"]),
        "test_record_sha256": sha(ROOT / TESTS), "tests_completed": tests["tests_run"],
        "preflight_sha256": sha(ROOT / PREFLIGHT),
    })
    print(json.dumps({"status": "passed_before_training", "protected_files": len(current),
                      "source_git_commit": commit, "prior_files_preserved": len(old)}), flush=True)


def dense_exposure_tables(data):
    """Recompute known rows and positive cells from immutable NPZ arrays."""
    known = np.zeros((len(data["items"]), 19), np.int64)
    positive = np.zeros_like(known)
    eligible = np.zeros(len(data["items"]), np.int64)
    for index, item in enumerate(data["items"]):
        if index >= data["full_count"] or item["domain"] != "dacl":
            continue
        row = data["dense"].get(item["image"])
        require(row is not None, "An eligible original full DACL row lost dense supervision")
        masks, mask_known = dense_fields(row)
        require(np.array_equal(mask_known, np.asarray(row["known"], dtype=np.uint8))
                and not masks[mask_known == 0].any(), "Unknown dense channel became asserted foreground")
        known[index] = mask_known
        positive[index] = masks.sum((1, 2), dtype=np.int64) * mask_known
        eligible[index] = 1
    require(int(eligible.sum()) == 6225, "Exactly original6225 full TRAIN rows must be eligible")
    return eligible, known, positive


def verify_history(history, data, arrays, protocol, control_history):
    require(isinstance(history, list) and len(history) == EPOCHS
            and isinstance(control_history, list) and len(control_history) == EPOCHS,
            "Six completed candidate epochs and the existing six-epoch control are required")
    require(isinstance(arrays, np.ndarray) and arrays.dtype == np.int64
            and arrays.shape == (EPOCHS, DRAW_COUNT)
            and np.array_equal(arrays, data["epoch_draws"])
            and int(arrays.min()) >= 0 and int(arrays.max()) < len(data["items"]),
            "The exact original int64 control draw arrays must be preserved")
    labels = np.asarray([row["targets"] for row in data["items"]], np.int64)
    domains = np.asarray([DOMAINS[row["domain"]] for row in data["items"]], np.int64)
    eligible, dense_known, dense_positive = dense_exposure_tables(data)
    measured_epochs = []
    for index, row in enumerate(history):
        integer(row.get("epoch"), "Invalid actual epoch number", positive=True)
        indices = arrays[index]
        digest = hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()
        require(row["epoch"] == index + 1 and row.get("sampled_row_indices_sha256") == digest
                and control_history[index].get("sampled_row_indices_sha256") == digest,
                "Observed candidate or historical control draw order differs")
        expected_domains = {name: int((domains[indices] == number).sum()) for name, number in DOMAINS.items()}
        require(row.get("sampled_domain_counts") == expected_domains
                and control_history[index].get("sampled_domain_counts") == expected_domains,
                "Original source exposure changed")
        expected_photo = {name: {"positive": int((labels[indices, column] == 1).sum()),
                                "negative": int((labels[indices, column] == 0).sum()),
                                "unknown": int((labels[indices, column] == -1).sum())}
                          for column, name in enumerate(data["classes"])}
        require(row.get("sampled_photo_target_counts") == expected_photo
                and control_history[index].get("sampled_photo_target_counts") == expected_photo,
                "Original asserted/unknown photo label exposure changed")
        expected_eligible = int(eligible[indices].sum())
        expected_known = dense_known[indices].sum(0).tolist()
        expected_positive = dense_positive[indices].sum(0).tolist()
        require(row.get("sampled_dense_eligible_rows") == expected_eligible
                and expected_eligible == protocol["eligible_draws_by_epoch"][index]
                and row.get("sampled_dense_known_class_rows") == expected_known
                and row.get("sampled_dense_positive_cells") == expected_positive,
                "Observed native19 known/positive exposure differs from actual NPZ masks")
        updates = row["optimizer_step_diagnostics"]
        for key in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps"):
            integer(updates.get(key), "Invalid actual optimizer/AMP count")
        require(updates["attempted_batches"] == DRAW_COUNT // 8 == row.get("teacher_forward_batches")
                and updates["actual_optimizer_steps"] + updates["amp_skipped_steps"] == updates["attempted_batches"]
                and 0 < updates["actual_optimizer_steps"] <= updates["attempted_batches"],
                "Actual optimizer/AMP/teacher batch budget differs")
        expected_lr = [5e-6 + (initial - 5e-6) * (1 + math.cos(math.pi * index / EPOCHS)) / 2
                       for initial in (4e-5, 1e-4)]
        observed_lr = row.get("optimizer_learning_rates")
        require(isinstance(observed_lr, list) and len(observed_lr) == 2
                and all(math.isclose(finite(value, "Nonfinite observed learning rate"), expected,
                                     rel_tol=1e-12, abs_tol=1e-15)
                        for value, expected in zip(observed_lr, expected_lr)), "Declared cosine learning rates changed")
        require(row.get("teacher_state_unchanged") is True and row.get("bn_buffers_unchanged") is True,
                "Teacher or BatchNorm state changed")
        require(finite(row.get("train_loss"), "Nonfinite training loss") > 0,
                "Actual training loss must be positive")
        measured_epochs.append({
            "epoch": index + 1, "sampled_domain_counts": expected_domains,
            "dense_eligible_rows": expected_eligible, "dense_known_class_rows": expected_known,
            "dense_positive_cells": expected_positive, "optimizer_step_diagnostics": updates,
        })
    attempts = sum(row["optimizer_step_diagnostics"]["attempted_batches"] for row in history)
    require(attempts == EPOCHS * (DRAW_COUNT // 8) == 10686, "Actual six-epoch attempt budget differs")
    return {
        "completed_epochs": EPOCHS, "sampled_draws": EPOCHS * DRAW_COUNT,
        "attempted_batches": attempts,
        "actual_optimizer_steps": sum(row["optimizer_step_diagnostics"]["actual_optimizer_steps"] for row in history),
        "amp_skipped_steps": sum(row["optimizer_step_diagnostics"]["amp_skipped_steps"] for row in history),
        "actual_teacher_forward_batches": sum(row["teacher_forward_batches"] for row in history),
        "original_control_draw_order_and_photo_labels_preserved": True,
        "dense_exposure_recomputed_from_npz_masks": True,
        "horizontal_flips_preserve_recomputed_cell_counts": True,
        "epochs": measured_epochs,
    }


def verify_checkpoints(data, protocol, training, history, run):
    full_sha, public_sha = sha(run / "dense-best.pt"), sha(run / "best.pt")
    require(training.get("weights_sha256") == public_sha and training.get("dense_weights_sha256") == full_sha,
            "Selected dense or public checkpoint SHA256 changed")
    full = torch.load(run / "dense-best.pt", map_location="cpu", weights_only=True)
    public = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    split_sha = sha(run / "SPLIT.json")
    selected = min(history, key=lambda row: (row["worst_target_error"], row["sum_target_errors"]))
    for checkpoint, architecture, count in ((full, ARCH, 328), (public, AUX_ARCH, 324)):
        require(checkpoint.get("architecture") == architecture and checkpoint.get("classes") == data["classes"]
                and checkpoint.get("auxiliary_classes") == list(AUX_CLASSES) and checkpoint.get("imgsz") == 640
                and checkpoint.get("split_sha256") == split_sha == training["split_sha256"]
                and checkpoint.get("study_protocol_sha256") == sha(ROOT / PROTOCOL)
                and checkpoint.get("epoch") == selected["epoch"]
                and checkpoint.get("worst_target_error") == selected["worst_target_error"]
                and checkpoint.get("selection_split") == "val"
                and len(checkpoint.get("state_dict", {})) == count
                and all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                        for value in checkpoint["state_dict"].values()),
                "Selected checkpoint metadata or finite state contract differs")
    require(full.get("dense_classes") == list(AUX_CLASSES) and full.get("dense_initialization_seed") == SEED
            and public.get("derived_from_dense_weights_sha256") == full_sha
            and public.get("training_only_head_removed") is True
            and read(run / "VALIDATION.json") == selected,
            "Selected full328 and derived324 provenance or epoch selection differs")
    dense_model = DenseAuxiliaryClassifier(7, pretrained=False, dense_seed=SEED)
    dense_model.load_state_dict(full["state_dict"], strict=True)
    dense_model.eval()
    require(not zero_dense_projection(dense_model), "The actual selected dense projection never learned")
    exported = original_state(dense_model)
    require(set(exported) == set(public["state_dict"])
            and all(torch.equal(value, public["state_dict"][name]) for name, value in exported.items()),
            "The derived324 export differs from selected original tensors")
    legacy_model = AuxiliaryClassifier(7, pretrained=False)
    legacy_model.load_state_dict(public["state_dict"], strict=True)
    legacy_model.eval()
    require(batchnorm_state_sha256(dense_model) == batchnorm_state_sha256(legacy_model)
            == training["initial_bn_buffer_sha256"], "Selected checkpoint BatchNorm buffers changed")
    with Image.open(ROOT / data["items"][0]["image"]) as image:
        inputs = image_transform(640)(image.convert("RGB"))[None]
    with torch.no_grad():
        outputs = dense_model.forward_dense_training(inputs)
        dense_public = dense_model(inputs)
        legacy_outputs = legacy_model.forward_training(inputs)
        legacy_public = legacy_model(inputs)
    require([list(value.shape) for value in outputs] == [[1, 7], [1, 7, 80, 80], [1, 19], [1, 19, 80, 80]]
            and all(bool(torch.isfinite(value).all()) for value in (*outputs, *legacy_outputs))
            and torch.equal(dense_public, outputs[0]) and torch.equal(legacy_public, legacy_outputs[0])
            and torch.equal(dense_public, legacy_public)
            and all(torch.equal(left, right) for left, right in zip(outputs[:3], legacy_outputs)),
            "Trained full328 and derived324 actual CPU outputs are not exactly equal")
    return {
        "weights_sha256": public_sha, "dense_weights_sha256": full_sha, "selected_epoch": selected["epoch"],
        "cpu_strict_328_state_reload": True, "cpu_strict_324_state_reload": True,
        "derived_public_state_tensors_equal": True, "trained_public_outputs_exactly_equal": True,
        "trained_original_three_outputs_exactly_equal": True, "all_outputs_finite": True,
        "public_shape": [1, 7], "private_spatial_shape": [1, 7, 80, 80],
        "auxiliary_shape": [1, 19], "dense_auxiliary_shape": [1, 19, 80, 80],
        "input_scope": "One original TRAIN photo; CPU float32; no heldout inference for this contract check",
    }


def after_training(protocol):
    require(not (ROOT / OUTPUT).exists(), "Preserve completed verification")
    before, ledger = read(ROOT / BEFORE), read(ROOT / LEDGER)
    require(before.get("status") == "passed" and before.get("protocol_sha256") == sha(ROOT / PROTOCOL)
            and before.get("source_sha256") == protocol["source_sha256"]
            and before.get("input_ledger_sha256") == sha(ROOT / LEDGER)
            and before.get("test_record_sha256") == sha(ROOT / TESTS)
            and before.get("preflight_sha256") == sha(ROOT / PREFLIGHT),
            "Before-training source/input/test/preflight binding changed")
    current = snapshots(ledger["snapshots_before"])
    require(current == ledger["snapshots_before"], "Protected input SHA/size/mtime changed during training/evaluation")
    verify_git_sources({**protocol["source_sha256"], PROTOCOL: sha(ROOT / PROTOCOL)}, before["source_git_commit"])
    tests, preflight = verify_tests(protocol), verify_preflight()
    previous = read(ROOT / PREVIOUS_PROOF)
    verify_source_inventory(protocol, previous)
    data = load_data()
    verify_prepared_data(data, protocol)
    with np.load(ROOT / DRAWS, allow_pickle=False) as arrays:
        draws = arrays["control"].copy()
    run = ROOT / "runs" / NAME
    training, history = read(run / "TRAINING.json"), read(run / "history.json")
    require(training.get("status") == "complete" and training.get("actual_epochs") == EPOCHS
            and training.get("study_protocol_sha256") == sha(ROOT / PROTOCOL)
            and training.get("source_git_commit") == before["source_git_commit"]
            and training.get("source_sha256") == protocol["source_sha256"]
            and training.get("architecture") == AUX_ARCH and training.get("training_model_architecture") == ARCH
            and training.get("source_test_used") is False and training.get("historical_control_retrained") is False
            and training.get("original_sampling_and_targets_unchanged") is True,
            "Completed actual dense training metadata differs")
    measured = verify_history(history, data, draws, protocol, read(ROOT / "runs" / CONTROL / "history.json"))
    require(training.get("attempted_batches") == measured["attempted_batches"]
            and training.get("actual_optimizer_steps") == measured["actual_optimizer_steps"]
            and training.get("actual_teacher_forward_batches") == measured["actual_teacher_forward_batches"],
            "Actual training update/teacher summaries differ")
    initializer = torch.load(ROOT / INITIAL, map_location="cpu", weights_only=True)
    teacher = AuxiliaryClassifier(7, pretrained=False)
    teacher.load_state_dict(initializer["state_dict"], strict=True)
    require(training.get("initial_bn_buffer_sha256") == training.get("final_bn_buffer_sha256")
            == batchnorm_state_sha256(teacher)
            and training.get("teacher_state_sha256") == training.get("final_teacher_state_sha256")
            == state_sha256(teacher.state_dict()), "Teacher or BN summary differs from the original fixed initializer")
    require(training.get("original_photo_positive_weights") == data["sampling_data"]["photo_weights"].tolist()
            and training.get("original_pixel_positive_weights") == data["supervision_weights"]["pixel_weights"].tolist()
            and training.get("original_auxiliary_positive_weights") == data["supervision_weights"]["auxiliary_weights"].tolist()
            and training.get("dense_positive_weights") == data["dense_weights"].tolist(),
            "Original or declared dense loss weights changed")
    require(not (run / "TARGET-TEST.json").exists() and not list(run.glob("target-test-*.json")),
            "Source TEST inference must remain outside this repeated study")
    checkpoints = verify_checkpoints(data, protocol, training, history, run)
    result = {
        "schema": "facility_dense_auxiliary_study_verification_v1", "status": "passed",
        "protocol_sha256": sha(ROOT / PROTOCOL), "source_git_commit": before["source_git_commit"],
        "runtime_source_count": len(protocol["source_sha256"]), "source_sha256": protocol["source_sha256"],
        "git_blob_bytes_verified": True, "protected_file_count": len(current),
        "prior_protected_files_preserved": ledger["prior_protected_file_count"],
        "original_input_files_preserved": ledger["original_input_files"], "new_dense_mask_files_preserved": 6225,
        "all_original_and_new_input_sha_size_mtime_preserved": True, "individual_input_paths_published": False,
        "input_ledger_sha256": sha(ROOT / LEDGER), "before_training_sha256": sha(ROOT / BEFORE),
        "test_record_sha256": sha(ROOT / TESTS), "tests": tests,
        "preflight_sha256": sha(ROOT / PREFLIGHT), "actual_train_gpu_preflight": preflight,
        "actual_sampling": measured, "actual_completed_training_epochs": EPOCHS,
        "reused_control_epochs_not_recounted": EPOCHS, **checkpoints,
        "public_output_equals_training_photo_output": True, "frozen_teacher_and_bn_preserved": True,
        "app_profile_sha256": current[APP_PROFILE]["sha256"], "app_profile_unchanged": True,
        "independent_new_photos": 0, "new_public_photo_targets": 0, "original_public_pixel_targets_changed": False,
        "training_only_auxiliary_spatial_classes": list(AUX_CLASSES),
        "source_test_inference_executed": False, "app_model_promoted": False,
        "research_models_deployed": False, "verification_training_epochs": 0,
        "accuracy_measured_by_this_check": False, "independent_field_safety_verified": False,
    }
    write_new(ROOT / OUTPUT, result)
    print(json.dumps({key: result[key] for key in (
        "status", "actual_completed_training_epochs", "weights_sha256", "protected_file_count")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = validate_protocol(read(ROOT / PROTOCOL))
    (before_training if args.before else after_training)(protocol)


if __name__ == "__main__":
    main()
