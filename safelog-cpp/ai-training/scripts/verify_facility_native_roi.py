"""Verify completed native ROI checkpoints and preserved TRAIN pixels on CPU.

Local hashes prove consistency with recorded files, not publisher authenticity,
expert truth, field accuracy, or structural safety. No held-out image inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.presence_classifier import MEAN, STD
from scripts.facility_native_roi_study import (
    AUDIT, MANIFESTS, MIN_TEST_COUNTS, PROTOCOL, SOURCE_FILES, protected_hashes, validate_protocol,
)
from scripts.preflight_facility_detail import select_train_replay
from scripts.report_facility_resolution import validate_histories
from scripts.verify_facility_resolution import (
    git_blob, hash_map, local_path, read, require, sha, verify_cpu_reload,
)

RUNTIME_SOURCES = tuple(SOURCE_FILES)
TEST_SOURCES = (
    "tests/test_facility_native_roi_data.py",
    "tests/test_facility_native_roi_study.py",
    "tests/test_facility_native_roi_report.py",
)
PROTOCOL_PATH = PROTOCOL
PREFLIGHT_PATH = "runs/facility-native-roi-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-native-roi-source-before-training.json"
LEDGER_PATH = "runs/facility-native-roi-data/ledger.json"
OUTPUT_PATH = "reports/facility-native-roi-study-verification.json"
APP_PROFILE = "reports/facility-inference-profile.json"
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def verify_frozen_sources(snapshot, protocol, preflight, root, protocol_sha,
                          preflight_sha, blob_reader=None):
    require(snapshot.get("schema") == "facility_native_roi_source_before_training_v1"
            and snapshot.get("declared_before_training") is True
            and type(snapshot.get("new_completed_training_epochs_at_snapshot")) is int
            and snapshot["new_completed_training_epochs_at_snapshot"] == 0,
            "A recorded before-training source snapshot is required")
    commit = snapshot.get("source_git_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[a-f0-9]{40}", commit),
            "Recorded Git commit must be a full object ID")
    for document in (snapshot, protocol, preflight):
        hash_map(document.get("source_sha256"), RUNTIME_SOURCES)
    sources = protocol["source_sha256"]
    require(snapshot["source_sha256"] == preflight["source_sha256"] == sources,
            "Recorded source inventories differ")
    require(snapshot.get("protocol_sha256") == preflight.get("protocol_sha256") == protocol_sha
            and snapshot.get("preflight_sha256") == preflight_sha,
            "Snapshot protocol/preflight binding differs")
    reader = blob_reader or (lambda c, p: git_blob(root, c, p))
    for relative, expected in sources.items():
        require(sha(local_path(root, relative)) == expected,
                "Current source differs from before-training bytes")
        blob = reader(commit, relative)
        require(isinstance(blob, bytes) and hashlib.sha256(blob).hexdigest() == expected,
                "Committed source bytes differ from before-training record")
    return {"source_git_commit": commit, "runtime_source_count": len(sources),
            "source_sha256": dict(sources), "git_blob_bytes_verified": True,
            "working_runtime_sources_unchanged": True}


def validate_preflight(preflight, protocol, protocol_sha, current_protected):
    fixed = {"schema": "facility_native_roi_preflight_v1", "status": "passed",
             "protocol_sha256": protocol_sha,
             "actual_labels_masks_known_aux_domains_row_indices_equal": True,
             "mask_label_known_aux_index_replay_equal": True,
             "unchanged_image_tensors_equal": True, "unique_train_rows": 8,
             "new_training_epochs": 0, "source_test_inference_executed": False,
             "app_model_promoted": False, "accuracy_measured": False,
             "label_changes": 0, "new_masks": 0}
    for key, expected in fixed.items():
        require(type(preflight.get(key)) is type(expected) and preflight[key] == expected,
                "Actual preflight replay/update contract differs")
    hash_map(current_protected, current_protected.keys())
    require(preflight.get("protected_file_sha256") == current_protected
            and current_protected.get(PROTOCOL_PATH) == protocol_sha
            and current_protected.get(APP_PROFILE) == protocol["app_profile_sha256"],
            "Protected preflight data/protocol/app bytes differ")
    require(preflight.get("source_sha256") == protocol["source_sha256"],
            "Preflight was executed against different runtime sources")
    require(isinstance(preflight.get("replay_row_indices_sha256"), str)
            and _SHA.fullmatch(preflight["replay_row_indices_sha256"]),
            "Actual replay row digest is missing")
    require(set(preflight.get("pairs", {})) == {"control", "native"},
            "Both actual preflight variants are required")
    counts = []
    for row in preflight["pairs"].values():
        expected = {"actual_batch_size": 8, "actual_optimizer_steps": 1,
                    "amp_skipped_steps": 0, "public_shape": [8, 7],
                    "pixel_shape": [8, 7, 80, 80], "auxiliary_shape": [8, 19],
                    "all_outputs_finite": True, "cpu_reload_verified": True,
                    "strict_factory_reload_verified": True,
                    "public_output_equals_training_photo_output": True}
        for key, value in expected.items():
            require(type(row.get(key)) is type(value) and row[key] == value,
                    "Actual preflight AMP/shape/CPU reload proof differs")
        require(type(row.get("parameter_count")) is int and row["parameter_count"] > 0
                and type(row.get("peak_cuda_allocated_bytes")) is int
                and row["peak_cuda_allocated_bytes"] > 0
                and isinstance(row.get("disposable_checkpoint_sha256"), str)
                and _SHA.fullmatch(row["disposable_checkpoint_sha256"]),
                "Measured preflight parameter/resource/checkpoint proof is missing")
        transfer = row.get("initial_state_transfer", {})
        require(transfer.get("shared_state_tensors_equal") is True
                and type(transfer.get("shared_state_tensor_count")) is int
                and transfer["shared_state_tensor_count"] > 0
                and transfer.get("strict_state_load") is True
                and type(transfer.get("new_state_tensor_count")) is int
                and transfer["new_state_tensor_count"] == 0
                and type(transfer.get("additional_parameters")) is int
                and transfer["additional_parameters"] == 0
                and transfer.get("new_output_projection_zero") is None,
                "Preflight introduced parameters or changed initializer state")
        counts.append(row["parameter_count"])
    require(counts[0] == counts[1], "Paired parameter counts differ")
    require(preflight["pairs"]["control"]["initial_state_transfer"]
            == preflight["pairs"]["native"]["initial_state_transfer"],
            "Initial state inventories differ across the pair")
    return {"passed": True, "actual_train_rows": 8, "parameter_count": counts[0],
            "disposable_amp_optimizer_steps": 2, "actual_non_image_replay_identical": True}


def validate_test_counts(record):
    require(set(MIN_TEST_COUNTS) == set(TEST_SOURCES), "Focused suite inventory differs from its declared minimum")
    require(type(record.get("tests_run")) is int and record["tests_run"] >= sum(MIN_TEST_COUNTS.values())
            and all(type(record.get(k)) is int and record[k] == 0
                    for k in ("failures", "errors", "skipped")),
            "All focused suites must actually execute with no failures/errors/skips")
    by_module = record.get("tests_by_module", {})
    require(isinstance(by_module, dict) and set(by_module) == set(MIN_TEST_COUNTS)
            and all(type(by_module[k]) is int and by_module[k] >= minimum
                    for k, minimum in MIN_TEST_COUNTS.items())
            and sum(by_module.values()) == record["tests_run"]
            and type(record.get("expected_tests_collected")) is int
            and record["expected_tests_collected"] == record["tests_run"],
            "Collected full-suite counts differ from actually executed focused tests")
    return {key: record[key] for key in ("tests_run", "failures", "errors", "skipped",
                                          "expected_tests_collected", "tests_by_module")}


def validate_test_results(record, root):
    require(record.get("schema") == "facility_native_roi_test_results_v1"
            and record.get("status") == "passed"
            and record.get("before_after_sources_equal") is True,
            "Actual passed test-result record is required")
    counts = validate_test_counts(record)
    hash_map(record.get("test_source_sha256"), TEST_SOURCES)
    hash_map(record.get("source_sha256"), RUNTIME_SOURCES)
    for relative, expected in {**record["test_source_sha256"], **record["source_sha256"]}.items():
        require(sha(local_path(root, relative)) == expected, "Tested source bytes changed")
    return {**counts, **{key: record[key] for key in ("test_source_sha256", "source_sha256")}}


def validate_run_metadata(training, checkpoint, history, variant, protocol, protocol_sha,
                          initial_state, weights_sha, split_sha):
    require(variant in ("control", "native"), "Unknown native ROI variant")
    expected = {"status": "complete", "actual_epochs": protocol["requested_epochs"],
                "architecture": protocol["architecture_by_variant"][variant],
                "imgsz": 640, "model_variant": variant, "classes": protocol["classes"],
                "auxiliary_classes": protocol["auxiliary_classes"],
                "auxiliary_source_class_count": 19, "validation_batch_size": 16,
                "validation_domains": ["dacl", "damsegment", "codebrim"],
                "train_dacl": 6225, "train_damsegment": 1585, "train_codebrim": 6438,
                "val_dacl": 710, "val_damsegment": 424,
                "initial_weights_sha256": protocol["initial_weights_sha256"],
                "core_spatial_manifest_sha256": protocol["core_spatial_manifest_sha256"],
                "spatial_manifest_sha256": protocol["paired_manifest_sha256"][variant],
                "spatial_manifest_path": protocol["paired_manifest_paths"][variant],
                "auxiliary_manifest_sha256": protocol["auxiliary_manifest_sha256"],
                "source_sha256": protocol["source_sha256"], "study_protocol_sha256": protocol_sha,
                "study_protocol_path": PROTOCOL_PATH,
                "native_roi_recipe": protocol["native_roi_recipe"],
                "paired_data_audit_sha256": protocol["paired_data_audit_sha256"],
                "photo_pooling_grid": [80, 80], "source_pixel_target_grid": [80, 80],
                "raw_spatial_grid": [80, 80], "new_photo_targets": 0,
                "new_pixel_targets": 0, "app_model_promoted": False,
                "target_ranking_weight": 0., "weights_sha256": weights_sha, "split_sha256": split_sha}
    for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
                "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions"):
        expected[key] = protocol[key]
    for key, value in expected.items():
        require(type(training.get(key)) is type(value) and training[key] == value,
                f"Actual native ROI training condition differs: {key}")
    transfer = {"shared_state_tensors_equal": True, "shared_state_tensor_count": len(initial_state),
                "new_state_tensor_count": 0, "new_output_projection_zero": None,
                "strict_state_load": True, "additional_parameters": 0}
    require(training.get("initial_state_transfer") == transfer, "Strict initializer transfer differs")
    require(training.get("target_ranking") == {"weight": 0.,
                "classes": ["concrete_crack", "concrete_spalling"], "sampling_changed": False,
                "new_labels_asserted": 0, "public_outputs_changed": False},
            "Ranking/new-label state changed")
    preservation = training.get("paired_label_preservation", {})
    require(all(preservation.get(k) is True for k in ("full_rows_equal", "photo_targets_equal",
                "pixel_target_paths_equal", "sample_weights_equal", "auxiliary_crop_targets_remain_unknown"))
            and type(preservation.get("row_count")) is int and preservation["row_count"] == 26289
            and type(preservation.get("changed_dacl_detail_rows")) is int
            and 0 < preservation["changed_dacl_detail_rows"] <= 10586,
            "Original full/crop targets, masks, sampling or unknown auxiliary targets changed")
    require(len(history) == protocol["requested_epochs"], "Declared epochs were not completed")
    for key in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps"):
        actual = sum(row["optimizer_step_diagnostics"][key] for row in history)
        require(type(training.get("optimizer_step_diagnostics", {}).get(key)) is int
                and training["optimizer_step_diagnostics"][key] == actual,
                "Optimizer summary differs from actual epochs")
    require(training["optimizer_step_diagnostics"]["actual_optimizer_steps"] > 0,
            "No actual optimizer updates were recorded")
    require(type(training.get("elapsed_training_minutes")) in (int, float)
            and math.isfinite(training["elapsed_training_minutes"]) and training["elapsed_training_minutes"] > 0
            and type(training.get("peak_cuda_allocated_bytes")) is int
            and training["peak_cuda_allocated_bytes"] > 0, "Measured training resources are missing")
    for key, value in {"architecture": expected["architecture"], "classes": protocol["classes"],
                       "auxiliary_classes": list(AUX_CLASSES), "imgsz": 640, "mean": MEAN, "std": STD,
                       "selection_split": "val", "split_sha256": split_sha}.items():
        require(type(checkpoint.get(key)) is type(value) and checkpoint[key] == value,
                "Completed checkpoint metadata differs")
    epoch = checkpoint.get("epoch")
    require(type(epoch) is int and 1 <= epoch <= len(history), "Selected checkpoint epoch is invalid")
    error = checkpoint.get("worst_target_error")
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1
            and error == history[epoch - 1]["worst_target_error"] == training.get("best_worst_target_error"),
            "Selected checkpoint measurement differs from actual history")
    state = checkpoint.get("state_dict")
    require(isinstance(state, dict) and set(state) == set(initial_state), "Checkpoint state inventory changed")
    for key, tensor in state.items():
        reference = initial_state[key]
        require(isinstance(tensor, torch.Tensor) and tensor.shape == reference.shape
                and tensor.dtype == reference.dtype and bool(torch.isfinite(tensor).all()),
                "Completed checkpoint state shape/dtype/finite contract differs")
    return {"imgsz": 640, "architecture": expected["architecture"], "actual_epochs": training["actual_epochs"],
            "selected_epoch": epoch, "weights_sha256": weights_sha, "state_tensor_count": len(state),
            "new_state_tensor_count": 0, "strict_state_inventory_verified": True,
            "optimizer_step_diagnostics": dict(training["optimizer_step_diagnostics"])}


def verify_prepared_data(root, protocol):
    """Rehash actual paired PNGs and all recorded original TRAIN inputs."""
    from scripts.prepare_facility_native_roi import RECIPE, eligible, pixel_sha, validate_box, validate_pair
    root = Path(root).resolve()
    audit = read(local_path(root, AUDIT)); ledger_path = local_path(root, LEDGER_PATH)
    ledger = read(ledger_path)
    require(audit.get("schema") == "facility_native_roi_data_audit_v1"
            and audit.get("status") == "prepared" and audit.get("private_ledger_sha256") == sha(ledger_path)
            and ledger.get("schema") == "facility_native_roi_ledger_v1"
            and ledger.get("local_only") is True and ledger.get("split") == "train"
            and ledger.get("all_inputs_preserved") is True
            and ledger.get("recipe") == audit.get("recipe") == RECIPE,
            "Prepared TRAIN ledger/audit binding differs")
    require(sha(root / AUDIT) == protocol["paired_data_audit_sha256"]
            and ledger.get("base_manifest_sha256") == protocol["core_spatial_manifest_sha256"]
            and ledger.get("derived_manifest_sha256") == protocol["paired_manifest_sha256"],
            "Ledger belongs to different original/paired manifests")
    before, after = ledger.get("input_snapshots_before"), ledger.get("input_snapshots_after")
    require(isinstance(before, dict) and before and before == after,
            "Original source byte/size/mtime preservation is missing")
    for relative, evidence in before.items():
        require(isinstance(evidence, dict) and set(evidence) == {"sha256", "size_bytes", "mtime_ns"}
                and isinstance(evidence["sha256"], str) and _SHA.fullmatch(evidence["sha256"])
                and type(evidence["size_bytes"]) is int and evidence["size_bytes"] >= 0
                and type(evidence["mtime_ns"]) is int, "Original input fingerprint is invalid")
        path = local_path(root, relative); stat = path.stat()
        require(stat.st_size == evidence["size_bytes"] and stat.st_mtime_ns == evidence["mtime_ns"]
                and sha(path) == evidence["sha256"], "A recorded original TRAIN input changed")
    base = read(root / "data/facility-spatial-training/train.json")
    pair = {v: read(root / p) for v, p in MANIFESTS.items()}
    require(all(r[key] in before for r in base["items"] for key in ("image", "pixel_target")),
            "The preservation ledger omits an original TRAIN image or mask")
    pair_proof = validate_pair(base, pair["control"], pair["native"])
    for variant, path in MANIFESTS.items():
        require(sha(root / path) == protocol["paired_manifest_sha256"][variant], "Paired manifest SHA changed")
    parents = ledger.get("parents"); rows = ledger.get("changed_rows")
    require(isinstance(parents, list) and isinstance(rows, list) and rows,
            "Actual parent/crop ledger inventory is missing")
    by_parent = {p["parent_image"]: p for p in parents}
    originals = {r["image"] for r in base["items"][:base["full_count"]] if r["domain"] == "dacl"}
    require(len(by_parent) == len(parents) == len(originals) and set(by_parent) == originals,
            "Geometry eligibility does not cover every original DACL TRAIN parent")
    for parent in parents:
        require(type(parent.get("eligible")) is bool
                and parent["eligible"] == eligible(parent.get("source_size"), parent.get("processed_size")),
                "ROI eligibility was not determined solely from source geometry")
    expected_indices = [i for i, r in enumerate(base["items"]) if i >= base["full_count"]
                        and r["domain"] == "dacl" and by_parent[r["parent_image"]]["eligible"]]
    require(all(type(r.get("row_index")) is int for r in rows)
            and [r.get("row_index") for r in rows] == expected_indices
            and len(rows) == pair_proof["changed_rows"] == audit.get("changed_rows"),
            "Every existing eligible positive/negative/unknown crop must be included in order")
    output_bytes = 0; output_bytes_by_variant = {"control": 0, "native": 0}
    for row in rows:
        index = row["row_index"]; original = base["items"][index]; parent = by_parent[original["parent_image"]]
        require(row.get("parent_image") == original["parent_image"] and row.get("original_image") == original["image"]
                and row.get("pixel_target") == original["pixel_target"]
                and row.get("box_in_parent") == original["box_in_parent"]
                and row.get("source_size") == parent["source_size"]
                and row.get("processed_size") == parent["processed_size"],
                "Ledger changed original crop geometry or mask identity")
        mask = local_path(root, row["pixel_target"])
        require(row.get("pixel_target_sha256") == sha(mask), "An original coarse mask changed")
        box = validate_box(row["box_in_parent"], row["processed_size"])
        sx = row["source_size"][0] / row["processed_size"][0]
        sy = row["source_size"][1] / row["processed_size"][1]
        projected = row.get("native_box")
        require(isinstance(projected, list) and len(projected) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in projected)
                and projected == [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy],
                "Equivalent half-open floating native ROI geometry changed")
        for variant in ("control", "native"):
            evidence = row.get(variant, {}); relative = pair[variant]["items"][index]["image"]
            require(evidence.get("image") == relative, "Paired PNG path/order differs")
            path = local_path(root, relative)
            require(evidence.get("sha256") == sha(path)
                    and type(evidence.get("size_bytes")) is int
                    and evidence["size_bytes"] == path.stat().st_size, "A prepared ROI PNG changed")
            with Image.open(path) as image:
                require(image.format == "PNG" and image.mode == "RGB" and image.size == (640, 640)
                        and image.info == {},
                        "Prepared ROI must be an RGB 640-square PNG")
                # Verify the decoded pixels as well as the encoded file bytes.
                require(pixel_sha(image) == evidence.get("pixel_sha256"),
                        "Prepared ROI decoded pixel digest differs")
            output_bytes += path.stat().st_size
            output_bytes_by_variant[variant] += path.stat().st_size
    require(audit.get("derived_pngs") == 2*len(rows) and audit.get("input_files_checked") == len(before)
            and audit.get("changed_dacl_parents") == pair_proof["changed_dacl_parents"]
            and audit.get("derived_png_bytes_total") == output_bytes
            and audit.get("derived_png_bytes_by_variant") == output_bytes_by_variant,
            "Prepared public aggregate counts differ from the actual ledger")
    return {"status": "passed", **pair_proof, "derived_pngs": 2*len(rows),
            "derived_png_hashes_verified": True, "derived_png_pixels_verified": True,
            "equivalent_half_open_roi_geometry_verified": True,
            "input_files_checked": len(before), "all_original_input_sha_size_mtime_preserved": True,
            "output_png_bytes": output_bytes, "output_png_bytes_by_variant": output_bytes_by_variant,
            "source_metadata_absent_in_derived_pngs": True, "ledger_sha256": sha(ledger_path),
            "data_audit_sha256": sha(root / AUDIT), "individual_paths_or_boxes_published": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-results", type=Path, required=True)
    args = parser.parse_args(argv); tests_path = args.test_results.resolve()
    require(tests_path.is_relative_to((ROOT / "runs").resolve()) and tests_path.is_file(),
            "Test evidence must be an existing local runs file")
    require(not (ROOT / OUTPUT_PATH).exists(), "Preserve existing completion evidence")
    torch.set_num_threads(1)
    protocol = validate_protocol(read(ROOT / PROTOCOL_PATH), ROOT)
    proof_paths = [ROOT / p for p in (PROTOCOL_PATH, PREFLIGHT_PATH, SOURCE_RECORD_PATH, LEDGER_PATH, AUDIT)] + [tests_path]
    proof_before = {path: sha(path) for path in proof_paths}; protocol_sha = proof_before[ROOT / PROTOCOL_PATH]
    preflight = read(ROOT / PREFLIGHT_PATH); snapshot = read(ROOT / SOURCE_RECORD_PATH)
    before = protected_hashes(ROOT)
    sources = verify_frozen_sources(snapshot, protocol, preflight, ROOT, protocol_sha, proof_before[ROOT / PREFLIGHT_PATH])
    preflight_proof = validate_preflight(preflight, protocol, protocol_sha, before)
    tests = validate_test_results(read(tests_path), ROOT)
    data_proof = verify_prepared_data(ROOT, protocol)
    names = [protocol["control"], protocol["treatment"]]
    require(not any((ROOT / "reports" / f"{name}-target-test.json").exists()
                    for name in [protocol["reference"]] + names), "No-test study refuses held-out inference records")
    histories = [read(ROOT / "runs" / name / "history.json") for name in names]
    run_before = {ROOT / "runs" / name / file: sha(ROOT / "runs" / name / file)
                  for name in names for file in ("history.json", "TRAINING.json", "best.pt", "SPLIT.json")}
    sampling = validate_histories(*histories, protocol["requested_epochs"], protocol["draws_per_epoch"])
    manifest = read(ROOT / "data/facility-spatial-training/train.json")
    auxiliary = read(ROOT / "data/facility-auxiliary-training/train.json")
    selected, _ = select_train_replay(manifest, auxiliary)
    initial = torch.load(ROOT / "runs" / protocol["reference"] / "best.pt", map_location="cpu", weights_only=True)
    entries, trainings = [], []
    for variant, name, history in zip(("control", "native"), names, histories):
        run = ROOT / "runs" / name; weights = run / "best.pt"
        training = read(run / "TRAINING.json"); trainings.append(training)
        require(sha(run / "SPLIT.json") == initial["split_sha256"], "Original split bytes changed")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        entry = validate_run_metadata(training, checkpoint, history, variant, protocol, protocol_sha,
                                      initial["state_dict"], sha(weights), sha(run / "SPLIT.json"))
        require(training["paired_label_preservation"]["changed_dacl_detail_rows"] == data_proof["changed_rows"],
                "Training recorded a different set of detail interventions")
        entry["variant"] = variant
        # Both models are the original AUX640 factory; raw and loss maps are 80.
        entry["cpu_reload"] = verify_cpu_reload(weights, selected[0], ROOT, preflight_proof["parameter_count"], "control")
        entries.append(entry)
    for key in ("classes", "split_sha256", "core_spatial_manifest_sha256", "auxiliary_manifest_sha256",
                "expected_sampling", "expected_label_sampling", "photo_positive_weights", "pixel_positive_weights",
                "auxiliary_positive_weights", "additional_validation", "additional_test", "total_loss_formula", "loss",
                "paired_label_preservation"):
        require(key in trainings[0] and trainings[0][key] == trainings[1].get(key),
                "Paired original targets/data/loss/sampling conditions differ")
    require(protected_hashes(ROOT) == before, "CPU proof changed protected experiment/app bytes")
    verify_frozen_sources(snapshot, protocol, preflight, ROOT, protocol_sha, proof_before[ROOT / PREFLIGHT_PATH])
    validate_test_results(read(tests_path), ROOT)
    require(all(sha(path) == digest for path, digest in {**proof_before, **run_before}.items()),
            "Input evidence changed during completion verification")
    result = {"schema": "facility_native_roi_study_verification_v1", "status": "passed",
              "verified_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_sha,
              "preflight_sha256": proof_before[ROOT / PREFLIGHT_PATH],
              "source_before_training_sha256": proof_before[ROOT / SOURCE_RECORD_PATH], **sources,
              "preflight": preflight_proof, "test_results_sha256": proof_before[tests_path], "tests": tests,
              "prepared_data_integrity": data_proof, "actual_sampling_verification": sampling,
              "experiments": entries, "actual_completed_training_epochs": sum(e["actual_epochs"] for e in entries),
              "protected_file_sha256": before, "protected_files_unchanged": True,
              "verification_training_epochs": 0, "source_test_inference_executed": False,
              "app_model_promoted": False, "deployed": False, "accuracy_measured_by_verifier": False,
              "additional_expert_confirmed_labels": 0, "label_changes": 0,
              "new_photo_targets": 0, "new_pixel_targets": 0,
              "scope": "Local recorded TRAIN pixel/source/checkpoint/CPU proof; not authenticated truth or app accuracy"}
    with (ROOT / OUTPUT_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": "passed", "runtime_sources": len(RUNTIME_SOURCES), "tests_run": tests["tests_run"],
                      "actual_completed_training_epochs": result["actual_completed_training_epochs"],
                      "paired_pngs_verified": data_proof["derived_pngs"], "cpu_train_photo_forwards": 2,
                      "test_inference": False, "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
