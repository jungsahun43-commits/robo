"""CPU completion proof for the predeclared resolution pair; no held-out inference.

Hashes establish equality to the locally recorded experiment, not authenticated
publisher provenance, independent expert truth, or application accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.presence_classifier import PresenceClassifier, MEAN, STD
from safelog_ai.resolution_classifier import RECIPE
from scripts.facility_resolution_study import SOURCE_FILES, validate_protocol
from scripts.preflight_facility_detail import select_train_replay
from scripts.report_facility_resolution import validate_histories

RUNTIME_SOURCES = SOURCE_FILES + (
    "scripts/preflight_facility_resolution.py", "scripts/preflight_facility_detail.py",
    "scripts/report_facility_detail.py", "scripts/report_facility_discrimination.py",
    "scripts/report_facility_small_region.py", "scripts/evaluate_facility_target.py",
    "scripts/analyze_facility_target.py",
)
TEST_SOURCES = (
    "tests/test_facility_resolution_classifier.py", "tests/test_facility_resolution_study.py",
    "tests/test_facility_resolution_sources.py", "tests/test_facility_resolution_report.py",
    "tests/test_facility_resolution_verification.py",
)
VERIFIER_SOURCE = "scripts/verify_facility_resolution.py"
PROTOCOL_PATH = "reports/facility-resolution-study-protocol.json"
PREFLIGHT_PATH = "runs/facility-resolution-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-resolution-source-before-training.json"
OUTPUT_PATH = "reports/facility-resolution-study-verification.json"
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON keys in proof")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON number in proof")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def local_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative
            and ":" not in relative and not PurePosixPath(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Expected a relative internal proof path")
    root = Path(root).resolve(); path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or escaped proof file")
    return path


def hash_map(values, keys):
    require(isinstance(values, dict) and set(values) == set(keys)
            and all(isinstance(v, str) and _SHA.fullmatch(v) for v in values.values()),
            "Proof source hash inventory differs")


def git_blob(root, commit, relative):
    # Binary stdout is essential: shell text decoding changes CRLF blob hashes.
    repo = Path(root).resolve().parents[1]
    result = subprocess.run(["git", "-c", f"safe.directory={repo}", "show",
                             f"{commit}:safelog-cpp/ai-training/{relative}"],
                            cwd=repo, capture_output=True, check=False)
    require(result.returncode == 0, "Recorded Git source blob is unavailable")
    return result.stdout


def verify_frozen_sources(snapshot, protocol, preflight, root, protocol_sha,
                          preflight_sha, blob_reader=None):
    """Compare dynamic snapshot commit, actual Git bytes and current 18 files."""
    require(snapshot.get("schema") == "facility_resolution_source_before_training_v1"
            and snapshot.get("declared_before_training") is True
            and snapshot.get("git_blob_bytes_match_protocol_sources") is True
            and type(snapshot.get("new_completed_training_epochs_at_snapshot")) is int
            and snapshot["new_completed_training_epochs_at_snapshot"] == 0,
            "A before-training source snapshot is required")
    commit = snapshot.get("source_git_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[a-f0-9]{40}", commit),
            "Recorded Git commit must be a complete object ID")
    for document in (snapshot, protocol, preflight):
        hash_map(document.get("source_sha256"), RUNTIME_SOURCES)
    sources = protocol["source_sha256"]
    require(snapshot["source_sha256"] == preflight["source_sha256"] == sources,
            "Recorded source inventories differ")
    require(snapshot.get("protocol_sha256") == preflight.get("protocol_sha256") == protocol_sha
            and snapshot.get("preflight_sha256") == preflight_sha
            and snapshot.get("initial_weights_sha256") == protocol["initial_weights_sha256"],
            "Source snapshot protocol/preflight/initializer binding differs")
    reader = blob_reader or (lambda c, p: git_blob(root, c, p))
    for relative, expected in sources.items():
        require(sha(local_path(root, relative)) == expected,
                "Current runtime source differs from before-training bytes")
        blob = reader(commit, relative)
        require(isinstance(blob, bytes) and hashlib.sha256(blob).hexdigest() == expected,
                "Committed runtime bytes differ from recorded source")
    return {"source_git_commit": commit, "runtime_source_count": len(sources),
            "source_sha256": dict(sources), "git_blob_bytes_verified": True,
            "working_runtime_sources_unchanged": True}


def protected_hashes(root, protocol):
    paths = {"initial": f"runs/{protocol['reference']}/best.pt",
             "core": "data/facility-spatial-training/train.json",
             "auxiliary": "data/facility-auxiliary-training/train.json",
             "app_profile": "reports/facility-inference-profile.json", "protocol": PROTOCOL_PATH}
    return {key: sha(local_path(root, path)) for key, path in paths.items()}


def validate_preflight(preflight, protocol, protocol_sha, current_protected):
    expected = {"schema": "facility_resolution_preflight_v1", "status": "passed",
                "training_epochs": 0, "accuracy_measured": False,
                "actual_train_source_cases": 8, "bounded_replay_batches_per_variant": 3,
                "constructor_rng_equal": True, "640_initial_outputs_identical": True,
                "mask_label_known_index_replay_equal": True,
                "resized_image_tensors_expected_to_differ": True,
                "protected_files_unchanged": True, "protocol_sha256": protocol_sha}
    for key, value in expected.items():
        require(type(preflight.get(key)) is type(value) and preflight[key] == value,
                "Required actual preflight proof differs")
    hash_map(current_protected, ("initial", "core", "auxiliary", "app_profile", "protocol"))
    require(preflight.get("protected_file_sha256") == current_protected,
            "Protected initial/data/app/protocol bytes changed")
    require(current_protected["initial"] == protocol["initial_weights_sha256"]
            and current_protected["core"] == protocol["core_spatial_manifest_sha256"]
            and current_protected["auxiliary"] == protocol["auxiliary_manifest_sha256"]
            and current_protected["protocol"] == protocol_sha, "Protected protocol hashes differ")
    require(set(preflight.get("variants", {})) == {"control", "highres"},
            "Both actual preflight variants are required")
    counts = []
    for variant in ("control", "highres"):
        row = preflight["variants"][variant]; size = protocol["imgsz_by_variant"][variant]
        fixed = {"imgsz": size, "raw_map_shape": [8, 7, 80 if size == 640 else 120,
                                                        80 if size == 640 else 120],
                 "loss_and_pool_grid": [80, 80], "photo_shape": [8, 7], "auxiliary_shape": [8, 19],
                 "actual_optimizer_steps": 1, "validation_like_train_repeated_batch_size": 16,
                 "validation_size_forward_finite": True, "all_gradients_finite": True,
                 "disposable_serialization_and_factory_cpu_reload_verified": True}
        for key, value in fixed.items():
            require(type(row.get(key)) is type(value) and row[key] == value,
                    "Actual preflight shape/update/reload proof differs")
        require(type(row.get("parameter_count")) is int and row["parameter_count"] > 0
                and type(row.get("peak_cuda_allocated_bytes")) is int and row["peak_cuda_allocated_bytes"] > 0
                and type(row.get("loss")) in (int, float) and math.isfinite(row["loss"]) and row["loss"] > 0,
                "Measured preflight resource/loss proof missing")
        require(isinstance(row.get("disposable_serialized_checkpoint_sha256"), str)
                and _SHA.fullmatch(row["disposable_serialized_checkpoint_sha256"]),
                "Disposable checkpoint proof missing")
        counts.append(row["parameter_count"])
    require(counts[0] == counts[1], "Resolution comparison introduced parameters")
    replay = preflight.get("replay", {})
    require(set(replay) == {"control", "highres"}
            and all(isinstance(replay[v], list) and len(replay[v]) == 3 for v in replay),
            "Three actual replay batches per variant are required")
    for left, right in zip(replay["control"], replay["highres"]):
        for row in (left, right):
            hash_map(row, ("non_image_batch_sha256", "image_tensor_sha256", "indices_sha256"))
        require(all(left[k] == right[k] for k in ("non_image_batch_sha256", "indices_sha256"))
                and left["image_tensor_sha256"] != right["image_tensor_sha256"],
                "Mask/target/index replay or resolution input proof differs")
    return {"actual_train_source_cases": 8, "replay_batches_per_variant": 3,
            "parameter_count": counts[0], "passed": True}


def validate_test_results(result, root):
    require(result.get("schema") == "facility_resolution_test_results_v1"
            and result.get("status") == "passed" and result.get("before_after_sources_equal") is True,
            "Actual passed test-result record is required")
    require(type(result.get("tests_run")) is int and result["tests_run"] >= 37
            and all(type(result.get(k)) is int and result[k] == 0 for k in ("failures", "errors", "skipped")),
            "At least 37 executed tests with no failures/errors/skips are required")
    hash_map(result.get("test_source_sha256"), TEST_SOURCES)
    hash_map(result.get("source_sha256"), (VERIFIER_SOURCE,))
    for relative, expected in {**result["test_source_sha256"], **result["source_sha256"]}.items():
        require(sha(local_path(root, relative)) == expected, "Tested verifier/test bytes changed")
    return {key: result[key] for key in ("tests_run", "failures", "errors", "skipped",
                                          "test_source_sha256", "source_sha256")}


def validate_run_metadata(training, checkpoint, history, variant, protocol, protocol_sha,
                          initial_state, weights_sha, split_sha):
    expected = {"status": "complete", "actual_epochs": protocol["requested_epochs"],
                "architecture": protocol["architecture_by_variant"][variant],
                "imgsz": protocol["imgsz_by_variant"][variant], "model_variant": variant,
                "classes": protocol["classes"], "auxiliary_classes": protocol["auxiliary_classes"],
                "auxiliary_source_class_count": 19, "validation_batch_size": 16,
                "initial_weights_sha256": protocol["initial_weights_sha256"],
                "core_spatial_manifest_sha256": protocol["core_spatial_manifest_sha256"],
                "spatial_manifest_sha256": protocol["core_spatial_manifest_sha256"],
                "auxiliary_manifest_sha256": protocol["auxiliary_manifest_sha256"],
                "source_sha256": protocol["source_sha256"], "study_protocol_sha256": protocol_sha,
                "resolution_architecture": RECIPE, "photo_pooling_grid": [80, 80],
                "source_pixel_target_grid": [80, 80],
                "raw_spatial_grid": [80, 80] if variant == "control" else [120, 120],
                "new_photo_targets": 0, "new_pixel_targets": 0, "app_model_promoted": False,
                "target_ranking_weight": 0., "weights_sha256": weights_sha, "split_sha256": split_sha}
    for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
                "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions"):
        expected[key] = protocol[key]
    for key, value in expected.items():
        require(type(training.get(key)) is type(value) and training[key] == value,
                "Actual training condition differs from declared recipe")
    transfer = {"shared_state_tensors_equal": True, "shared_state_tensor_count": len(initial_state),
                "new_state_tensor_count": 0, "new_output_projection_zero": None,
                "strict_state_load": True, "additional_parameters": 0}
    require(training.get("initial_state_transfer") == transfer, "Initial shared-state transfer differs")
    require(training.get("target_ranking") == {"weight": 0.,
                "classes": ["concrete_crack", "concrete_spalling"], "sampling_changed": False,
                "new_labels_asserted": 0, "public_outputs_changed": False}, "Ranking/new-label state changed")
    for key in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps"):
        actual = sum(row["optimizer_step_diagnostics"][key] for row in history)
        require(type(training.get("optimizer_step_diagnostics", {}).get(key)) is int
                and training["optimizer_step_diagnostics"][key] == actual, "Optimizer summary differs from actual epochs")
    require(training["optimizer_step_diagnostics"]["actual_optimizer_steps"] > 0,
            "No actual training optimizer updates")
    require(type(training.get("elapsed_training_minutes")) in (int, float)
            and math.isfinite(training["elapsed_training_minutes"]) and training["elapsed_training_minutes"] > 0
            and type(training.get("peak_cuda_allocated_bytes")) is int and training["peak_cuda_allocated_bytes"] > 0,
            "Completed training resource measurements missing")
    for key, value in {"architecture": expected["architecture"], "classes": protocol["classes"],
                       "auxiliary_classes": list(AUX_CLASSES), "imgsz": expected["imgsz"],
                       "mean": MEAN, "std": STD, "selection_split": "val", "split_sha256": split_sha}.items():
        require(type(checkpoint.get(key)) is type(value) and checkpoint[key] == value,
                "Completed checkpoint metadata differs")
    epoch = checkpoint.get("epoch")
    require(type(epoch) is int and 1 <= epoch <= len(history), "Selected checkpoint epoch is invalid")
    error = checkpoint.get("worst_target_error")
    require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1
            and error == history[epoch - 1]["worst_target_error"] == training.get("best_worst_target_error"),
            "Selected checkpoint measurement differs from training history")
    state = checkpoint.get("state_dict")
    require(isinstance(state, dict) and set(state) == set(initial_state), "Checkpoint introduced or omitted state tensors")
    for key, tensor in state.items():
        reference = initial_state[key]
        require(isinstance(tensor, torch.Tensor) and tensor.shape == reference.shape
                and tensor.dtype == reference.dtype and bool(torch.isfinite(tensor).all()),
                "Completed checkpoint state shape/dtype/finite contract differs")
    return {"imgsz": expected["imgsz"], "architecture": expected["architecture"],
            "actual_epochs": training["actual_epochs"], "selected_epoch": epoch,
            "weights_sha256": weights_sha, "state_tensor_count": len(state),
            "new_state_tensor_count": 0, "strict_state_inventory_verified": True,
            "optimizer_step_diagnostics": dict(training["optimizer_step_diagnostics"])}


def verify_cpu_reload(weights, item, root, expected_parameters, variant):
    """One deterministic original TRAIN photo, CPU inference only, no updates."""
    provider = PresenceClassifier(weights, device="cpu")
    require(provider.device.type == "cpu", "Completion proof must use CPU")
    with Image.open(local_path(root, item["image"])) as original:
        batch = provider.transform(original.convert("RGB")).unsqueeze(0)
    raw = []
    hook = provider.model.segmentation_head.register_forward_hook(
        lambda module, args, output: raw.append(list(output.shape)))
    try:
        with torch.inference_mode():
            photo, maps, auxiliary = provider.model.forward_training(batch)
            public = provider.model(batch)
    finally:
        hook.remove()
    size = 80 if variant == "control" else 120
    require(raw == [[1, 7, size, size]] * 2, "Reloaded native map shape differs")
    require(list(photo.shape) == [1, 7] and list(maps.shape) == [1, 7, 80, 80]
            and list(auxiliary.shape) == [1, 19] and list(public.shape) == [1, 7]
            and all(bool(torch.isfinite(v).all()) for v in (photo, maps, auxiliary, public))
            and torch.equal(photo, public), "Reloaded CPU public/training output contract differs")
    require(sum(p.numel() for p in provider.model.parameters()) == expected_parameters,
            "Reloaded parameter count differs from preflight")
    return {"device": "cpu", "strict_factory_reload_verified": True, "train_photos_forwarded": 1,
            "input_shape": list(batch.shape), "native_map_shape": raw[0],
            "photo_shape": list(photo.shape), "loss_and_pool_map_shape": list(maps.shape),
            "auxiliary_shape": list(auxiliary.shape), "public_shape": list(public.shape),
            "all_outputs_finite": True, "public_output_equals_training_photo_output": True,
            "parameter_count": expected_parameters}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-results", type=Path, required=True)
    args = parser.parse_args(argv)
    tests_path = args.test_results.resolve()
    require(tests_path.is_relative_to((ROOT / "runs").resolve()) and tests_path.is_file(),
            "Test evidence must be an existing local runs file")
    require(not (ROOT / OUTPUT_PATH).exists(), "Preserve existing completion evidence")
    torch.set_num_threads(1)
    protocol = validate_protocol(read(ROOT / PROTOCOL_PATH), ROOT)
    require(protocol.get("resolution_architecture") == RECIPE, "Declared pooling/projection recipe differs")
    proof_files = (ROOT / PROTOCOL_PATH, ROOT / PREFLIGHT_PATH, ROOT / SOURCE_RECORD_PATH, tests_path)
    proof_before = {path: sha(path) for path in proof_files}
    protocol_sha = proof_before[ROOT / PROTOCOL_PATH]; preflight = read(ROOT / PREFLIGHT_PATH)
    snapshot = read(ROOT / SOURCE_RECORD_PATH); before = protected_hashes(ROOT, protocol)
    sources = verify_frozen_sources(snapshot, protocol, preflight, ROOT, protocol_sha,
                                    proof_before[ROOT / PREFLIGHT_PATH])
    preflight_proof = validate_preflight(preflight, protocol, protocol_sha, before)
    tests = validate_test_results(read(tests_path), ROOT)
    names = [protocol["control"], protocol["treatment"]]
    require(not any((ROOT / "reports" / f"{name}-target-test.json").exists()
                    for name in [protocol["reference"]] + names), "No-test study refuses held-out inference records")
    histories = [read(ROOT / "runs" / name / "history.json") for name in names]
    run_files = [ROOT / "runs" / name / file for name in names
                 for file in ("history.json", "TRAINING.json", "best.pt", "SPLIT.json")]
    run_before = {path: sha(path) for path in run_files}
    sampling = validate_histories(*histories, protocol["requested_epochs"], protocol["draws_per_epoch"])
    manifest = read(ROOT / "data/facility-spatial-training/train.json")
    auxiliary = read(ROOT / "data/facility-auxiliary-training/train.json")
    selected, _ = select_train_replay(manifest, auxiliary)
    initial = torch.load(ROOT / "runs" / protocol["reference"] / "best.pt", map_location="cpu", weights_only=True)
    entries = []; trainings = []
    for variant, name, history in zip(("control", "highres"), names, histories):
        run = ROOT / "runs" / name; weights = run / "best.pt"
        training = read(run / "TRAINING.json"); trainings.append(training)
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        entry = validate_run_metadata(training, checkpoint, history, variant, protocol, protocol_sha,
                                      initial["state_dict"], sha(weights), sha(run / "SPLIT.json"))
        entry["variant"] = variant
        entry["cpu_reload"] = verify_cpu_reload(weights, selected[0], ROOT, preflight_proof["parameter_count"], variant)
        entries.append(entry)
    for key in ("classes", "split_sha256", "expected_sampling", "expected_label_sampling",
                "photo_positive_weights", "pixel_positive_weights", "auxiliary_positive_weights",
                "additional_validation", "additional_test", "total_loss_formula", "loss"):
        require(key in trainings[0] and trainings[0][key] == trainings[1].get(key),
                "Paired original targets/data/loss conditions differ")
    require(protected_hashes(ROOT, protocol) == before, "CPU proof changed protected experiment/app bytes")
    verify_frozen_sources(snapshot, protocol, preflight, ROOT, protocol_sha,
                          proof_before[ROOT / PREFLIGHT_PATH])
    validate_test_results(read(tests_path), ROOT)
    require(all(sha(path) == digest for path, digest in {**proof_before, **run_before}.items()),
            "Input evidence changed during CPU completion proof")
    result = {"schema": "facility_resolution_study_verification_v1", "status": "passed",
              "verified_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_sha,
              "preflight_sha256": proof_before[ROOT / PREFLIGHT_PATH],
              "source_before_training_sha256": proof_before[ROOT / SOURCE_RECORD_PATH],
              **sources, "preflight": preflight_proof, "test_results_sha256": proof_before[tests_path],
              "tests": tests, "actual_sampling_verification": sampling, "experiments": entries,
              "actual_completed_training_epochs": sum(e["actual_epochs"] for e in entries),
              "protected_file_sha256": before, "protected_files_unchanged": True,
              "verification_training_epochs": 0, "source_test_inference_executed": False,
              "app_model_promoted": False, "deployed": False, "accuracy_measured_by_verifier": False,
              "additional_expert_confirmed_labels": 0, "label_changes": 0,
              "new_photo_targets": 0, "new_pixel_targets": 0,
              "scope": "Local source/metadata/state/CPU contract proof; not independent truth or app accuracy"}
    (ROOT / OUTPUT_PATH).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "runtime_sources": len(RUNTIME_SOURCES),
                      "tests_run": tests["tests_run"], "actual_completed_training_epochs": result["actual_completed_training_epochs"],
                      "cpu_train_photo_forwards": 2, "test_inference": False, "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
