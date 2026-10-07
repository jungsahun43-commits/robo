"""Verify real sampling/updates, immutable original inputs and CPU contracts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_rc_positive import (NAME, PROTOCOL, MANIFEST, DRAWS, INITIAL, EPOCHS, DRAW_COUNT,
    REPLACEMENTS, NEW_DOMAINS, read, sha, validate_protocol)
from scripts.fetch_rc2119 import write_new
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, ARCH
from safelog_ai.presence_classifier import image_transform
from safelog_ai.frozen_batchnorm import batchnorm_state_sha256

BEFORE = "runs/facility-rc-positive-before.json"
LEDGER = "runs/facility-rc-positive-input-ledger.json"
TESTS = "runs/facility-rc-positive-tests.json"
OUTPUT = "reports/facility-rc-positive-study-verification.json"
GIT = "C:/Program Files/Git/cmd/git.exe"


def require(condition, message):
    if not condition: raise ValueError(message)


def git(arguments, data=None):
    return subprocess.run([GIT, "-c", "safe.directory=" + str(ROOT.parents[1]).replace("\\", "/"), *arguments],
        cwd=ROOT.parents[1], input=data, capture_output=True, check=True).stdout


def verify_git_sources(sources, commit):
    names = list(sources)
    raw = git(["cat-file", "--batch"], "".join(f"{commit}:safelog-cpp/ai-training/{p}\n" for p in names).encode())
    stream = io.BytesIO(raw)
    for path in names:
        header = stream.readline().split()
        require(len(header) == 3 and header[1] == b"blob", "Runtime source absent from declared Git commit")
        content = stream.read(int(header[2])); require(stream.read(1) == b"\n", "Malformed Git blob stream")
        require(hashlib.sha256(content).hexdigest() == sources[path], "Committed runtime bytes differ: " + path)


def snapshots(paths):
    def inspect(name):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink(), "Input path leaves workspace")
        status = path.stat()
        return name, {"sha256": sha(path), "size_bytes": status.st_size, "mtime_ns": status.st_mtime_ns}
    result = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (name, item) in enumerate(pool.map(inspect, sorted(set(paths))), 1):
            result[name] = item
            if i % 10000 == 0: print(f"Protected file SHA/size/mtime verified {i}", flush=True)
    return result


def before_training(protocol):
    require(not (ROOT / BEFORE).exists() and not (ROOT / "runs" / NAME).exists(), "Preserve existing training snapshot")
    previous = read(ROOT / "reports/facility-semantic-study-verification.json")
    old_ledger_path = ROOT / "runs/facility-semantic-preflight/protected-inputs.json"
    require(sha(old_ledger_path) == previous["prepared_data_integrity"]["ledger_sha256"], "Established original input ledger changed")
    old = read(old_ledger_path)["snapshots_before"]
    paths = set(old) | set(previous["protected_file_sha256"]) | set(protocol["source_sha256"]) | set(protocol["input_sha256"])
    paths.update((PROTOCOL, TESTS, "runs/facility-rc-positive-preflight/preflight.json"))
    for row in read(ROOT / MANIFEST)["items"]:
        paths.update(row[k] for k in ("image", "pixel_target", "source_image", "source_mask", "source_annotation"))
    current = snapshots(paths)
    require(all(current[p] == expected for p, expected in old.items()), "An original prepared input SHA/size/mtime changed")
    require(all(current[p]["sha256"] == expected for p, expected in previous["protected_file_sha256"].items()), "An original run/profile/protocol changed")
    commit = git(["rev-parse", "HEAD"]).decode().strip()
    verify_git_sources({**protocol["source_sha256"], PROTOCOL: sha(ROOT / PROTOCOL)}, commit)
    tests = read(ROOT / TESTS)
    require(tests["tests_run"] == 26 and tests["failures"] == tests["errors"] == tests["skipped"] == 0
            and tests["source_sha256"] == protocol["source_sha256"], "Actual frozen-code focused tests required")
    write_new(ROOT / LEDGER, {"schema": "rc_positive_input_ledger_v1", "local_only": True,
        "snapshots_before": current, "original_input_files": len(old), "previous_ledger_sha256": sha(old_ledger_path)})
    write_new(ROOT / BEFORE, {"status": "passed", "protocol_sha256": sha(ROOT / PROTOCOL),
        "source_git_commit": commit, "git_blob_bytes_verified": True, "source_sha256": protocol["source_sha256"],
        "input_ledger_sha256": sha(ROOT / LEDGER), "protected_files": len(current),
        "original_inputs_preserved": len(old), "test_record_sha256": sha(ROOT / TESTS)})
    print(json.dumps({"status": "passed_before_training", "protected_files": len(current), "source_git_commit": commit}), flush=True)


def verify_history(history, data, arrays):
    items = data["items"] + read(ROOT / MANIFEST)["items"]
    core_count = len(data["items"])
    labels = np.asarray([r["targets"] for r in items], np.int64)
    domains = np.asarray([NEW_DOMAINS[r["domain"]] for r in items], np.int64)
    require(len(history) == EPOCHS, "Six actually completed epochs are required")
    for i, row in enumerate(history):
        indices = arrays[i]
        require(row["epoch"] == i + 1 and row["sampled_row_indices_sha256"] == hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(), "Observed draw order differs")
        expected_domains = {d: int((domains[indices] == k).sum()) for d, k in NEW_DOMAINS.items()}
        require(row["sampled_domain_counts"] == expected_domains and expected_domains["rc2119"] == REPLACEMENTS, "Observed source exposure differs")
        for k, label in enumerate(data["classes"]):
            expected = {"positive": int((labels[indices, k] == 1).sum()), "negative": int((labels[indices, k] == 0).sum()), "unknown": int((labels[indices, k] == -1).sum())}
            require(row["sampled_photo_target_counts"][label] == expected, "Known/unknown label counts differ")
        cells = sum(sum(items[int(index)]["positive_cells"]) for index in indices if index >= core_count)
        require(row["asserted_new_positive_cells_sampled"] == cells, "Actual asserted foreground exposure differs")
        updates = row["optimizer_step_diagnostics"]
        require(updates["attempted_batches"] == DRAW_COUNT // 8 == row["teacher_forward_batches"]
                and updates["actual_optimizer_steps"] + updates["amp_skipped_steps"] == updates["attempted_batches"]
                and 0 < updates["actual_optimizer_steps"] <= updates["attempted_batches"], "Actual update/AMP budget inconsistent")
        require(row["teacher_state_unchanged"] and row["bn_buffers_unchanged"], "Teacher or BN changed")
    return {"completed_epochs": EPOCHS, "sampled_draws": EPOCHS * DRAW_COUNT,
            "new_positive_source_draws": EPOCHS * REPLACEMENTS,
            "actual_optimizer_steps": sum(r["optimizer_step_diagnostics"]["actual_optimizer_steps"] for r in history),
            "amp_skipped_steps": sum(r["optimizer_step_diagnostics"]["amp_skipped_steps"] for r in history)}


def after_training(protocol):
    require(not (ROOT / OUTPUT).exists(), "Preserve completed verification")
    before = read(ROOT / BEFORE); ledger = read(ROOT / LEDGER)
    require(before["protocol_sha256"] == sha(ROOT / PROTOCOL) and before["input_ledger_sha256"] == sha(ROOT / LEDGER), "Before-training proof binding changed")
    current = snapshots(ledger["snapshots_before"])
    require(current == ledger["snapshots_before"], "Protected file SHA/size/mtime changed during training/evaluation")
    verify_git_sources({**protocol["source_sha256"], PROTOCOL: sha(ROOT / PROTOCOL)}, before["source_git_commit"])
    from scripts.facility_rc_positive import load_data
    data = load_data(); arrays = np.load(ROOT / DRAWS, allow_pickle=False)
    require(np.array_equal(arrays["control"], data["epoch_draws"]) and np.array_equal(arrays["candidate"], data["candidate_draws"]), "Declared paired arrays changed")
    run = ROOT / "runs" / NAME; training = read(run / "TRAINING.json"); history = read(run / "history.json")
    require(training["status"] == "complete" and training["actual_epochs"] == EPOCHS
            and training["study_protocol_sha256"] == sha(ROOT / PROTOCOL)
            and training["source_git_commit"] == before["source_git_commit"]
            and training["source_sha256"] == protocol["source_sha256"], "Completed actual training metadata differs")
    measured = verify_history(history, data, arrays["candidate"])
    require(training["actual_optimizer_steps"] == measured["actual_optimizer_steps"], "Actual update summary differs")
    require(training["initial_bn_buffer_sha256"] == training["final_bn_buffer_sha256"]
            and training["teacher_state_sha256"] == training["final_teacher_state_sha256"], "Frozen state summary differs")
    weights_sha = sha(run / "best.pt")
    require(weights_sha == training["weights_sha256"], "Selected trained checkpoint changed")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    require(checkpoint["architecture"] == ARCH and checkpoint["study_protocol_sha256"] == sha(ROOT / PROTOCOL)
            and len(checkpoint["state_dict"]) == 324 and all(torch.isfinite(t).all() for t in checkpoint["state_dict"].values()), "Checkpoint contract/state invalid")
    model = AuxiliaryClassifier(7, pretrained=False); model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
    require(batchnorm_state_sha256(model) == training["initial_bn_buffer_sha256"], "Selected checkpoint BN differs")
    with Image.open(ROOT / data["items"][0]["image"]) as image: inputs = image_transform(640)(image.convert("RGB"))[None]
    with torch.no_grad(): outputs = model.forward_training(inputs); public = model(inputs)
    require([list(x.shape) for x in outputs] == [[1, 7], [1, 7, 80, 80], [1, 19]]
            and all(torch.isfinite(x).all() for x in outputs) and torch.equal(public, outputs[0]), "Actual CPU/public output contract failed")
    tests = read(ROOT / TESTS)
    result = {"schema": "facility_rc_positive_study_verification_v1", "status": "passed", "protocol_sha256": sha(ROOT / PROTOCOL),
        "source_git_commit": before["source_git_commit"], "runtime_source_count": len(protocol["source_sha256"]),
        "source_sha256": protocol["source_sha256"], "git_blob_bytes_verified": True,
        "protected_file_count": len(current), "original_input_files_preserved": ledger["original_input_files"],
        "all_original_and_new_input_sha_size_mtime_preserved": True, "individual_input_paths_published": False,
        "input_ledger_sha256": sha(ROOT / LEDGER), "before_training_sha256": sha(ROOT / BEFORE),
        "test_record_sha256": sha(ROOT / TESTS), "tests": tests, "actual_sampling": measured,
        "actual_completed_training_epochs": EPOCHS, "reused_control_epochs_not_recounted": 6,
        "weights_sha256": weights_sha, "cpu_strict_324_state_reload": True, "all_outputs_finite": True,
        "public_output_equals_training_photo_output": True, "frozen_teacher_and_bn_preserved": True,
        "new_negative_photo_targets": 0, "new_background_negative_pixels": 0, "new_auxiliary_labels": 0,
        "source_test_inference_executed": False, "app_model_promoted": False, "verification_training_epochs": 0}
    write_new(ROOT / OUTPUT, result)
    print(json.dumps({k: result[k] for k in ("status", "actual_completed_training_epochs", "weights_sha256", "protected_file_count")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--before", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(4)
    protocol = validate_protocol(read(ROOT / PROTOCOL))
    (before_training if args.before else after_training)(protocol)


if __name__ == "__main__": main()
