"""One fixed student BatchNorm-statistics intervention with a completed control."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, ARCH
from safelog_ai.frozen_batchnorm import batchnorm_inventory, batchnorm_state_sha256
from scripts import facility_retention_strength_study as previous
from scripts import facility_retention_study as original
from scripts.facility_subtype_study import require, local_path, write, FixedEpochSampler
from scripts.train_facility_target import read, sha

PROTOCOL = "reports/facility-batchnorm-study-protocol.json"
REFERENCE = previous.REFERENCE
CONTROL = previous.RUN_NAME
RUN_NAME = "facility-presence-target-batchnorm-frozen"
VARIANT = "frozen"
VARIANTS = (VARIANT,)
NAMES = {VARIANT: RUN_NAME}
INITIAL_SHA = previous.INITIAL_SHA
TEACHER_STATE_SHA = previous.TEACHER_STATE_SHA
INITIAL_BN_BUFFERS_SHA = "3def7c0bc305c6dfcc546e682fe364a75a385047cfdc2577076a1f7676430975"
DRAW_ARCHIVE = previous.DRAW_ARCHIVE
PLAN = previous.PLAN
PRIOR_PROTOCOL = previous.PROTOCOL
PRIOR_VERIFICATION = "reports/facility-retention-strength-study-verification.json"
PRIOR_COMPARISON = "reports/facility-retention-strength-study-comparison.json"
PRIOR_PREFLIGHT = "runs/facility-retention-strength-preflight.json"
PRIOR_INPUT_LEDGER = "runs/facility-retention-strength-preflight/protected-inputs.json"
PRIOR_PROTOCOL_SHA = "b9a5dfce931d1c871ed03f00371155fffd32ea7147080bc6f4429b8e68a1f22c"
PRIOR_VERIFICATION_SHA = "27d3f96098dd2e8685b4a09e211b84ce9d0832208e166fc4a4b0b3a017600c2a"
PRIOR_COMPARISON_SHA = "e121e74c4f5095de0bfad0d034049021348c0504ff1379547be90c1a18eb17be"
CONTROL_WEIGHTS_SHA = "07739923273a5733f43696b3c2a09d89503a760d3af9ca4a173bb351c452e712"
CONTROL_ARTIFACTS = previous.CONTROL_ARTIFACTS
NEW_SOURCES = (
    "safelog_ai/frozen_batchnorm.py", "scripts/facility_batchnorm_study.py", "scripts/train_facility_batchnorm.py",
    "scripts/preflight_facility_batchnorm.py", "scripts/verify_facility_batchnorm.py",
    "scripts/test_facility_batchnorm.py", "scripts/run_facility_batchnorm.py",
    "scripts/report_facility_batchnorm.py", "scripts/plot_facility_batchnorm.py",
    "tests/test_facility_batchnorm_study.py", "tests/test_facility_batchnorm_verification.py",
    "tests/test_facility_batchnorm_report.py",
)
SOURCE_FILES = previous.SOURCE_FILES + NEW_SOURCES
MIN_TEST_COUNTS = {"tests/test_facility_batchnorm_study.py": 14,
                   "tests/test_facility_batchnorm_verification.py": 8,
                   "tests/test_facility_batchnorm_report.py": 10}
RETENTION_GATE = deepcopy(previous.RETENTION_GATE)
BATCHNORM_POLICY = {
    "version": "student_tracked_batchnorm_eval_affine_trainable_v1", "layer_count": 47,
    "channel_count": 12328, "buffer_tensor_count": 141, "affine_parameter_tensor_count": 94,
    "student_only": True, "apply_after_each_model_train": True, "all_student_batchnorm_eval": True,
    "training_forward_uses_existing_running_statistics": True,
    "training_forward_uses_current_batch_statistics": False,
    "tracked_running_mean_variance_counter_bitwise_unchanged": True,
    "affine_parameters_trainable": True, "backbone_parameters_trainable": True, "head_parameters_trainable": True,
    "eps_and_momentum_unchanged": True, "teacher_policy_unchanged": True,
    "buffers": ["running_mean", "running_var", "num_batches_tracked"],
    "student_additional_parameters": 0, "student_architecture_changed": False,
    "scope": "Changes student TRAIN normalization and running-buffer updates; preserves learnable affine and all other losses",
}
RECIPE = deepcopy(previous.RECIPE)
RECIPE.update(version="known_other_five_teacher_retention_batchnorm_v1",
    weight_by_variant={VARIANT: 4.0}, sampler_by_variant={VARIANT: "original_control"},
    reused_control_run=CONTROL, reused_control_weight=4.0)
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def fixed_values():
    value = previous.fixed_values()
    value.pop("previous_treatment", None); value.pop("previous_treatment_weights_sha256", None)
    value.update(schema="facility_batchnorm_study_protocol_v1", control=CONTROL, treatment=RUN_NAME,
        architecture_by_variant={VARIANT: ARCH}, imgsz_by_variant={VARIANT: 640},
        distillation_recipe=deepcopy(RECIPE), distillation_weight_by_variant={VARIANT: 4.0},
        paired_draws_keys={VARIANT: "control"}, batchnorm_policy=deepcopy(BATCHNORM_POLICY),
        initial_batchnorm_buffers_sha256=INITIAL_BN_BUFFERS_SHA,
        previous_protocol_sha256=PRIOR_PROTOCOL_SHA, previous_verification_sha256=PRIOR_VERIFICATION_SHA,
        previous_comparison_sha256=PRIOR_COMPARISON_SHA, reused_control_weights_sha256=CONTROL_WEIGHTS_SHA,
        control_comparison_scope="Reuse the completed normal-BatchNorm weight-four six-epoch control; only student BatchNorm policy changes")
    return value


def completed_records(root=ROOT):
    root = Path(root).resolve()
    training = read(root / f"runs/{CONTROL}/TRAINING.json")
    require(training.get("status") == "complete" and type(training.get("actual_epochs")) is int
            and training["actual_epochs"] == 6 and training.get("model_variant") == "strong"
            and type(training.get("distillation_weight")) is float and training["distillation_weight"] == 4.0
            and training.get("study_protocol_sha256") == PRIOR_PROTOCOL_SHA
            and training.get("weights_sha256") == CONTROL_WEIGHTS_SHA,
            "Only the verified completed normal-BatchNorm weight-four run may be reused")
    require(sha(local_path(root, f"runs/{CONTROL}/best.pt")) == CONTROL_WEIGHTS_SHA, "Completed weight-four checkpoint changed")
    return {f"runs/{CONTROL}/{name}": sha(local_path(root, f"runs/{CONTROL}/{name}")) for name in CONTROL_ARTIFACTS}


def expected_protected_hashes(root=ROOT):
    root = Path(root).resolve()
    for name, expected in ((PRIOR_PROTOCOL, PRIOR_PROTOCOL_SHA), (PRIOR_VERIFICATION, PRIOR_VERIFICATION_SHA),
                           (PRIOR_COMPARISON, PRIOR_COMPARISON_SHA)):
        require(sha(local_path(root, name)) == expected, f"Previous frozen strength evidence changed: {name}")
    proof = read(root / PRIOR_VERIFICATION); protocol = read(root / PRIOR_PROTOCOL)
    comparison = read(root / PRIOR_COMPARISON)
    require(proof.get("status") == "passed" and proof.get("runtime_source_count") == 74
            and proof.get("actual_completed_training_epochs") == 6
            and proof.get("working_runtime_sources_unchanged") is True
            and proof.get("both_teachers_unchanged_eval_frozen_no_grad") is True
            and proof.get("control_training_repeated") is False,
            "Completed source74 strength proof is required")
    require(set(proof["source_sha256"]) == set(previous.SOURCE_FILES)
            and proof["source_sha256"] == protocol["source_sha256"]
            and proof["protocol_sha256"] == PRIOR_PROTOCOL_SHA, "Previous source74/protocol binding differs")
    require(comparison.get("schema") == "facility_retention_strength_study_comparison_v1"
            and comparison.get("protocol_sha256") == PRIOR_PROTOCOL_SHA
            and comparison.get("technical_verification_sha256") == PRIOR_VERIFICATION_SHA
            and comparison.get("technical_verification", {}).get("status") == "passed",
            "The verified previous strength source-VAL comparison is required")
    protected = dict(proof["protected_file_sha256"]); protected.update(proof["source_sha256"])
    protected.update(completed_records(root))
    protected.update({PRIOR_PROTOCOL: PRIOR_PROTOCOL_SHA, PRIOR_VERIFICATION: PRIOR_VERIFICATION_SHA,
        PRIOR_COMPARISON: PRIOR_COMPARISON_SHA, PRIOR_PREFLIGHT: proof["preflight_sha256"],
        PRIOR_INPUT_LEDGER: proof["prepared_data_integrity"]["ledger_sha256"]})
    for name, expected in protected.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid protected SHA256")
        require(sha(local_path(root, name)) == expected, f"Protected previous/original file changed: {name}")
    return dict(sorted(protected.items()))


def protected_hashes(root=ROOT):
    return expected_protected_hashes(root)


def validate_protocol(protocol, root=ROOT, args=None):
    root = Path(root).resolve()
    require(isinstance(protocol, dict), "Expected the fixed BatchNorm protocol")
    for key, expected in fixed_values().items():
        require(json.dumps(protocol.get(key), sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False), f"Declared BatchNorm condition differs: {key}")
    sources = protocol.get("source_sha256", {})
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "Runtime source inventory differs")
    for name, expected in sources.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid runtime SHA256")
        require(sha(local_path(root, name)) == expected, f"Frozen runtime source changed: {name}")
    require(protocol.get("protected_file_sha256") == expected_protected_hashes(root), "Protected previous/original inventory differs")
    require(protocol.get("reused_control_artifact_sha256") == completed_records(root), "Reused completed control artifacts changed")
    if args is not None:
        require(args.variant == VARIANT and args.name == RUN_NAME, "Only the single frozen-BatchNorm candidate can train")
        for argument, field in (("seed", "seed"), ("epochs", "requested_epochs"), ("patience", "patience"),
             ("batch", "batch_size"), ("draws_per_epoch", "draws_per_epoch"), ("backbone_lr", "backbone_lr"),
             ("head_lr", "head_lr"), ("auxiliary_weight", "auxiliary_weight")):
            require(getattr(args, argument, None) == protocol[field], f"CLI condition differs: {argument}")
        require(Path(args.initial).resolve() == (root / f"runs/{REFERENCE}/best.pt").resolve(),
                "BatchNorm candidate must freshly initialize from the original 0773")
        require(Path(args.auxiliary_manifest).resolve() == (root / "data/facility-auxiliary-training/train.json").resolve(),
                "Use the original auxiliary labels")
        return {"imgsz": 640, "architecture": ARCH}
    return deepcopy(protocol)


def prepare_data(root, protocol, variant=VARIANT, auxiliary_path=None):
    require(variant == VARIANT, "Only the frozen-BatchNorm candidate has fresh training inputs")
    data = original.prepare_data(root, protocol, "control", auxiliary_path)
    data["control_reuse_proof"] = {"historical_control_reused": True, "reused_control_run": CONTROL,
        "reused_control_weight": 4.0, "reused_control_retrained": False, "all_original_control_draws_used": True,
        "new_training_epochs": 6}
    return data


def declare():
    torch.set_num_threads(4)
    path = ROOT / PROTOCOL
    require(not path.exists(), "Preserve the predeclared protocol")
    require(not (ROOT / "runs" / RUN_NAME).exists(), "Cannot declare after the BatchNorm candidate starts")
    protocol = fixed_values()
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(local_path(ROOT, name)) for name in SOURCE_FILES},
        protected_file_sha256=expected_protected_hashes(ROOT), reused_control_artifact_sha256=completed_records(ROOT))
    validate_protocol(protocol, ROOT); prepare_data(ROOT, protocol)
    checkpoint = torch.load(ROOT / f"runs/{REFERENCE}/best.pt", map_location="cpu", weights_only=True)
    model = AuxiliaryClassifier(7, pretrained=False); model.load_state_dict(checkpoint["state_dict"], strict=True)
    inventory = batchnorm_inventory(model)
    for key in ("layer_count", "channel_count", "buffer_tensor_count", "affine_parameter_tensor_count"):
        require(inventory[key] == BATCHNORM_POLICY[key], "Original BatchNorm architecture inventory changed")
    require(batchnorm_state_sha256(model) == INITIAL_BN_BUFFERS_SHA, "Original BatchNorm buffer state changed")
    write(path, protocol)
    print(json.dumps({"status": "declared", "runtime_sources": len(SOURCE_FILES), "new_training_epochs": 6,
        "reused_control_training_epochs": 6, "protocol_sha256": sha(path), "new_labels": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", required=True)
    parser.parse_args(); declare()
