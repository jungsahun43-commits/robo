"""One lower initial non-backbone LR candidate with the completed BN control."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import facility_batchnorm_study as previous
from scripts.facility_subtype_study import require, local_path, write, FixedEpochSampler
from scripts.train_facility_target import read, sha

PROTOCOL = "reports/facility-head-lr-study-protocol.json"
REFERENCE = previous.REFERENCE
CONTROL = previous.RUN_NAME
RUN_NAME = "facility-presence-target-head-lr-low"
VARIANT = "low_lr"
VARIANTS = (VARIANT,)
NAMES = {VARIANT: RUN_NAME}
INITIAL_SHA = previous.INITIAL_SHA
TEACHER_STATE_SHA = previous.TEACHER_STATE_SHA
INITIAL_BN_BUFFERS_SHA = previous.INITIAL_BN_BUFFERS_SHA
DRAW_ARCHIVE = previous.DRAW_ARCHIVE
PLAN = previous.PLAN
PRIOR_PROTOCOL = previous.PROTOCOL
PRIOR_VERIFICATION = "reports/facility-batchnorm-study-verification.json"
PRIOR_COMPARISON = "reports/facility-batchnorm-study-comparison.json"
PRIOR_PREFLIGHT = "runs/facility-batchnorm-preflight.json"
PRIOR_INPUT_LEDGER = "runs/facility-batchnorm-preflight/protected-inputs.json"
PRIOR_PROTOCOL_SHA = "449a41f805b9c33810f83561fbabb20031fe9261ae9abc547f9a8e5f0d9814ff"
PRIOR_VERIFICATION_SHA = "c9dd057fd36e12e7d6c4615454f5b67776fac4cf18ea687d0692bd8cbd87de84"
PRIOR_COMPARISON_SHA = "f402f1d7b36604aaba4f2c84d050073592e8245e7ea13481c2d4fdce7ee5ebcf"
CONTROL_WEIGHTS_SHA = "1d61498dcaa89c5e6009ca693aaec7113a8d3c24061c7b6426fd590227f6ea94"
CONTROL_ARTIFACTS = previous.CONTROL_ARTIFACTS
NEW_SOURCES = (
    "scripts/facility_head_lr_study.py", "scripts/train_facility_head_lr.py",
    "scripts/preflight_facility_head_lr.py", "scripts/verify_facility_head_lr.py",
    "scripts/test_facility_head_lr.py", "scripts/run_facility_head_lr.py",
    "scripts/report_facility_head_lr.py", "scripts/plot_facility_head_lr.py",
    "tests/test_facility_head_lr_study.py", "tests/test_facility_head_lr_verification.py",
    "tests/test_facility_head_lr_report.py",
)
SOURCE_FILES = previous.SOURCE_FILES + NEW_SOURCES
MIN_TEST_COUNTS = {"tests/test_facility_head_lr_study.py": 8,
                   "tests/test_facility_head_lr_verification.py": 8,
                   "tests/test_facility_head_lr_report.py": 8}
RETENTION_GATE = deepcopy(previous.RETENTION_GATE)
BATCHNORM_POLICY = deepcopy(previous.BATCHNORM_POLICY)
OPTIMIZER_SCHEDULE = {"name": "CosineAnnealingLR", "T_max": 6, "eta_min": .000005, "weight_decay": .0002}
CONTROL_HEAD_LR = .00025
CANDIDATE_HEAD_LR = .0001
BACKBONE_LR = .00004
RECIPE = deepcopy(previous.RECIPE)
RECIPE.update(version="known_other_five_teacher_retention_head_lr_v1",
    weight_by_variant={VARIANT: 4.0}, sampler_by_variant={VARIANT: "original_control"}, reused_control_run=CONTROL)
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def cosine_curve(initial_lr, epochs=6, eta_min=.000005):
    require(not isinstance(initial_lr, bool) and isinstance(initial_lr, Real)
            and math.isfinite(float(initial_lr)) and initial_lr > 0
            and type(epochs) is int and epochs > 0
            and not isinstance(eta_min, bool) and isinstance(eta_min, Real)
            and math.isfinite(float(eta_min)) and 0 < eta_min <= initial_lr, "Invalid declared cosine LR curve")
    return [float(eta_min + (initial_lr - eta_min) * (1 + math.cos(math.pi * step / epochs)) / 2)
            for step in range(epochs + 1)]


HEAD_LR_POLICY = {
    "parameter_group": "all non-backbone student parameters",
    "control_initial_head_lr": CONTROL_HEAD_LR, "candidate_initial_head_lr": CANDIDATE_HEAD_LR,
    "backbone_initial_lr": BACKBONE_LR, "scheduler": deepcopy(OPTIMIZER_SCHEDULE),
    "step0_to6_curves": {"control_head": cosine_curve(CONTROL_HEAD_LR),
        "candidate_head": cosine_curve(CANDIDATE_HEAD_LR), "backbone": cosine_curve(BACKBONE_LR)},
    "scheduler_step_timing": "After each completed epoch; epoch1 uses step0, epoch6 uses step5, final uses step6",
    "control_curve_provenance": "Derived from frozen control protocol and scheduler source; no previous per-epoch LR observations",
    "candidate_curve_provenance": "Observed optimizer_learning_rates before each new epoch and optimizer_final_learning_rates after step6",
    "initial_head_lr_ratio": .4, "uniform_ratio_across_curve": False,
    "batchnorm_teacher_losses_and_original_sampling_unchanged": True,
}
LR_POLICY = HEAD_LR_POLICY


def fixed_values():
    value = previous.fixed_values()
    value.update(schema="facility_head_lr_study_protocol_v1", control=CONTROL, treatment=RUN_NAME,
        architecture_by_variant={VARIANT: previous.ARCH}, imgsz_by_variant={VARIANT: 640},
        distillation_recipe=deepcopy(RECIPE), distillation_weight_by_variant={VARIANT: 4.0},
        paired_draws_keys={VARIANT: "control"}, head_lr=CANDIDATE_HEAD_LR,
        head_lr_policy=deepcopy(HEAD_LR_POLICY), optimizer_schedule=deepcopy(OPTIMIZER_SCHEDULE),
        previous_protocol_sha256=PRIOR_PROTOCOL_SHA, previous_verification_sha256=PRIOR_VERIFICATION_SHA,
        previous_comparison_sha256=PRIOR_COMPARISON_SHA, reused_control_weights_sha256=CONTROL_WEIGHTS_SHA,
        control_comparison_scope="Reuse the completed frozen-BN weight-four six-epoch control; only initial non-backbone LR changes")
    return value


def completed_records(root=ROOT):
    root = Path(root).resolve()
    training = read(root / f"runs/{CONTROL}/TRAINING.json")
    require(training.get("status") == "complete" and type(training.get("actual_epochs")) is int
            and training["actual_epochs"] == 6 and training.get("model_variant") == "frozen"
            and type(training.get("distillation_weight")) is float and training["distillation_weight"] == 4.0
            and training.get("head_lr") == CONTROL_HEAD_LR and training.get("backbone_lr") == BACKBONE_LR
            and training.get("batchnorm_policy") == BATCHNORM_POLICY
            and training.get("batchnorm_state_preservation", {}).get("buffers_unchanged") is True
            and training.get("study_protocol_sha256") == PRIOR_PROTOCOL_SHA
            and training.get("weights_sha256") == CONTROL_WEIGHTS_SHA,
            "Only the completed frozen-BN weight-four original-head-LR control may be reused")
    require(sha(local_path(root, f"runs/{CONTROL}/best.pt")) == CONTROL_WEIGHTS_SHA, "Completed frozen-BN checkpoint changed")
    return {f"runs/{CONTROL}/{name}": sha(local_path(root, f"runs/{CONTROL}/{name}")) for name in CONTROL_ARTIFACTS}


def expected_protected_hashes(root=ROOT):
    root = Path(root).resolve()
    for name, expected in ((PRIOR_PROTOCOL, PRIOR_PROTOCOL_SHA), (PRIOR_VERIFICATION, PRIOR_VERIFICATION_SHA),
                           (PRIOR_COMPARISON, PRIOR_COMPARISON_SHA)):
        require(sha(local_path(root, name)) == expected, f"Previous frozen BN evidence changed: {name}")
    proof = read(root / PRIOR_VERIFICATION); protocol = read(root / PRIOR_PROTOCOL)
    comparison = read(root / PRIOR_COMPARISON)
    require(proof.get("status") == "passed" and proof.get("runtime_source_count") == 86
            and proof.get("actual_completed_training_epochs") == 6 and proof.get("working_runtime_sources_unchanged") is True
            and proof.get("both_teachers_unchanged_eval_frozen_no_grad") is True
            and proof.get("candidate_original_batchnorm_buffers_preserved_affine_learnable") is True
            and proof.get("control_training_repeated") is False, "Completed source86 BN proof is required")
    require(set(proof["source_sha256"]) == set(previous.SOURCE_FILES)
            and proof["source_sha256"] == protocol["source_sha256"] and proof["protocol_sha256"] == PRIOR_PROTOCOL_SHA,
            "Previous source86/protocol binding differs")
    require(comparison.get("schema") == "facility_batchnorm_study_comparison_v1"
            and comparison.get("protocol_sha256") == PRIOR_PROTOCOL_SHA
            and comparison.get("technical_verification_sha256") == PRIOR_VERIFICATION_SHA
            and comparison.get("technical_verification", {}).get("status") == "passed",
            "Verified previous BN source-VAL comparison is required")
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
    require(isinstance(protocol, dict), "Expected the fixed head-LR protocol")
    for key, expected in fixed_values().items():
        require(json.dumps(protocol.get(key), sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False), f"Declared head-LR condition differs: {key}")
    sources = protocol.get("source_sha256", {})
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "Runtime source inventory differs")
    for name, expected in sources.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid runtime SHA256")
        require(sha(local_path(root, name)) == expected, f"Frozen runtime source changed: {name}")
    require(protocol.get("protected_file_sha256") == expected_protected_hashes(root), "Protected previous/original inventory differs")
    require(protocol.get("reused_control_artifact_sha256") == completed_records(root), "Reused completed BN control artifacts changed")
    if args is not None:
        require(args.variant == VARIANT and args.name == RUN_NAME, "Only the single lower-head-LR candidate can train")
        for argument, field in (("seed", "seed"), ("epochs", "requested_epochs"), ("patience", "patience"),
             ("batch", "batch_size"), ("draws_per_epoch", "draws_per_epoch"), ("backbone_lr", "backbone_lr"),
             ("head_lr", "head_lr"), ("auxiliary_weight", "auxiliary_weight")):
            require(type(getattr(args, argument, None)) is type(protocol[field])
                    and getattr(args, argument) == protocol[field], f"CLI condition differs: {argument}")
        require(Path(args.initial).resolve() == (root / f"runs/{REFERENCE}/best.pt").resolve(),
                "Head-LR candidate must freshly initialize from the original 0773")
        require(Path(args.auxiliary_manifest).resolve() == (root / "data/facility-auxiliary-training/train.json").resolve(),
                "Use the original auxiliary labels")
        return {"imgsz": 640, "architecture": previous.ARCH}
    return deepcopy(protocol)


def prepare_data(root, protocol, variant=VARIANT, auxiliary_path=None):
    require(variant == VARIANT, "Only the lower-head-LR candidate has fresh training inputs")
    data = previous.prepare_data(root, protocol, "frozen", auxiliary_path)
    data["control_reuse_proof"] = {"historical_control_reused": True, "reused_control_run": CONTROL,
        "reused_control_head_lr": CONTROL_HEAD_LR, "reused_control_retrained": False,
        "all_original_control_draws_used": True, "new_training_epochs": 6}
    return data


def declare():
    torch.set_num_threads(4)
    path = ROOT / PROTOCOL
    require(not path.exists(), "Preserve the predeclared protocol")
    require(not (ROOT / "runs" / RUN_NAME).exists(), "Cannot declare after the head-LR candidate starts")
    protocol = fixed_values()
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(local_path(ROOT, name)) for name in SOURCE_FILES},
        protected_file_sha256=expected_protected_hashes(ROOT), reused_control_artifact_sha256=completed_records(ROOT))
    validate_protocol(protocol, ROOT); prepare_data(ROOT, protocol)
    write(path, protocol)
    print(json.dumps({"status": "declared", "runtime_sources": len(SOURCE_FILES), "new_training_epochs": 6,
        "reused_control_training_epochs": 6, "protocol_sha256": sha(path), "new_labels": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", required=True)
    parser.parse_args(); declare()
