"""One fixed weight-four follow-up, reusing the completed weight-zero control.

Only one fresh six-epoch student is trained. Previously completed epochs are
reported as reused evidence and are never counted as new training again.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.retention_distillation import state_sha256
from scripts import facility_retention_study as frozen
from scripts.facility_subtype_study import require, local_path, write, FixedEpochSampler
from scripts.train_facility_target import read, sha

PROTOCOL = "reports/facility-retention-strength-study-protocol.json"
REFERENCE = frozen.REFERENCE
CONTROL = frozen.NAMES["control"]
PREVIOUS = frozen.NAMES["distill"]
RUN_NAME = "facility-presence-target-retention-strong"
VARIANT = "strong"
VARIANTS = (VARIANT,)
NAMES = {VARIANT: RUN_NAME}
INITIAL_SHA = frozen.INITIAL_SHA
TEACHER_STATE_SHA = frozen.TEACHER_STATE_SHA
DRAW_ARCHIVE = frozen.DRAW_ARCHIVE
PLAN = frozen.PLAN
PRIOR_PROTOCOL = frozen.PROTOCOL
PRIOR_VERIFICATION = "reports/facility-retention-study-verification.json"
PRIOR_COMPARISON = "reports/facility-retention-study-comparison.json"
PRIOR_PREFLIGHT = "runs/facility-retention-preflight.json"
PRIOR_INPUT_LEDGER = "runs/facility-retention-preflight/protected-inputs.json"
PRIOR_ADAPTER_TESTS = "runs/facility-retention-report-adapter-tests.json"
PRIOR_ADAPTER_TESTS_SHA = "672d393a8b4c6f230fbb96973a8b53a4e6cec874c7ac906b01e316e2547c1eef"
PRIOR_PROTOCOL_SHA = "8f3438426a3b43c167041b0c85ae1a8abf059ab846a9772ddb2dbf713be5f394"
PRIOR_VERIFICATION_SHA = "8aa268860e7a38fefb8cd21878f1dfd6a5dc5d263e2880aa31b103f83e05e770"
PRIOR_COMPARISON_SHA = "a5e2d128a3e4cf2a01474f4ac35836650792d5fb2bff81362c251e2dbf6ed555"
CONTROL_WEIGHTS_SHA = "e1f80be9cba61df5e27a5d36552a9c811d1a4fc472992464a7b8e1e67a5bee9d"
PREVIOUS_WEIGHTS_SHA = "c27c3faddb134d6b8f7e6dfafc91ea321781e42436e4023caf80d42983f2fc48"
CONTROL_ARTIFACTS = (
    "TRAINING.json", "history.json", "SPLIT.json", "best.pt", "VALIDATION.json",
    "TARGET-SELECTION.json", "ERROR-SIZE-AUDIT.json",
    "validation-dacl.json", "validation-damsegment.json", "validation-codebrim.json",
    "target-validation-dacl-grid1.json", "target-validation-damsegment-grid1.json",
    "target-validation-codebrim-grid1.json",
)
NEW_SOURCES = (
    "scripts/facility_retention_strength_study.py", "scripts/train_facility_retention_strength.py",
    "scripts/preflight_facility_retention_strength.py", "scripts/verify_facility_retention_strength.py",
    "scripts/test_facility_retention_strength.py", "scripts/run_facility_retention_strength.py",
    "scripts/report_facility_retention_strength.py", "scripts/plot_facility_retention_strength.py",
    "tests/test_facility_retention_strength_study.py", "tests/test_facility_retention_strength_verification.py",
    "tests/test_facility_retention_strength_report.py",
)
SOURCE_FILES = frozen.SOURCE_FILES + (
    "scripts/report_facility_retention_results.py", "tests/test_facility_retention_report_adapter.py",
) + NEW_SOURCES
MIN_TEST_COUNTS = {"tests/test_facility_retention_strength_study.py": 10,
                   "tests/test_facility_retention_strength_verification.py": 8,
                   "tests/test_facility_retention_strength_report.py": 10}
RETENTION_GATE = deepcopy(frozen.RETENTION_GATE)
RECIPE = deepcopy(frozen.RECIPE)
RECIPE.update(version="known_other_five_teacher_retention_strength_v1",
    weight_by_variant={VARIANT: 4.0}, sampler_by_variant={VARIANT: "original_control"},
    historical_control_reused=True, reused_control_run=CONTROL,
    reused_control_weight=0.0, reused_control_retrained=False)


def fixed_values():
    value = frozen.fixed_values()
    value.update(schema="facility_retention_strength_study_protocol_v1", control=CONTROL, treatment=RUN_NAME,
        previous_treatment=PREVIOUS, architecture_by_variant={VARIANT: frozen.ARCH},
        imgsz_by_variant={VARIANT: 640}, distillation_recipe=deepcopy(RECIPE),
        distillation_weight_by_variant={VARIANT: 4.0}, paired_draws_keys={VARIANT: "control"},
        previous_protocol_sha256=PRIOR_PROTOCOL_SHA, previous_verification_sha256=PRIOR_VERIFICATION_SHA,
        previous_comparison_sha256=PRIOR_COMPARISON_SHA,
        previous_report_adapter_tests_sha256=PRIOR_ADAPTER_TESTS_SHA,
        new_training_epochs=6, reused_control_training_epochs=6, reused_control_retrained=False,
        reused_control_weights_sha256=CONTROL_WEIGHTS_SHA, previous_treatment_weights_sha256=PREVIOUS_WEIGHTS_SHA,
        control_comparison_scope="Reuse the already completed original-draw weight-zero six-epoch control; no second control training")
    return value


def completed_records(root=ROOT):
    """Bind the previously completed control and weight-one run documents."""
    root = Path(root).resolve()
    records = {}
    for name, variant, weight, expected_weights in ((CONTROL, "control", 0.0, CONTROL_WEIGHTS_SHA),
            (PREVIOUS, "distill", 1.0, PREVIOUS_WEIGHTS_SHA)):
        training = read(root / f"runs/{name}/TRAINING.json")
        require(training.get("status") == "complete" and type(training.get("actual_epochs")) is int
                and training["actual_epochs"] == 6 and training.get("model_variant") == variant
                and type(training.get("distillation_weight")) is float and training["distillation_weight"] == weight
                and training.get("study_protocol_sha256") == PRIOR_PROTOCOL_SHA
                and training.get("weights_sha256") == expected_weights,
                "Only verified completed previous retention runs may be reused")
        require(sha(local_path(root, f"runs/{name}/best.pt")) == expected_weights, "Previous checkpoint changed")
        for artifact in CONTROL_ARTIFACTS:
            relative = f"runs/{name}/{artifact}"
            records[relative] = sha(local_path(root, relative))
    return dict(sorted(records.items()))


def expected_protected_hashes(root=ROOT):
    root = Path(root).resolve()
    require(isinstance(PRIOR_COMPARISON_SHA, str) and frozen.previous._SHA.fullmatch(PRIOR_COMPARISON_SHA),
            "The completed previous final comparison must be bound before declaration")
    for name, expected in ((PRIOR_PROTOCOL, PRIOR_PROTOCOL_SHA), (PRIOR_VERIFICATION, PRIOR_VERIFICATION_SHA),
                           (PRIOR_COMPARISON, PRIOR_COMPARISON_SHA)):
        require(sha(local_path(root, name)) == expected, f"Previous frozen retention evidence changed: {name}")
    proof = read(root / PRIOR_VERIFICATION); protocol = read(root / PRIOR_PROTOCOL)
    comparison = read(root / PRIOR_COMPARISON)
    require(proof.get("status") == "passed" and proof.get("runtime_source_count") == 61
            and proof.get("actual_completed_training_epochs") == 12
            and proof.get("working_runtime_sources_unchanged") is True
            and proof.get("both_teachers_unchanged_eval_frozen_no_grad") is True,
            "Completed source61 retention proof is required")
    require(set(proof["source_sha256"]) == set(frozen.SOURCE_FILES)
            and proof["source_sha256"] == protocol["source_sha256"]
            and proof["protocol_sha256"] == PRIOR_PROTOCOL_SHA, "Previous source61/protocol binding differs")
    require(comparison.get("schema") == "facility_retention_study_comparison_v1"
            and comparison.get("protocol_sha256") == PRIOR_PROTOCOL_SHA
            and comparison.get("technical_verification_sha256") == PRIOR_VERIFICATION_SHA
            and comparison.get("technical_verification", {}).get("status") == "passed",
            "The verified previous source-VAL comparison is required")
    protected = dict(proof["protected_file_sha256"]); protected.update(proof["source_sha256"])
    protected.update(completed_records(root))
    protected.update({PRIOR_PROTOCOL: PRIOR_PROTOCOL_SHA, PRIOR_VERIFICATION: PRIOR_VERIFICATION_SHA,
        PRIOR_COMPARISON: PRIOR_COMPARISON_SHA, PRIOR_PREFLIGHT: proof["preflight_sha256"],
        PRIOR_INPUT_LEDGER: proof["prepared_data_integrity"]["ledger_sha256"],
        PRIOR_ADAPTER_TESTS: PRIOR_ADAPTER_TESTS_SHA})
    adapter_tests = read(root / PRIOR_ADAPTER_TESTS)
    require(adapter_tests.get("status") == "passed" and type(adapter_tests.get("tests_run")) is int
            and adapter_tests["tests_run"] == 2 and all(type(adapter_tests.get(key)) is int
                and adapter_tests[key] == 0 for key in ("failures", "errors", "skipped")),
            "Completed previous report adapter tests are required")
    for name, expected in protected.items():
        require(isinstance(expected, str) and frozen.previous._SHA.fullmatch(expected), "Invalid protected SHA256")
        require(sha(local_path(root, name)) == expected, f"Protected previous/original file changed: {name}")
    return dict(sorted(protected.items()))


def protected_hashes(root=ROOT):
    return expected_protected_hashes(root)


def validate_protocol(protocol, root=ROOT, args=None):
    root = Path(root).resolve()
    require(isinstance(protocol, dict), "Expected the fixed retention-strength protocol")
    for key, expected in fixed_values().items():
        require(json.dumps(protocol.get(key), sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False),
                f"Declared retention-strength condition differs: {key}")
    sources = protocol.get("source_sha256", {})
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "Runtime source inventory differs")
    for name, expected in sources.items():
        require(isinstance(expected, str) and frozen.previous._SHA.fullmatch(expected), "Invalid runtime SHA256")
        require(sha(local_path(root, name)) == expected, f"Frozen runtime source changed: {name}")
    require(protocol.get("protected_file_sha256") == expected_protected_hashes(root),
            "Protected previous/original inventory differs")
    require(protocol.get("reused_control_artifact_sha256") == completed_records(root),
            "Reused completed control/treatment artifact bytes changed")
    if args is not None:
        require(args.variant == VARIANT and args.name == RUN_NAME, "Only the single strong candidate can train")
        for argument, field in (("seed", "seed"), ("epochs", "requested_epochs"), ("patience", "patience"),
             ("batch", "batch_size"), ("draws_per_epoch", "draws_per_epoch"), ("backbone_lr", "backbone_lr"),
             ("head_lr", "head_lr"), ("auxiliary_weight", "auxiliary_weight")):
            require(getattr(args, argument, None) == protocol[field], f"CLI condition differs: {argument}")
        require(Path(args.initial).resolve() == (root / f"runs/{REFERENCE}/best.pt").resolve(),
                "Strong candidate must freshly initialize from the original 0773")
        require(Path(args.auxiliary_manifest).resolve() == (root / "data/facility-auxiliary-training/train.json").resolve(),
                "Use the original auxiliary labels")
        return {"imgsz": 640, "architecture": frozen.ARCH}
    return deepcopy(protocol)


def prepare_data(root, protocol, variant=VARIANT, auxiliary_path=None):
    require(variant == VARIANT, "Only the strong candidate has fresh training inputs")
    data = frozen.prepare_data(root, protocol, "control", auxiliary_path)
    data["control_reuse_proof"] = {"historical_control_reused": True,
        "reused_control_run": CONTROL, "reused_control_retrained": False,
        "all_original_control_draws_used": True, "new_training_epochs": 6}
    return data


def declare():
    torch.set_num_threads(4)
    path = ROOT / PROTOCOL
    require(not path.exists(), "Preserve the predeclared protocol")
    require(not (ROOT / "runs" / RUN_NAME).exists(), "Cannot declare after the strong candidate starts")
    protocol = fixed_values()
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(local_path(ROOT, name)) for name in SOURCE_FILES},
        protected_file_sha256=expected_protected_hashes(ROOT), reused_control_artifact_sha256=completed_records(ROOT))
    validate_protocol(protocol, ROOT)
    prepare_data(ROOT, protocol)
    checkpoint = torch.load(ROOT / f"runs/{REFERENCE}/best.pt", map_location="cpu", weights_only=True)
    require(state_sha256(checkpoint["state_dict"]) == TEACHER_STATE_SHA, "Frozen teacher tensor state changed")
    write(path, protocol)
    print(json.dumps({"status": "declared", "runtime_sources": len(SOURCE_FILES), "new_training_epochs": 6,
        "reused_control_training_epochs": 6, "protocol_sha256": sha(path), "new_labels": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", required=True)
    parser.parse_args(); declare()
