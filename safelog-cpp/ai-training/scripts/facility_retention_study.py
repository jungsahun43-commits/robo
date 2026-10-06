"""One predeclared original-draw pair with known-label teacher retention loss."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import ARCH
from safelog_ai.retention_distillation import RETAINED_CLASSES, STATE_HASH_RECIPE, TEMPERATURE, state_sha256
from safelog_ai.spalling_sampler import draw_sha256
from scripts.facility_resolution_study import prepare_data as original_data
from scripts import facility_subtype_study as previous
from scripts.facility_subtype_study import FixedEpochSampler, require, local_path, write
from scripts.train_facility_target import read, sha

PROTOCOL = "reports/facility-retention-study-protocol.json"
REFERENCE = previous.REFERENCE
VARIANTS = ("control", "distill")
NAMES = {"control": "facility-presence-target-retention-control",
         "distill": "facility-presence-target-retention-distill"}
DRAW_ARCHIVE = previous.DRAW_ARCHIVE
PLAN = previous.PLAN
INITIAL_SHA = previous.INITIAL_SHA
TEACHER_STATE_SHA = "caa5f3133fffb6426c0a7fe77804d79727784f26b0102b142859801f27cb4ea8"
PRIOR_PROTOCOL = previous.PROTOCOL
PRIOR_VERIFICATION = "reports/facility-subtype-study-verification.json"
PRIOR_COMPARISON = "reports/facility-subtype-study-comparison.json"
PRIOR_INPUT_LEDGER = "runs/facility-subtype-preflight/protected-inputs.json"
PRIOR_PROTOCOL_SHA = "2b5a18b0722c2511ca6609399550dde4c5227d9b2e7ccf4bc8ab1e547781626e"
PRIOR_VERIFICATION_SHA = "b4dbc1630c0b391b792d353ddcedd001092d2ec65592588560f80e250d6ee590"
PRIOR_COMPARISON_SHA = "ac895d508e2d05ab634356585a4cdeff9963665063ccaa726714f6ca592bb743"
NEW_SOURCES = (
    "safelog_ai/retention_distillation.py", "scripts/facility_retention_study.py",
    "scripts/train_facility_retention.py", "scripts/preflight_facility_retention.py",
    "scripts/verify_facility_retention.py", "scripts/test_facility_retention.py",
    "scripts/run_facility_retention_pair.py", "scripts/report_facility_retention.py",
    "scripts/plot_facility_retention.py", "tests/test_facility_retention_study.py",
    "tests/test_facility_retention_verification.py", "tests/test_facility_retention_report.py",
)
SOURCE_FILES = previous.SOURCE_FILES + NEW_SOURCES
MIN_TEST_COUNTS = {"tests/test_facility_retention_study.py": 15,
                   "tests/test_facility_retention_verification.py": 9,
                   "tests/test_facility_retention_report.py": 10}
RETENTION_GATE = {
    "maximum_known_other_class_ap_drop_vs_reference": .02,
    "minimum_dacl_exposed_rebar_ap_recovery_vs_control": .02,
    "maximum_primary_worst_error_regression_vs_reference_and_control": .02,
}
RECIPE = {
    "version": "known_other_five_teacher_retention_v1", "candidate_count": 1,
    "teacher_run": REFERENCE, "teacher_weights_sha256": INITIAL_SHA,
    "teacher_state_sha256": TEACHER_STATE_SHA,
    "teacher_architecture": ARCH, "teacher_eval": True, "teacher_requires_grad": False,
    "teacher_forward_both_arms": True, "teacher_input": "Same augmented 640 TRAIN batch tensor as the student",
    "teacher_no_grad": True, "teacher_logits_detached_in_loss": True,
    "teacher_state_hash_recipe": STATE_HASH_RECIPE,
    "temperature": TEMPERATURE, "temperature_squared_scaling": True,
    "retained_classes": list(RETAINED_CLASSES), "retained_indices": [2, 3, 4, 5, 6],
    "excluded_primary_indices": [0, 1], "mask": "Original known photo entries only; publisher unknown stays excluded",
    "direction": "Bernoulli KL(teacher || student)", "reduction": "Mean over known original other-five entries",
    "loss_compute_dtype": "float32", "weight_by_variant": {"control": 0.0, "distill": 1.0},
    "sampler_by_variant": {"control": "original_control", "distill": "original_control"},
    "subtype_treatment_draws_used": False, "native_roi_pixels_used": False,
    "student_additional_parameters": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
    "new_auxiliary_targets": 0, "teacher_probabilities_are_asserted_labels": False,
    "input_size": 640, "coarse_mask_grid": [80, 80],
}


def fixed_values():
    value = previous.fixed_values()
    value.pop("subtype_sampling_recipe")
    value.update(schema="facility_retention_study_protocol_v1", control=NAMES["control"],
        treatment=NAMES["distill"], architecture_by_variant={v: ARCH for v in VARIANTS},
        imgsz_by_variant={v: 640 for v in VARIANTS}, distillation_recipe=deepcopy(RECIPE),
        retention_candidate_gate=deepcopy(RETENTION_GATE),
        paired_draws_keys={v: "control" for v in VARIANTS}, teacher_weights_sha256=INITIAL_SHA,
        teacher_state_sha256=TEACHER_STATE_SHA,
        distillation_weight_by_variant=deepcopy(RECIPE["weight_by_variant"]),
        previous_protocol_sha256=PRIOR_PROTOCOL_SHA, previous_verification_sha256=PRIOR_VERIFICATION_SHA,
        previous_comparison_sha256=PRIOR_COMPARISON_SHA)
    return value


def expected_protected_hashes(root=ROOT):
    root = Path(root).resolve()
    for name, expected in ((PRIOR_PROTOCOL, PRIOR_PROTOCOL_SHA), (PRIOR_VERIFICATION, PRIOR_VERIFICATION_SHA),
                           (PRIOR_COMPARISON, PRIOR_COMPARISON_SHA)):
        require(sha(local_path(root, name)) == expected, f"Previous frozen study evidence changed: {name}")
    proof = read(root / PRIOR_VERIFICATION)
    protocol = read(root / PRIOR_PROTOCOL)
    require(proof.get("status") == "passed" and proof.get("runtime_source_count") == 49
            and proof.get("actual_completed_training_epochs") == 12
            and proof.get("working_runtime_sources_unchanged") is True,
            "Completed source49 subtype proof is required")
    require(set(proof["source_sha256"]) == set(previous.SOURCE_FILES)
            and proof["source_sha256"] == protocol["source_sha256"]
            and proof["protocol_sha256"] == PRIOR_PROTOCOL_SHA, "Previous source49/protocol binding differs")
    protected = dict(proof["protected_file_sha256"])
    protected.update(proof["source_sha256"])
    protected.update({PRIOR_PROTOCOL: PRIOR_PROTOCOL_SHA, PRIOR_VERIFICATION: PRIOR_VERIFICATION_SHA,
                      PRIOR_COMPARISON: PRIOR_COMPARISON_SHA})
    for name, expected in protected.items():
        require(isinstance(expected, str) and previous._SHA.fullmatch(expected), "Invalid protected SHA256")
        require(sha(local_path(root, name)) == expected, f"Protected previous/original file changed: {name}")
    return dict(sorted(protected.items()))


def protected_hashes(root=ROOT):
    return expected_protected_hashes(root)


def validate_protocol(protocol, root=ROOT, args=None):
    root = Path(root).resolve()
    require(isinstance(protocol, dict), "Expected the fixed retention protocol")
    for key, expected in fixed_values().items():
        require(json.dumps(protocol.get(key), sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False),
                f"Declared retention condition differs: {key}")
    sources = protocol.get("source_sha256", {})
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "Runtime source inventory differs")
    for name, expected in sources.items():
        require(isinstance(expected, str) and previous._SHA.fullmatch(expected), "Invalid runtime SHA256")
        require(sha(local_path(root, name)) == expected, f"Frozen runtime source changed: {name}")
    require(protocol.get("protected_file_sha256") == expected_protected_hashes(root),
            "Protected previous/original inventory differs")
    if args is not None:
        require(args.variant in VARIANTS and args.name == NAMES[args.variant], "Run name differs from fixed pair")
        fields = {"seed": "seed", "epochs": "requested_epochs", "patience": "patience", "batch": "batch_size",
            "draws_per_epoch": "draws_per_epoch", "backbone_lr": "backbone_lr", "head_lr": "head_lr",
            "auxiliary_weight": "auxiliary_weight"}
        for argument, field in fields.items():
            require(getattr(args, argument, None) == protocol[field], f"CLI condition differs: {argument}")
        require(Path(args.initial).resolve() == (root / f"runs/{REFERENCE}/best.pt").resolve(),
                "Use the original fixed initializer path")
        require(Path(args.auxiliary_manifest).resolve() == (root / "data/facility-auxiliary-training/train.json").resolve(),
                "Use the original auxiliary labels")
        if hasattr(args, "temperature"):
            require(type(args.temperature) in (int, float) and args.temperature == TEMPERATURE,
                    "CLI teacher temperature differs")
        if hasattr(args, "distillation_weight"):
            require(type(args.distillation_weight) in (int, float)
                    and args.distillation_weight == RECIPE["weight_by_variant"][args.variant], "CLI distillation weight differs")
        return {"imgsz": 640, "architecture": ARCH}
    return deepcopy(protocol)


def validate_control_draws(draws, sampling, epoch_records, rows, expected_shape=(6, 14248)):
    require(isinstance(draws, np.ndarray) and draws.shape == expected_shape and draws.dtype == np.dtype("int64")
            and np.all(draws >= 0) and np.all(draws < rows), "Invalid original control draw dimensions or bounds")
    weights = sampling.detach().cpu() if isinstance(sampling, torch.Tensor) else torch.as_tensor(sampling)
    require(weights.ndim == 1 and len(weights) == rows and weights.dtype == torch.float64
            and torch.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0,
            "Expected the unchanged original double sampler weights")
    require(isinstance(epoch_records, list) and len(epoch_records) == expected_shape[0], "Original control epoch records missing")
    generator = torch.Generator().manual_seed(56); records = []
    for epoch, (indices, expected) in enumerate(zip(draws, epoch_records), 1):
        digest = draw_sha256(indices)
        require(expected.get("epoch") == epoch and digest == expected.get("control_order_sha256"),
                "Prepared original control epoch hash changed")
        regenerated = torch.multinomial(weights, expected_shape[1], replacement=True, generator=generator).numpy()
        require(np.array_equal(indices, regenerated), "Original control differs from fixed seed56 draw replay")
        records.append({"epoch": epoch, "draws": expected_shape[1], "control_order_sha256": digest,
            "distill_order_sha256": digest, "ordered_row_indices_equal": True})
    return records


def prepare_data(root, protocol, variant, auxiliary_path=None):
    root = Path(root).resolve()
    require(variant in VARIANTS, "Unknown fixed retention variant")
    data = original_data(root, protocol, auxiliary_path)
    require(sha(root / PLAN) == protocol["sampler_plan_sha256"] == previous.PLAN_SHA
            and sha(root / DRAW_ARCHIVE) == protocol["paired_draws_sha256"] == previous.DRAW_SHA,
            "Prepared original control archive/plan changed")
    plan = read(root / PLAN)
    weights = {"original_photo_positive_weights": data["sampling_data"]["photo_weights"],
        "original_pixel_positive_weights": data["supervision_weights"]["pixel_weights"],
        "original_auxiliary_positive_weights": data["supervision_weights"]["auxiliary_weights"]}
    for key, tensor in weights.items():
        require(tensor.tolist() == plan[key], f"Original all-class loss weights changed: {key}")
    with np.load(root / DRAW_ARCHIVE, allow_pickle=False) as archive:
        require(set(archive.files) == {"control", "treatment"}, "Original private draw archive keys changed")
        # No treatment indices are loaded or substituted in this new study.
        draws = archive["control"].copy()
    records = validate_control_draws(draws, data["sampling_data"]["sampling"], plan["epochs"], len(data["items"]))
    history = read(root / "runs/facility-presence-target-subtype-control/history.json")
    require(len(history) == 6 and all(row["sampled_row_indices_sha256"] == record["control_order_sha256"]
        for row, record in zip(history, records)), "Completed original control row-order proof changed")
    draws.flags.writeable = False
    data.update(epoch_draws=draws, control_epoch_draws=draws, paired_draws={"epochs": records},
        pair_proof={"full_rows_equal": True, "photo_targets_equal": True, "pixel_target_paths_equal": True,
            "auxiliary_targets_equal": True, "sample_weights_equal": True, "all_loss_weights_equal": True,
            "row_count": len(data["items"]), "original_full_rows": data["full_count"],
            "every_position_domain_full_crop_all_seven_photo_targets_equal": True,
            "ordered_row_indices_identical_each_epoch": True, "all_auxiliary_label_exposure_equal": True,
            "original_control_draws_equal_completed_subtype_control": True,
            "subtype_treatment_draws_used": False, "native_roi_pixels_used": False,
            "auxiliary_crop_targets_remain_unknown": True})
    return data


def declare():
    torch.set_num_threads(4)
    path = ROOT / PROTOCOL
    require(not path.exists(), "Preserve the predeclared protocol")
    require(not any((ROOT / "runs" / name).exists() for name in NAMES.values()), "Cannot declare after training starts")
    protocol = fixed_values()
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(local_path(ROOT, name)) for name in SOURCE_FILES},
        protected_file_sha256=expected_protected_hashes(ROOT))
    validate_protocol(protocol, ROOT)
    prepare_data(ROOT, protocol, "control")
    checkpoint = torch.load(ROOT / f"runs/{REFERENCE}/best.pt", map_location="cpu", weights_only=True)
    require(state_sha256(checkpoint["state_dict"]) == TEACHER_STATE_SHA, "Frozen teacher tensor state changed")
    write(path, protocol)
    print(json.dumps({"status": "declared", "runtime_sources": len(SOURCE_FILES),
        "planned_training_epochs": 12, "protocol_sha256": sha(path), "new_labels": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", required=True)
    parser.parse_args(); declare()
