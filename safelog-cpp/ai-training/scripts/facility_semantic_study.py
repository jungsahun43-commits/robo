"""One fixed ImageNet-feature residual candidate retaining the entire base model."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.semantic_residual_classifier import ARCH, OFFICIAL_URL, WEIGHT_ENUM, WEIGHT_SHA_PREFIX
from scripts import facility_head_lr_study as previous
from scripts.facility_subtype_study import require, local_path, write, FixedEpochSampler
from scripts.train_facility_target import read, sha

PROTOCOL = "reports/facility-semantic-study-protocol.json"
REFERENCE = previous.REFERENCE
CONTROL = previous.RUN_NAME
RUN_NAME = "facility-presence-target-semantic-features"
VARIANT = "semantic"
VARIANTS = (VARIANT,)
NAMES = {VARIANT: RUN_NAME}
INITIAL_SHA = previous.INITIAL_SHA
TEACHER_STATE_SHA = previous.TEACHER_STATE_SHA
INITIAL_BN_BUFFERS_SHA = previous.INITIAL_BN_BUFFERS_SHA
DRAW_ARCHIVE = previous.DRAW_ARCHIVE
PLAN = previous.PLAN
WEIGHTS_PATH = "data/pretrained/convnext-tiny-imagenet1k-v1.pth"
METADATA_PATH = "reports/facility-semantic-pretrained-weights.json"
PRIOR_PROTOCOL = previous.PROTOCOL
PRIOR_VERIFICATION = "reports/facility-head-lr-study-verification.json"
PRIOR_COMPARISON = "reports/facility-head-lr-study-comparison.json"
PRIOR_PREFLIGHT = "runs/facility-head-lr-preflight.json"
PRIOR_INPUT_LEDGER = "runs/facility-head-lr-preflight/protected-inputs.json"
PRIOR_PROTOCOL_SHA = "e67f693bef3142f1c8f86f937e8b53509a9b3041b260029446db75bed2bd74f0"
PRIOR_VERIFICATION_SHA = "6fc0c0d6a5b1f1ff1585f88fbed72a33a722515c84685e638997fafcaff57980"
PRIOR_COMPARISON_SHA = "eaa4250ccd8a71b7af09ee2293e6eb213b805deb36322d7c6284f3f0a87dd773"
CONTROL_WEIGHTS_SHA = "058e6de039da9bb2b35c78e35c1cecf64099c8ac187597804fce99c8ce5afcb7"
CONTROL_ARTIFACTS = previous.CONTROL_ARTIFACTS
NEW_SOURCES = (
    "safelog_ai/semantic_residual_classifier.py", "safelog_ai/research_presence.py",
    "scripts/facility_semantic_study.py", "scripts/fetch_facility_semantic_weights.py",
    "scripts/train_facility_semantic.py", "scripts/evaluate_facility_semantic.py",
    "scripts/preflight_facility_semantic.py", "scripts/verify_facility_semantic.py", "scripts/test_facility_semantic.py",
    "scripts/run_facility_semantic.py", "scripts/report_facility_semantic.py", "scripts/plot_facility_semantic.py",
    "tests/test_facility_semantic_study.py", "tests/test_facility_semantic_verification.py", "tests/test_facility_semantic_report.py",
)
SOURCE_FILES = previous.SOURCE_FILES + NEW_SOURCES
MIN_TEST_COUNTS = {"tests/test_facility_semantic_study.py": 8,
                   "tests/test_facility_semantic_verification.py": 8,
                   "tests/test_facility_semantic_report.py": 8}
RETENTION_GATE = deepcopy(previous.RETENTION_GATE)
BATCHNORM_POLICY = deepcopy(previous.BATCHNORM_POLICY)
OPTIMIZER_SCHEDULE = deepcopy(previous.OPTIMIZER_SCHEDULE)
RECIPE = deepcopy(previous.RECIPE)
RECIPE.update(version="known_other_five_teacher_retention_semantic_features_v1",
    weight_by_variant={VARIANT: 4.0}, sampler_by_variant={VARIANT: "original_control"}, reused_control_run=CONTROL)
SEMANTIC_POLICY = {
    "version": "frozen_imagenet_convnext_tiny_zero_residual_heads_v1", "candidate_count": 1,
    "original_backbone_and_heads_retained": True, "original_state_tensor_count": 324,
    "new_semantic_head_state_tensor_count": 6, "new_semantic_head_parameter_count": 21345,
    "new_semantic_heads_initialized_zero": True, "initial_public_maps_aux_equal_original": True,
    "encoder_architecture": "convnext_tiny", "weight_enum": WEIGHT_ENUM, "official_url": OFFICIAL_URL,
    "pretrained_weights_path": WEIGHTS_PATH, "pretrained_metadata_path": METADATA_PATH,
    "encoder_parameters_frozen": True, "encoder_always_eval": True, "encoder_forward_no_grad": True,
    "encoder_state_bitwise_unchanged": True, "encoder_stochastic_depth_disabled_by_eval": True,
    "encoder_excluded_from_optimizer": True, "official_classifier_zero_pool_norm_transferred": True,
    "imagenet_classifier_logits_used": False, "low_feature_layer": "features.3", "low_channels": 192,
    "low_feature_grid_for_640": [80, 80], "pooled_channels": 768,
    "map_residual": "Original7x80 maps plus zero-initialized Conv2d192to7 before original top32 pooling",
    "photo_residual": "Original global7 logits plus zero-initialized Linear768to7",
    "auxiliary_residual": "Original auxiliary19 logits plus zero-initialized Linear768to19",
    "original_mix_and_top32_unchanged": True, "new_heads_trainable": True,
    "whole640_original_rgb_normalization": True, "official224_center_crop_used": False,
    "new_photo_targets": 0, "new_pixel_targets": 0, "new_auxiliary_targets": 0,
    "new_independent_facility_photos": 0,
    "scope": "Adds a frozen ImageNet feature/compute package and learnable residual heads; no original backbone replacement or pure capacity-cause claim",
}
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def pretrained_metadata(root=ROOT):
    root = Path(root).resolve()
    path = local_path(root, METADATA_PATH); record = read(path)
    require(record.get("schema") == "facility_semantic_pretrained_weights_v1" and record.get("status") == "verified"
            and record.get("weights_path") == WEIGHTS_PATH and record.get("official_url") == OFFICIAL_URL
            and record.get("weight_enum") == WEIGHT_ENUM and record.get("expected_filename_sha256_prefix") == WEIGHT_SHA_PREFIX,
            "Verified official semantic metadata is required")
    digest = record.get("weights_sha256"); encoder = record.get("encoder_state_sha256")
    require(isinstance(digest, str) and _SHA.fullmatch(digest) and digest.startswith(WEIGHT_SHA_PREFIX)
            and isinstance(encoder, str) and _SHA.fullmatch(encoder), "Official full-file/encoder SHA is invalid")
    weights = local_path(root, WEIGHTS_PATH)
    require(sha(weights) == digest and weights.stat().st_size == record.get("file_size_bytes"), "Verified official weight bytes changed")
    require(record.get("torch_version") == str(torch.__version__) and record.get("torchvision_version") == str(torchvision.__version__)
            and sha(Path(torchvision.__file__).parent / "models/convnext.py") == record.get("torchvision_convnext_source_sha256"),
            "Installed semantic model dependency version/source changed")
    require(record.get("encoder_state_tensor_count") == 180 and record.get("encoder_parameter_count") == 27820128
            and record.get("strict_encoder_load") is True and record.get("official_pooling_norm_transferred") is True
            and record.get("all_encoder_parameters_frozen") is True and record.get("encoder_eval") is True,
            "Official encoder inventory/load proof differs")
    return record


def semantic_policy(root=ROOT):
    record = pretrained_metadata(root)
    return {**deepcopy(SEMANTIC_POLICY), "pretrained_weights_sha256": record["weights_sha256"],
        "pretrained_metadata_sha256": sha(Path(root) / METADATA_PATH), "encoder_state_sha256": record["encoder_state_sha256"],
        "encoder_state_tensor_count": record["encoder_state_tensor_count"], "encoder_parameter_count": record["encoder_parameter_count"]}


def fixed_values(root=ROOT):
    record = pretrained_metadata(root)
    value = previous.fixed_values(); value.pop("head_lr_policy", None)
    value.update(schema="facility_semantic_study_protocol_v1", control=CONTROL, treatment=RUN_NAME,
        architecture_by_variant={VARIANT: ARCH}, imgsz_by_variant={VARIANT: 640},
        distillation_recipe=deepcopy(RECIPE), distillation_weight_by_variant={VARIANT: 4.0}, paired_draws_keys={VARIANT: "control"},
        semantic_features_policy=semantic_policy(root), semantic_model_inventory=deepcopy(record["encoder_inventory"]),
        semantic_encoder_state_sha256=record["encoder_state_sha256"], semantic_pretrained_weights_path=WEIGHTS_PATH,
        semantic_pretrained_weights_sha256=record["weights_sha256"], semantic_pretrained_metadata_path=METADATA_PATH,
        semantic_pretrained_metadata_sha256=sha(Path(root) / METADATA_PATH),
        optimizer_policy={"control_head_lr": .0001, "candidate_head_lr": .0001, "backbone_lr_both": .00004,
            "schedule": deepcopy(OPTIMIZER_SCHEDULE), "both_curves_observed_in_epoch_histories": True,
            "learning_rates_changed": False},
        previous_protocol_sha256=PRIOR_PROTOCOL_SHA, previous_verification_sha256=PRIOR_VERIFICATION_SHA,
        previous_comparison_sha256=PRIOR_COMPARISON_SHA, reused_control_weights_sha256=CONTROL_WEIGHTS_SHA,
        control_comparison_scope="Reuse completed original-backbone frozen-BN head-LR1e-4 weight-four control; add a frozen ImageNet residual feature package")
    return value


def completed_records(root=ROOT):
    root = Path(root).resolve()
    training = read(root / f"runs/{CONTROL}/TRAINING.json")
    require(training.get("status") == "complete" and type(training.get("actual_epochs")) is int and training["actual_epochs"] == 6
            and training.get("model_variant") == "low_lr" and training.get("distillation_weight") == 4.0
            and training.get("head_lr") == .0001 and training.get("backbone_lr") == .00004
            and training.get("batchnorm_policy") == BATCHNORM_POLICY
            and training.get("batchnorm_state_preservation", {}).get("buffers_unchanged") is True
            and training.get("study_protocol_sha256") == PRIOR_PROTOCOL_SHA and training.get("weights_sha256") == CONTROL_WEIGHTS_SHA,
            "Only the completed exact original-model head-LR control may be reused")
    require(sha(local_path(root, f"runs/{CONTROL}/best.pt")) == CONTROL_WEIGHTS_SHA, "Completed head-LR checkpoint changed")
    return {f"runs/{CONTROL}/{name}": sha(local_path(root, f"runs/{CONTROL}/{name}")) for name in CONTROL_ARTIFACTS}


def expected_protected_hashes(root=ROOT):
    root = Path(root).resolve()
    record = pretrained_metadata(root)
    for name, expected in ((PRIOR_PROTOCOL, PRIOR_PROTOCOL_SHA), (PRIOR_VERIFICATION, PRIOR_VERIFICATION_SHA), (PRIOR_COMPARISON, PRIOR_COMPARISON_SHA)):
        require(sha(local_path(root, name)) == expected, f"Previous frozen head-LR evidence changed: {name}")
    proof = read(root / PRIOR_VERIFICATION); protocol = read(root / PRIOR_PROTOCOL); comparison = read(root / PRIOR_COMPARISON)
    require(proof.get("status") == "passed" and proof.get("runtime_source_count") == 97 and proof.get("actual_completed_training_epochs") == 6
            and proof.get("working_runtime_sources_unchanged") is True and proof.get("both_teachers_unchanged_eval_frozen_no_grad") is True
            and proof.get("candidate_original_batchnorm_buffers_preserved_affine_learnable") is True
            and proof.get("control_training_repeated") is False, "Completed source97 head-LR proof is required")
    require(set(proof["source_sha256"]) == set(previous.SOURCE_FILES) and proof["source_sha256"] == protocol["source_sha256"]
            and proof["protocol_sha256"] == PRIOR_PROTOCOL_SHA, "Previous source97/protocol binding differs")
    require(comparison.get("schema") == "facility_head_lr_study_comparison_v1" and comparison.get("protocol_sha256") == PRIOR_PROTOCOL_SHA
            and comparison.get("technical_verification_sha256") == PRIOR_VERIFICATION_SHA
            and comparison.get("technical_verification", {}).get("status") == "passed", "Verified previous head-LR source-VAL comparison is required")
    protected = dict(proof["protected_file_sha256"]); protected.update(proof["source_sha256"]); protected.update(completed_records(root))
    protected.update({PRIOR_PROTOCOL: PRIOR_PROTOCOL_SHA, PRIOR_VERIFICATION: PRIOR_VERIFICATION_SHA,
        PRIOR_COMPARISON: PRIOR_COMPARISON_SHA, PRIOR_PREFLIGHT: proof["preflight_sha256"],
        PRIOR_INPUT_LEDGER: proof["prepared_data_integrity"]["ledger_sha256"],
        WEIGHTS_PATH: record["weights_sha256"], METADATA_PATH: sha(root / METADATA_PATH)})
    for name, expected in protected.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid protected SHA256")
        require(sha(local_path(root, name)) == expected, f"Protected previous/original/pretrained file changed: {name}")
    return dict(sorted(protected.items()))


def protected_hashes(root=ROOT):
    return expected_protected_hashes(root)


def validate_protocol(protocol, root=ROOT, args=None):
    root = Path(root).resolve()
    require(isinstance(protocol, dict), "Expected the fixed semantic residual protocol")
    for key, expected in fixed_values(root).items():
        require(json.dumps(protocol.get(key), sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False),
                f"Declared semantic condition differs: {key}")
    sources = protocol.get("source_sha256", {})
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "Runtime source inventory differs")
    for name, expected in sources.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid runtime SHA256")
        require(sha(local_path(root, name)) == expected, f"Frozen runtime source changed: {name}")
    require(protocol.get("protected_file_sha256") == expected_protected_hashes(root), "Protected previous/original/pretrained inventory differs")
    require(protocol.get("reused_control_artifact_sha256") == completed_records(root), "Reused completed head-LR control artifacts changed")
    if args is not None:
        require(args.variant == VARIANT and args.name == RUN_NAME, "Only the single semantic residual candidate can train")
        for argument, field in (("seed", "seed"), ("epochs", "requested_epochs"), ("patience", "patience"), ("batch", "batch_size"),
             ("draws_per_epoch", "draws_per_epoch"), ("backbone_lr", "backbone_lr"), ("head_lr", "head_lr"), ("auxiliary_weight", "auxiliary_weight")):
            require(type(getattr(args, argument, None)) is type(protocol[field]) and getattr(args, argument) == protocol[field],
                    f"CLI condition differs: {argument}")
        require(Path(args.initial).resolve() == (root / f"runs/{REFERENCE}/best.pt").resolve(), "Semantic base must freshly initialize from original0773")
        require(Path(args.auxiliary_manifest).resolve() == (root / "data/facility-auxiliary-training/train.json").resolve(), "Use original auxiliary labels")
        return {"imgsz": 640, "architecture": ARCH}
    return deepcopy(protocol)


def prepare_data(root, protocol, variant=VARIANT, auxiliary_path=None):
    require(variant == VARIANT, "Only the semantic candidate has fresh training inputs")
    data = previous.prepare_data(root, protocol, "low_lr", auxiliary_path)
    data["control_reuse_proof"] = {"historical_control_reused": True, "reused_control_run": CONTROL,
        "reused_control_head_lr": .0001, "reused_control_retrained": False, "all_original_control_draws_used": True, "new_training_epochs": 6}
    return data


def declare():
    torch.set_num_threads(4)
    path = ROOT / PROTOCOL
    require(not path.exists(), "Preserve the predeclared protocol")
    require(not (ROOT / "runs" / RUN_NAME).exists(), "Cannot declare after the semantic candidate starts")
    protocol = fixed_values(ROOT)
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(), source_sha256={name: sha(local_path(ROOT, name)) for name in SOURCE_FILES},
        protected_file_sha256=expected_protected_hashes(ROOT), reused_control_artifact_sha256=completed_records(ROOT))
    validate_protocol(protocol, ROOT); prepare_data(ROOT, protocol)
    write(path, protocol)
    print(json.dumps({"status": "declared", "runtime_sources": len(SOURCE_FILES), "new_training_epochs": 6,
        "reused_control_training_epochs": 6, "protocol_sha256": sha(path), "new_labels": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", required=True)
    parser.parse_args(); declare()
