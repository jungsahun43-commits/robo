"""Fixed six-plus-six TRAIN study of one native-tag negative subtype sampler.

The historical full/crop inputs, targets and loss weights are retained. Private
paired draws were prepared before this study; no VAL outcome changes the recipe.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np
import torch
from torch.utils.data import Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import ARCH, AUX_CLASSES
from safelog_ai.spalling_sampler import (FACTOR, RELATED_TAGS, ELIGIBLE_STRATA,
    coupling_strata, eligible_rows, paired_epoch, draw_counts, draw_sha256)
from scripts.facility_resolution_study import (CLASSES, INITIAL_SHA,
    prepare_data as original_data)
from scripts.facility_native_roi_study import SOURCE_FILES as FROZEN_NATIVE_SOURCES
from scripts.report_facility_detail import GATE
from scripts.train_facility_target import DOMAINS, read, sha

PROTOCOL = "reports/facility-subtype-study-protocol.json"
PLAN = "reports/facility-spalling-sampler-dry-run.json"
PLAN_TESTS = "reports/facility-spalling-sampler-tests.json"
DRAW_ARCHIVE = "runs/facility-spalling-sampler-plan/paired-draws.npz"
PLAN_LEDGER = "runs/facility-spalling-sampler-v1-evidence/protected-ledger.json"
REFERENCE = "facility-presence-target-roi-control"
VARIANTS = ("control", "negative")
NAMES = {"control": "facility-presence-target-subtype-control",
         "negative": "facility-presence-target-subtype-negative"}
ARCHIVE_KEYS = {"control": "control", "negative": "treatment"}
PLAN_SHA = "54c406c09f5bf4b01c994fb3344f1b6b50d0d3b3a927048a08f0f0fa2f8b5a12"
PLAN_TESTS_SHA = "c6c271c901c9b86c0cb928fdb37a476eeec1a1d7b39c329d3d4b4ff36f982ecf"
DRAW_SHA = "18b46d4021962a7d8fa22fc9475bbf7d956d76c30779a449d08a2eaa85d3ed3b"
CORE_SHA = "8216d8c1c958e59401b7ec20809afa681eb2b1ccb9640fec953597c5db61c681"
AUX_SHA = "53544bf83e1b32e3a7b8775c5f8348594359bf6e01eade1c8571242e758f4b61"
APP_SHA = "ecc09958022c659ef37385d1633177c034d27cda8034cadc3ab4af2ab8abb035"
NEW_SOURCES = (
    "scripts/facility_subtype_study.py", "scripts/train_facility_subtype.py",
    "scripts/preflight_facility_subtype.py", "scripts/verify_facility_subtype.py",
    "scripts/test_facility_subtype.py", "scripts/run_facility_subtype_pair.py",
    "scripts/report_facility_subtype.py", "scripts/plot_facility_subtype.py",
    "tests/test_facility_subtype_study.py", "tests/test_facility_subtype_verification.py",
    "tests/test_facility_subtype_report.py",
)
PRIOR_TESTS = ("tests/test_facility_native_roi_data.py",
               "tests/test_facility_native_roi_study.py",
               "tests/test_facility_native_roi_report.py")
SOURCE_FILES = tuple(dict.fromkeys(FROZEN_NATIVE_SOURCES + (
    "safelog_ai/spalling_sampler.py", "scripts/prepare_facility_spalling_sampler.py",
    "tests/test_spalling_sampler.py", "scripts/report_facility_native_roi_results.py",
    "tests/test_facility_native_roi_report_adapter.py",
) + PRIOR_TESTS + NEW_SOURCES))
MIN_TEST_COUNTS = {"tests/test_facility_subtype_study.py": 10,
                   "tests/test_facility_subtype_verification.py": 8,
                   "tests/test_facility_subtype_report.py": 10}
RECIPE = {
    "version": "original_native_tag_negative_subtype_v1", "candidate_count": 1,
    "eligible_factor": FACTOR, "eligible_other_four": list(RELATED_TAGS),
    "eligibility": "Original full DACL TRAIN Spalling target 0 and at least one original Rockpocket/WConccor/Hollowareas/Cavity tag 1",
    "eligible_original_full_dacl_parents": 666,
    "control": "Frozen six seed56 original-weight replacement draw arrays",
    "negative": "Prepared seed59 maximal coupling within full DACL Crack/Spalling joint00 or10",
    "replacement_probability": "(factor-1)*p/(1+(factor-1)*p); exact boundary no-op",
    "replacement_distribution": "Original sampler weights conditional on eligible rows in the same stratum",
    "eligible_strata": list(ELIGIBLE_STRATA), "coupling_numpy_seed": 59,
    "every_position_domain_full_crop_two_photo_targets_equal": True,
    "original_eligible_positions_retained": True,
    "non_dacl_crops_and_spalling_positive_positions_unchanged": True,
    "original_sampler_weights_and_all_loss_weights_unchanged": True,
    "other_five_photo_and_auxiliary_label_exposure_may_change": True,
    "individual_scores_used_for_eligibility": False,
    "new_photo_targets": 0, "new_pixel_targets": 0, "new_auxiliary_targets": 0,
    "input_size": 640, "coarse_mask_grid": [80, 80],
    "native_roi_pixels_used": False,
}
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    """Create evidence once; preserve any earlier execution record."""
    with Path(path).open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def local_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative
            and ":" not in relative and not PurePosixPath(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Expected a relative internal evidence path")
    root = Path(root).resolve(); path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or escaped evidence path")
    return path


def fixed_values():
    return {"schema": "facility_subtype_study_protocol_v1", "declared_before_training": True,
        "reference": REFERENCE, "control": NAMES["control"], "treatment": NAMES["negative"],
        "architecture_by_variant": {v: ARCH for v in VARIANTS},
        "imgsz_by_variant": {v: 640 for v in VARIANTS},
        "classes": CLASSES, "auxiliary_classes": list(AUX_CLASSES),
        "seed": 56, "requested_epochs": 6, "patience": 6, "batch_size": 8,
        "draws_per_epoch": 14248, "backbone_lr": .00004, "head_lr": .00025,
        "auxiliary_weight": .5, "target_ranking_weight": 0., "domain_proportions": [.7, .1, .2],
        "loader_randomness": {"sampler_seed": 56, "training_worker_seed": 57, "post_model_seed": 58,
            "validation_worker_seeds": {d: 156 + k for d, k in DOMAINS.items()}},
        "initial_weights_sha256": INITIAL_SHA, "core_spatial_manifest_sha256": CORE_SHA,
        "auxiliary_manifest_sha256": AUX_SHA, "app_profile_sha256": APP_SHA,
        "subtype_sampling_recipe": deepcopy(RECIPE), "research_candidate_gate": deepcopy(GATE),
        "paired_draws_path": DRAW_ARCHIVE, "paired_draws_sha256": DRAW_SHA,
        "private_draw_archive_sha256": DRAW_SHA, "sampler_plan_path": PLAN,
        "sampler_plan_sha256": PLAN_SHA, "sampler_plan_tests_sha256": PLAN_TESTS_SHA,
        "source_test_inference_executed": False, "app_model_promoted": False}


def expected_protected_hashes(root=ROOT):
    """Preserve the already published dry-run and original frozen study inputs.

    The current dry-run's public inventory excludes individual image paths.
    The new preflight separately binds every original TRAIN image and mask.
    """
    root = Path(root).resolve()
    require(sha(local_path(root, PLAN)) == PLAN_SHA, "Prepared sampler plan changed")
    plan = read(root / PLAN)
    result = dict(plan["protected_input_sha256"])
    result.update(plan["source_sha256"]); result.update(plan["test_source_sha256"])
    result.update({PLAN: PLAN_SHA, PLAN_TESTS: PLAN_TESTS_SHA, DRAW_ARCHIVE: DRAW_SHA})
    # Prior tests and the preserved first numerical proof stay unchanged too.
    previous = read(root / "reports/facility-native-roi-study-verification.json")
    for name in PRIOR_TESTS:
        result[name] = previous["tests"]["test_source_sha256"][name]
    result[PLAN_LEDGER] = plan["numerical_repair_note"]["preserved_v1_protected_ledger_sha256"]
    for name, expected in result.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid protected SHA256")
        require(sha(local_path(root, name)) == expected, f"Protected original input changed: {name}")
    return dict(sorted(result.items()))


def protected_hashes(root=ROOT):
    return expected_protected_hashes(root)


def validate_protocol(protocol, root=ROOT, args=None):
    root = Path(root).resolve()
    require(isinstance(protocol, dict), "Expected the fixed subtype protocol")
    for key, expected in fixed_values().items():
        # Canonical JSON also distinguishes 0 from false and 1 from true in
        # nested recipe flags; ordinary Python container equality does not.
        require(json.dumps(protocol.get(key), sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False),
                f"Declared subtype condition differs: {key}")
    sources = protocol.get("source_sha256", {})
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_FILES), "Runtime source inventory differs")
    for name, expected in sources.items():
        require(isinstance(expected, str) and _SHA.fullmatch(expected), "Invalid runtime SHA256")
        require(sha(local_path(root, name)) == expected, f"Frozen runtime source changed: {name}")
    protected = protocol.get("protected_file_sha256")
    require(isinstance(protected, dict) and protected == expected_protected_hashes(root),
            "Protected original/sampler inventory differs")
    for field, path in (("core_spatial_manifest_sha256", "data/facility-spatial-training/train.json"),
                        ("auxiliary_manifest_sha256", "data/facility-auxiliary-training/train.json"),
                        ("app_profile_sha256", "reports/facility-inference-profile.json")):
        require(sha(local_path(root, path)) == protocol[field], f"Protected data changed: {field}")
    require(sha(local_path(root, f"runs/{REFERENCE}/best.pt")) == INITIAL_SHA, "Original initializer changed")
    if args is not None:
        require(args.variant in VARIANTS and args.name == NAMES[args.variant], "Run name differs from fixed pair")
        fields = {"seed": "seed", "epochs": "requested_epochs", "patience": "patience",
            "batch": "batch_size", "draws_per_epoch": "draws_per_epoch", "backbone_lr": "backbone_lr",
            "head_lr": "head_lr", "auxiliary_weight": "auxiliary_weight"}
        for argument, field in fields.items():
            require(getattr(args, argument, None) == protocol[field], f"CLI condition differs: {argument}")
        require(Path(args.initial).resolve() == (root / f"runs/{REFERENCE}/best.pt").resolve(),
                "Use the original fixed initializer path")
        require(Path(args.auxiliary_manifest).resolve() == (root / "data/facility-auxiliary-training/train.json").resolve(),
                "Use the original auxiliary labels")
        return {"imgsz": 640, "architecture": ARCH}
    return deepcopy(protocol)


def validate_paired_draws(control, treatment, items, full_count, auxiliary, sampling,
                          epoch_records, expected_shape=(6, 14248), replay=True):
    """Verify private arrays, unchanged positions and fixed RNG regeneration."""
    require(isinstance(expected_shape, tuple) and len(expected_shape) == 2
            and all(type(v) is int and v > 0 for v in expected_shape), "Invalid paired draw dimensions")
    for draws in (control, treatment):
        require(isinstance(draws, np.ndarray) and draws.shape == expected_shape
                and draws.dtype == np.dtype("int64") and np.all(draws >= 0) and np.all(draws < len(items)),
                "Invalid paired draw shape, integer dtype or row bounds")
    require(isinstance(epoch_records, list) and len(epoch_records) == expected_shape[0], "Paired epoch records missing")
    eligible = eligible_rows(items, full_count, auxiliary)
    strata = coupling_strata(items, full_count)
    weights = sampling.detach().cpu().numpy() if isinstance(sampling, torch.Tensor) else sampling
    require(isinstance(weights, np.ndarray) and weights.shape == (len(items),)
            and weights.dtype.kind == "f" and np.isfinite(weights).all() and (weights >= 0).all()
            and weights.sum() > 0, "Invalid unchanged original sampler weights")
    generator = torch.Generator().manual_seed(56)
    coupling_rng = np.random.default_rng(59)
    records = []
    for index, (base, paired, expected) in enumerate(zip(control, treatment, epoch_records), 1):
        require(expected.get("epoch") == index, "Paired epoch order changed")
        require(draw_sha256(base) == expected.get("control_order_sha256")
                and draw_sha256(paired) == expected.get("treatment_order_sha256"), "Prepared epoch draw hash changed")
        require(np.array_equal(strata[base], strata[paired]), "Source/full-crop/two-target stratum changed")
        require(np.array_equal(base[eligible[base]], paired[eligible[base]]), "Original eligible draws were replaced")
        changed = base != paired
        require(np.isin(strata[base[changed]], ELIGIBLE_STRATA).all()
                and not eligible[base[changed]].any() and eligible[paired[changed]].all(),
                "Replacement violates the declared negative subtype scope")
        require(int(changed.sum()) == expected.get("changed_positions")
                and int(eligible[base].sum()) == expected.get("eligible_draws_control")
                and int(eligible[paired].sum()) == expected.get("eligible_draws_treatment"),
                "Prepared changed/eligible exposure changed")
        require(draw_counts(base, strata) == draw_counts(paired, strata), "Paired aggregate strata changed")
        if replay:
            regenerated = torch.multinomial(torch.as_tensor(weights, dtype=torch.float64), expected_shape[1],
                replacement=True, generator=generator).numpy()
            require(np.array_equal(base, regenerated), "Prepared control differs from original seed56 sampler")
            regenerated_pair = paired_epoch(base, weights, eligible, strata, coupling_rng, FACTOR)
            require(np.array_equal(paired, regenerated_pair), "Prepared treatment differs from original seed59 coupling")
        records.append({"epoch": index, "control_order_sha256": draw_sha256(base),
            "treatment_order_sha256": draw_sha256(paired), "changed_positions": int(changed.sum()),
            "eligible_draws_control": int(eligible[base].sum()),
            "eligible_draws_treatment": int(eligible[paired].sum()),
            "stratum_counts": draw_counts(base, strata),
            "every_position_domain_full_crop_and_two_targets_equal": True})
    return {"epochs": records, "eligible_rows": eligible, "strata": strata,
        "total_changed_positions": sum(row["changed_positions"] for row in records),
        "total_eligible_draws_control": sum(row["eligible_draws_control"] for row in records),
        "total_eligible_draws_treatment": sum(row["eligible_draws_treatment"] for row in records)}


class FixedEpochSampler(Sampler[int]):
    """Replay exactly one prepared one-based epoch without consuming any RNG."""
    def __init__(self, epoch_draws):
        require(isinstance(epoch_draws, np.ndarray) and epoch_draws.ndim == 2
                and epoch_draws.dtype == np.dtype("int64") and min(epoch_draws.shape) > 0
                and np.all(epoch_draws >= 0), "Expected prepared integer epoch draws")
        self.epoch_draws = epoch_draws.copy()
        self.epoch_draws.flags.writeable = False
        self.epoch = None

    def set_epoch(self, epoch):
        require(type(epoch) is int and 1 <= epoch <= len(self.epoch_draws), "Epoch must be one-based and declared")
        self.epoch = epoch

    def __iter__(self):
        require(self.epoch is not None, "Set the declared epoch before loader iteration")
        return iter(self.epoch_draws[self.epoch - 1].tolist())

    def __len__(self):
        return self.epoch_draws.shape[1]


def prepare_data(root, protocol, variant, auxiliary_path=None):
    root = Path(root).resolve()
    require(variant in VARIANTS, "Unknown fixed sampler variant")
    data = original_data(root, protocol, auxiliary_path)
    plan = read(root / PLAN)
    require(sha(root / PLAN) == protocol["sampler_plan_sha256"] == PLAN_SHA
            and sha(root / DRAW_ARCHIVE) == protocol["paired_draws_sha256"] == DRAW_SHA,
            "Prepared sampler plan/archive bytes changed")
    require(plan.get("status") == "prepared_dry_run" and plan.get("eligible_factor") == FACTOR
            and plan.get("planned_epochs_each_arm") == 6 and plan.get("draws_per_epoch") == 14248
            and plan.get("individual_errors_used_for_eligibility") is False,
            "Original single-candidate sampler preparation is required")
    expected_weights = {"original_photo_positive_weights": data["sampling_data"]["photo_weights"],
        "original_pixel_positive_weights": data["supervision_weights"]["pixel_weights"],
        "original_auxiliary_positive_weights": data["supervision_weights"]["auxiliary_weights"]}
    for key, actual in expected_weights.items():
        require(actual.tolist() == plan[key], f"Original all-class loss weights changed: {key}")
    with np.load(root / DRAW_ARCHIVE, allow_pickle=False) as archive:
        require(set(archive.files) == {"control", "treatment"}, "Unexpected private draw archive keys")
        control, treatment = archive["control"].copy(), archive["treatment"].copy()
    proof = validate_paired_draws(control, treatment, data["items"], data["full_count"],
        data["auxiliary"], data["sampling_data"]["sampling"], plan["epochs"])
    require(int(proof["eligible_rows"].sum()) == 666 and proof["total_changed_positions"] == 1204
            and proof["total_eligible_draws_control"] == 3539
            and proof["total_eligible_draws_treatment"] == 4743, "Prepared total eligibility/exposure changed")
    for name in ("facility-presence-target-native-roi-control", "facility-presence-target-native-roi-native"):
        history = read(root / f"runs/{name}/history.json")
        require(len(history) == 6 and all(row["sampled_row_indices_sha256"] == expected["control_order_sha256"]
            for row, expected in zip(history, proof["epochs"])), "Completed original control row order changed")
    epoch_draws = control if variant == "control" else treatment
    epoch_draws.flags.writeable = False
    control.flags.writeable = False
    data.update(epoch_draws=epoch_draws, control_epoch_draws=control, eligible_rows=proof["eligible_rows"],
        paired_draws={key: value for key, value in proof.items() if key not in ("eligible_rows", "strata")},
        pair_proof={"full_rows_equal": True, "photo_targets_equal": True,
            "pixel_target_paths_equal": True, "auxiliary_targets_equal": True,
            "sample_weights_equal": True, "all_loss_weights_equal": True,
            "row_count": len(data["items"]), "original_full_rows": data["full_count"],
            "native_roi_pixels_used": False, "every_position_domain_full_crop_two_photo_targets_equal": True,
            "original_eligible_positions_retained": True,
            "non_dacl_crops_and_spalling_positive_positions_unchanged": True,
            "original_control_draws_equal_completed_native_pair": True,
            "other_five_photo_and_auxiliary_label_exposure_may_change": True,
            "eligible_original_full_dacl_parents": 666, "changed_positions_all_epochs": 1204,
            "eligible_draws_control_all_epochs": 3539, "eligible_draws_treatment_all_epochs": 4743,
            "auxiliary_crop_targets_remain_unknown": True})
    return data


def declare():
    # Avoid excessive small-tensor CPU thread overhead during source/weight
    # preparation. The original weights and both prepared arrays are checked.
    torch.set_num_threads(4)
    path = ROOT / PROTOCOL
    require(not path.exists(), "Preserve the predeclared protocol")
    require(not any((ROOT / "runs" / name).exists() for name in NAMES.values()), "Cannot declare after training starts")
    protocol = fixed_values()
    protocol.update(declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={file: sha(local_path(ROOT, file)) for file in SOURCE_FILES},
        protected_file_sha256=expected_protected_hashes(ROOT))
    validate_protocol(protocol, ROOT)
    # Both archives are replayed by prepare_data. Their original source inputs
    # are identical, so a second full source scan would not add a proof.
    prepare_data(ROOT, protocol, "control")
    write(path, protocol)
    print(json.dumps({"status": "declared", "runtime_sources": len(SOURCE_FILES),
        "planned_training_epochs": 12, "protocol_sha256": sha(path), "new_labels": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true", required=True)
    parser.parse_args(); declare()
