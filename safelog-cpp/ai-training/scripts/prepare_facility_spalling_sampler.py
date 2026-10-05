"""Record a CPU-only paired sampler dry-run; never start training or inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.spalling_sampler import (FACTOR, RELATED_TAGS, ELIGIBLE_STRATA,
    coupling_probabilities, coupling_strata, eligible_rows, paired_epoch,
    draw_counts, draw_sha256)
from scripts.facility_resolution_study import (CLASSES, INITIAL_SHA, build_sampling,
                                               supervision_weights)
from scripts.audit_facility_spalling_tag_scores import INPUTS as TAG_INPUTS

REPORT = "reports/facility-spalling-sampler-dry-run.json"
TEST_RECORD = "reports/facility-spalling-sampler-tests.json"
PLAN = "runs/facility-spalling-sampler-plan"
V1_EVIDENCE = "runs/facility-spalling-sampler-v1-evidence"
SOURCE_PATHS = ("safelog_ai/spalling_sampler.py", "scripts/prepare_facility_spalling_sampler.py",
                "tests/test_spalling_sampler.py")
DIRECT = {
    "core": "data/facility-spatial-training/train.json",
    "auxiliary": "data/facility-auxiliary-training/train.json",
    "tag_audit": "reports/facility-spalling-tag-score-audit.json",
    "tag_tests": "reports/facility-spalling-tag-score-tests.json",
    "prior_audit": "reports/facility-spalling-train-audit.json",
    "prior_tests": "reports/facility-spalling-train-audit-tests.json",
    "prior_private": "runs/facility-spalling-train-audit/SPALLING-AUDIT.json",
    "native_protocol": "reports/facility-native-roi-study-protocol.json",
    "native_verification": "reports/facility-native-roi-study-verification.json",
    "native_comparison": "reports/facility-native-roi-study-comparison.json",
    "control_history": "runs/facility-presence-target-native-roi-control/history.json",
    "native_history": "runs/facility-presence-target-native-roi-native/history.json",
    "control_training": "runs/facility-presence-target-native-roi-control/TRAINING.json",
    "native_training": "runs/facility-presence-target-native-roi-native/TRAINING.json",
}
EPOCHS = 6
DRAWS = 14248
SAMPLER_SEED = 56
COUPLING_SEED = 59
EXPECTED_TESTS = 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), "Protected path escapes the repository")
    stat = path.stat()
    return {"sha256": sha(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def inventory():
    # Reuse the established protected input set, frozen SOURCE30, all existing
    # Python code/tests, and both completed native ROI study records.
    private = read(DIRECT["prior_private"])
    frozen = read(DIRECT["native_verification"])
    paths = set(private["provenance"]["input_snapshots_before"])
    paths.update(frozen["source_sha256"])
    paths.update(frozen["protected_file_sha256"])
    paths.update(TAG_INPUTS.values())
    paths.update(DIRECT.values())
    paths.update(SOURCE_PATHS)
    if (ROOT / V1_EVIDENCE).is_dir():
        paths.update(f"{V1_EVIDENCE}/{name}" for name in SOURCE_PATHS)
        paths.update(f"{V1_EVIDENCE}/{name}" for name in (
            "facility-spalling-sampler-dry-run.json", "facility-spalling-sampler-tests.json",
            "protected-ledger.json", "facility-spalling-sampler-plan/paired-draws.npz"))
    for directory in ("scripts", "safelog_ai", "tests"):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).glob("*.py"))
    return sorted(paths)


def snapshots(paths):
    return {name: snapshot(name) for name in paths}


def snapshot_sha256(values):
    """Bind the complete private inventory without publishing photo paths."""
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def public_input_hashes(values):
    # Only metadata/artifact and program paths appear in aggregate reports.
    names = set(DIRECT.values()) | set(TAG_INPUTS.values()) | set(SOURCE_PATHS)
    names.update(read(DIRECT["native_verification"])["source_sha256"])
    names.update(read(DIRECT["native_verification"])["protected_file_sha256"])
    return {name: values[name]["sha256"] for name in sorted(names)}


def write_new(relative, value):
    path = ROOT / relative
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def bound_inputs(before):
    documents = {key: read(name) for key, name in DIRECT.items()}
    core, aux, audit = (documents[key] for key in ("core", "auxiliary", "tag_audit"))
    frozen, protocol = (documents[key] for key in ("native_verification", "native_protocol"))
    require(frozen.get("status") == "passed" and frozen.get("runtime_source_count") == 30
            and frozen.get("working_runtime_sources_unchanged") is True,
            "Completed native ROI SOURCE30 verification is required")
    require(len(frozen["source_sha256"]) == 30
            and frozen["source_sha256"] == protocol["source_sha256"], "Frozen runtime source inventory changed")
    for name, expected in {**frozen["source_sha256"], **frozen["protected_file_sha256"]}.items():
        require(before[name]["sha256"] == expected, f"Frozen original input changed: {name}")
    require(frozen.get("actual_completed_training_epochs") == 12
            and frozen.get("actual_sampling_verification", {}).get("epochs_compared") == 6
            and frozen["actual_sampling_verification"].get("actual_ordered_row_index_hashes_identical_each_epoch") is True,
            "The existing completed six-plus-six native ROI pair must remain bound")
    require(protocol.get("seed") == SAMPLER_SEED and protocol.get("requested_epochs") == EPOCHS
            and protocol.get("draws_per_epoch") == DRAWS and protocol.get("domain_proportions") == [.7, .1, .2]
            and protocol.get("initial_weights_sha256") == INITIAL_SHA, "Original paired recipe changed")
    require(before[DIRECT["native_protocol"]]["sha256"] == frozen["protocol_sha256"]
            and documents["native_comparison"]["protocol_sha256"] == frozen["protocol_sha256"],
            "Completed study report belongs to another protocol")
    require(audit.get("status") == "audited" and audit.get("dacl_train_photos") == 6225
            and audit.get("model_weights_sha256") == INITIAL_SHA
            and audit.get("group_composition_matches_bound_original_source_audit") is True,
            "Complete full-DACL frozen-model tag diagnostic is required")
    require(audit["input_sha256"] == {key: before[name]["sha256"] for key, name in TAG_INPUTS.items()},
            "Full-tag diagnostic source or input bytes changed")
    require(audit["focused_test_record_sha256"] == before[DIRECT["tag_tests"]]["sha256"],
            "Full-tag diagnostic test record changed")
    tag_tests = documents["tag_tests"]
    require(tag_tests.get("status") == "passed" and tag_tests.get("tests_run") == 6
            and all(tag_tests.get(key) == 0 for key in ("failures", "errors", "skipped")),
            "Actual original full-tag focused tests must remain passed")
    require(core.get("split") == "train" and core.get("classes") == CLASSES
            and core.get("full_count") == DRAWS and len(core.get("items", [])) == 26289,
            "Original core TRAIN order, class set or population changed")
    require(aux.get("split") == "train" and tuple(aux.get("classes", ())) == AUX_CLASSES
            and len(aux.get("items", [])) == 6225, "Original full-DACL native tag population changed")
    require(before[DIRECT["core"]]["sha256"] == protocol["core_spatial_manifest_sha256"]
            and before[DIRECT["auxiliary"]]["sha256"] == protocol["auxiliary_manifest_sha256"],
            "Original photo/pixel/native target bytes changed")
    source_audit = read(TAG_INPUTS["source_audit"])
    groups = read(TAG_INPUTS["label_groups"])
    require(source_audit.get("status") == "complete" and source_audit.get("train_photos") == 6225
            and tuple(source_audit["native_classes"]) == AUX_CLASSES
            and source_audit["manifest_sha256"]["spatial"] == before[DIRECT["core"]]["sha256"]
            and source_audit["manifest_sha256"]["auxiliary"] == before[DIRECT["auxiliary"]]["sha256"],
            "Original source annotation audit is not bound to core/native targets")
    require(groups["sampling_strata_photo_counts"]["related_other_four_without_Spalling"] == 666
            and audit["disjoint_four_groups"]["spalling_negative_other_four_present"]["photos"] == 666,
            "The fixed original eligible group must contain exactly 666 parents")
    package = read(TAG_INPUTS["package"])
    content = dict(package)
    content_sha = content.pop("package_content_sha256", None)
    require(content_sha == hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "Original TRAIN130 package canonical identity changed")
    require(package.get("weights_sha256") == INITIAL_SHA and len(package.get("cases", [])) == 130
            and package["provenance"]["manifest_sha256"] == before[DIRECT["core"]]["sha256"]
            and package["provenance"]["thresholds_sha256"] == before[TAG_INPUTS["selection"]]["sha256"],
            "Original TRAIN130 package/model/selection binding changed")
    selection = read(TAG_INPUTS["selection"])
    require(selection.get("weights_sha256") == INITIAL_SHA and selection.get("selection_split") == "val"
            and selection.get("classes") == CLASSES and before[TAG_INPUTS["weights"]]["sha256"] == INITIAL_SHA,
            "Frozen0773 model and already-selected VAL cutoffs changed")
    prior = documents["prior_audit"]
    require(prior.get("status") == "audited" and prior.get("tests_run") == 14
            and prior.get("all_original_sha_size_mtime_preserved") is True
            and prior.get("private_details_sha256") == before[DIRECT["prior_private"]]["sha256"],
            "Original selected-case read-only audit must remain bound")
    for name, expected in documents["prior_private"]["provenance"]["input_snapshots_before"].items():
        normalized = {"sha256": expected["sha256"], "size": expected["size_bytes"],
                      "mtime_ns": expected["mtime_ns"]}
        require(before[name] == normalized, f"Established protected input metadata changed: {name}")
    auxiliary = {}
    for item in aux["items"]:
        require(item.get("domain") == "dacl" and item.get("split") == "train"
                and item["image"] not in auxiliary, "Native tag identity/split changed")
        auxiliary[item["image"]] = item["targets"]
    eligible = eligible_rows(core["items"], core["full_count"], auxiliary)
    strata = coupling_strata(core["items"], core["full_count"])
    require(int(eligible.sum()) == 666, "Actual original eligible parent membership differs")
    require(Counter(strata[eligible].tolist()) == {"dacl/full/00": 421, "dacl/full/10": 245},
            "Original eligible Crack/Spalling joint composition changed")
    for item in core["items"][:core["full_count"]]:
        if item["domain"] == "dacl":
            tags = auxiliary[item["image"]]
            require(item["targets"][0] == int(bool(tags[AUX_CLASSES.index("Crack")]
                                                        or tags[AUX_CLASSES.index("ACrack")])),
                    "Native Crack/ACrack and original photo Crack differ")
    return documents, eligible, strata


def run_tests():
    require(not (ROOT / TEST_RECORD).exists(), "Preserve existing sampler test evidence")
    paths = inventory()
    before = snapshots(paths)
    spec = importlib.util.spec_from_file_location("test_spalling_sampler", ROOT / SOURCE_PATHS[2])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    collected = suite.countTestCases()
    require(collected == EXPECTED_TESTS, "Collect all six new focused sampler tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = snapshots(paths)
    passed = (result.wasSuccessful() and result.testsRun == EXPECTED_TESTS
              and not result.skipped and before == after)
    record = {"schema": "facility_spalling_sampler_tests_v1", "status": "passed" if passed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
        "expected_tests_collected": collected, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "source_sha256": {name: before[name]["sha256"] for name in SOURCE_PATHS[:2]},
        "test_source_sha256": {SOURCE_PATHS[2]: before[SOURCE_PATHS[2]]["sha256"]},
        "protected_inputs": len(before), "protected_snapshot_before_sha256": snapshot_sha256(before),
        "protected_snapshot_after_sha256": snapshot_sha256(after),
        "protected_input_sha256": public_input_hashes(before),
        "all_input_sha256_size_mtime_preserved": before == after,
        "old_focused_tests_rerun": 0, "training_performed": False, "inference_performed": False,
        "scope": "New synthetic sampler probability, boundary, pairing and scope tests only"}
    write_new(TEST_RECORD, record)
    require(passed, "Sampler focused tests failed; retain the recorded evidence")
    print(json.dumps({"status": "passed", "tests_run": EXPECTED_TESTS, "protected_inputs": len(before)}))


def prepare():
    require(not (ROOT / REPORT).exists() and not (ROOT / PLAN).exists(),
            "Preserve existing sampler plan/report; refusing overwrite")
    paths = inventory()
    before = snapshots(paths)
    tests = read(TEST_RECORD)
    require(tests.get("status") == "passed" and tests.get("tests_run") == EXPECTED_TESTS
            and tests.get("expected_tests_collected") == EXPECTED_TESTS
            and all(tests.get(key) == 0 for key in ("failures", "errors", "skipped"))
            and tests.get("all_input_sha256_size_mtime_preserved") is True
            and tests["protected_snapshot_after_sha256"] == snapshot_sha256(before)
            and tests["protected_input_sha256"] == public_input_hashes(before),
            "Actual source-bound focused tests are required")
    documents, eligible, strata = bound_inputs(before)
    core, aux = documents["core"], documents["auxiliary"]
    original = build_sampling(core["items"], core["full_count"], CLASSES, proportions=(.7, .1, .2))
    weights = original["sampling"].numpy().copy()
    supervision = supervision_weights(core, aux)
    training = documents["control_training"]
    require(training.get("actual_epochs") == EPOCHS and training.get("seed") == SAMPLER_SEED
            and training.get("draws_per_epoch") == DRAWS and training.get("domain_proportions") == [.7, .1, .2],
            "Actual native control budget/seed/proportions changed")
    require(original["photo_weights"].tolist() == training["photo_positive_weights"]
            and supervision["pixel_weights"].tolist() == training["pixel_positive_weights"]
            and supervision["auxiliary_weights"].tolist() == training["auxiliary_positive_weights"],
            "Original photo/pixel/native loss weights differ from completed control")
    history = documents["control_history"]
    native_history = documents["native_history"]
    require(len(history) == len(native_history) == EPOCHS, "Exactly six original paired sampling epochs are required")
    for name in ("control_training", "native_training"):
        require(documents[name]["initial_weights_sha256"] == INITIAL_SHA
                and documents[name]["source_sha256"] == documents["native_verification"]["source_sha256"],
                "Existing paired training initializer/runtime provenance changed")
    generator = torch.Generator(device="cpu").manual_seed(SAMPLER_SEED)
    rng = np.random.default_rng(COUPLING_SEED)
    labels = original["labels"].numpy()
    domains = original["domains"].numpy()
    row_type = np.arange(len(weights)) >= core["full_count"]
    control, treatment, epochs = [], [], []
    for epoch, (actual, actual_native) in enumerate(zip(history, native_history), 1):
        base = torch.multinomial(original["sampling"], DRAWS, replacement=True,
                                 generator=generator).numpy()
        base_sha = draw_sha256(base)
        require(actual["epoch"] == actual_native["epoch"] == epoch
                and base_sha == actual["sampled_row_indices_sha256"]
                == actual_native["sampled_row_indices_sha256"],
                "CPU control row order does not match both actual native study histories")
        paired = paired_epoch(base, weights, eligible, strata, rng)
        changed = base != paired
        require(np.array_equal(domains[base], domains[paired])
                and np.array_equal(row_type[base], row_type[paired])
                and np.array_equal(labels[base, :2], labels[paired, :2]),
                "A paired position changed its source/full-crop/original Crack-Spalling targets")
        require(np.all(~eligible[base[changed]]) and np.all(eligible[paired[changed]])
                and np.all(np.isin(strata[base[changed]], ELIGIBLE_STRATA)),
                "Replacement escaped the fixed original eligible stratum")
        counts = draw_counts(base, strata)
        paired_counts = draw_counts(paired, strata)
        domain_counts = {name: int((domains[base] == index).sum()) for index, name in enumerate(("dacl", "damsegment", "codebrim"))}
        row_counts = {"full": int((~row_type[base]).sum()), "crop": int(row_type[base].sum())}
        full_joint = {domain: {state: counts.get(f"{domain}/full/{state}", 0)
                              for state in ("00", "10", "01", "11", "unknown")}
                      for domain in ("dacl", "damsegment", "codebrim")}
        for key, value in counts.items():
            domain, kind, state = key.split("/", 2)
            if kind == "full" and state.startswith("unknown:"):
                full_joint[domain]["unknown"] += value
        require(counts == paired_counts and domain_counts == actual["sampled_domain_counts"]
                and row_counts == actual["sampled_row_type_counts"]
                and full_joint == actual["sampled_full_target_joint_counts"],
                "Actual original source/full/crop/joint sampling counts changed")
        epochs.append({"epoch": epoch, "draws": DRAWS,
            "control_order_sha256": base_sha, "treatment_order_sha256": draw_sha256(paired),
            "actual_control_history_order_sha256": actual["sampled_row_indices_sha256"],
            "actual_native_history_order_sha256": actual_native["sampled_row_indices_sha256"],
            "actual_control_order_matches": True, "actual_native_order_matches": True,
            "changed_positions": int(changed.sum()), "unchanged_positions": int((~changed).sum()),
            "eligible_draws_control": int(eligible[base].sum()),
            "eligible_draws_treatment": int(eligible[paired].sum()),
            "eligible_draws_by_stratum": {key: {
                "control": int((eligible[base] & (strata[base] == key)).sum()),
                "treatment": int((eligible[paired] & (strata[paired] == key)).sum())}
                for key in ELIGIBLE_STRATA},
            "domain_counts": domain_counts, "full_crop_counts": row_counts,
            "full_target_joint_counts": full_joint, "stratum_counts_control": counts,
            "stratum_counts_treatment": paired_counts, "all_stratum_counts_equal": True,
            "every_position_domain_full_crop_and_two_targets_equal": True,
            "all_original_eligible_draws_retained": True,
            "all_non_dacl_crops_and_spalling_positive_positions_unchanged": True})
        control.append(base.copy())
        treatment.append(paired)
    strata_plan = {}
    for key in ELIGIBLE_STRATA:
        members = strata == key
        p = float(weights[members & eligible].sum() / weights[members].sum())
        strata_plan[key] = {"original_rows": int(members.sum()), "eligible_original_parents": int((members & eligible).sum()),
                           "original_stratum_mass": float(weights[members].sum()),
                           **coupling_probabilities(p)}
    numerical_repair_note = None
    if (ROOT / V1_EVIDENCE).is_dir():
        old_report = read(f"{V1_EVIDENCE}/facility-spalling-sampler-dry-run.json")
        old_test_record = read(f"{V1_EVIDENCE}/facility-spalling-sampler-tests.json")
        old_archive = ROOT / V1_EVIDENCE / "facility-spalling-sampler-plan/paired-draws.npz"
        require(sha(old_archive) == old_report["private_draw_archive_sha256"]
                and sha(ROOT / V1_EVIDENCE / "facility-spalling-sampler-tests.json")
                    == old_report["focused_test_record_sha256"],
                "Preserved first sampler artifacts lost their original binding")
        for name, expected in {**old_report["source_sha256"], **old_report["test_source_sha256"]}.items():
            require(sha(ROOT / V1_EVIDENCE / name) == expected,
                    "Preserved first sampler source or test bytes changed")
        old_ledger = read(f"{V1_EVIDENCE}/protected-ledger.json")
        require(snapshot_sha256(old_ledger["snapshots"]) == old_ledger["snapshot_sha256"]
                == old_report["protected_snapshot_before_sha256"]
                == old_test_record["protected_snapshot_before_sha256"],
                "Preserved first sampler private protection ledger changed")
        with np.load(old_archive, allow_pickle=False) as old_draws:
            require(np.array_equal(old_draws["control"], np.stack(control))
                    and np.array_equal(old_draws["treatment"], np.stack(treatment)),
                    "Numerical boundary repair changed actual paired draw arrays")
        numerical_repair_note = {
            "status": "corrected_before_any_new_training",
            "reason": "Use algebraically stable interior replacement probability to avoid cancellation when eligible probability is almost one; add a near-one assertion in the existing boundary test",
            "first_plan_actual_stratum_probabilities": {key: old_report["stratum_probabilities"][key]["original_eligible_probability"]
                                                       for key in ELIGIBLE_STRATA},
            "first_plan_actual_probabilities_were_valid": True,
            "preserved_v1_evidence_path": V1_EVIDENCE,
            "preserved_v1_source_sha256": old_report["source_sha256"],
            "preserved_v1_test_source_sha256": old_report["test_source_sha256"],
            "preserved_v1_report_sha256": sha(ROOT / V1_EVIDENCE / "facility-spalling-sampler-dry-run.json"),
            "preserved_v1_test_record_sha256": old_report["focused_test_record_sha256"],
            "preserved_v1_private_draw_archive_sha256": old_report["private_draw_archive_sha256"],
            "preserved_v1_protected_ledger_sha256": sha(ROOT / V1_EVIDENCE / "protected-ledger.json"),
            "preserved_v1_protected_snapshot_sha256": old_ledger["snapshot_sha256"],
            "all_six_by_14248_control_draw_arrays_exactly_equal": True,
            "all_six_by_14248_treatment_draw_arrays_exactly_equal": True,
            "corrected_source_and_tests_bound_before_final_publication": True,
            "new_training_started_before_repair": False,
        }
    after = snapshots(paths)
    require(before == after, "Protected original inputs/source metadata changed during sampler planning")
    output = ROOT / PLAN
    output.mkdir(exist_ok=False)
    archive = output / "paired-draws.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, control=np.stack(control), treatment=np.stack(treatment))
    with np.load(archive, allow_pickle=False) as restored:
        require(np.array_equal(restored["control"], np.stack(control))
                and np.array_equal(restored["treatment"], np.stack(treatment)),
                "Private draw archive does not retain the exact paired indices")
    report = {"schema": "facility_spalling_sampler_dry_run_v1", "status": "prepared_dry_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU-only fixed sampler plan using original full TRAIN tags and actual completed control row order; no new performance result",
        "candidate_count": 1, "eligible_factor": FACTOR, "eligible_other_four": list(RELATED_TAGS),
        "original_rows": len(core["items"]), "original_full_rows": core["full_count"],
        "original_full_dacl_parents": 6225, "eligible_original_full_dacl_parents": int(eligible.sum()),
        "original_class_order": CLASSES, "original_auxiliary_class_order": list(AUX_CLASSES),
        "domain_proportions": [.7, .1, .2], "sampler_seed": SAMPLER_SEED,
        "separate_coupling_numpy_seed": COUPLING_SEED, "draws_per_epoch": DRAWS,
        "planned_epochs_each_arm": EPOCHS, "draw_hash_recipe": "SHA256 of row-order concatenated little-endian int64 bytes",
        "coupling": "Retain all base eligible rows; replace only noneligible base rows with (pt-p)/(1-p), sampling original-weight eligible rows in the same full DACL Crack/Spalling joint00 or10 stratum; pt=1.5p/(1+0.5p)",
        "conditional_stratum_reweighting_preserves_original_stratum_mass": True,
        "stratum_probabilities": strata_plan, "epochs": epochs,
        "total_changed_positions": sum(row["changed_positions"] for row in epochs),
        "total_eligible_draws_control": sum(row["eligible_draws_control"] for row in epochs),
        "total_eligible_draws_treatment": sum(row["eligible_draws_treatment"] for row in epochs),
        "all_six_actual_control_and_native_order_hashes_match": True,
        "all_six_source_full_crop_crack_spalling_draw_counts_equal": True,
        "every_position_original_two_photo_targets_equal": True,
        "original_photo_positive_weights": original["photo_weights"].tolist(),
        "original_pixel_positive_weights": supervision["pixel_weights"].tolist(),
        "original_auxiliary_positive_weights": supervision["auxiliary_weights"].tolist(),
        "original_loss_weights_preserved": True,
        "other_five_photo_labels_and_auxiliary_exposure_may_change": True,
        "other_class_ap_regression_gate_required_for_future_training": True,
        "future_pair_scope": "One fixed new six-plus-six-epoch comparison only; requires a separately declared authorized training protocol before execution",
        "completed_native_pair_training_epochs": 12, "new_training_epochs": 0,
        "training_epochs_budget_spent": 0, "new_optimizer_steps": 0, "gpu_used": False,
        "new_inference_performed": False, "new_labels": 0, "label_changes": 0,
        "new_photo_targets": 0, "new_pixel_targets": 0, "new_auxiliary_targets": 0,
        "individual_errors_used_for_eligibility": False, "original_0773_or_completed_native_val_performance_changed": False,
        "individual_validation_or_test_photos_annotations_predictions_read": False,
        "app_profile_changed": False, "app_model_promoted": False,
        "protected_inputs": len(before), "all_input_sha256_size_mtime_preserved": True,
        "protected_snapshot_before_sha256": snapshot_sha256(before),
        "protected_snapshot_after_sha256": snapshot_sha256(after),
        "protected_input_sha256": public_input_hashes(before),
        "input_sha256": {key: before[name]["sha256"] for key, name in DIRECT.items()},
        "source_sha256": {name: before[name]["sha256"] for name in SOURCE_PATHS[:2]},
        "test_source_sha256": {SOURCE_PATHS[2]: before[SOURCE_PATHS[2]]["sha256"]},
        "focused_test_record_sha256": sha(ROOT / TEST_RECORD), "focused_tests_passed": EXPECTED_TESTS,
        "private_draw_archive_path": archive.relative_to(ROOT).as_posix(),
        "private_draw_archive_sha256": sha(archive), "private_draw_archive_shape_each_arm": [EPOCHS, DRAWS],
        "individual_draw_row_indices_or_photo_ids_published": False,
        "numerical_repair_note": numerical_repair_note,
        "limitations": ["This is a sampler dry-run and supplies no new VAL performance or causal error explanation.",
                        "Exact Crack/Spalling/source/full-crop preservation does not preserve other photo or auxiliary label exposure.",
                        "Original loss weights remain fixed even when other-class exposure shifts; future AP regression checks are mandatory."]}
    write_new(REPORT, report)
    require(snapshots(paths) == before, "Protected inputs changed while saving the new plan")
    print(json.dumps({"status": report["status"], "epochs": EPOCHS, "draws_each": DRAWS,
        "eligible_parents": int(eligible.sum()), "changed_positions": report["total_changed_positions"],
        "eligible_draws_control": report["total_eligible_draws_control"],
        "eligible_draws_treatment": report["total_eligible_draws_treatment"],
        "actual_control_order_hashes_match": True, "training_epochs_budget_spent": 0,
        "protected_inputs": len(before)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only:
        run_tests()
    else:
        prepare()


if __name__ == "__main__":
    main()
