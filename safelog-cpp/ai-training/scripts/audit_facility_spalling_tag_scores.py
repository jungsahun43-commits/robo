"""Describe frozen DACL TRAIN spalling errors by original native photo tags.

Reads existing full-TRAIN predictions and metadata only. No image, native annotation,
VAL/TEST prediction, inference, target editing, or training is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_facility_train_review import CLASSES, verify_cache

INITIAL_SHA = "0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a"
APP_PROFILE_SHA = "ecc09958022c659ef37385d1633177c034d27cda8034cadc3ab4af2ab8abb035"
AUX_CLASSES = ("ACrack", "Bearing", "Cavity", "Crack", "Drainage", "EJoint",
               "Efflorescence", "ExposedRebars", "Graffiti", "Hollowareas", "JTape",
               "PEquipment", "Restformwork", "Rockpocket", "Rust", "Spalling",
               "WConccor", "Weathering", "Wetspot")
OTHER_FOUR = ("Rockpocket", "WConccor", "Hollowareas", "Cavity")
CUTOFF = 0.4649081826210022
TEST_NAME = "test_spalling_tag_scores"
INPUTS = {
    "cache": "runs/facility-roi-0773-train-review-predictions.json",
    "core": "data/facility-spatial-training/train.json",
    "auxiliary": "data/facility-auxiliary-training/train.json",
    "package": "runs/facility-train-review-roi-0773/TRAIN-REVIEW.json",
    "weights": "runs/facility-presence-target-roi-control/best.pt",
    "selection": "runs/facility-presence-target-roi-control/TARGET-SELECTION.json",
    "split": "runs/facility-presence-target-roi-control/SPLIT.json",
    "source_audit": "reports/facility-resolution-source-audit.json",
    "label_groups": "reports/facility-train-label-groups.json",
    "app_profile": "reports/facility-inference-profile.json",
    "audit_source": "scripts/audit_facility_spalling_tag_scores.py",
    "audit_tests": "tests/test_spalling_tag_scores.py",
    "cache_verifier_source": "scripts/build_facility_train_review.py",
}
REPORT = ROOT / "reports/facility-spalling-tag-score-audit.json"
TEST_RECORD = ROOT / "reports/facility-spalling-tag-score-tests.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot():
    return {key: {"sha256": sha(ROOT / name), "size": (ROOT / name).stat().st_size,
                  "mtime_ns": (ROOT / name).stat().st_mtime_ns}
            for key, name in INPUTS.items()}


def write(path, value):
    require(not path.exists(), "Preserve existing diagnostic evidence; refusing overwrite")
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def labels(value, size, allow_unknown=False):
    valid = (-1, 0, 1) if allow_unknown else (0, 1)
    require(isinstance(value, (list, tuple)) and len(value) == size
            and all(type(v) is int and v in valid for v in value), "Invalid original tag/target vector")


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    result = {}
    for name, fraction in (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1)):
        index = (len(ordered) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        result[name] = ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return result


def describe(rows):
    counts = {key: 0 for key in ("TP", "FN", "FP", "TN")}
    for row in rows:
        counts[row["outcome"]] += 1
    positive, negative = counts["TP"] + counts["FN"], counts["FP"] + counts["TN"]
    return {"photos": len(rows), "known_positive": positive, "known_negative": negative,
            "unknown_photo_targets": 0, "confusion": counts,
            "descriptive_train_fnr": counts["FN"] / positive if positive else None,
            "descriptive_train_fpr": counts["FP"] / negative if negative else None,
            "score_quantiles": {"all": quantiles([r["score"] for r in rows]),
                                "known_positive": quantiles([r["score"] for r in rows if r["target"]]),
                                "known_negative": quantiles([r["score"] for r in rows if not r["target"]]),
                                "FN": quantiles([r["score"] for r in rows if r["outcome"] == "FN"]),
                                "FP": quantiles([r["score"] for r in rows if r["outcome"] == "FP"])}}


def summarize(known_train_items, cached_scores_map, aux_by_image, aux_classes, cutoff):
    """Pure aggregate of exactly matching DACL TRAIN inputs; never returns identifiers.

    `aux_by_image` maps each image to its unchanged auxiliary manifest item.
    Full non-DACL TRAIN rows can be present in `known_train_items` and score map;
    auxiliary membership must equal the DACL full-photo subset exactly.
    """
    require(tuple(aux_classes) == AUX_CLASSES, "Original nineteen-class order changed")
    require(type(cutoff) in (float, int) and math.isfinite(cutoff) and 0 <= cutoff <= 1,
            "Invalid frozen cutoff")
    require(isinstance(known_train_items, list) and known_train_items, "Empty TRAIN population")
    images = [item.get("image") for item in known_train_items]
    require(all(isinstance(i, str) and i for i in images) and len(images) == len(set(images)),
            "Duplicate or missing full TRAIN membership")
    require(set(cached_scores_map) == set(images), "Scoring membership must equal full TRAIN")
    dacl = [item for item in known_train_items if item.get("domain") == "dacl"]
    require(set(aux_by_image) == {item["image"] for item in dacl}, "Auxiliary membership mismatch")
    require(dacl, "No DACL TRAIN photos")
    for item in known_train_items:
        require(item.get("domain") in ("dacl", "damsegment", "codebrim"), "Unknown source domain")
        require(item.get("split", "train") == "train" and item.get("target_split", "train") == "train"
                and "parent_image" not in item, "Expected original full TRAIN rows only")
        labels(item["targets"], len(CLASSES), allow_unknown=True)
        scores = cached_scores_map[item["image"]]
        require(isinstance(scores, (list, tuple)) and len(scores) == len(CLASSES)
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in scores),
                "Invalid seven-class cached probabilities")
    indices = {tag: AUX_CLASSES.index(tag) for tag in AUX_CLASSES}
    rows = []
    for item in dacl:
        aux = aux_by_image[item["image"]]
        require(aux.get("image") == item["image"] and aux.get("domain") == "dacl"
                and aux.get("split") == "train", "Auxiliary item identity/split changed")
        labels(aux.get("targets"), len(AUX_CLASSES))
        target = item["targets"][1]
        require(target in (0, 1) and target == aux["targets"][indices["Spalling"]],
                "Original photo Spalling target differs from native tag")
        score = cached_scores_map[item["image"]][1]
        pred = score >= cutoff
        outcome = ("TP" if pred else "FN") if target else ("FP" if pred else "TN")
        tags = {tag for tag in AUX_CLASSES if aux["targets"][indices[tag]]}
        rows.append({"target": target, "score": score, "outcome": outcome, "tags": tags,
                     "other": bool(tags.intersection(OTHER_FOUR)),
                     "crack": bool(tags.intersection(("Crack", "ACrack")))})
    disjoint = {}
    for target in (1, 0):
        for other in (True, False):
            name = f"spalling_{'positive' if target else 'negative'}_other_four_{'present' if other else 'absent'}"
            disjoint[name] = describe([r for r in rows if r["target"] == target and r["other"] == other])
    negative_other = [r for r in rows if not r["target"] and r["other"]]
    crack_partition = {f"crack_or_acrack_{'present' if crack else 'absent'}":
                       describe([r for r in negative_other if r["crack"] == crack]) for crack in (True, False)}
    overlapping = {tag: {"all": describe([r for r in rows if tag in r["tags"]]),
                         "spalling_positive": describe([r for r in rows if tag in r["tags"] and r["target"]]),
                         "spalling_negative": describe([r for r in rows if tag in r["tags"] and not r["target"]])}
                   for tag in OTHER_FOUR}
    overall = describe(rows)
    require(sum(g["photos"] for g in disjoint.values()) == overall["photos"], "Disjoint photo accounting failed")
    require(all(sum(g["confusion"][k] for g in disjoint.values()) == overall["confusion"][k]
                for k in ("TP", "FN", "FP", "TN")), "Disjoint confusion accounting failed")
    return {"overall": overall, "disjoint_four_groups": disjoint,
            "negative_other_four_crack_partition": crack_partition,
            "non_disjoint_native_tags": overlapping,
            "group_policy": {"other_four": list(OTHER_FOUR), "disjoint_groups_partition_all_dacl": True,
                             "crack_partition_partitions_negative_other_four_only": True,
                             "native_tag_groups_overlap_do_not_sum": True,
                             "absence_is_not_normal_or_safe_ground_truth": True}}


def run_tests():
    require(not TEST_RECORD.exists(), "Existing focused test record must be preserved")
    before = snapshot()
    spec = importlib.util.spec_from_file_location(TEST_NAME, ROOT / INPUTS["audit_tests"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    collected = suite.countTestCases()
    require(collected == 6, "Expected all six focused tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = snapshot()
    passed = result.wasSuccessful() and result.testsRun == collected and not result.skipped and before == after
    record = {"schema": "facility_spalling_tag_score_tests_v1", "status": "passed" if passed else "failed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
              "expected_tests_collected": collected, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "all_input_sha256_size_mtime_preserved": before == after,
              "audit_source_sha256": before["audit_source"]["sha256"],
              "test_source_sha256": before["audit_tests"]["sha256"],
              "protected_input_sha256": {k: v["sha256"] for k, v in before.items()},
              "model_inference_performed": False, "training_performed": False}
    write(TEST_RECORD, record)
    require(passed, "Focused tests failed; preserve record and resolve before actual audit")
    print(json.dumps({"status": record["status"], "tests_run": result.testsRun}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.test_only:
        run_tests()
        return
    require(not REPORT.exists(), "Existing aggregate audit must be preserved")
    before = snapshot()
    tests = read(TEST_RECORD)
    require(tests.get("status") == "passed" and tests.get("tests_run") == 6
            and tests.get("expected_tests_collected") == 6
            and all(tests.get(k) == 0 for k in ("failures", "errors", "skipped"))
            and tests.get("all_input_sha256_size_mtime_preserved") is True,
            "Actual complete focused test evidence required")
    require(tests.get("protected_input_sha256") == {k: v["sha256"] for k, v in before.items()},
            "Protected source or input changed after focused tests")
    document = {key: read(ROOT / INPUTS[key]) for key in
                ("cache", "core", "auxiliary", "package", "selection", "source_audit", "label_groups")}
    core, aux, package, cache = (document[k] for k in ("core", "auxiliary", "package", "cache"))
    require(core.get("split") == "train" and core.get("classes") == CLASSES and core.get("full_count") == 14248,
            "Original full TRAIN manifest changed")
    originals = core["items"][:core["full_count"]]
    require(len(originals) == 14248, "Full TRAIN population incomplete")
    canonical = deepcopy(package)
    content_sha = canonical.pop("package_content_sha256", None)
    require(content_sha == hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "TRAIN review package canonical content changed")
    require(package.get("schema") == "facility_train_review_v1" and package.get("split") == "train"
            and package.get("classes") == CLASSES and package.get("weights_sha256") == INITIAL_SHA
            and package.get("original_full_train_count") == 14248 and package.get("scored_cache_count") == 14248
            and package.get("detail_rows_not_reviewed") == 0, "Unexpected TRAIN review package scope")
    require(before["weights"]["sha256"] == INITIAL_SHA, "Immutable0773 model bytes changed")
    require(before["app_profile"]["sha256"] == APP_PROFILE_SHA, "Existing app profile bytes changed")
    for key, field in (("cache", "cache_sha256"), ("core", "manifest_sha256"), ("selection", "thresholds_sha256")):
        require(before[key]["sha256"] == package["provenance"][field], "Package input provenance changed")
    require(before["cache_verifier_source"]["sha256"] == package["provenance"]["script_sha256"],
            "Original package builder source changed")
    require(cache.get("manifest_sha256") == before["core"]["sha256"]
            and cache.get("split_sha256") == before["split"]["sha256"]
            and cache.get("classes") == CLASSES and cache.get("run") == "facility-presence-target-roi-control",
            "TRAIN cache original-model provenance changed")
    require(cache.get("images") == [i["image"] for i in originals], "Cache full TRAIN order changed")
    scores = verify_cache(cache, originals, CLASSES, INITIAL_SHA)
    original_by_image = {i["image"]: i for i in originals}
    cases = package.get("cases", [])
    require(len(cases) == 130 and len({i["image"] for i in cases}) == 130,
            "Original review case membership changed")
    for case in cases:
        item = original_by_image.get(case.get("image"))
        require(item is not None and case.get("domain") == item["domain"]
                and case.get("original_targets") == item["targets"]
                and case.get("probabilities") == scores[item["image"]],
                "Review case target/domain/score differs from immutable full TRAIN cache")
    selection = document["selection"]
    require(selection.get("weights_sha256") == INITIAL_SHA and selection.get("classes") == CLASSES
            and selection.get("selection_split") == "val" and selection["selected"]["grid"] == 1
            and selection["selected"]["per_class"]["concrete_spalling"]["threshold"] == CUTOFF
            and package["thresholds"]["concrete_spalling"] == CUTOFF,
            "Frozen previously selected grid1 cutoff changed")
    source = document["source_audit"]
    groups = document["label_groups"]
    require(source.get("status") == "complete" and source.get("train_photos") == 6225
            and tuple(source["native_classes"]) == AUX_CLASSES
            and source["manifest_sha256"]["spatial"] == before["core"]["sha256"]
            and source["manifest_sha256"]["auxiliary"] == before["auxiliary"]["sha256"],
            "Bound original source audit mismatch")
    require(aux.get("split") == "train" and tuple(aux.get("classes", ())) == AUX_CLASSES
            and len(aux.get("items", [])) == 6225, "Original auxiliary TRAIN manifest changed")
    aux_by_image = {i["image"]: i for i in aux["items"]}
    require(len(aux_by_image) == 6225, "Duplicate auxiliary TRAIN membership")
    aggregate = summarize(originals, scores, aux_by_image, aux["classes"], CUTOFF)
    require(aggregate["overall"]["photos"] == 6225, "Unexpected full DACL population")
    cohorts = [i for i in package["sampling"] if i["domain"] == "dacl" and i["task"] == "concrete_spalling"]
    expected = {i["outcome"]: i["available"] for i in cohorts}
    require(len(cohorts) == 4 and set(expected) == {"TP", "FN", "FP", "TN"}
            and expected == aggregate["overall"]["confusion"], "Cache errors differ from original TRAIN sampling census")
    require(groups.get("schema") == "facility_train_label_groups_v1" and groups.get("status") == "audited"
            and groups.get("train_photos") == 6225 and groups.get("native_tag_count") == 19
            and groups["source_manifest_sha256"]["spatial"] == before["core"]["sha256"]
            and groups["source_manifest_sha256"]["auxiliary"] == before["auxiliary"]["sha256"],
            "Original label composition audit provenance changed")
    strata = groups["sampling_strata_photo_counts"]
    disjoint = aggregate["disjoint_four_groups"]
    for group, key in (("spalling_positive_other_four_absent", "Spalling_without_other_four"),
                       ("spalling_positive_other_four_present", "Spalling_with_any_other_four"),
                       ("spalling_negative_other_four_present", "related_other_four_without_Spalling")):
        require(disjoint[group]["photos"] == strata[key], "Original composition stratum changed")
    require(aggregate["negative_other_four_crack_partition"]["crack_or_acrack_absent"]["photos"]
            == strata["related_other_four_without_Spalling_or_crack"], "Crack-negative source composition changed")
    for tag in OTHER_FOUR:
        require(aggregate["non_disjoint_native_tags"][tag]["all"]["photos"]
                == groups["native_tag_photo_counts"][tag], "Original native tag count changed")
    after = snapshot()
    require(before == after, "Protected input/source bytes or metadata changed during audit")
    report = {"schema": "facility_spalling_tag_score_audit_v1", "status": "audited",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Frozen0773 full DACL TRAIN descriptive error census by unchanged native photo tags; not new generalization performance",
              "classes": CLASSES, "native_classes": list(AUX_CLASSES), "model_weights_sha256": INITIAL_SHA,
              "frozen_grid1_spalling_cutoff": CUTOFF, "full_train_scoring_cache_rows": 14248,
              "dacl_train_photos": 6225, **aggregate,
              "full_train_cache_targets_order_membership_and_domains_verified": True,
              "existing130_review_cases_match_original_full_train_cache": True,
              "native_spalling_photo_target_matches_original_seven_class_target": True,
              "overall_confusion_matches_original_train_review_sampling_census": True,
              "group_composition_matches_bound_original_source_audit": True,
              "all_input_sha256_size_mtime_preserved": True,
              "input_sha256": {k: v["sha256"] for k, v in before.items()},
              "focused_test_record_sha256": sha(TEST_RECORD), "focused_tests_passed": 6,
              "policy": {"uses_existing_full_train_predictions": True, "new_model_inference_performed": False,
                         "source_images_or_native_annotations_read": False,
                         "individual_validation_or_test_images_annotations_predictions_read": False,
                         "new_cutoff_fitting_performed": False, "original_labels_or_app_profile_changed": False,
                         "training_performed": False, "individual_photo_ids_paths_targets_or_predictions_published": False,
                         "model_promotion_performed": False, "causal_error_explanation_proven": False},
              "limitations": ["TRAIN predictions use a model trained on these sources; these rates do not estimate unseen factory performance.",
                              "Native tag coexistence is not polygon overlap, label error or proof that a tag caused an error.",
                              "Non-disjoint native-tag groups overlap and their counts must not be added.",
                              "Tag absence is a publisher label statement, not normality, all-defect absence or structural safety.",
                              "This census avoids the hard-plus-random selection of the local130 review but remains limited to existing DACL TRAIN photos.",
                              "All original photo/annotation verification is reused from the bound source audit; no image or annotation is newly read."]}
    write(REPORT, report)
    print(json.dumps({"status": "audited", "dacl_photos": 6225, "confusion": aggregate["overall"]["confusion"],
                      "groups": {k: {"photos": v["photos"], "confusion": v["confusion"],
                                     "train_fnr": v["descriptive_train_fnr"], "train_fpr": v["descriptive_train_fpr"]}
                                 for k, v in disjoint.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
