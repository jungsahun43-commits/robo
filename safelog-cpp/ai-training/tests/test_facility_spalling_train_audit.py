"""Synthetic checks for the read-only TRAIN spalling audit."""
from copy import deepcopy
import unittest

import numpy as np

from safelog_ai.review_feedback import CLASSES, canonical_sha256
from safelog_ai.spalling_geometry import SourceRaster, unavailable_source_raster
from scripts.audit_facility_spalling_train import cohort_rows, replay_supervision


WEIGHTS_SHA256 = "0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a"


def signed(package):
    result = deepcopy(package)
    result.pop("package_content_sha256", None)
    result["package_content_sha256"] = canonical_sha256(result)
    return result


def synthetic_cohort():
    """A tiny selected cohort; its counts are not population error rates."""
    thresholds = {CLASSES[0]: 0.6, CLASSES[1]: 0.5}
    cases, originals, scores = [], [], {}
    specifications = (
        ("fp", "dacl", 0, 0.9, CLASSES[1]),
        ("fn", "damsegment", 1, 0.2, CLASSES[1]),
        ("tp", "codebrim", 1, 0.5, CLASSES[0]),
        ("tn", "dacl", 0, 0.49, CLASSES[0]),
        ("unknown", "damsegment", -1, 0.9, CLASSES[0]),
    )
    for name, domain, spalling_target, probability, selected_task in specifications:
        image = f"data/facility/train/{name}.jpg"
        targets = [0, spalling_target, -1, -1, -1, -1, -1]
        probabilities = [0.1, probability, 0.3, 0.4, 0.1, 0.2, 0.05]
        original = {"image": image, "domain": domain, "targets": targets,
                    "pixel_target": f"data/facility/masks/{name}.npz"}
        index = CLASSES.index(selected_task)
        predicted = int(probabilities[index] >= thresholds[selected_task])
        outcome = ("TP" if predicted else "FN") if targets[index] else (
            "FP" if predicted else "TN")
        cases.append({
            "case_id": f"case-{name}", "image": image, "domain": domain,
            "original_targets": list(targets), "probabilities": list(probabilities),
            "selected_for": [{"task": selected_task, "outcome": outcome,
                              "probability": probabilities[index],
                              "threshold": thresholds[selected_task]}],
        })
        originals.append(original)
        scores[image] = list(probabilities)
    package = signed({"schema": "facility_train_review_v1", "split": "train",
                      "classes": list(CLASSES), "weights_sha256": WEIGHTS_SHA256,
                      "thresholds": thresholds, "cases": cases})
    return package, originals, scores


def synthetic_source():
    fine = np.zeros((7, 640, 640), dtype=bool)
    for index, coordinate in ((0, 0), (2, 8), (4, 16)):
        fine[index, coordinate, coordinate] = True
    coarse = fine.reshape(7, 80, 8, 80, 8).max((2, 4))
    union = fine.any(axis=0)
    return SourceRaster("synthetic", fine, coarse,
                        np.array([True] * 5 + [False] * 2), {}, union, ~union)


class SpallingTrainAuditTests(unittest.TestCase):
    def test_cohort_classifies_all_outcomes_and_distinguishes_direct_selection(self):
        package, originals, scores = synthetic_cohort()
        before = deepcopy((package, originals, scores))
        rows = cohort_rows(package, originals, scores)
        self.assertEqual([row["outcome"] for row in rows], ["FP", "FN", "TP", "TN", "UNKNOWN"])
        self.assertEqual([row["direct_spalling_selection"] for row in rows],
                         [True, True, False, False, False])
        self.assertEqual(sum(row["direct_spalling_selection"] for row in rows), 2)
        self.assertEqual(len(rows), 5)
        for row, case, original in zip(rows, package["cases"], originals):
            self.assertEqual(row["case_id"], case["case_id"])
            self.assertEqual(row["image"], case["image"])
            self.assertEqual(row["domain"], case["domain"])
            self.assertEqual(row["original_target"], original["targets"][1])
            self.assertEqual(row["probability"], scores[case["image"]][1])
            self.assertEqual(row["threshold"], 0.5)
            self.assertEqual(row["case"], case)
            self.assertEqual(row["item"], original)
            self.assertIsNot(row["case"], case)
            self.assertIsNot(row["item"], original)
        self.assertEqual((package, originals, scores), before)
        # Returned nested storage must not allow callers to relabel original inputs.
        rows[0]["case"]["original_targets"][1] = 1
        rows[0]["item"]["targets"][1] = 1
        self.assertEqual((package, originals, scores), before)

    def test_cohort_rejects_membership_target_domain_and_cache_mismatch(self):
        scenarios = (
            "duplicate_case_id", "duplicate_image", "missing_original", "duplicate_original",
            "target_mismatch", "domain_mismatch", "missing_score", "extra_score",
            "score_mismatch", "score_shape", "score_nonfinite", "score_bool",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                package, originals, scores = synthetic_cohort()
                first_image = originals[0]["image"]
                if scenario == "duplicate_case_id":
                    package["cases"][1]["case_id"] = package["cases"][0]["case_id"]
                elif scenario == "duplicate_image":
                    duplicate = deepcopy(package["cases"][0])
                    duplicate["case_id"] = "case-duplicate-image"
                    package["cases"].append(duplicate)
                elif scenario == "missing_original":
                    package["cases"][0]["image"] = "data/facility/val/heldout.jpg"
                elif scenario == "duplicate_original":
                    originals.append(deepcopy(originals[0]))
                elif scenario == "target_mismatch":
                    package["cases"][0]["original_targets"][1] = 1
                elif scenario == "domain_mismatch":
                    package["cases"][0]["domain"] = "codebrim"
                elif scenario == "missing_score":
                    del scores[first_image]
                elif scenario == "extra_score":
                    scores["data/facility/val/heldout.jpg"] = [0.1] * 7
                elif scenario == "score_mismatch":
                    scores[first_image][6] = 0.7
                elif scenario == "score_shape":
                    scores[first_image] = [0.1] * 6
                elif scenario == "score_nonfinite":
                    scores[first_image][2] = float("nan")
                elif scenario == "score_bool":
                    scores[first_image][2] = False
                package = signed(package)
                with self.assertRaises(ValueError):
                    cohort_rows(package, originals, scores)

    def test_cohort_rejects_package_and_selection_tampering(self):
        scenarios = (
            "hash", "split", "classes", "weights", "threshold_bool", "threshold_missing",
            "selection_task", "selection_outcome", "selection_probability",
            "selection_threshold", "unknown_selection", "bool_target", "case_probability_shape",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                package, originals, scores = synthetic_cohort()
                selection = package["cases"][0]["selected_for"][0]
                if scenario == "hash":
                    package["package_content_sha256"] = "f" * 64
                elif scenario == "split":
                    package["split"] = "val"
                elif scenario == "classes":
                    package["classes"] = list(reversed(CLASSES))
                elif scenario == "weights":
                    package["weights_sha256"] = "a" * 64
                elif scenario == "threshold_bool":
                    package["thresholds"][CLASSES[1]] = True
                elif scenario == "threshold_missing":
                    del package["thresholds"][CLASSES[1]]
                elif scenario == "selection_task":
                    selection["task"] = CLASSES[2]
                elif scenario == "selection_outcome":
                    selection["outcome"] = "TP"
                elif scenario == "selection_probability":
                    selection["probability"] = 0.8
                elif scenario == "selection_threshold":
                    selection["threshold"] = 0.4
                elif scenario == "unknown_selection":
                    package["cases"][-1]["selected_for"].append(
                        {"task": CLASSES[1], "outcome": "FP", "probability": 0.9, "threshold": 0.5})
                elif scenario == "bool_target":
                    package["cases"][0]["original_targets"][0] = False
                elif scenario == "case_probability_shape":
                    package["cases"][0]["probabilities"] = [0.1] * 6
                if scenario != "hash":
                    package = signed(package)
                with self.assertRaises(ValueError):
                    cohort_rows(package, originals, scores)

    def test_replay_keeps_unknown_distinct_from_asserted_negative_without_source(self):
        source = unavailable_source_raster()
        targets = [0, -1, 1, 0, -1, 1, 0]
        masks, known, conflicts = replay_supervision(source, targets)
        self.assertEqual(masks.shape, (7, 80, 80))
        self.assertEqual(masks.dtype, np.uint8)
        self.assertEqual(known.shape, (7,))
        self.assertEqual(known.dtype, np.uint8)
        np.testing.assert_array_equal(known, [1, 0, 0, 1, 0, 0, 1])
        self.assertFalse(masks.any())
        self.assertEqual(conflicts, [])
        self.assertEqual(targets, [0, -1, 1, 0, -1, 1, 0])

    def test_replay_clears_conflicting_supervision_without_changing_targets(self):
        source = synthetic_source()
        source_before = deepcopy(source)
        targets = [0, 1, 1, 0, -1, 1, 0]
        masks, known, conflicts = replay_supervision(source, targets)
        self.assertEqual(conflicts, [0, 1])
        np.testing.assert_array_equal(known, [0, 0, 1, 1, 0, 0, 1])
        expected = np.zeros((7, 80, 80), dtype=np.uint8)
        expected[2, 1, 1] = 1
        np.testing.assert_array_equal(masks, expected)
        self.assertEqual(targets, [0, 1, 1, 0, -1, 1, 0])
        for attribute in ("fine_masks", "any_masks", "class_available", "annotation_union", "source_unmarked"):
            np.testing.assert_array_equal(getattr(source, attribute), getattr(source_before, attribute))
        masks[2, 1, 1] = 0
        self.assertTrue(source.any_masks[2, 1, 1])

    def test_replay_rejects_invalid_targets_and_corrupted_source_arrays(self):
        source = unavailable_source_raster()
        for targets in ([0] * 6, [0] * 8, [False] * 7, [0] * 6 + [2],
                        [0] * 6 + [0.0], tuple([0] * 7)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                replay_supervision(source, targets)
        with self.assertRaises(ValueError):
            replay_supervision(object(), [0] * 7)
        corruptions = (
            ("any_masks", np.zeros((7, 79, 80), dtype=bool)),
            ("any_masks", np.full((7, 80, 80), 2, dtype=np.uint8)),
            ("class_available", np.zeros(6, dtype=bool)),
        )
        for attribute, value in corruptions:
            invalid_source = deepcopy(source)
            object.__setattr__(invalid_source, attribute, value)
            with self.subTest(attribute=attribute, shape=value.shape), self.assertRaises(ValueError):
                replay_supervision(invalid_source, [0] * 7)


if __name__ == "__main__":
    unittest.main()
