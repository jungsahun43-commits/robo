"""Focused tests for a metadata-only frozen TRAIN tag/error census."""
from copy import deepcopy
import math
import unittest

from scripts.audit_facility_spalling_tag_scores import AUX_CLASSES, CLASSES, summarize


class SpallingTagScoresTests(unittest.TestCase):
    def fixture(self):
        specs = [(1, .2, ("Spalling", "Rockpocket", "Cavity")),
                 (1, .8, ("Spalling",)),
                 (0, .8, ("Rockpocket", "Cavity", "Crack")),
                 (0, .8, ("Hollowareas",)),
                 (0, .2, ("WConccor", "ACrack")),
                 (0, .2, ())]
        items, scores, aux = [], {}, {}
        for k, (target, score, tags) in enumerate(specs):
            image = f"train-{k}"
            items.append({"image": image, "targets": [0, target, 0, 0, 0, 0, 0], "domain": "dacl"})
            scores[image] = [0., score, 0., 0., 0., 0., 0.]
            aux[image] = {"image": image, "domain": "dacl", "split": "train",
                          "targets": [int(tag in tags) for tag in AUX_CLASSES]}
        return items, scores, aux

    def test_full_census_partitions_do_not_inherit_review_sampling_bias(self):
        items, scores, aux = self.fixture()
        result = summarize(items, scores, aux, AUX_CLASSES, .5)
        self.assertEqual(result["overall"]["photos"], 6)
        self.assertEqual(result["overall"]["confusion"], {"TP": 1, "FN": 1, "FP": 2, "TN": 2})
        groups = result["disjoint_four_groups"]
        self.assertEqual(sum(g["photos"] for g in groups.values()), 6)
        negative = groups["spalling_negative_other_four_present"]
        self.assertEqual(negative["photos"], 3)
        self.assertEqual(negative["descriptive_train_fpr"], 2 / 3)
        self.assertEqual(result["negative_other_four_crack_partition"]["crack_or_acrack_present"]["photos"], 2)
        self.assertEqual(result["negative_other_four_crack_partition"]["crack_or_acrack_absent"]["photos"], 1)

    def test_overlapping_tag_counts_are_explicit_and_not_added(self):
        items, scores, aux = self.fixture()
        result = summarize(items, scores, aux, AUX_CLASSES, .5)
        tags = result["non_disjoint_native_tags"]
        self.assertEqual(tags["Rockpocket"]["all"]["photos"], 2)
        self.assertEqual(tags["Cavity"]["all"]["photos"], 2)
        self.assertGreater(sum(t["all"]["photos"] for t in tags.values()),
                           sum(g["photos"] for name, g in result["disjoint_four_groups"].items() if name.endswith("present")))
        self.assertIs(result["group_policy"]["native_tag_groups_overlap_do_not_sum"], True)

    def test_rejects_unknown_spalling_and_tampered_native_tags(self):
        for kind in ("unknown", "photo_mismatch", "aux_bool", "aux_float", "aux_unknown", "class_order"):
            with self.subTest(kind=kind):
                items, scores, aux = self.fixture()
                classes = AUX_CLASSES
                if kind == "unknown": items[0]["targets"][1] = -1
                if kind == "photo_mismatch": aux["train-0"]["targets"][AUX_CLASSES.index("Spalling")] = 0
                if kind == "aux_bool": aux["train-0"]["targets"][0] = False
                if kind == "aux_float": aux["train-0"]["targets"][0] = 0.
                if kind == "aux_unknown": aux["train-0"]["targets"][0] = -1
                if kind == "class_order": classes = tuple(reversed(AUX_CLASSES))
                with self.assertRaises(ValueError): summarize(items, scores, aux, classes, .5)

    def test_rejects_invalid_probabilities_cutoff_or_membership(self):
        for kind in ("nan", "infinite", "over_one", "boolean_score", "short_score", "missing_score",
                     "extra_score", "missing_aux", "extra_aux", "duplicate", "heldout", "crop", "boolean_cutoff"):
            with self.subTest(kind=kind):
                items, scores, aux = self.fixture()
                cutoff = .5
                if kind == "nan": scores["train-0"][1] = math.nan
                if kind == "infinite": scores["train-0"][1] = math.inf
                if kind == "over_one": scores["train-0"][1] = 1.01
                if kind == "boolean_score": scores["train-0"][1] = True
                if kind == "short_score": scores["train-0"].pop()
                if kind == "missing_score": del scores["train-0"]
                if kind == "extra_score": scores["extra"] = [0.] * 7
                if kind == "missing_aux": del aux["train-0"]
                if kind == "extra_aux": aux["extra"] = deepcopy(aux["train-0"])
                if kind == "duplicate": items.append(deepcopy(items[0]))
                if kind == "heldout": aux["train-0"]["split"] = "val"
                if kind == "crop": items[0]["parent_image"] = "other"
                if kind == "boolean_cutoff": cutoff = True
                with self.assertRaises(ValueError): summarize(items, scores, aux, AUX_CLASSES, cutoff)

    def test_inputs_original_labels_and_cached_scores_are_preserved(self):
        items, scores, aux = self.fixture()
        # Other sources have no native19 targets and stay outside DACL aggregates.
        items.append({"image": "code-patch", "domain": "codebrim", "targets": [0, 1, 0, 0, -1, 0, -1]})
        scores["code-patch"] = [.1] * len(CLASSES)
        before = deepcopy((items, scores, aux))
        result = summarize(items, scores, aux, AUX_CLASSES, .5)
        self.assertEqual((items, scores, aux), before)
        self.assertEqual(result["overall"]["photos"], 6)
        self.assertNotIn("train-", str(result))
        self.assertNotIn("code-patch", str(result))

    def test_cutoff_equality_empty_denominators_and_quantiles(self):
        items, scores, aux = self.fixture()
        scores["train-0"][1] = .5
        result = summarize(items, scores, aux, AUX_CLASSES, .5)
        positive = result["disjoint_four_groups"]["spalling_positive_other_four_present"]
        self.assertEqual(positive["confusion"]["TP"], 1)
        self.assertIsNone(positive["descriptive_train_fpr"])
        self.assertEqual(positive["score_quantiles"]["all"]["median"], .5)
        self.assertIsNone(positive["score_quantiles"]["FP"])
        negative = result["disjoint_four_groups"]["spalling_negative_other_four_present"]
        self.assertAlmostEqual(negative["score_quantiles"]["all"]["p10"], .32)
        self.assertIsNone(negative["descriptive_train_fnr"])


if __name__ == "__main__":
    unittest.main()
