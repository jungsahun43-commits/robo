"""Reporting-only AP support normalization, using synthetic saved-VAL fixtures."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest

from scripts import report_facility_native_roi_results as adapter


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class NativeROIReportAdapterTests(unittest.TestCase):
    def entry(self):
        labels = ("concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
                  "wet_surface", "efflorescence", "surface_cavity")
        supports = {"dacl": 710, "damsegment": 424, "codebrim": 611}
        known = {"dacl": set(range(7)), "damsegment": {0, 1}, "codebrim": {0, 1, 2, 3, 5}}
        ranking = {}
        for domain, support in supports.items():
            ranking[domain] = {}
            for index, label in enumerate(labels):
                asserted = index in known[domain]
                ranking[domain][label] = {"ap": .75 if asserted else None,
                    "known_photos": support if asserted else 0,
                    "positive_photos": 12 if asserted else 0,
                    "status": "measured_on_asserted_source_labels" if asserted
                        else "not_measured_source_labels_unknown"}
        # The original loader sums saved float 0/1 targets, yielding exact
        # integer-valued floats even though these are photo support counts.
        ranking["dacl"]["concrete_crack"]["positive_photos"] = sum([1.] * 211 + [0.] * 499)
        ranking["damsegment"]["wet_surface"]["positive_photos"] = sum([0.] * 424)
        return {"run": "synthetic-run", "ranking_ap": ranking,
            "weights_sha256": "a" * 64, "worst_error": .2, "target_passed": False,
            "per_class": {"concrete_crack": {"threshold": .3,
                "domains": {"dacl": {"tp": 200, "fn": 11, "fp": 2, "tn": 497}}}},
            "small_dacl_polygon_area_below_one_percent": {
                "concrete_crack": {"positive_photos": 93, "false_negatives": 11, "fnr": 11 / 93}},
            "provenance": {"saved_validation_sha256": "b" * 64}}

    def test_exact_float_target_sums_normalize_only_ap_support_and_preserve_input(self):
        entry = self.entry()
        before = canonical(entry)
        normalized, conversions = adapter.normalize_ap_positive_support(entry)
        self.assertEqual(conversions, 2)
        self.assertIsNot(normalized, entry)
        self.assertEqual(canonical(entry), before)
        self.assertIs(type(entry["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"]), float)
        self.assertIs(type(normalized["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"]), int)
        self.assertEqual(normalized["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"], 211)
        self.assertIs(type(normalized["ranking_ap"]["damsegment"]["wet_surface"]["positive_photos"]), int)
        self.assertEqual(normalized["ranking_ap"]["damsegment"]["wet_surface"]["positive_photos"], 0)
        self.assertIsNone(normalized["ranking_ap"]["damsegment"]["wet_surface"]["ap"])
        expected = copy.deepcopy(entry)
        expected["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"] = 211
        expected["ranking_ap"]["damsegment"]["wet_surface"]["positive_photos"] = 0
        self.assertEqual(canonical(normalized), canonical(expected))

    def test_invalid_positive_or_known_support_is_rejected(self):
        for invalid in (True, float("nan"), float("inf"), 211.5, -1, 711, "211.0"):
            entry = self.entry()
            entry["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"] = invalid
            with self.subTest(positive_support=repr(invalid)), self.assertRaises(ValueError):
                adapter.normalize_ap_positive_support(entry)
        for invalid in (True, float("nan"), float("inf"), 710., -1, "710", None):
            entry = self.entry()
            entry["ranking_ap"]["dacl"]["concrete_crack"]["known_photos"] = invalid
            with self.subTest(known_support=repr(invalid)), self.assertRaises(ValueError):
                adapter.normalize_ap_positive_support(entry)

    def test_original_loader_validation_precedes_normalization_and_patch_restores_on_error(self):
        calls = []
        def rejecting_loader(name):
            calls.append(name)
            raise ValueError("original saved-score validation failed")
        with self.assertRaisesRegex(ValueError, "original saved-score validation failed"):
            adapter.adapted_load_run(rejecting_loader, "synthetic-run")
        self.assertEqual(calls, ["synthetic-run"])
        training = {"status": "complete", "actual_epochs": 6,
                    "weights_sha256": "a" * 64, "fixed_recipe": {"labels": 7}}
        entry = self.entry()
        before = canonical((training, entry))
        def original_loader(name):
            calls.append(name)
            return training, entry
        returned_training, normalized, conversions = adapter.adapted_load_run(original_loader, "synthetic-run")
        self.assertIs(returned_training, training)
        self.assertEqual(conversions, 2)
        self.assertIs(type(normalized["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"]), int)
        module = SimpleNamespace(load_run=original_loader)
        normalizations = []
        with self.assertRaisesRegex(RuntimeError, "synthetic report failure"):
            with adapter.patched_load_run(module, normalizations):
                self.assertIsNot(module.load_run, original_loader)
                patched_training, patched_entry = module.load_run("synthetic-run")
                self.assertIs(patched_training, training)
                self.assertIs(type(patched_entry["ranking_ap"]["dacl"]["concrete_crack"]["positive_photos"]), int)
                raise RuntimeError("synthetic report failure")
        self.assertIs(module.load_run, original_loader)
        self.assertEqual(normalizations, [{"run": "synthetic-run", "normalized_support_fields": 2}])
        self.assertEqual(canonical((training, entry)), before)

    def test_reporting_provenance_preserves_metrics_labels_gates_and_records_conversion_count(self):
        entry, converted = adapter.normalize_ap_positive_support(self.entry())
        result = {"experiments": [entry], "classes": ["concrete_crack", "concrete_spalling"],
            "comparisons": {"maximum_error_treatment_minus_control_pp": -.2,
                "ranking_ap_changes": [{"class": "concrete_crack", "treatment_minus_control_ap": .01}]},
            "research_gate": {"research_candidate_nominated": False, "maximum_other_ap_drop": .02},
            "error_rows": [{"class": "concrete_crack", "fn": 11, "positive_photos": 211}],
            "labels": {"concrete_crack": "표면 균열 의심"},
            "deployed": False, "source_test_inference_executed": False}
        normalizations = [{"run": "synthetic-run", "normalized_support_fields": converted},
                          {"run": "synthetic-second", "normalized_support_fields": 0}]
        evidence = {"adapter_source_sha256": "a" * 64}
        before = canonical((result, normalizations, evidence))
        expected_digest = hashlib.sha256(json.dumps(result, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        updated = adapter.add_reporting_provenance(result, normalizations, evidence)
        self.assertEqual(canonical((result, normalizations, evidence)), before)
        self.assertIsNot(updated, result)
        self.assertNotIn("reporting_provenance", result)
        self.assertEqual(set(updated), set(result) | {"reporting_provenance"})
        self.assertEqual(canonical({key: value for key, value in updated.items()
                                   if key != "reporting_provenance"}), canonical(result))
        provenance = updated["reporting_provenance"]
        self.assertEqual(provenance["normalized_support_fields"], converted)
        self.assertEqual(provenance["normalizations_by_run"], normalizations)
        self.assertEqual(provenance["adapter_source_sha256"], evidence["adapter_source_sha256"])
        self.assertEqual(provenance["original_metric_payload_sha256"], expected_digest)
        self.assertEqual(provenance["metric_payload_sha256_after_reporting_metadata"], expected_digest)
        self.assertIs(provenance["metrics_and_labels_unchanged"], True)
        self.assertIs(provenance["confusion_ap_threshold_gate_code_unchanged"], True)
        self.assertIs(provenance["training_recipe_changed"], False)


if __name__ == "__main__":
    unittest.main()
