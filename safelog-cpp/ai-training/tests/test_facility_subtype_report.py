"""Synthetic contract checks for subtype reporting; no inference or new results."""
from __future__ import annotations

import copy
import unittest

from scripts import plot_facility_subtype as plot
from scripts import report_facility_subtype as report
from scripts.report_facility_detail import GATE
from scripts.report_facility_discrimination import KNOWN_INDICES
from scripts.report_facility_small_region import CLASSES, COUNTS, DOMAINS, TARGETS


def fixture():
    """Fabricated finite aggregates for contract tests, never written as results."""
    names = list(plot.RUNS)
    protocol = {"reference": names[0], "control": names[1], "treatment": names[2],
                "research_candidate_gate": copy.deepcopy(GATE)}
    entries = []
    for index, name in enumerate(names):
        per_class, small, ranking = {}, {}, {}
        for task, small_n, small_misses in zip(TARGETS, (93, 105), ((29, 28, 25), (42, 40, 37))):
            domains = {}
            for domain in DOMAINS:
                positive = 211 if task == TARGETS[0] else 339
                if domain != "dacl":
                    positive = 200
                negative = COUNTS[domain] - positive
                fn = int(positive * (.20, .21, .18)[index])
                fp = int(negative * (.19, .22, .16)[index])
                domains[domain] = {"tp": positive-fn, "fn": fn, "fp": fp, "tn": negative-fp,
                    "positive_photos": positive, "negative_photos": negative,
                    "fnr": fn/positive, "fpr": fp/negative}
            per_class[task] = {"threshold": .45, "domains": domains}
            fn = small_misses[index]
            small[task] = {"positive_photos": small_n, "false_negatives": fn, "fnr": fn/small_n}
        for domain in DOMAINS:
            ranking[domain] = {}
            for k, task in enumerate(CLASSES):
                known = k in KNOWN_INDICES[domain]
                positive = per_class[task]["domains"][domain]["positive_photos"] if task in TARGETS else 12
                ranking[domain][task] = {"ap": .8 if known else None,
                    "known_photos": COUNTS[domain] if known else 0,
                    "positive_photos": positive if known else 0,
                    "status": "measured_on_asserted_source_labels" if known else "not_measured_source_labels_unknown"}
        worst = max(point[rate] for task in per_class.values() for point in task["domains"].values()
                    for rate in ("fnr", "fpr"))
        entries.append({"run": name, "selected_grid": 1, "configured_grids": [1], "test_executed": False,
            "test_result": None, "per_class": per_class, "worst_error": worst, "target_passed": False,
            "ranking_ap": ranking, "small_dacl_polygon_area_below_one_percent": small,
            "imgsz": 640, "actual_epochs": 12 if index == 0 else 6})
    return entries, protocol


def histories_fixture():
    # Actual predeclared aggregate plan, with fabricated optimizer counts only.
    plan = report.read(report.ROOT / "reports/facility-spalling-sampler-dry-run.json")
    protocol = {"requested_epochs": 6, "draws_per_epoch": 14248, "batch_size": 8}
    histories = [[], []]
    for expected in plan["epochs"]:
        for index, variant in enumerate(("control", "treatment")):
            order = expected[f"{variant}_order_sha256"]
            histories[index].append({"epoch": expected["epoch"], "sampled_row_indices_sha256": order,
                "sampled_domain_counts": copy.deepcopy(expected["domain_counts"]),
                "sampled_row_type_counts": copy.deepcopy(expected["full_crop_counts"]),
                "sampled_full_target_joint_counts": copy.deepcopy(expected["full_target_joint_counts"]),
                "subtype_sampling": {"eligible_draws": expected[f"eligible_draws_{variant}"],
                    "changed_positions_from_control": 0 if index == 0 else expected["changed_positions"],
                    "declared_draw_sha256": order, "draw_hash_matches_prepared": True},
                "optimizer_step_diagnostics": {"attempted_batches": 1781,
                    "actual_optimizer_steps": 1780, "amp_skipped_steps": 1}})
    return *histories, protocol, plan


class SubtypeReportTests(unittest.TestCase):
    def test_loader_checks_precede_representation_normalization(self):
        calls = []
        def fail(name):
            calls.append(name)
            raise ValueError("saved score provenance failed")
        with self.assertRaisesRegex(ValueError, "saved score provenance failed"):
            report.normalized_load_run("synthetic", fail)
        self.assertEqual(calls, ["synthetic"])
        entries, _ = fixture(); entry = entries[0]; before = copy.deepcopy(entry)
        entry["ranking_ap"]["dacl"][TARGETS[0]]["positive_photos"] = 211.
        before = copy.deepcopy(entry)
        training = {"weights_sha256": "a"*64}
        returned, normalized, count = report.normalized_load_run("synthetic", lambda _: (training, entry))
        self.assertIs(returned, training)
        self.assertEqual(count, 1)
        self.assertEqual(entry, before)
        self.assertIs(type(normalized["ranking_ap"]["dacl"][TARGETS[0]]["positive_photos"]), int)
        expected = copy.deepcopy(entry); expected["ranking_ap"]["dacl"][TARGETS[0]]["positive_photos"] = 211
        self.assertEqual(normalized, expected)

    def test_nonintegral_bool_nonfinite_and_out_of_bounds_support_rejected(self):
        for invalid in (True, -1, 211.5, float("nan"), float("inf"), 711., "211"):
            entries, _ = fixture()
            entries[0]["ranking_ap"]["dacl"][TARGETS[0]]["positive_photos"] = invalid
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                report.normalized_load_run("synthetic", lambda _: ({}, entries[0]))

    def test_exact_integral_saved_float_counts_pass_unchanged_strict_gate(self):
        entries, protocol = fixture()
        original = report.validate_measurements(entries, protocol)
        changed = copy.deepcopy(entries)
        for entry in changed:
            for rows in entry["ranking_ap"].values():
                for point in rows.values():
                    point["positive_photos"] = float(point["positive_photos"])
        adapted = [report.normalize_ap_positive_support(entry)[0] for entry in changed]
        self.assertEqual(report.validate_measurements(adapted, protocol), original)
        self.assertTrue(original["research_gate"]["research_candidate_nominated"])
        self.assertEqual(original["verified_known_class_ap_measurements"], 42)

    def test_measured_count_rate_and_scope_tampering_rejected(self):
        entries, protocol = fixture()
        for mutation in ("count", "rate", "scope", "headline"):
            changed = copy.deepcopy(entries)
            point = changed[2]["per_class"][TARGETS[0]]["domains"]["dacl"]
            if mutation == "count":
                point["fn"] += 1
            elif mutation == "rate":
                point["fnr"] += .01
            elif mutation == "scope":
                changed[2]["test_executed"] = True
            else:
                changed[2]["target_passed"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_measurements(changed, protocol)

    def test_unknown_ap_and_changed_small_support_rejected(self):
        entries, protocol = fixture()
        entries[2]["ranking_ap"]["damsegment"]["wet_surface"] = {
            "ap": .9, "known_photos": 424, "positive_photos": 0,
            "status": "measured_on_asserted_source_labels"}
        with self.assertRaises(ValueError):
            report.validate_measurements(entries, protocol)
        entries, protocol = fixture()
        entries[2]["small_dacl_polygon_area_below_one_percent"][TARGETS[0]]["positive_photos"] = 94
        with self.assertRaises(ValueError):
            report.validate_measurements(entries, protocol)

    def test_other_known_ap_and_small_damage_regression_guard_remains_active(self):
        entries, protocol = fixture()
        entries[2]["ranking_ap"]["dacl"]["rust_stain"]["ap"] -= .021
        self.assertFalse(report.validate_measurements(entries, protocol)["research_gate"]["research_candidate_nominated"])
        entries, protocol = fixture()
        point = entries[2]["small_dacl_polygon_area_below_one_percent"][TARGETS[0]]
        point.update(false_negatives=32, fnr=32/93)
        self.assertFalse(report.validate_measurements(entries, protocol)["research_gate"]["research_candidate_nominated"])

    def test_five_percent_boundary_is_not_a_strict_pass(self):
        entries, protocol = fixture()
        for entry in entries:
            for task in TARGETS:
                for domain in DOMAINS:
                    positive, negative = 200, COUNTS[domain]-200
                    entry["per_class"][task]["domains"][domain] = {
                        "tp": 190, "fn": 10, "fp": 0, "tn": negative,
                        "positive_photos": positive, "negative_photos": negative,
                        "fnr": .05, "fpr": 0.}
                    entry["ranking_ap"][domain][task]["positive_photos"] = positive
                small = entry["small_dacl_polygon_area_below_one_percent"][task]
                small.update(false_negatives=6, fnr=6/small["positive_photos"])
            entry.update(worst_error=.05, target_passed=False)
        report.validate_measurements(entries, protocol)
        entries[2]["target_passed"] = True
        with self.assertRaises(ValueError):
            report.validate_measurements(entries, protocol)

    def test_prepared_order_changes_are_required_and_preserved_categories_pass(self):
        control, treatment, protocol, plan = histories_fixture()
        actual = report.validate_histories(control, treatment, protocol, plan)
        self.assertEqual((actual["eligible_draws_control"], actual["eligible_draws_treatment"],
                          actual["changed_positions"]), (3539, 4743, 1204))
        self.assertFalse(actual["actual_ordered_row_indices_identical_asserted"])
        self.assertFalse(actual["other_photo_and_auxiliary_label_exposure_identical_asserted"])
        self.assertTrue(actual["actual_source_full_crop_two_target_counts_identical_each_epoch"])
        treatment[0]["sampled_row_indices_sha256"] = control[0]["sampled_row_indices_sha256"]
        with self.assertRaises(ValueError):
            report.validate_histories(control, treatment, protocol, plan)

    def test_exposure_retuning_categories_and_optimizer_counts_rejected(self):
        for mutation in ("exposure", "domain", "updates", "epoch"):
            control, treatment, protocol, plan = histories_fixture()
            if mutation == "exposure":
                treatment[1]["subtype_sampling"]["eligible_draws"] += 1
            elif mutation == "domain":
                treatment[1]["sampled_domain_counts"]["dacl"] += 1
            elif mutation == "updates":
                treatment[1]["optimizer_step_diagnostics"]["actual_optimizer_steps"] += 1
            else:
                treatment[1]["epoch"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_histories(control, treatment, protocol, plan)

    def test_plot_matches_strict_measurements_and_rejects_altered_rates(self):
        entries, _ = fixture()
        aggregate = {"schema": "facility_subtype_study_comparison_v1", "experiments": entries,
            "deployed": False, "source_test_inference_executed": False, "label_changes": 0,
            "additional_expert_confirmed_labels": 0}
        rows = plot.measurements(aggregate)
        self.assertEqual([r["maximum_fnr_fpr"] for r in rows], [e["worst_error"] for e in entries])
        self.assertEqual([r["small_false_negatives"] for r in rows], [[29, 42], [28, 40], [25, 37]])
        entries[2]["per_class"][TARGETS[0]]["domains"]["dacl"]["fpr"] += .01
        with self.assertRaises(ValueError):
            plot.measurements(aggregate)


if __name__ == "__main__":
    unittest.main()
