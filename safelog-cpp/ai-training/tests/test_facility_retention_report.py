"""Retention report contracts on synthetic aggregates; no inference or results."""
from __future__ import annotations

import copy
import unittest

from scripts import plot_facility_retention as plot
from scripts import report_facility_retention as report
from tests.test_facility_subtype_report import fixture as original_fixture


def fixture():
    entries, protocol = original_fixture()
    for entry, name in zip(entries, plot.RUNS):
        entry["run"] = name
    protocol.update(reference=plot.RUNS[0], control=plot.RUNS[1], treatment=plot.RUNS[2])
    for entry, ap in zip(entries, (.8, .75, .79)):
        entry["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = ap
    return entries, protocol


def histories_fixture():
    plan = report.read(report.ROOT / "reports/facility-spalling-sampler-dry-run.json")
    protocol = {"requested_epochs": 6, "draws_per_epoch": 14248, "batch_size": 8}
    histories = [[], []]
    for point in plan["epochs"]:
        order = point["control_order_sha256"]
        counts = {label: {"positive": 0, "negative": 14248, "unknown": 0} for label in report.CLASSES}
        # Some original other-class entries unknown: they must be excluded.
        counts["wet_surface"] = {"positive": 0, "negative": 12000, "unknown": 2248}
        known = sum(counts[label]["positive"]+counts[label]["negative"] for label in report.OTHER_CLASSES)
        for index, variant in enumerate(report.VARIANTS):
            weight = float(index)
            histories[index].append({"epoch": point["epoch"], "sampled_row_indices_sha256": order,
                "sampled_domain_counts": copy.deepcopy(point["domain_counts"]),
                "sampled_row_type_counts": copy.deepcopy(point["full_crop_counts"]),
                "sampled_full_target_joint_counts": copy.deepcopy(point["full_target_joint_counts"]),
                "sampled_photo_target_counts": copy.deepcopy(counts),
                "fixed_sampling": {"changed_positions_from_control": 0, "declared_draw_sha256": order,
                    "draw_hash_matches_prepared": True},
                "retention_distillation": {"weight": weight, "temperature": 2.,
                    "unweighted_mean_batch_loss": .025, "weighted_mean_batch_loss": weight*.025,
                    "teacher_forward_batches": 1781, "known_other_class_entries": known,
                    "contributing_batches": 1781, "excluded_primary_classes": report.TARGETS,
                    "teacher_eval": True, "teacher_frozen": True, "teacher_state_unchanged": True,
                    "teacher_gradients_absent": True},
                "optimizer_step_diagnostics": {"attempted_batches": 1781,
                    "actual_optimizer_steps": 1780, "amp_skipped_steps": 1}})
    return *histories, protocol, plan


def teacher_fixture():
    state = "a"*64; weights = "b"*64
    protocol = {"teacher_state_sha256": state, "initial_weights_sha256": weights}
    training = {"actual_teacher_forward_batches": 10686, "teacher_state_preservation": {
        "initial_state_sha256": state, "final_state_sha256": state, "initial_weights_sha256": weights,
        "unchanged": True, "eval_mode": True, "all_parameters_frozen": True, "no_parameter_gradients": True,
        "teacher_forward_both_arms": True, "student_parameter_count": 3244151,
        "teacher_parameter_count": 3244151, "teacher_state_tensor_count": 324}}
    return training, protocol


class RetentionReportTests(unittest.TestCase):
    def test_loader_provenance_then_exact_integral_support_normalization(self):
        def failing(_):
            raise ValueError("saved cache provenance failed")
        with self.assertRaisesRegex(ValueError, "saved cache provenance failed"):
            report.normalized_load_run("synthetic", failing)
        entries, _ = fixture(); entry = entries[0]
        entry["ranking_ap"]["dacl"][report.TARGETS[0]]["positive_photos"] = 211.
        before = copy.deepcopy(entry)
        _, normalized, converted = report.normalized_load_run("synthetic", lambda _: ({}, entry))
        self.assertEqual(converted, 1)
        self.assertEqual(entry, before)
        self.assertIs(type(normalized["ranking_ap"]["dacl"][report.TARGETS[0]]["positive_photos"]), int)
        for invalid in (True, -1, 211.5, float("nan"), float("inf"), 711., "211"):
            entry = copy.deepcopy(before)
            entry["ranking_ap"]["dacl"][report.TARGETS[0]]["positive_photos"] = invalid
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                report.normalized_load_run("synthetic", lambda _: ({}, entry))

    def test_original_measured_rates_unknown_ap_and_research_gate_unchanged(self):
        entries, protocol = fixture()
        measured = report.validate_measurements(entries, protocol)
        self.assertEqual(measured["verified_known_class_ap_measurements"], 42)
        floats = copy.deepcopy(entries)
        for entry in floats:
            for rows in entry["ranking_ap"].values():
                for point in rows.values():
                    point["positive_photos"] = float(point["positive_photos"])
        self.assertEqual(report.validate_measurements([report.normalize_ap_positive_support(e)[0] for e in floats], protocol), measured)
        for mutation in ("rate", "unknown", "strict", "test"):
            changed = copy.deepcopy(entries)
            if mutation == "rate":
                changed[2]["per_class"][report.TARGETS[0]]["domains"]["dacl"]["fnr"] += .01
            elif mutation == "unknown":
                changed[2]["ranking_ap"]["damsegment"]["wet_surface"]["ap"] = .9
            elif mutation == "strict":
                changed[2]["target_passed"] = True
            else:
                changed[2]["test_executed"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_measurements(changed, protocol)

    def test_retention_checks_every_known_other_class_not_average(self):
        entries, _ = fixture()
        gate = report.retention_measurements(entries, report.RETENTION_GATE)
        self.assertTrue(gate["retention_candidate_nominated"])
        self.assertEqual(gate["other_class_ap_measurements"], 24)
        self.assertEqual(len(gate["per_source_other_class_ap"]), 8)
        self.assertFalse(any(row["domain"] == "damsegment" for row in gate["per_source_other_class_ap"]))
        entries[2]["ranking_ap"]["codebrim"]["efflorescence"]["ap"] = .779
        gate = report.retention_measurements(entries, report.RETENTION_GATE)
        self.assertFalse(gate["all_known_other_class_reference_retention_passed"])
        self.assertFalse(gate["retention_candidate_nominated"])
        self.assertAlmostEqual(gate["worst_other_class_ap_decline_by_model"][2]["maximum_other_ap_drop_vs_reference"], .021)

    def test_exposed_recovery_and_reference_retention_boundaries(self):
        entries, _ = fixture()
        entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .769
        self.assertFalse(report.retention_measurements(entries, report.RETENTION_GATE)["dacl_exposed_rebar_recovery_vs_control_passed"])
        entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .78
        entries[1]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .76
        self.assertTrue(report.retention_measurements(entries, report.RETENTION_GATE)["retention_candidate_nominated"])
        changed = dict(report.RETENTION_GATE); changed["maximum_known_other_class_ap_drop_vs_reference"] = .03
        with self.assertRaises(ValueError):
            report.retention_measurements(entries, changed)

    def test_primary_maximum_guard_against_reference_and_control(self):
        entries, _ = fixture()
        worst = min(entries[0]["worst_error"], entries[1]["worst_error"])
        entries[2]["worst_error"] = worst+.02
        self.assertTrue(report.retention_measurements(entries, report.RETENTION_GATE)["primary_maximum_error_regression_guard_passed"])
        entries[2]["worst_error"] = worst+.020001
        self.assertFalse(report.retention_measurements(entries, report.RETENTION_GATE)["retention_candidate_nominated"])
        entries[2]["worst_error"] = float("nan")
        with self.assertRaises(ValueError):
            report.retention_measurements(entries, report.RETENTION_GATE)

    def test_retention_does_not_promote_research_or_strict_five_percent(self):
        entries, protocol = fixture()
        # Primary rates/size unchanged: no0.5pp or2-small-case improvement.
        entries[2]["per_class"] = copy.deepcopy(entries[0]["per_class"])
        entries[2]["worst_error"] = entries[0]["worst_error"]
        entries[2]["small_dacl_polygon_area_below_one_percent"] = copy.deepcopy(entries[0]["small_dacl_polygon_area_below_one_percent"])
        measured = report.validate_measurements(entries, protocol)
        self.assertFalse(measured["research_gate"]["research_candidate_nominated"])
        self.assertFalse(entries[2]["target_passed"])
        self.assertTrue(report.retention_measurements(entries, report.RETENTION_GATE)["retention_candidate_nominated"])

    def test_teacher_state_eval_gradient_and_forward_evidence_required(self):
        training, protocol = teacher_fixture()
        actual = report.validate_teacher(training, protocol)
        self.assertFalse(actual["teacher_predictions_are_new_truth"])
        for mutation in ("hash", "gradient", "mode", "forwards", "reference"):
            changed = copy.deepcopy(training)
            if mutation == "hash":
                changed["teacher_state_preservation"]["final_state_sha256"] = "c"*64
            elif mutation == "gradient":
                changed["teacher_state_preservation"]["no_parameter_gradients"] = False
            elif mutation == "mode":
                changed["teacher_state_preservation"]["eval_mode"] = False
            elif mutation == "forwards":
                changed["actual_teacher_forward_batches"] = 10685
            else:
                changed["teacher_state_preservation"]["initial_weights_sha256"] = "c"*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_teacher(changed, protocol)

    def test_identical_original_control_schedule_and_all_seven_exposure_required(self):
        control, distill, protocol, plan = histories_fixture()
        actual = report.validate_histories(control, distill, protocol, plan)
        self.assertTrue(actual["actual_ordered_row_index_hashes_identical_each_epoch"])
        self.assertTrue(actual["actual_all_seven_photo_target_exposure_identical_each_epoch"])
        self.assertEqual(actual["actual_teacher_forward_batches_each_arm"], 10686)
        for mutation in ("subtype", "otherclass", "count"):
            c, d = copy.deepcopy(control), copy.deepcopy(distill)
            if mutation == "subtype":
                d[0]["sampled_row_indices_sha256"] = plan["epochs"][0]["treatment_order_sha256"]
            elif mutation == "otherclass":
                d[0]["sampled_photo_target_counts"]["rust_stain"].update(positive=1, negative=14247)
            else:
                d[0]["sampled_row_type_counts"]["full"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_histories(c, d, protocol, plan)

    def test_unknown_teacher_mask_weighted_loss_and_frozen_forward_checks(self):
        for mutation in ("unknown", "weight", "forward", "gradient"):
            control, distill, protocol, plan = histories_fixture()
            audit = distill[1]["retention_distillation"]
            if mutation == "unknown":
                audit["known_other_class_entries"] += 1
                control[1]["retention_distillation"]["known_other_class_entries"] += 1
            elif mutation == "weight":
                audit["weighted_mean_batch_loss"] += .001
            elif mutation == "forward":
                control[1]["retention_distillation"]["teacher_forward_batches"] -= 1
            else:
                audit["teacher_gradients_absent"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_histories(control, distill, protocol, plan)

    def test_plot_reconstructs_real_aggregate_values_and_rejects_false_retention_claim(self):
        entries, _ = fixture()
        aggregate = {"schema": "facility_retention_study_comparison_v1", "experiments": entries,
            "deployed": False, "source_test_inference_executed": False, "label_changes": 0,
            "additional_expert_confirmed_labels": 0, "retention_gate": report.retention_measurements(entries, report.RETENTION_GATE)}
        rows = plot.measurements(aggregate)
        self.assertEqual([row["dacl_exposed_rebar_ap"] for row in rows], [.8, .75, .79])
        self.assertAlmostEqual(rows[1]["maximum_other_ap_drop_vs_reference"], .05)
        aggregate["retention_gate"]["retention_candidate_nominated"] = False
        with self.assertRaises(ValueError):
            plot.measurements(aggregate)


if __name__ == "__main__":
    unittest.main()
