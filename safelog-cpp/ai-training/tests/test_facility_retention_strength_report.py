"""Synthetic reporting contracts for weight4; no new inference or results."""
from __future__ import annotations
import copy
import unittest

from scripts import report_facility_retention_strength as report
from scripts import plot_facility_retention_strength as plot
from tests.test_facility_retention_report import fixture as three_fixture
from tests.test_facility_retention_report import histories_fixture as original_histories
from tests.test_facility_retention_report import teacher_fixture


def fixture():
    entries, protocol = three_fixture()
    strong = copy.deepcopy(entries[2]); strong["run"] = report.RUNS[3]
    strong["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .795
    entries.append(strong)
    for entry in entries:
        entry["actual_epochs"] = 6
    protocol.update(treatment=report.RUNS[3], retention_candidate_gate=copy.deepcopy(report.RETENTION_GATE))
    return entries, protocol


def aggregate_fixture():
    entries, protocol = fixture()
    return {"schema": "facility_retention_strength_study_comparison_v1", "experiments": entries,
        **report.measurements(entries, protocol), "actual_new_completed_training_epochs": 6,
        "new_control_training_epochs": 0, "previous_pair_completed_epochs_counted_again": False,
        "weight_chosen_after_previous_source_val_results": True, "repeated_source_val_adaptation": True,
        "source_test_inference_executed": False, "app_model_promoted": False, "deployed": False,
        "teacher_predictions_are_new_truth": False}


def histories_fixture():
    control, strong, protocol, plan = original_histories()
    for row in strong:
        audit = row["retention_distillation"]
        audit.update(weight=4., weighted_mean_batch_loss=4.*audit["unweighted_mean_batch_loss"])
    return control, strong, protocol, plan


def training_fixture():
    teacher, proto = teacher_fixture()
    proto.update(seed=56, requested_epochs=6, patience=6, batch_size=8, draws_per_epoch=14248,
        backbone_lr=.00004, head_lr=.00025, auxiliary_weight=.5, loader_randomness={"fixture": True},
        domain_proportions=[.7, .1, .2], architecture_by_variant={"strong": "lraspp_mobilenet_facility_auxiliary_v1"},
        source_sha256={}, core_spatial_manifest_sha256="c"*64, auxiliary_manifest_sha256="d"*64,
        sampler_plan_sha256="e"*64, paired_draws_sha256="f"*64, distillation_recipe={"fixture": True})
    training = {key: proto[key] for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
        "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    training.update(teacher, status="complete", actual_epochs=6, imgsz=640, model_variant="strong", distillation_weight=4.,
        architecture=proto["architecture_by_variant"]["strong"], initial_weights_sha256=proto["initial_weights_sha256"],
        source_sha256={}, study_protocol_sha256="1"*64, core_spatial_manifest_sha256=proto["core_spatial_manifest_sha256"],
        spatial_manifest_sha256=proto["core_spatial_manifest_sha256"], auxiliary_manifest_sha256=proto["auxiliary_manifest_sha256"],
        sampling_intervention_applied=False, paired_draws_key="control", sampler_plan_sha256=proto["sampler_plan_sha256"],
        private_draw_archive_sha256=proto["paired_draws_sha256"], distillation_recipe=proto["distillation_recipe"])
    return training, proto


class RetentionStrengthReportTests(unittest.TestCase):
    def test_four_model_support_counts_preserve_original_gates(self):
        entries, proto = fixture(); measured = report.measurements(entries, proto)
        self.assertEqual(measured["verified_known_class_ap_measurements"], 56)
        self.assertEqual(measured["verified_known_other_class_ap_measurements"], 32)
        self.assertEqual(len(measured["four_model_other_class_ap"]), 8)
        self.assertEqual(len(measured["worst_other_class_ap_decline_by_model"]), 4)
        self.assertTrue(measured["retention_gate"]["retention_candidate_nominated"])
        self.assertEqual(len(measured["error_rows"]), 24)

    def test_saved_integral_float_support_only_normalizes_after_validation(self):
        entries, proto = fixture(); original = report.measurements(entries, proto)
        for entry in entries:
            for source in entry["ranking_ap"].values():
                for point in source.values():
                    point["positive_photos"] = float(point["positive_photos"])
        before = copy.deepcopy(entries)
        normalized = [report.normalize_ap_positive_support(entry)[0] for entry in entries]
        self.assertEqual(entries, before)
        self.assertEqual(report.measurements(normalized, proto), original)
        for invalid in (True, 12.5, -1., float("nan"), float("inf")):
            entry = copy.deepcopy(entries[0]); entry["ranking_ap"]["dacl"]["exposed_rebar"]["positive_photos"] = invalid
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                report.normalize_ap_positive_support(entry)
        def fail(_):
            raise ValueError("original provenance failure")
        with self.assertRaisesRegex(ValueError, "original provenance failure"):
            report.normalized_load_run("synthetic", fail)

    def test_any_known_other_ap_regression_remains_a_failure(self):
        entries, proto = fixture()
        entries[3]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .779
        entries[3]["ranking_ap"]["codebrim"]["efflorescence"]["ap"] = .779
        measured = report.measurements(entries, proto)
        self.assertFalse(measured["retention_gate"]["retention_candidate_nominated"])
        failed = [row for row in measured["retention_gate"]["per_source_other_class_ap"] if not row["retained_vs_reference"]]
        self.assertEqual(len(failed), 2)

    def test_weight1_increment_is_descriptive_and_not_a_new_promotion_gate(self):
        entries, proto = fixture()
        entries[3]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .78
        measured = report.measurements(entries, proto)
        self.assertTrue(measured["retention_gate"]["retention_candidate_nominated"])
        self.assertLess(measured["increment_vs_weight1"]["dacl_exposed_rebar_weight4_minus_weight1_ap"], 0)
        self.assertTrue(measured["increment_vs_weight1"]["descriptive_only_no_additional_promotion_gate"])

    def test_rate_unknown_label_and_strict_five_percent_tampering_rejected(self):
        for mutation in ("rate", "unknown", "strict", "gate"):
            entries, proto = fixture()
            if mutation == "rate":
                entries[3]["per_class"][report.TARGETS[0]]["domains"]["dacl"]["fnr"] += .01
            elif mutation == "unknown":
                entries[2]["ranking_ap"]["damsegment"]["wet_surface"]["ap"] = .9
            elif mutation == "strict":
                entries[3]["target_passed"] = True
            else:
                proto["retention_candidate_gate"]["maximum_known_other_class_ap_drop_vs_reference"] = .03
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.measurements(entries, proto)

    def test_same_original_draws_and_reused_control_cost_six_new_epochs_only(self):
        control, strong, proto, plan = histories_fixture()
        actual = report.validate_histories(control, strong, proto, plan)
        self.assertEqual(actual["new_completed_training_epochs"], 6)
        self.assertEqual(actual["new_control_training_epochs"], 0)
        self.assertTrue(actual["control_reused_from_completed_prior_study"])
        self.assertTrue(actual["actual_ordered_row_index_hashes_identical_each_epoch"])
        self.assertEqual(actual["teacher_forward_batches_each_arm"], 10686)

    def test_old_weight1_or_treatment_draws_cannot_enter_new_weight4_proof(self):
        for mutation in ("weight", "order", "unknown", "teacher"):
            control, strong, proto, plan = histories_fixture()
            if mutation == "weight":
                strong[0]["retention_distillation"]["weight"] = 1.
            elif mutation == "order":
                strong[0]["sampled_row_indices_sha256"] = plan["epochs"][0]["treatment_order_sha256"]
            elif mutation == "unknown":
                strong[0]["retention_distillation"]["known_other_class_entries"] += 1
            else:
                strong[0]["retention_distillation"]["teacher_state_unchanged"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_histories(control, strong, proto, plan)

    def test_strength_metadata_and_original_frozen_teacher_required(self):
        training, proto = training_fixture()
        measured = report.validate_training(training, proto, "1"*64)
        self.assertFalse(measured["teacher_predictions_are_new_truth"])
        for mutation in ("weight", "plan", "teacher", "source"):
            changed = copy.deepcopy(training)
            if mutation == "weight":
                changed["distillation_weight"] = 1.
            elif mutation == "plan":
                changed.pop("sampler_plan_sha256")
            elif mutation == "teacher":
                changed["teacher_state_preservation"]["no_parameter_gradients"] = False
            else:
                changed["study_protocol_sha256"] = "2"*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_training(changed, proto, "1"*64)

    def test_plot_reconstructs_four_model_ap_and_requires_adaptive_scope(self):
        aggregate = aggregate_fixture(); rows = plot.plotted_measurements(aggregate)
        self.assertEqual([row["dacl_exposed_rebar_ap"] for row in rows], [.8, .75, .79, .795])
        self.assertEqual([row["newly_trained_this_followup"] for row in rows], [False, False, False, True])
        aggregate["weight_chosen_after_previous_source_val_results"] = False
        with self.assertRaises(ValueError):
            plot.plotted_measurements(aggregate)

    def test_plot_rejects_duplicate_costs_and_altered_retention_results(self):
        for mutation in ("cost", "duplicate", "count", "retention"):
            aggregate = aggregate_fixture()
            if mutation == "cost":
                aggregate["new_control_training_epochs"] = 6
            elif mutation == "duplicate":
                aggregate["previous_pair_completed_epochs_counted_again"] = True
            elif mutation == "count":
                aggregate["verified_known_class_ap_measurements"] = 84
            else:
                aggregate["retention_gate"]["retention_candidate_nominated"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plot.plotted_measurements(aggregate)


if __name__ == "__main__":
    unittest.main()
