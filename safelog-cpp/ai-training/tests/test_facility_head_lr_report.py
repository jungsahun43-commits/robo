"""Synthetic head-LR reporting contracts; no training or fabricated results."""
import copy
import unittest
from scripts import report_facility_head_lr as report
from scripts import plot_facility_head_lr as plot
from scripts.facility_head_lr_study import HEAD_LR_POLICY, OPTIMIZER_SCHEDULE
from tests.test_facility_batchnorm_report import fixture as old_fixture
from tests.test_facility_batchnorm_report import training_fixture as old_training
from tests.test_facility_batchnorm_report import histories_fixture as old_histories


def fixture():
    entries, proto = old_fixture()
    for entry, name in zip(entries, report.RUNS):
        entry["run"] = name
    for entry, ap in zip(entries, (.701, .707, .727)):
        entry["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = ap
    proto.update(reference=report.RUNS[0], control=report.RUNS[1], treatment=report.RUNS[2])
    return entries, proto


def lr_fixture():
    control, history, proto, plan = old_histories()
    training, metadata_proto = old_training()
    proto.update(metadata_proto, head_lr=.0001, backbone_lr=.00004, head_lr_policy=copy.deepcopy(HEAD_LR_POLICY))
    proto["architecture_by_variant"] = {"low_lr": proto["architecture_by_variant"]["frozen"]}
    training.update(head_lr=.0001, model_variant="low_lr", head_lr_policy=copy.deepcopy(HEAD_LR_POLICY),
        optimizer_schedule=copy.deepcopy(OPTIMIZER_SCHEDULE), optimizer_final_learning_rates={"head": .000005, "backbone": .000005})
    curves = proto["head_lr_policy"]["step0_to6_curves"]
    for index, row in enumerate(history):
        row["optimizer_learning_rates"] = {"head": curves["candidate_head"][index], "backbone": curves["backbone"][index]}
    return control, history, training, {"head_lr": .00025, "backbone_lr": .00004}, proto, plan


def proof_fixture(measured):
    return {"candidate_actual_rates_verified": True, "reused_control_declared_rates_verified": True,
        "reused_control_rates_are_observations": False, "candidate_observed_epochs": 6,
        "candidate_before_epoch_observations": [{"epoch": i+1, **row} for i, row in enumerate(measured["candidate_before_epoch_observations"])],
        "candidate_final_after_step6_observation": measured["candidate_final_observed_rates"],
        "declared_step0_to6_curves": measured["step0_to6_curves"]}


class HeadLRReportTests(unittest.TestCase):
    def test_new_actual_lr_curve_and_old_derived_curve_have_unchanged_floor(self):
        _, history, training, control, proto, _ = lr_fixture()
        actual = report.learning_rate_measurements(history, training, control, proto)
        self.assertTrue(actual["candidate_actual_rates_verified"])
        self.assertFalse(actual["reused_control_rates_are_observations"])
        self.assertFalse(actual["uniform_ratio_across_curve"])
        self.assertEqual(actual["candidate_final_observed_rates"], {"head": .000005, "backbone": .000005})
        curves = actual["step0_to6_curves"]
        self.assertAlmostEqual(curves["candidate_head"][0]/curves["control_head"][0], .4)
        self.assertEqual(curves["candidate_head"][6]/curves["control_head"][6], 1.)

    def test_changed_head_backbone_floor_or_forged_rate_rejected(self):
        for mutation in ("head", "backbone", "floor", "nonfinite", "constant_ratio"):
            _, history, training, control, proto, _ = lr_fixture()
            if mutation == "head":
                history[1]["optimizer_learning_rates"]["head"] *= .9
            elif mutation == "backbone":
                history[1]["optimizer_learning_rates"]["backbone"] *= .9
            elif mutation == "floor":
                training["optimizer_final_learning_rates"]["head"] = .000002
            elif mutation == "nonfinite":
                history[0]["optimizer_learning_rates"]["head"] = float("nan")
            else:
                proto["head_lr_policy"]["uniform_ratio_across_curve"] = True
                training["head_lr_policy"] = copy.deepcopy(proto["head_lr_policy"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.learning_rate_measurements(history, training, control, proto)

    def test_forged_completion_proof_cannot_claim_old_lr_observations(self):
        _, history, training, control, proto, _ = lr_fixture()
        actual = report.learning_rate_measurements(history, training, control, proto)
        report.validate_learning_rate_proof(proof_fixture(actual), actual)
        for mutation in ("old_observed", "rate", "count"):
            proof = copy.deepcopy(proof_fixture(actual))
            if mutation == "old_observed":
                proof["reused_control_rates_are_observations"] = True
            elif mutation == "rate":
                proof["candidate_before_epoch_observations"][1]["head"] += .000001
            else:
                proof["candidate_observed_epochs"] = 5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_learning_rate_proof(proof, actual)

    def test_fixed_recovery_gate_uses_already_strong_new_matched_control(self):
        entries, proto = fixture()
        measured = report.measurements(entries, proto)
        self.assertEqual(measured["verified_known_class_ap_measurements"], 42)
        self.assertEqual(measured["verified_known_other_class_ap_measurements"], 24)
        self.assertTrue(measured["retention_gate"]["retention_candidate_nominated"])
        entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .720
        measured = report.measurements(entries, proto)
        self.assertFalse(measured["retention_gate"]["retention_candidate_nominated"])
        self.assertFalse(measured["retention_gate"]["dacl_exposed_rebar_recovery_vs_control_passed"])

    def test_saved_floating_support_normalization_does_not_modify_true_metrics(self):
        entries, proto = fixture(); measured = report.measurements(entries, proto)
        for entry in entries:
            for source in entry["ranking_ap"].values():
                for point in source.values():
                    point["positive_photos"] = float(point["positive_photos"])
        before = copy.deepcopy(entries)
        self.assertEqual(report.measurements([report.normalize_ap_positive_support(e)[0] for e in entries], proto), measured)
        self.assertEqual(entries, before)
        for invalid in (True, 12.5, float("nan"), float("inf")):
            entry = copy.deepcopy(entries[0]); entry["ranking_ap"]["dacl"]["exposed_rebar"]["positive_photos"] = invalid
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                report.normalize_ap_positive_support(entry)

    def test_unknown_source_labels_primary_counts_and_strict_gate_cannot_change(self):
        for mutation in ("unknown", "rate", "strict", "gate"):
            entries, proto = fixture()
            if mutation == "unknown":
                entries[2]["ranking_ap"]["damsegment"]["wet_surface"]["ap"] = .9
            elif mutation == "rate":
                entries[2]["per_class"][report.TARGETS[0]]["domains"]["dacl"]["fnr"] += .01
            elif mutation == "strict":
                entries[2]["target_passed"] = True
            else:
                proto["retention_candidate_gate"]["minimum_dacl_exposed_rebar_ap_recovery_vs_control"] = .01
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.measurements(entries, proto)

    def test_both_bn_buffers_teacher_and_original_unknown_mask_preserved(self):
        control, history, training, _, proto, plan = lr_fixture()
        teacher, bn = report.validate_training(training, proto, "1"*64)
        self.assertFalse(teacher["teacher_predictions_are_new_truth"])
        self.assertTrue(bn["buffers_unchanged"])
        actual = report.validate_history(control, history, proto, plan, bn["initial_buffers_sha256"])
        self.assertEqual(actual["new_completed_training_epochs"], 6)
        self.assertEqual(actual["new_control_training_epochs"], 0)
        history[0]["retention_distillation"]["known_other_class_entries"] += 1
        with self.assertRaises(ValueError):
            report.validate_history(control, history, proto, plan, bn["initial_buffers_sha256"])

    def test_plot_labels_head_lr_scope_and_rejects_new_bn_policy_change(self):
        entries, proto = fixture()
        aggregate = {"schema": "facility_head_lr_study_comparison_v1", "experiments": entries, **report.measurements(entries, proto),
            "deployed": False, "source_test_inference_executed": False, "label_changes": 0, "additional_expert_confirmed_labels": 0,
            "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0, "previous_training_epochs_counted_again": False,
            "head_lr_chosen_after_previous_source_val_results": True, "repeated_source_val_adaptation": True, "training_normalization_changed": False}
        rows = plot.measurements(aggregate)
        self.assertEqual([row["dacl_exposed_rebar_ap"] for row in rows], [.701, .707, .727])
        aggregate["training_normalization_changed"] = True
        with self.assertRaises(ValueError):
            plot.measurements(aggregate)


if __name__ == "__main__":
    unittest.main()
