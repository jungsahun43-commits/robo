"""Synthetic BN-report contracts; never publish fixtures as measured results."""
from __future__ import annotations
import copy
import unittest
from scripts import report_facility_batchnorm as report
from scripts import plot_facility_batchnorm as plot
from tests.test_facility_retention_report import fixture as original_fixture
from tests.test_facility_retention_report import histories_fixture as original_histories
from tests.test_facility_retention_report import teacher_fixture


def fixture():
    entries, proto = original_fixture()
    for entry, name in zip(entries, report.RUNS):
        entry["run"] = name; entry["actual_epochs"] = 6
    entries[1]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .79
    entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .815
    proto.update(reference=report.RUNS[0], control=report.RUNS[1], treatment=report.RUNS[2], retention_candidate_gate=copy.deepcopy(report.RETENTION_GATE))
    return entries, proto


def bn_fixture():
    proto = {"batchnorm_policy": {"fixture": "eval after model.train; affine trainable"}, "initial_batchnorm_buffers_sha256": "a"*64}
    training = {"batchnorm_policy": copy.deepcopy(proto["batchnorm_policy"]), "batchnorm_state_preservation": {
        "initial_buffers_sha256": "a"*64, "final_buffers_sha256": "a"*64, "buffers_unchanged": True,
        "layer_count": 47, "channel_count": 12328, "buffer_tensor_count": 141, "affine_parameter_tensor_count": 94,
        "affine_parameters_trainable": True, "backbone_parameters_trainable": True, "head_parameters_trainable": True,
        "eps_and_momentum_unchanged": True, "all_batchnorm_eval": True, "policy_applied_after_each_model_train": True}}
    return training, proto


def histories_fixture():
    control, frozen, proto, plan = original_histories()
    for history in (control, frozen):
        for row in history:
            audit = row["retention_distillation"]
            audit.update(weight=4., weighted_mean_batch_loss=4.*audit["unweighted_mean_batch_loss"])
    for row in frozen:
        row["batchnorm_training"] = {"buffer_sha256": "a"*64, "buffers_unchanged": True,
            "actual_eval_layers": 47, "affine_trainable_tensors": 94, "affine_gradient_tensors": 94,
            "affine_gradient_nonzero_tensors": 3, "affine_gradients_finite": True, "policy_applied_after_model_train": True}
    return control, frozen, proto, plan


def training_fixture():
    teacher, proto = teacher_fixture()
    bn, bn_proto = bn_fixture(); proto.update(bn_proto)
    proto.update(seed=56, requested_epochs=6, patience=6, batch_size=8, draws_per_epoch=14248,
        backbone_lr=.00004, head_lr=.00025, auxiliary_weight=.5, loader_randomness={"fixture": True},
        domain_proportions=[.7, .1, .2], architecture_by_variant={"frozen": "lraspp_mobilenet_facility_auxiliary_v1"},
        source_sha256={}, core_spatial_manifest_sha256="c"*64, auxiliary_manifest_sha256="d"*64,
        sampler_plan_sha256="e"*64, paired_draws_sha256="f"*64, distillation_recipe={"fixture": True})
    training = {key: proto[key] for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
        "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    training.update(teacher); training.update(bn)
    training.update(status="complete", actual_epochs=6, imgsz=640, model_variant="frozen", distillation_weight=4.,
        architecture=proto["architecture_by_variant"]["frozen"], initial_weights_sha256=proto["initial_weights_sha256"], source_sha256={},
        study_protocol_sha256="1"*64, core_spatial_manifest_sha256=proto["core_spatial_manifest_sha256"],
        spatial_manifest_sha256=proto["core_spatial_manifest_sha256"], auxiliary_manifest_sha256=proto["auxiliary_manifest_sha256"],
        sampling_intervention_applied=False, paired_draws_key="control", sampler_plan_sha256=proto["sampler_plan_sha256"],
        private_draw_archive_sha256=proto["paired_draws_sha256"], distillation_recipe=proto["distillation_recipe"])
    return training, proto


def aggregate_fixture():
    entries, proto = fixture()
    return {"schema": "facility_batchnorm_study_comparison_v1", "experiments": entries, **report.measurements(entries, proto),
        "deployed": False, "source_test_inference_executed": False, "label_changes": 0, "additional_expert_confirmed_labels": 0,
        "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0,
        "previous_training_epochs_counted_again": False, "batchnorm_policy_chosen_after_previous_source_val_results": True,
        "repeated_source_val_adaptation": True, "training_normalization_changed": True}


class BatchNormReportTests(unittest.TestCase):
    def test_original_bn_buffers_47_layers_and_trainable_affine_policy_required(self):
        training, proto = bn_fixture(); actual = report.validate_batchnorm(training, proto)
        self.assertTrue(actual["training_normalization_uses_original_running_statistics"])
        self.assertFalse(actual["affine_or_backbone_head_parameter_freezing_asserted"])
        for mutation in ("hash", "affine", "backbone", "mode", "inventory"):
            changed = copy.deepcopy(training)
            if mutation == "hash":
                changed["batchnorm_state_preservation"]["final_buffers_sha256"] = "b"*64
            elif mutation == "affine":
                changed["batchnorm_state_preservation"]["affine_parameters_trainable"] = False
            elif mutation == "backbone":
                changed["batchnorm_state_preservation"]["backbone_parameters_trainable"] = False
            elif mutation == "mode":
                changed["batchnorm_state_preservation"]["all_batchnorm_eval"] = False
            else:
                changed["batchnorm_state_preservation"]["layer_count"] = 46
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_batchnorm(changed, proto)

    def test_original_measurements_42_all_and24_other_counts_preserved(self):
        entries, proto = fixture(); measured = report.measurements(entries, proto)
        self.assertEqual(measured["verified_known_class_ap_measurements"], 42)
        self.assertEqual(measured["verified_known_other_class_ap_measurements"], 24)
        self.assertTrue(measured["retention_gate"]["retention_candidate_nominated"])
        self.assertEqual(len(measured["error_rows"]), 18)

    def test_recovery_guard_uses_new_normal_bn_weight4_matched_control(self):
        entries, proto = fixture()
        entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .80
        # Relative to old weight0=.75 this would pass; against actual new
        # normal-BN weight4=.79 it has only0.01 recovery and must fail.
        measured = report.measurements(entries, proto)
        self.assertFalse(measured["retention_gate"]["dacl_exposed_rebar_recovery_vs_control_passed"])
        self.assertFalse(measured["retention_gate"]["retention_candidate_nominated"])
        entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .81
        self.assertTrue(report.measurements(entries, proto)["retention_gate"]["retention_candidate_nominated"])

    def test_saved_float_support_normalizes_without_changing_metrics(self):
        entries, proto = fixture(); measured = report.measurements(entries, proto)
        for entry in entries:
            for source in entry["ranking_ap"].values():
                for point in source.values():
                    point["positive_photos"] = float(point["positive_photos"])
        before = copy.deepcopy(entries)
        normalized = [report.normalize_ap_positive_support(entry)[0] for entry in entries]
        self.assertEqual(entries, before); self.assertEqual(report.measurements(normalized, proto), measured)
        for invalid in (True, 12.5, -1., float("nan"), float("inf")):
            entry = copy.deepcopy(entries[0]); entry["ranking_ap"]["dacl"]["exposed_rebar"]["positive_photos"] = invalid
            with self.subTest(value=repr(invalid)), self.assertRaises(ValueError):
                report.normalize_ap_positive_support(entry)
        def failing(_):
            raise ValueError("original provenance failure")
        with self.assertRaisesRegex(ValueError, "original provenance failure"):
            report.normalized_load_run("synthetic", failing)

    def test_unchanged_primary_small_unknown_ap_and_strict_guards(self):
        for mutation in ("rate", "unknown", "strict", "small", "gate"):
            entries, proto = fixture()
            if mutation == "rate":
                entries[2]["per_class"][report.TARGETS[0]]["domains"]["dacl"]["fnr"] += .01
            elif mutation == "unknown":
                entries[2]["ranking_ap"]["damsegment"]["wet_surface"]["ap"] = .9
            elif mutation == "strict":
                entries[2]["target_passed"] = True
            elif mutation == "small":
                entries[2]["small_dacl_polygon_area_below_one_percent"][report.TARGETS[0]]["positive_photos"] += 1
            else:
                proto["retention_candidate_gate"]["maximum_known_other_class_ap_drop_vs_reference"] = .03
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.measurements(entries, proto)

    def test_actual_bn_epoch_buffers_and_last_batch_affine_gradient_evidence(self):
        control, frozen, proto, plan = histories_fixture()
        actual = report.validate_history(control, frozen, proto, plan, "a"*64)
        self.assertTrue(actual["original_buffers_verified_all_six_epochs"])
        self.assertEqual(actual["new_completed_training_epochs"], 6)
        self.assertEqual(actual["new_control_training_epochs"], 0)
        self.assertEqual(actual["teacher_forward_batches_each_arm"], 10686)
        self.assertIn("final minibatch", actual["affine_gradient_evidence_scope"])

    def test_bn_buffer_updates_affine_freeze_or_missing_gradient_cannot_pass(self):
        for mutation in ("buffer", "eval", "gradient", "finite", "policy"):
            control, frozen, proto, plan = histories_fixture(); bn = frozen[1]["batchnorm_training"]
            if mutation == "buffer":
                bn["buffer_sha256"] = "b"*64
            elif mutation == "eval":
                bn["actual_eval_layers"] = 46
            elif mutation == "gradient":
                bn["affine_gradient_tensors"] = 93
            elif mutation == "finite":
                bn["affine_gradients_finite"] = False
            else:
                bn["policy_applied_after_model_train"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_history(control, frozen, proto, plan, "a"*64)

    def test_original_draws_and_known_teacher_mask_remain_required(self):
        for mutation in ("order", "weight", "unknown", "label"):
            control, frozen, proto, plan = histories_fixture()
            if mutation == "order":
                frozen[0]["sampled_row_indices_sha256"] = plan["epochs"][0]["treatment_order_sha256"]
            elif mutation == "weight":
                frozen[0]["retention_distillation"]["weight"] = 1.
            elif mutation == "unknown":
                frozen[0]["retention_distillation"]["known_other_class_entries"] += 1
            else:
                frozen[0]["sampled_photo_target_counts"]["rust_stain"].update(positive=1, negative=14247)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_history(control, frozen, proto, plan, "a"*64)

    def test_new_student_metadata_keeps_teacher_original_and_bn_trainable(self):
        training, proto = training_fixture(); teacher, bn = report.validate_training(training, proto, "1"*64)
        self.assertFalse(teacher["teacher_predictions_are_new_truth"])
        self.assertTrue(bn["buffers_unchanged"])
        training["teacher_state_preservation"]["no_parameter_gradients"] = False
        with self.assertRaises(ValueError):
            report.validate_training(training, proto, "1"*64)

    def test_plot_recovers_actual_values_and_requires_cost_normalization_scope(self):
        aggregate = aggregate_fixture(); rows = plot.measurements(aggregate)
        self.assertEqual([row["dacl_exposed_rebar_ap"] for row in rows], [.8, .79, .815])
        for mutation in ("cost", "adaptation", "normalization", "retention"):
            changed = copy.deepcopy(aggregate)
            if mutation == "cost":
                changed["new_control_training_epochs"] = 6
            elif mutation == "adaptation":
                changed["repeated_source_val_adaptation"] = False
            elif mutation == "normalization":
                changed["training_normalization_changed"] = False
            else:
                changed["retention_gate"]["retention_candidate_nominated"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                plot.measurements(changed)


if __name__ == "__main__":
    unittest.main()
