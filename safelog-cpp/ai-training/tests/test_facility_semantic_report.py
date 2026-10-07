"""Semantic report/plot invariants with synthetic metadata and aggregate counts."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts import report_facility_semantic as report
from scripts import plot_facility_semantic as plot
from tests.test_facility_head_lr_report import fixture as old_fixture, lr_fixture


def fixture():
    entries, protocol = old_fixture()
    for entry, name in zip(entries, report.RUNS): entry["run"] = name
    protocol.update(reference=report.RUNS[0], control=report.RUNS[1], treatment=report.RUNS[2])
    for entry, ap in zip(entries, (.701, .7129, .7329)): entry["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = ap
    return entries, protocol


def semantic_metadata():
    _, _, training, _, protocol, _ = lr_fixture()
    inventory = {"parameter_count": 31085624, "trainable_parameter_count": 3265496,
        "frozen_encoder_parameter_count": 27820128, "new_semantic_head_parameter_count": 21345,
        "state_tensor_count": 510, "original_state_tensor_count": 324, "encoder_state_tensor_count": 180,
        "new_head_state_tensor_count": 6, "all_encoder_parameters_frozen": True, "encoder_eval": True,
        "all_encoder_modules_eval": True, "new_semantic_heads_trainable": True, "new_semantic_heads_zero": True,
        "low_channels": 192, "pooled_channels": 768, "low_feature_grid_for_640": [80, 80]}
    policy = {"encoder": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "encoder_frozen": True}
    protocol.update(semantic_model_inventory=copy.deepcopy(inventory), semantic_features_policy=policy,
        semantic_encoder_state_sha256="a"*64, semantic_pretrained_weights_sha256="b"*64,
        architecture_by_variant={"semantic": "semantic_test_arch"})
    training.update(model_variant="semantic", architecture="semantic_test_arch", semantic_features_policy=copy.deepcopy(policy),
        semantic_model_inventory=inventory, initial_state_transfer={"shared_state_tensors_equal": True,
            "shared_state_tensor_count": 324, "original_state_tensor_count": 324,
            "new_semantic_head_state_tensor_count": 6, "new_semantic_heads_zero": True,
            "full_original_model_state_preserved": True, "strict_state_load": True,
            "additional_trainable_parameters": 21345, "new_state_tensor_count": 186},
        frozen_semantic_state_preservation={"initial_encoder_state_sha256": "a"*64,
            "final_encoder_state_sha256": "a"*64, "encoder_state_unchanged": True,
            "encoder_all_eval": True, "encoder_parameters_frozen": True, "encoder_gradients_absent": True,
            "head_parameters_trainable": True, "semantic_heads_changed_from_zero": {"map": True, "photo": True, "aux": True},
            "encoder_forward_batches": 10686},
        semantic_pretrained_transfer={"weights_sha256": "b"*64, "encoder_state_sha256": "a"*64,
            "encoder_state_tensor_count": 180, "encoder_parameter_count": 27820128,
            "strict_encoder_load": True, "official_pooling_norm_transferred": True,
            "discarded_image_net_classifier_tensors": ["classifier.2.weight", "classifier.2.bias"],
            "all_encoder_parameters_frozen": True, "encoder_eval": True, "new_semantic_heads_zero": True})
    training["teacher_state_preservation"]["student_parameter_count"] = inventory["parameter_count"]
    return training, protocol


def histories():
    control, candidate, training, _, protocol, plan = lr_fixture()
    actual, sem_proto = semantic_metadata(); protocol.update(sem_proto)
    control_training = copy.deepcopy(training)
    for index, row in enumerate(control): row["optimizer_learning_rates"] = copy.deepcopy(candidate[index]["optimizer_learning_rates"])
    for row in candidate:
        row["semantic_features_training"] = {"encoder_state_sha256": "a"*64,
            "encoder_state_unchanged": True, "encoder_all_eval": True, "encoder_parameters_frozen": True,
            "encoder_gradients_absent": True, "semantic_head_gradient_tensors": 6,
            "semantic_head_gradient_nonzero_tensors": 6, "semantic_head_gradients_finite": True,
            "semantic_heads_changed_from_zero": {"map": True, "photo": True, "aux": True}, "encoder_forward_batches": 1781}
    return control, candidate, control_training, actual, protocol, plan


def aggregate():
    entries, protocol = fixture(); training, semantic_protocol = semantic_metadata()
    for index, entry in enumerate(entries):
        entry.update(title=("초기", "재사용", "새 특징")[index], actual_epochs=6, best_epoch=1, imgsz=640)
        if index: entry["resources"] = {"elapsed_training_minutes": 12.5+index, "peak_cuda_allocated_bytes": 2*1024**3,
            "optimizer_step_diagnostics": {"attempted_batches": 10686, "actual_optimizer_steps": 10680, "amp_skipped_steps": 6}}
    return {"schema": "facility_semantic_study_comparison_v1", "experiments": entries, **report.measurements(entries, protocol),
        "semantic_feature_verification": report.validate_semantic_features(training, semantic_protocol),
        "technical_verification": {"runtime_source_count": 112, "tests_run": 32},
        "deployed": False, "app_model_promoted": False, "source_test_inference_executed": False,
        "label_changes": 0, "additional_expert_confirmed_labels": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
        "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0,
        "previous_training_epochs_counted_again": False, "repeated_source_val_adaptation": True,
        "equal_compute_or_parameter_budget_asserted": False}


class SemanticReportTests(unittest.TestCase):
    def test_completion_requires_zero_initial_equality_frozen_encoder_learned_heads_and_offline_cpu(self):
        training, protocol = semantic_metadata(); training["weights_sha256"] = "c"*64
        protocol.update(reused_control_weights_sha256="d"*64, previous_protocol_sha256="e"*64, previous_verification_sha256="f"*64)
        protected = {"profile.json": "1"*64}; inventory = copy.deepcopy(training["semantic_model_inventory"])
        inventory["new_semantic_heads_zero"] = False
        point = {"variant": "semantic", "weights_sha256": "c"*64, "actual_epochs": 6, "imgsz": 640,
            "architecture": "semantic_test_arch", "strict_state_inventory_verified": True,
            "state_tensor_count": 510, "original_state_tensor_count": 324, "new_state_tensor_count": 186,
            "semantic_model_inventory": inventory, "initial_state_transfer": copy.deepcopy(training["initial_state_transfer"]),
            "semantic_pretrained_transfer": copy.deepcopy(training["semantic_pretrained_transfer"]),
            "frozen_semantic_state_preservation": copy.deepcopy(training["frozen_semantic_state_preservation"]),
            "actual_checkpoint_bn_buffers_and_configuration_verified": True,
            "batchnorm_state_preservation": copy.deepcopy(training["batchnorm_state_preservation"]),
            "cpu_reload": {"device": "cpu", "strict_factory_reload_verified": True, "all_outputs_finite": True,
                "public_output_equals_training_photo_output": True, "photo_shape": [1, 7],
                "loss_and_pool_map_shape": [1, 7, 80, 80], "auxiliary_shape": [1, 19],
                "offline_construction_verified": True, "frozen_encoder_state_sha256": "a"*64,
                "parameter_count": 31085624, "state_tensor_count": 510}}
        proof = {"schema": "facility_semantic_study_verification_v1", "status": "passed", "protocol_sha256": "2"*64,
            "source_sha256": protocol["source_sha256"], "runtime_source_count": len(protocol["source_sha256"]),
            "source_git_commit": "3"*40, "git_blob_bytes_verified": True, "working_runtime_sources_unchanged": True,
            "protected_files_unchanged": True, "existing_control_preserved": True, "protected_file_sha256": protected,
            "original324_and_zero_initial_outputs_verified": True, "frozen_semantic_encoder_and_learned_heads_verified": True,
            "candidate_original_batchnorm_buffers_preserved_affine_learnable": True, "both_teachers_unchanged_eval_frozen_no_grad": True,
            "parameter_and_compute_budget_identical": False, "actual_completed_training_epochs": 6, "reused_control_epochs": 6,
            "control_training_repeated": False, "verification_training_epochs": 0, "source_test_inference_executed": False,
            "app_model_promoted": False, "deployed": False, "accuracy_measured_by_verifier": False,
            "additional_expert_confirmed_labels": 0, "label_changes": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
            "preflight": {"passed": True, "zero_initial_outputs_identical_to_original": True,
                "semantic_model_inventory": copy.deepcopy(training["semantic_model_inventory"])},
            "tests": {"tests_run": 33, "expected_tests_collected": 33, "failures": 0, "errors": 0, "skipped": 0,
                "source_sha256": protocol["source_sha256"]},
            "prepared_data_integrity": {"status": "passed", "all_original_input_sha_size_mtime_preserved": True,
                "original_train_images_masks_annotations_verified": True, "individual_input_paths_published": False},
            "experiments": [point], "reused_control": {"weights_sha256": "d"*64, "reused": True, "training_repeated": False,
                "previous_protocol_sha256": "e"*64, "previous_verification_sha256": "f"*64,
                "cpu_reload": {"strict_factory_reload_verified": True, "all_outputs_finite": True,
                    "public_output_equals_training_photo_output": True}}}
        self.assertEqual(report.validate_technical(proof, protocol, "2"*64, training, protected)["actual_completed_training_epochs"], 6)
        for mutation in ("initial_outputs", "total324", "still_zero", "encoder", "offline", "shape", "new_truth", "retrain", "count", "skip"):
            p = copy.deepcopy(proof)
            if mutation == "initial_outputs": p["preflight"]["zero_initial_outputs_identical_to_original"] = False
            elif mutation == "total324": p["experiments"][0]["state_tensor_count"] = 324
            elif mutation == "still_zero": p["experiments"][0]["semantic_model_inventory"]["new_semantic_heads_zero"] = True
            elif mutation == "encoder": p["experiments"][0]["frozen_semantic_state_preservation"]["encoder_parameters_frozen"] = False
            elif mutation == "offline": p["experiments"][0]["cpu_reload"]["offline_construction_verified"] = False
            elif mutation == "shape": p["experiments"][0]["cpu_reload"]["photo_shape"] = [1, 8]
            elif mutation == "new_truth": p["new_photo_targets"] = 1
            elif mutation == "retrain": p["control_training_repeated"] = True
            elif mutation == "count": p["tests"]["tests_run"] = 32
            elif mutation == "skip": p["tests"]["skipped"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): report.validate_technical(p, protocol, "2"*64, training, protected)

    def test_all_known_ap_support_and_exact_existing_retention_gate_are_preserved(self):
        entries, protocol = fixture(); before = copy.deepcopy(entries)
        measured = report.measurements(entries, protocol)
        self.assertEqual(measured["verified_known_class_ap_measurements"], 42)
        self.assertEqual(measured["verified_known_other_class_ap_measurements"], 24)
        self.assertTrue(measured["retention_gate"]["dacl_exposed_rebar_recovery_vs_control_passed"])
        entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .725
        measured = report.measurements(entries, protocol)
        self.assertTrue(measured["retention_gate"]["all_known_other_class_reference_retention_passed"])
        self.assertFalse(measured["retention_gate"]["retention_candidate_nominated"])
        self.assertFalse(measured["retention_gate"]["dacl_exposed_rebar_recovery_vs_control_passed"])
        self.assertEqual(entries[:2], before[:2])

    def test_unknown_labels_counts_rates_and_relaxed_recovery_gate_cannot_enter(self):
        for mutation in ("unknown", "support", "rate", "threshold_flag", "gate", "order", "small"):
            entries, protocol = fixture()
            if mutation == "unknown": entries[2]["ranking_ap"]["damsegment"]["wet_surface"]["ap"] = 0.
            elif mutation == "support": entries[2]["ranking_ap"]["dacl"]["exposed_rebar"]["positive_photos"] += 1
            elif mutation == "rate": entries[2]["per_class"][report.TARGETS[0]]["domains"]["dacl"]["fnr"] += .001
            elif mutation == "threshold_flag": entries[2]["target_passed"] = True
            elif mutation == "gate": protocol["retention_candidate_gate"]["minimum_dacl_exposed_rebar_ap_recovery_vs_control"] = .01
            elif mutation == "order": entries.reverse()
            elif mutation == "small": entries[2]["small_dacl_polygon_area_below_one_percent"][report.TARGETS[0]]["positive_photos"] = 94
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): report.measurements(entries, protocol)

    def test_both_actual_optimizer_curves_match_without_old_head_lr_change_descriptor(self):
        control, candidate, ct, nt, protocol, _ = histories()
        measured = report.learning_rate_measurements([control, candidate], [ct, nt], protocol)
        self.assertTrue(measured["both_actual_rates_verified"]); self.assertTrue(measured["reused_control_rates_are_observations"])
        self.assertEqual(measured["before_epoch_observations_by_variant"]["control"], measured["before_epoch_observations_by_variant"]["semantic"])
        for mutation in ("old_head", "missing_observation", "changed_rate", "floor", "nan"):
            c, n, a, b, p, _ = histories()
            if mutation == "old_head": a["head_lr"] = .00025
            elif mutation == "missing_observation": c[0].pop("optimizer_learning_rates")
            elif mutation == "changed_rate": n[0]["optimizer_learning_rates"]["head"] += .000001
            elif mutation == "floor": b["optimizer_final_learning_rates"]["head"] = .000001
            elif mutation == "nan": n[0]["optimizer_learning_rates"]["head"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): report.learning_rate_measurements([c, n], [a, b], p)

    def test_original324_and_new_encoder180_heads6_are_separate_with_official_hashes(self):
        training, protocol = semantic_metadata(); before = copy.deepcopy(training)
        value = report.validate_semantic_features(training, protocol)
        self.assertEqual(value["inventory"]["state_tensor_count"], 510)
        self.assertEqual(value["original_base_state_tensors_transferred"], 324)
        self.assertEqual(training, before)
        for mutation in ("whole324", "unfrozen", "state_changed", "zero_final_heads", "no_forwards", "official_file", "pooling_norm", "base_transfer"):
            t, p = semantic_metadata()
            if mutation == "whole324": t["semantic_model_inventory"]["state_tensor_count"] = 324
            elif mutation == "unfrozen": t["frozen_semantic_state_preservation"]["encoder_parameters_frozen"] = False
            elif mutation == "state_changed": t["frozen_semantic_state_preservation"]["final_encoder_state_sha256"] = "c"*64
            elif mutation == "zero_final_heads": t["frozen_semantic_state_preservation"]["semantic_heads_changed_from_zero"]["aux"] = False
            elif mutation == "no_forwards": t["frozen_semantic_state_preservation"]["encoder_forward_batches"] = 0
            elif mutation == "official_file": t["semantic_pretrained_transfer"]["weights_sha256"] = "c"*64
            elif mutation == "pooling_norm": t["semantic_pretrained_transfer"]["official_pooling_norm_transferred"] = False
            elif mutation == "base_transfer": t["initial_state_transfer"]["shared_state_tensor_count"] = 510
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): report.validate_semantic_features(t, p)

    def test_teacher_base_bn_and_encoder_gradients_are_checked_with_known_source_mask(self):
        control, candidate, ct, training, protocol, plan = histories()
        teacher, bn, _ = report.validate_training(training, protocol, "1"*64)
        self.assertFalse(teacher["teacher_predictions_are_new_truth"])
        measured = report.validate_history(control, candidate, protocol, plan, bn["initial_buffers_sha256"])
        self.assertTrue(measured["frozen_semantic_encoder_verified_all_six_epochs"])
        for mutation in ("encoder_grad", "new_head_grad", "new_head_finite", "forwards", "unknown_teacher", "new_draw"):
            c, n, _, t, p, plan = histories()
            if mutation == "encoder_grad": n[0]["semantic_features_training"]["encoder_gradients_absent"] = False
            elif mutation == "new_head_grad": n[0]["semantic_features_training"]["semantic_head_gradient_nonzero_tensors"] = 0
            elif mutation == "new_head_finite": n[0]["semantic_features_training"]["semantic_head_gradients_finite"] = False
            elif mutation == "forwards": n[0]["semantic_features_training"]["encoder_forward_batches"] = 1
            elif mutation == "unknown_teacher": n[0]["retention_distillation"]["known_other_class_entries"] += 1
            elif mutation == "new_draw": n[0]["fixed_sampling"]["changed_positions_from_control"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): report.validate_history(c, n, p, plan, t["batchnorm_state_preservation"]["initial_buffers_sha256"])

    def test_exact_integral_ap_support_adapter_preserves_raw_metrics_and_rejects_fractional(self):
        entries, protocol = fixture(); expected = report.measurements(entries, protocol)
        for e in entries:
            for source in e["ranking_ap"].values():
                for point in source.values(): point["positive_photos"] = float(point["positive_photos"])
        before = copy.deepcopy(entries)
        self.assertEqual(report.measurements([report.normalize_ap_positive_support(e)[0] for e in entries], protocol), expected)
        self.assertEqual(entries, before)
        for value in (True, 1.5, float("nan"), float("inf")):
            e = copy.deepcopy(entries[0]); e["ranking_ap"]["dacl"]["exposed_rebar"]["positive_photos"] = value
            with self.subTest(value=repr(value)), self.assertRaises(ValueError): report.normalize_ap_positive_support(e)

    def test_plot_and_text_report_feature_package_compute_cost_and_all_failed_subgates(self):
        result = aggregate(); result["experiments"][2]["ranking_ap"]["dacl"]["exposed_rebar"]["ap"] = .725
        _, protocol = fixture(); result.update(report.measurements(result["experiments"], protocol))
        rows = plot.measurements(result); self.assertEqual(rows[1]["dacl_exposed_rebar_ap"], .7129)
        text = report.render(result)
        self.assertIn("31,085,624", text); self.assertIn("85/386", text); self.assertIn("FN 합계70", text)
        self.assertIn("동일 FLOPs", text); self.assertIn("미달", text); self.assertIn("철근 AP 회복: False", text)
        self.assertIn("원래 base324", text); self.assertIn("미확인", text); self.assertIn("보류 TEST", text)
        for mutation in ("compute", "new_data", "retrained_control", "gate"):
            changed = copy.deepcopy(result)
            if mutation == "compute": changed["equal_compute_or_parameter_budget_asserted"] = True
            elif mutation == "new_data": changed["new_photo_targets"] = 1
            elif mutation == "retrained_control": changed["new_control_training_epochs"] = 6
            elif mutation == "gate": changed["retention_gate"]["retention_candidate_nominated"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): plot.measurements(changed)

    def test_standalone_plot_png_provenance_matches_only_checked_aggregate_bytes(self):
        import json
        with tempfile.TemporaryDirectory(dir=report.ROOT / "runs") as directory:
            root = Path(directory); source, output = root / "synthetic.json", root / "synthetic.png"
            self.assertTrue(root.resolve().is_relative_to((report.ROOT / "runs").resolve()))
            source.write_text(json.dumps(aggregate()), encoding="utf-8")
            value = plot.main(["--input", str(source), "--output", str(output)])
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(value["source_aggregate_sha256"], plot.sha(source)); self.assertEqual(value["plot_sha256"], plot.sha(output))
            self.assertFalse(value["accuracy_estimated_by_plot"]); self.assertFalse(value["equal_compute_or_parameter_budget_asserted"])
            self.assertNotIn(str(root), json.dumps(value)); self.assertNotIn("case_id", json.dumps(value))
            with self.assertRaises(ValueError): plot.main(["--input", str(source), "--output", str(output)])


if __name__ == "__main__": unittest.main()
