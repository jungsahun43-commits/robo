"""Report a frozen ImageNet feature package with unchanged source-VAL gates.

The reused control is the completed low-head-LR model. Only the new candidate
adds six epochs; pretrained features, capacity and compute change together.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import CLASSES, DOMAINS, LABELS, TARGETS, read, require, sha
from scripts.report_facility_discrimination import verify_validation_cache
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_retention import RETENTION_GATE, normalized_load_run, retention_measurements
from scripts.report_facility_head_lr import validate_batchnorm, validate_history as previous_history

RUNS = ("facility-presence-target-roi-control", "facility-presence-target-head-lr-low",
        "facility-presence-target-semantic-features")
PROTOCOL_PATH = "reports/facility-semantic-study-protocol.json"
VERIFICATION_PATH = "reports/facility-semantic-study-verification.json"
PREFLIGHT_PATH = "runs/facility-semantic-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-semantic-source-before-training.json"
TEST_RECORD_PATH = "runs/facility-semantic-test-results.json"
OUTPUT_PATH = "reports/facility-semantic-study-comparison.json"
MARKDOWN_PATH = "reports/FACILITY_SEMANTIC_STUDY_RESULTS_KO.md"
PRIOR_COMPARISON = "reports/facility-head-lr-study-comparison.json"
APP_PROFILE = "reports/facility-inference-profile.json"
SCHEDULE = {"name": "CosineAnnealingLR", "T_max": 6, "eta_min": .000005, "weight_decay": .0002}
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def measurements(entries, protocol):
    require(isinstance(entries, list) and [entry.get("run") for entry in entries] == list(RUNS),
            "The original, reused low-head-LR and one semantic candidate are required")
    result = validate_measurements(entries, protocol)
    retention = retention_measurements(entries, protocol["retention_candidate_gate"])
    require(result["verified_known_class_ap_measurements"] == 42 and retention["other_class_ap_measurements"] == 24,
            "All original known/unknown AP states must be preserved")
    return {**result, "retention_gate": retention, "verified_known_other_class_ap_measurements": 24}


def learning_rate_measurements(histories, trainings, protocol):
    require(len(histories) == len(trainings) == 2 and protocol["head_lr"] == .0001 and protocol["backbone_lr"] == .00004,
            "Both arms keep the completed low-head-LR condition")
    curves = {key: [.000005 + (initial - .000005) * (1 + math.cos(math.pi * step / 6)) / 2 for step in range(7)]
              for key, initial in (("backbone", .00004), ("head", .0001))}
    observed, finals = {}, {}
    for variant, history, training in zip(("control", "semantic"), histories, trainings):
        require(training.get("optimizer_schedule") == SCHEDULE and len(history) == 6
                and training.get("head_lr") == .0001 and training.get("backbone_lr") == .00004,
                "Observed optimizer schedule or initial LR differs")
        rows = []
        for index, row in enumerate(history):
            values = row.get("optimizer_learning_rates", {})
            require(set(values) == set(curves) and all(type(v) in (int, float) and math.isfinite(v)
                    and math.isclose(v, curves[k][index], rel_tol=0., abs_tol=1e-15) for k, v in values.items()),
                    "Actual before-epoch learning rates differ from the fixed curve")
            rows.append({"epoch": index + 1, **values})
        final = training.get("optimizer_final_learning_rates", {})
        require(set(final) == set(curves) and all(type(v) in (int, float) and math.isfinite(v)
                and math.isclose(v, .000005, rel_tol=0., abs_tol=1e-15) for v in final.values()),
                "Actual final observed rates must retain the common floor")
        observed[variant] = rows; finals[variant] = dict(final)
    require(observed["control"] == observed["semantic"] and finals["control"] == finals["semantic"],
            "The actual paired optimizer curves differ")
    return {"both_actual_rates_verified": True, "reused_control_rates_are_observations": True,
            "observed_epochs_each_arm": 6, "before_epoch_observations_by_variant": observed,
            "final_after_step6_observations_by_variant": finals, "declared_step0_to6_curves": curves,
            "backbone_head_floor_schedule_unchanged": True}


def validate_semantic_features(training, protocol):
    require(training.get("semantic_features_policy") == protocol["semantic_features_policy"], "Semantic feature package changed")
    inventory = training.get("semantic_model_inventory", {})
    fields = ("parameter_count", "trainable_parameter_count", "frozen_encoder_parameter_count",
              "new_semantic_head_parameter_count", "state_tensor_count", "original_state_tensor_count",
              "encoder_state_tensor_count", "new_head_state_tensor_count")
    require(set(fields).issubset(inventory) and all(type(inventory[k]) is int and inventory[k] > 0 for k in fields)
            and inventory == protocol["semantic_model_inventory"],
            "Actual original/encoder/new-head inventories are required")
    require(all(inventory.get(k) is True for k in ("all_encoder_parameters_frozen", "encoder_eval", "all_encoder_modules_eval",
        "new_semantic_heads_trainable", "new_semantic_heads_zero")) and inventory.get("low_channels") == 192
        and inventory.get("pooled_channels") == 768 and inventory.get("low_feature_grid_for_640") == [80, 80],
        "Initial encoder features or zero/trainable head inventory differs")
    require(inventory["original_state_tensor_count"] == 324 and inventory["new_head_state_tensor_count"] == 6
            and inventory["new_semantic_head_parameter_count"] == 21345
            and inventory["parameter_count"] == 3244151 + inventory["frozen_encoder_parameter_count"] + 21345
            and inventory["trainable_parameter_count"] == 3244151 + 21345
            and inventory["state_tensor_count"] == 324 + inventory["encoder_state_tensor_count"] + 6,
            "The total student cannot be presented as the original324 states/3.24M parameters")
    transfer = training.get("initial_state_transfer", {})
    expected = {"shared_state_tensors_equal": True, "shared_state_tensor_count": 324,
        "original_state_tensor_count": 324, "new_semantic_head_state_tensor_count": 6,
        "new_semantic_heads_zero": True, "full_original_model_state_preserved": True,
        "strict_state_load": True, "additional_trainable_parameters": 21345,
        "new_state_tensor_count": inventory["state_tensor_count"] - 324}
    require(transfer == expected, "Initial original model and zero-head transfer differ")
    state = training.get("frozen_semantic_state_preservation", {})
    initial, final = state.get("initial_encoder_state_sha256"), state.get("final_encoder_state_sha256")
    require(isinstance(initial, str) and _SHA.fullmatch(initial)
            and initial == final == protocol["semantic_encoder_state_sha256"], "Frozen canonical encoder state changed")
    for key in ("encoder_state_unchanged", "encoder_all_eval", "encoder_parameters_frozen",
                "encoder_gradients_absent", "head_parameters_trainable"):
        require(state.get(key) is True, "The extra pretrained encoder must remain frozen/eval/no-grad")
    require(state.get("semantic_heads_changed_from_zero") == {"map": True, "photo": True, "aux": True}
            and type(state.get("encoder_forward_batches")) is int and state["encoder_forward_batches"] == 10686,
            "Actual three-head learning and encoder forward evidence is missing")
    require(isinstance(protocol.get("semantic_pretrained_weights_sha256"), str)
            and _SHA.fullmatch(protocol["semantic_pretrained_weights_sha256"]), "Official pretrained full file hash is required")
    pretrained = training.get("semantic_pretrained_transfer", {})
    require(pretrained.get("weights_sha256") == protocol["semantic_pretrained_weights_sha256"]
            and pretrained.get("encoder_state_sha256") == initial
            and pretrained.get("encoder_state_tensor_count") == inventory["encoder_state_tensor_count"]
            and pretrained.get("encoder_parameter_count") == inventory["frozen_encoder_parameter_count"]
            and all(pretrained.get(k) is True for k in ("strict_encoder_load", "official_pooling_norm_transferred",
                "all_encoder_parameters_frozen", "encoder_eval", "new_semantic_heads_zero"))
            and pretrained.get("discarded_image_net_classifier_tensors") == ["classifier.2.weight", "classifier.2.bias"],
            "Official pretrained transfer must include the frozen pooled LayerNorm and discard the ImageNet classifier")
    return {"inventory": dict(inventory), "official_pretrained_weights_sha256": protocol["semantic_pretrained_weights_sha256"],
            "canonical_encoder_state_sha256": initial, "encoder_state_unchanged": True,
            "encoder_frozen_eval_no_grad": True, "original_base_state_tensors_transferred": 324,
            "zero_initialized_new_head_state_tensors": 6, "semantic_heads_changed_from_zero": dict(state["semantic_heads_changed_from_zero"]),
            "encoder_forward_batches": state["encoder_forward_batches"], "equal_parameter_or_compute_budget_asserted": False}


def validate_teacher(training, protocol):
    state = training.get("teacher_state_preservation", {})
    require(state.get("initial_state_sha256") == state.get("final_state_sha256") == protocol["teacher_state_sha256"]
            and isinstance(state.get("initial_state_sha256"), str) and _SHA.fullmatch(state["initial_state_sha256"])
            and state.get("initial_weights_sha256") == protocol["initial_weights_sha256"], "Original frozen teacher hash differs")
    require(all(state.get(k) is True for k in ("unchanged", "eval_mode", "all_parameters_frozen", "no_parameter_gradients", "teacher_forward_both_arms"))
            and state.get("teacher_parameter_count") == 3244151 and state.get("teacher_state_tensor_count") == 324
            and state.get("student_parameter_count") == training["semantic_model_inventory"]["parameter_count"]
            and type(training.get("actual_teacher_forward_batches")) is int and training["actual_teacher_forward_batches"] == 10686,
            "Teacher identity/masking or actual enlarged student inventory differs")
    return {"teacher_weights_sha256": protocol["initial_weights_sha256"], "teacher_state_sha256": state["initial_state_sha256"],
            "actual_teacher_forward_batches": 10686, "unchanged_eval_frozen_no_gradients": True, "teacher_predictions_are_new_truth": False}


def validate_training(training, protocol, protocol_sha):
    expected = {k: protocol[k] for k in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
        "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    expected.update(status="complete", actual_epochs=6, imgsz=640, model_variant="semantic", distillation_weight=4.,
        architecture=protocol["architecture_by_variant"]["semantic"], initial_weights_sha256=protocol["initial_weights_sha256"],
        source_sha256=protocol["source_sha256"], study_protocol_sha256=protocol_sha,
        core_spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"], spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        auxiliary_manifest_sha256=protocol["auxiliary_manifest_sha256"], sampling_intervention_applied=False,
        paired_draws_key="control", sampler_plan_sha256=protocol["sampler_plan_sha256"], private_draw_archive_sha256=protocol["paired_draws_sha256"],
        distillation_recipe=protocol["distillation_recipe"])
    for key, value in expected.items():
        require(type(training.get(key)) is type(value) and training[key] == value, "Actual semantic training condition differs")
    return validate_teacher(training, protocol), validate_batchnorm(training, protocol), validate_semantic_features(training, protocol)


def validate_history(control, candidate, protocol, plan, initial_bn_sha):
    result = previous_history(control, candidate, protocol, plan, initial_bn_sha)
    for row in candidate:
        audit = row.get("semantic_features_training", {})
        require(audit.get("encoder_state_sha256") == protocol["semantic_encoder_state_sha256"]
                and all(audit.get(k) is True for k in ("encoder_state_unchanged", "encoder_all_eval", "encoder_parameters_frozen",
                    "encoder_gradients_absent")), "Actual per-epoch frozen encoder evidence differs")
        require(type(audit.get("semantic_head_gradient_tensors")) is int and audit["semantic_head_gradient_tensors"] == 6
                and type(audit.get("semantic_head_gradient_nonzero_tensors")) is int and 1 <= audit["semantic_head_gradient_nonzero_tensors"] <= 6
                and audit.get("semantic_head_gradients_finite") is True
                and audit.get("semantic_heads_changed_from_zero") == {"map": True, "photo": True, "aux": True}
                and type(audit.get("encoder_forward_batches")) is int
                and audit["encoder_forward_batches"] == row["optimizer_step_diagnostics"]["attempted_batches"],
                "Actual learnable semantic-head gradient/change evidence is missing")
    return {**result, "frozen_semantic_encoder_verified_all_six_epochs": True,
            "semantic_head_gradient_evidence_scope": "Final minibatch of each epoch, not every update"}


def validate_technical(proof, protocol, protocol_sha, training, protected):
    require(proof.get("schema") == "facility_semantic_study_verification_v1" and proof.get("status") == "passed"
            and proof.get("protocol_sha256") == protocol_sha and proof.get("source_sha256") == protocol["source_sha256"],
            "Matching completed semantic verification is required")
    require(type(proof.get("runtime_source_count")) is int and proof["runtime_source_count"] == len(protocol["source_sha256"])
            and isinstance(proof.get("source_git_commit"), str) and re.fullmatch(r"[a-f0-9]{40}", proof["source_git_commit"]),
            "Actual frozen source inventory/commit is missing")
    require(all(proof.get(k) is True for k in ("git_blob_bytes_verified", "working_runtime_sources_unchanged", "protected_files_unchanged", "existing_control_preserved"))
            and proof.get("protected_file_sha256") == protected, "Original source/control/app byte preservation proof differs")
    require(proof.get("original324_and_zero_initial_outputs_verified") is True
            and proof.get("frozen_semantic_encoder_and_learned_heads_verified") is True
            and proof.get("candidate_original_batchnorm_buffers_preserved_affine_learnable") is True
            and proof.get("both_teachers_unchanged_eval_frozen_no_grad") is True
            and proof.get("parameter_and_compute_budget_identical") is False,
            "Original324/zero-output/frozen-encoder proof or compute scope differs")
    preflight = proof.get("preflight", {})
    require(preflight.get("passed") is True and preflight.get("zero_initial_outputs_identical_to_original") is True
            and preflight.get("semantic_model_inventory") == training["semantic_model_inventory"],
            "Actual preflight initial-zero public/map/aux equality proof is missing")
    require(type(proof.get("actual_completed_training_epochs")) is int and proof["actual_completed_training_epochs"] == 6
            and type(proof.get("reused_control_epochs")) is int and proof["reused_control_epochs"] == 6
            and proof.get("control_training_repeated") is False and proof.get("verification_training_epochs") == 0,
            "Only one new six-epoch candidate is permitted")
    require(all(proof.get(k) is False for k in ("source_test_inference_executed", "app_model_promoted", "deployed", "accuracy_measured_by_verifier"))
            and all(type(proof.get(k)) is int and proof[k] == 0 for k in ("additional_expert_confirmed_labels", "label_changes", "new_photo_targets", "new_pixel_targets")),
            "Pretrained features cannot create truth, TEST accuracy or app deployment")
    tests = proof.get("tests", {})
    require(type(tests.get("tests_run")) is int and tests["tests_run"] > 0 and tests["tests_run"] == tests.get("expected_tests_collected")
            and all(type(tests.get(k)) is int and tests[k] == 0 for k in ("failures", "errors", "skipped"))
            and tests.get("source_sha256") == protocol["source_sha256"], "Actual complete focused tests must pass")
    data = proof.get("prepared_data_integrity", {})
    require(data.get("status") == "passed" and data.get("all_original_input_sha_size_mtime_preserved") is True
            and data.get("original_train_images_masks_annotations_verified") is True
            and data.get("individual_input_paths_published") is False, "Actual unchanged original TRAIN data proof is missing")
    points = proof.get("experiments", [])
    require(isinstance(points, list) and len(points) == 1, "One actual semantic checkpoint must be verified")
    point = points[0]
    require(point.get("weights_sha256") == training["weights_sha256"] and point.get("variant") == "semantic"
            and type(point.get("actual_epochs")) is int and point["actual_epochs"] == 6
            and point.get("architecture") == protocol["architecture_by_variant"]["semantic"] and point.get("imgsz") == 640
            and point.get("strict_state_inventory_verified") is True
            and type(point.get("state_tensor_count")) is int and point["state_tensor_count"] == training["semantic_model_inventory"]["state_tensor_count"]
            and type(point.get("original_state_tensor_count")) is int and point["original_state_tensor_count"] == 324
            and type(point.get("new_state_tensor_count")) is int and point["new_state_tensor_count"] == 186,
            "Actual enlarged checkpoint state inventory differs")
    measured_inventory = point.get("semantic_model_inventory", {})
    require({k: v for k, v in measured_inventory.items() if k != "new_semantic_heads_zero"}
                == {k: v for k, v in training["semantic_model_inventory"].items() if k != "new_semantic_heads_zero"}
            and measured_inventory.get("new_semantic_heads_zero") is False
            and point.get("frozen_semantic_state_preservation") == training["frozen_semantic_state_preservation"]
            and point.get("initial_state_transfer") == training["initial_state_transfer"]
            and point.get("semantic_pretrained_transfer") == training["semantic_pretrained_transfer"]
            and point.get("actual_checkpoint_bn_buffers_and_configuration_verified") is True
            and point.get("batchnorm_state_preservation") == training["batchnorm_state_preservation"],
            "Actual original324 transfer/frozen encoder/learned-head evidence differs")
    reload = point.get("cpu_reload", {})
    require(all(reload.get(k) is True for k in ("strict_factory_reload_verified", "all_outputs_finite", "public_output_equals_training_photo_output"))
            and reload.get("device") == "cpu" and reload.get("photo_shape") == [1, 7]
            and reload.get("loss_and_pool_map_shape") == [1, 7, 80, 80] and reload.get("auxiliary_shape") == [1, 19]
            and reload.get("offline_construction_verified") is True
            and reload.get("frozen_encoder_state_sha256") == protocol["semantic_encoder_state_sha256"]
            and reload.get("parameter_count") == training["semantic_model_inventory"]["parameter_count"]
            and reload.get("state_tensor_count") == 510,
            "Actual offline CPU7/80/19 factory reload proof is missing")
    reused = proof.get("reused_control", {})
    require(reused.get("weights_sha256") == protocol["reused_control_weights_sha256"]
            and reused.get("reused") is True and reused.get("training_repeated") is False
            and reused.get("previous_protocol_sha256") == protocol["previous_protocol_sha256"]
            and reused.get("previous_verification_sha256") == protocol["previous_verification_sha256"]
            and all(reused.get("cpu_reload", {}).get(k) is True for k in
                ("strict_factory_reload_verified", "all_outputs_finite", "public_output_equals_training_photo_output")),
            "Actual completed low-head-LR control identity/CPU proof differs")
    return {"status": "passed", "source_git_commit": proof["source_git_commit"], "runtime_source_count": proof["runtime_source_count"],
            "tests_run": tests["tests_run"], "actual_completed_training_epochs": 6}


def validate_learning_rate_proof(proof, measured):
    require(all(proof.get(k) is True for k in ("candidate_actual_rates_verified", "reused_control_actual_rates_verified",
        "reused_control_rates_are_observations", "both_actual_observed_curves_match"))
        and type(proof.get("candidate_observed_epochs")) is int and proof["candidate_observed_epochs"] == 6,
        "Actual reused/new observed LR proof is missing")
    require(proof.get("candidate_before_epoch_observations") == measured["before_epoch_observations_by_variant"]["semantic"]
            and proof.get("reused_control_before_epoch_observations") == measured["before_epoch_observations_by_variant"]["control"]
            and proof.get("candidate_final_after_step6_observation") == measured["final_after_step6_observations_by_variant"]["semantic"]
            and proof.get("reused_control_final_after_step6_observation") == measured["final_after_step6_observations_by_variant"]["control"]
            and proof.get("declared_step0_to6_curves") == measured["declared_step0_to6_curves"],
            "Completion LR proof differs from both actual saved observations")


def render(result):
    entries = result["experiments"]; retention = result["retention_gate"]; feature = result["semantic_feature_verification"]
    lines = ["# 동결 ImageNet 특징 보강 후속 실험 결과", "",
        "기존 저학습률 대조군에서 DACL 박락 오탐85/386과 작은 손상 FN 합계70이 남아, ImageNet ConvNeXt-Tiny의 다른 시각 표현을 추가하는 후보 한 개를 선택했다. 이 관찰은 선택 이유이며 개선 원인이나 개선 보장은 아니다.",
        "완료된 head LR1e-4 대조6epoch를 그대로 재사용했다. 새 후보만 원래0773에서6epoch 학습했다. 새 대조 학습은0epoch이며 과거6epoch를 이번 비용에 다시 더하지 않았다.",
        "원래 AuxiliaryClassifier의324개 state를 그대로 옮겼다. 동결된 ConvNeXt 저층192채널·80×80 특징에는 map7 head, 정규화된 pooled768 특징에는 photo7·aux19 head를 추가했다. 세 head는0으로 초기화해 학습 전7/80/19 출력이 원래 모델과 정확히 일치한다. 새 head의 학습과 encoder 상태 보존은 실제 검증 기록으로 확인한다.",
        "추가 encoder와 pooled LayerNorm은 eval·no_grad로 유지한다. 원래 BatchNorm47개·141버퍼는 기존 통계를 유지하고 γ/β94개·원래 backbone·head 가중치는 학습 가능하다. backbone4e-5/head1e-4, cosine floor5e-6,640입력·80마스크·draw14248·teacher weight4/T2·원래 known/unknown 손실과 표본 순서는 같다.",
        "ImageNet 사전학습·새 특징 경로·파라미터 용량·계산량이 함께 추가되는 방법 비교다. 동일 epoch·노출이며 동일 FLOPs·동일 용량이나 사전학습 특징만의 인과 효과를 주장하지 않는다. 새로운 시설 사진이나 손상 정답은 추가하지 않았다.", "",
        "![Frozen semantic feature comparison](facility-semantic-study-comparison.png)", "",
        "| 모델 | 이번 새 epoch | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한5% |", "|---|---:|---:|---:|---:|---|"]
    for index, entry in enumerate(entries):
        lines.append(f'| {entry["title"]} | {6 if index == 2 else 0} | {entry["actual_epochs"]} | {entry["best_epoch"]} | {100*entry["worst_error"]:.2f}% | {"통과" if entry["target_passed"] else "미달"} |')
    change = result["comparisons"]
    lines += ["", f'새 후보−초기 최대 오류 {change["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, 새 후보−재사용 대조 {change["maximum_error_treatment_minus_control_pp"]:+.2f}pp. 양수는 악화다.',
        "최대값은 균열·박락 × DACL710/Dam424/CODEBRIM611 × FNR/FPR의12개 비율 중 최대다. 전체 사진 오답 비율이나 앱 정확도가 아니다.", "",
        "## 독립된 기준과 실패 항목", "",
        f'유지 후보: **{"통과" if retention["retention_candidate_nominated"] else "미달"}**. 다른 알려진 AP 유지: {retention["all_known_other_class_reference_retention_passed"]}; 철근 AP 회복: {retention["dacl_exposed_rebar_recovery_vs_control_passed"]}; primary 최대 오류 guard: {retention["primary_maximum_error_regression_guard_passed"]}.',
        "기존 유지 조건은 초기 대비 다른 알려진 AP 하락0.02 이하, 이번 대조의 DACL 철근 AP 대비0.02 이상 회복, 초기·이번 대조 각각 대비 최대 target 오류 악화2pp 이하이다. 높은 대조 AP0.7129에도 회복 기준을 그대로 유지한다.",
        f'기존 연구 후보: **{"통과" if result["research_gate"]["research_candidate_nominated"] else "미달"}**. 초기·대조 각각 대비 최대 오류0.5pp 개선·target 악화2pp 이하·다른 알려진 AP 하락0.02 이하·작은 FN 합계2건 감소·항목별 작은 FNR 악화2pp 이하 조건은 바꾸지 않았다.',
        "유지·연구·엄격한5%는 별도 조건이다. 일부 AP 유지 통과를 전체 후보 통과로 표시하지 않는다.", "",
        "| 출처 | 다른 항목 | 초기 AP | 재사용 AP | 새 AP | 새−초기 | 새−대조 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in retention["per_source_other_class_ap"]:
        lines.append(f'| {row["domain"]} | {LABELS[row["class"]]} | {row["reference_ap"]:.4f} | {row["control_ap"]:.4f} | {row["treatment_ap"]:.4f} | {row["treatment_minus_reference_ap"]:+.4f} | {row["treatment_minus_control_ap"]:+.4f} |')
    lines += ["", "| 연구 비교 기준 | 최대 오류 개선 | target 회귀 guard | 다른 AP guard | 작은 FN2건 감소 | 작은 항목 FNR guard |", "|---|---|---|---|---|---|"]
    for name in ("initializer", "control"):
        point = result["research_gate"]["error_and_other_ap_gate"]["comparisons"][name]
        small = result["research_gate"]["small_area_comparisons"][name]
        passed = lambda value: "통과" if value else "미달"
        lines.append(f'| {"초기" if name == "initializer" else "재사용 대조"} | {passed(point["minimum_improvement_passed"])} | {passed(point["target_regression_guard_passed"])} | {passed(point["other_known_class_ap_guard_passed"])} | {passed(small["minimum_two_cases_improvement_passed"])} | {passed(small["small_area_per_class_regression_guard_passed"])} |')
    lines += ["", "전체7종의 알려진 조합14개×3모델=42AP, 다른5종의 알려진 조합8개×3모델=24AP를 동일 정답·분모에서 확인했다. AP는 순위 지표이며 정답률이 아니다. 미확인 항목을 정상·0점 AP로 바꾸지 않았다.", "",
        "| 작은 손상 | 초기 FN/양성 | 재사용 FN/양성 | 새 FN/양성 |", "|---|---:|---:|---:|"]
    for task in TARGETS:
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{e["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"]}/{e["small_dacl_polygon_area_below_one_percent"][task]["positive_photos"]}' for e in entries) + ' |')
    lines += ["", "분모는 균열93·박락105의198개 항목·사진 사례이며 같은 사진이 두 항목에 들어갈 수 있다. 고유198사진이나 물리적 손상 크기 정답으로 해석하지 않는다.", "",
        "## 출처별 관측 오류", "", "| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | FP/음성 | 오탐률 |", "|---|---|---|---:|---:|---:|---:|"]
    titles = {e["run"]: e["title"] for e in entries}
    for row in result["error_rows"]:
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% |')
    lines += ["", "Wilson95는 JSON에 함께 기록한 고정 예측·독립 사진 가정의 기술 통계다. 반복 VAL·epoch·임계값·후속 방법 선택을 보정한 현장 보장이나 모델 차이 유의성 검정이 아니다.", "",
        "## 실제 관측 LR와 비용", "", "대조·새 후보의 각 epoch 시작 전 LR6개와 마지막 step6 LR을 모두 실제 optimizer 관측으로 확인했다. 과거 대조 곡선을 새로 계산한 관측으로 표시하지 않는다.", "",
        "| 모델 | 기존/이번 학습 시간 | allocated peak | 실제 update | AMP skip |", "|---|---:|---:|---:|---:|"]
    for entry in entries[1:]:
        cost = entry["resources"]; steps = cost["optimizer_step_diagnostics"]
        lines.append(f'| {entry["title"]} | {cost["elapsed_training_minutes"]:.2f}분 | {cost["peak_cuda_allocated_bytes"]/1024**3:.3f}GiB | {steps["actual_optimizer_steps"]} | {steps["amp_skipped_steps"]} |')
    inventory = feature["inventory"]
    lines += ["", f'새 student 총 파라미터 {inventory["parameter_count"]:,}, 동결 encoder {inventory["frozen_encoder_parameter_count"]:,}, 새 학습 head {inventory["new_semantic_head_parameter_count"]:,}, 전체 state {inventory["state_tensor_count"]}개를 기록했다. 원래 base324와 새 encoder·head state를 구분한다.',
        f'공식 사전학습 파일 SHA `{feature["official_pretrained_weights_sha256"]}`, canonical encoder SHA `{feature["canonical_encoder_state_sha256"]}`를 대조했다. encoder는6epoch 동안 상태·eval·gradient 부재를 유지했다. 새 head의 gradient·nonzero는 각 epoch 마지막 minibatch의 증거이며 모든 update 검증은 아니다.',
        f'원본 TRAIN bytes·라벨·마스크·추출 배열·teacher 상태 및 {result["technical_verification"]["runtime_source_count"]}개 소스·실제 코드 테스트 {result["technical_verification"]["tests_run"]}개·offline CPU7/80/19 재로딩을 확인했다. 코드 테스트 수는 정확도 사례 수가 아니다.',
        "GPU 시간에는 자료 읽기와 epoch 검증·캐시 영향이 포함된다. 사전학습 파일 취득·preflight·검증 비용과 재사용 과거 학습은 새6epoch 시간에 합산하지 않았다. 추가 encoder의 CPU/모바일 지연이나 현장 처리량은 측정하지 않았다.", "",
        "## 적용 상태와 한계", "", "한 seed의 반복 공개 source-VAL 연구다. 보류 TEST 추론·앱 자동 승격·새 전문가 확정 라벨·정답 변경·새 시설 사진은 없다. 기본 facility-validation-v2와 프로필 SHA를 유지한다. teacher·ImageNet 특징은 새 손상 정답이 아니다.",
        "공장 시설 정확도·정밀 위치·구조 안전·미래 현장5% 미만을 측정하거나 보장하지 않는다. 새로운 특징이 적용됐다는 사실과 실제 성능 개선 여부를 구분한다.", "",
        "[고정 조건](facility-semantic-study-protocol.json), [실측 집계](facility-semantic-study-comparison.json), [기술 검증](facility-semantic-study-verification.json)", ""]
    if "preflight_cuda_forward_cost" in result:
        cost = result["preflight_cuda_forward_cost"]
        lines += ["## 사전 실행 GPU forward 실측", "",
            f'같은 TRAIN 단일 tensor1×3×640×640을 CUDA float32 eval/inference에서 원래 AUX·새 semantic 각각3회 예열·10회 측정했다. {cost["device_name"]}에서 원래 평균 {cost["original"]["mean_ms"]:.3f}ms·중앙 {cost["original"]["median_ms"]:.3f}ms, 새 평균 {cost["semantic"]["mean_ms"]:.3f}ms·중앙 {cost["semantic"]["median_ms"]:.3f}ms이다.',
            "모델 forward와 CPU dispatch/GPU 동기화를 포함하며 사진 디코딩·전처리·전송·네트워크는 제외했다. 학습 전0-head 모델의 단일 입력 실측이며 앱 Android 지연이나 현장 정확도 평가가 아니다.", ""]
    return "\n".join(lines)


def main():
    from scripts.facility_semantic_study import validate_protocol, protected_hashes
    from scripts.verify_facility_semantic import validate_test_results
    require(not any((ROOT / p).exists() for p in (OUTPUT_PATH, MARKDOWN_PATH)), "Preserve completed semantic reports")
    protocol = validate_protocol(read(ROOT / PROTOCOL_PATH), ROOT); protocol_sha = sha(ROOT / PROTOCOL_PATH)
    protected = protected_hashes(ROOT)
    require(not any((ROOT / "reports" / f"{name}-target-test.json").exists() for name in RUNS), "Source-VAL study refuses TEST results")
    loaded = [normalized_load_run(name) for name in RUNS]; trainings, entries = [v[0] for v in loaded], [v[1] for v in loaded]
    teacher, bn, semantic = validate_training(trainings[2], protocol, protocol_sha)
    control_bn = validate_batchnorm(trainings[1], protocol)
    require(trainings[0]["weights_sha256"] == protocol["initial_weights_sha256"], "Original0773 reference differs")
    histories = [read(ROOT / "runs" / name / "history.json") for name in RUNS[1:]]
    plan_path = ROOT / "reports/facility-spalling-sampler-dry-run.json"
    require(sha(plan_path) == protocol["sampler_plan_sha256"], "Original control draw plan changed")
    sampling = validate_history(*histories, protocol, read(plan_path), bn["initial_buffers_sha256"])
    rates = learning_rate_measurements(histories, trainings[1:], protocol)
    truth, normalizations = [], []
    for entry, load in zip(entries, loaded):
        run = ROOT / "runs" / entry["run"]; digests = {}; converted = 0
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run / f"validation-{domain}.json"), read(run / f"target-validation-{domain}-grid1.json"), entry, domain)
            wrapped, count = normalize_ap_positive_support({"ranking_ap": {domain: points}})
            entry["ranking_ap"][domain] = wrapped["ranking_ap"][domain]; converted += count; digests[domain] = digest
        truth.append(digests); normalizations.append({"run": entry["run"], "loader_support_fields_normalized": load[2], "verified_cache_support_fields_normalized": converted})
    require(truth[0] == truth[1] == truth[2], "Original full VAL truth/order changed")
    require(sha(ROOT / PRIOR_COMPARISON) == protocol["previous_comparison_sha256"], "Completed reused control report changed")
    prior = read(ROOT / PRIOR_COMPARISON)
    for old, entry in zip((prior["experiments"][0], prior["experiments"][2]), entries[:2]):
        require(all(old.get(k) == entry.get(k) for k in ("run", "weights_sha256", "per_class", "worst_error", "target_passed", "ranking_ap", "small_dacl_polygon_area_below_one_percent")), "Reused original/control measurements changed")
    for index, (entry, training, title) in enumerate(zip(entries, trainings, ("추가 학습 전 ROI 모델", "재사용 저학습률 대조군", "동결 ConvNeXt 특징 보강군"))):
        entry.update(title=title, imgsz=training["imgsz"], newly_trained_this_followup=index == 2)
        if index: entry["resources"] = {k: training[k] for k in ("elapsed_training_minutes", "peak_cuda_allocated_bytes", "optimizer_step_diagnostics")}
    measured = measurements(entries, protocol)
    for training, history in zip(trainings[1:], histories):
        require(all(type(training["optimizer_step_diagnostics"].get(k)) is int
                and training["optimizer_step_diagnostics"][k] == sum(row["optimizer_step_diagnostics"][k] for row in history)
                for k in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")), "Optimizer summary differs from actual epochs")
        require(type(training.get("elapsed_training_minutes")) in (int, float) and math.isfinite(training["elapsed_training_minutes"])
                and training["elapsed_training_minutes"] > 0 and type(training.get("peak_cuda_allocated_bytes")) is int
                and training["peak_cuda_allocated_bytes"] > 0, "Actual resource measurements are missing")
    proof_path = ROOT / VERIFICATION_PATH; proof = read(proof_path)
    technical = validate_technical(proof, protocol, protocol_sha, trainings[2], protected)
    validate_learning_rate_proof(proof.get("learning_rate_verification", {}), rates)
    from scripts.verify_facility_semantic import validate_forward_latency
    forward_cost = validate_forward_latency(proof.get("preflight", {}).get("actual_cuda_forward_latency", {}))
    orders = [row["sampled_row_indices_sha256"] for row in histories[1]]
    require(proof.get("actual_sampling_verification", {}).get("ordered_row_hashes_by_variant") == {"control": orders, "semantic": orders}
            and proof["actual_sampling_verification"].get("known_other_class_entries_each_arm") == sampling["known_other_class_entries_each_arm"],
            "Completion draw/known-mask proof differs from actual saved histories")
    require(proof.get("preflight_sha256") == sha(ROOT / PREFLIGHT_PATH)
            and proof.get("source_before_training_sha256") == sha(ROOT / SOURCE_RECORD_PATH)
            and proof.get("test_results_sha256") == sha(ROOT / TEST_RECORD_PATH), "Completion proof input bytes changed")
    require(validate_test_results(read(ROOT / TEST_RECORD_PATH), ROOT) == proof["tests"], "Actual test execution differs from completion proof")
    result = {"schema": "facility_semantic_study_comparison_v1", "protocol_sha256": protocol_sha, "experiments": entries,
        **measured, "semantic_feature_verification": semantic, "teacher_verification": teacher,
        "batchnorm_verification_by_variant": {"control": control_bn, "semantic": bn},
        "preflight_cuda_forward_cost": forward_cost,
        "learning_rate_verification": rates, "actual_sampling_verification": sampling, "original_validation_truth_sha256": truth[0],
        "technical_verification": technical, "technical_verification_sha256": sha(proof_path), "prepared_data_integrity": proof["prepared_data_integrity"],
        "app_profile_sha256": sha(ROOT / APP_PROFILE), "previous_comparison_sha256": protocol["previous_comparison_sha256"],
        "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0, "reused_control_training_epochs": 6,
        "previous_training_epochs_counted_again": False, "repeated_source_val_adaptation": True,
        "equal_compute_or_parameter_budget_asserted": False, "training_normalization_changed": False,
        "teacher_predictions_are_new_truth": False, "pretrained_features_are_new_damage_truth": False,
        "deployed": False, "app_model_promoted": False, "source_test_inference_executed": False,
        "additional_expert_confirmed_labels": 0, "label_changes": 0, "new_independent_photos": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
        "reporting_provenance": {"normalization": "Exact integral AP support floats only after original provenance validation", "normalizations_by_run": normalizations, "reporter_source_sha256": sha(Path(__file__))},
        "scope": "One frozen pretrained feature-package follow-up on repeated source-VAL; not factory accuracy or isolated semantic causality"}
    require(protected_hashes(ROOT) == protected, "Reporting changed original/control/profile bytes")
    for path, value in ((OUTPUT_PATH, result), (MARKDOWN_PATH, render(result))):
        with (ROOT / path).open("x", encoding="utf-8", newline="\n") as stream:
            if isinstance(value, str): stream.write(value)
            else: json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return result


if __name__ == "__main__": main()
