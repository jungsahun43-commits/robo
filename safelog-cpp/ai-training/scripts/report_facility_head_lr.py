"""Report one head-learning-rate follow-up with a reused frozen-BN control.

Only saved, validated source-VAL aggregates are published. Fixed teacher signals
are regularization, never new truth; there is no inference or app promotion.
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
from scripts.report_facility_retention import OTHER_CLASSES, RETENTION_GATE, normalized_load_run, retention_measurements, validate_teacher
from scripts.report_facility_resolution import validate_histories as identical_histories

RUNS = ("facility-presence-target-roi-control", "facility-presence-target-batchnorm-frozen",
        "facility-presence-target-head-lr-low")
PROTOCOL_PATH = "reports/facility-head-lr-study-protocol.json"
VERIFICATION_PATH = "reports/facility-head-lr-study-verification.json"
PREFLIGHT_PATH = "runs/facility-head-lr-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-head-lr-source-before-training.json"
TEST_RECORD_PATH = "runs/facility-head-lr-test-results.json"
OUTPUT_PATH = "reports/facility-head-lr-study-comparison.json"
MARKDOWN_PATH = "reports/FACILITY_HEAD_LR_STUDY_RESULTS_KO.md"
PRIOR_COMPARISON = "reports/facility-batchnorm-study-comparison.json"
APP_PROFILE = "reports/facility-inference-profile.json"


def learning_rate_measurements(history, training, control_training, protocol):
    """Check observed new rates; clearly label the source-derived old curve."""
    schedule = {"name": "CosineAnnealingLR", "T_max": 6, "eta_min": .000005, "weight_decay": .0002}
    require(training.get("optimizer_schedule") == schedule and training.get("head_lr_policy") == protocol["head_lr_policy"],
            "Actual optimizer/scheduler policy changed")
    require(type(control_training.get("head_lr")) in (int, float) and control_training["head_lr"] == .00025
            and type(control_training.get("backbone_lr")) in (int, float) and control_training["backbone_lr"] == .00004
            and protocol["head_lr"] == .0001 and protocol["backbone_lr"] == .00004,
            "Only the non-backbone initial LR may differ")
    curve = lambda initial: [.000005+(initial-.000005)*(1+math.cos(math.pi*step/6))/2 for step in range(7)]
    expected = {"control_head": curve(.00025), "candidate_head": curve(.0001), "backbone": curve(.00004)}
    policy = protocol["head_lr_policy"]
    require(policy.get("uniform_ratio_across_curve") is False, "A constant0.4 schedule scaling cannot be asserted")
    curves = policy.get("step0_to6_curves", {})
    require(set(curves) == set(expected), "The three exact declared schedule curves are required")
    for key, values in expected.items():
        require(isinstance(curves[key], list) and len(curves[key]) == 7
                and all(type(value) in (int, float) and math.isfinite(value)
                    and math.isclose(value, wanted, rel_tol=0., abs_tol=1e-15)
                    for value, wanted in zip(curves[key], values)), "Declared cosine schedule differs")
    require(isinstance(history, list) and len(history) == 6, "All six actual LR observations are required")
    observed = []
    for step, row in enumerate(history):
        point = row.get("optimizer_learning_rates", {})
        require(set(point) == {"backbone", "head"}, "Actual before-epoch optimizer LR observations are missing")
        for key, wanted in (("backbone", expected["backbone"][step]), ("head", expected["candidate_head"][step])):
            require(type(point[key]) in (int, float) and math.isfinite(point[key])
                    and math.isclose(point[key], wanted, rel_tol=0., abs_tol=1e-15), "Observed LR differs from fixed schedule")
        observed.append(dict(point))
    final = training.get("optimizer_final_learning_rates", {})
    require(set(final) == {"backbone", "head"} and all(type(value) in (int, float) and math.isfinite(value)
            and math.isclose(value, .000005, rel_tol=0., abs_tol=1e-15) for value in final.values()),
            "Actual rates after the sixth scheduler step must equal the unchanged floor")
    return {"candidate_actual_rates_verified": True, "reused_control_declared_rates_verified": True,
        "reused_control_rates_are_observations": False, "step0_to6_curves": expected,
        "candidate_before_epoch_observations": observed, "candidate_final_observed_rates": dict(final),
        "initial_head_lr_ratio": .4, "uniform_ratio_across_curve": False,
        "backbone_and_scheduler_floor_unchanged": True,
        "control_curve_provenance": "Derived from preserved control metadata, protocol and scheduler source; not old LR observations"}


def validate_learning_rate_proof(proof, measured):
    require(proof.get("candidate_actual_rates_verified") is True
            and proof.get("reused_control_declared_rates_verified") is True
            and proof.get("reused_control_rates_are_observations") is False
            and type(proof.get("candidate_observed_epochs")) is int and proof["candidate_observed_epochs"] == 6,
            "Actual new LR and source-derived control scope proof is required")
    require(proof.get("candidate_before_epoch_observations") == [{"epoch": i+1, **row} for i, row in enumerate(measured["candidate_before_epoch_observations"])]
            and proof.get("candidate_final_after_step6_observation") == measured["candidate_final_observed_rates"]
            and proof.get("declared_step0_to6_curves") == measured["step0_to6_curves"],
            "Actual completion LR observations differ from recorded optimizer rates")


def measurements(entries, protocol):
    require(isinstance(entries, list) and len(entries) == 3 and [entry.get("run") for entry in entries] == list(RUNS),
            "Original reference, reused frozen-BN head2.5e-4 and new head1e-4 are required")
    original = validate_measurements(entries, protocol)
    retention = retention_measurements(entries, protocol["retention_candidate_gate"])
    require(original["verified_known_class_ap_measurements"] == 42 and retention["other_class_ap_measurements"] == 24,
            "The three models must preserve all original known AP support")
    return {**original, "retention_gate": retention, "verified_known_other_class_ap_measurements": 24}


def validate_batchnorm(training, protocol):
    require(training.get("batchnorm_policy") == protocol["batchnorm_policy"], "Declared student BatchNorm policy changed")
    record = training.get("batchnorm_state_preservation", {})
    initial, final = record.get("initial_buffers_sha256"), record.get("final_buffers_sha256")
    require(isinstance(initial, str) and re.fullmatch(r"[a-f0-9]{64}", initial) and final == initial
            and initial == protocol["initial_batchnorm_buffers_sha256"],
            "Actual original student BN buffer hashes differ")
    expected = {"layer_count": 47, "channel_count": 12328, "buffer_tensor_count": 141,
                "affine_parameter_tensor_count": 94}
    for key, value in expected.items():
        require(type(record.get(key)) is int and record[key] == value, "Actual student BatchNorm inventory differs")
    for key in ("buffers_unchanged", "affine_parameters_trainable", "backbone_parameters_trainable", "head_parameters_trainable",
                "eps_and_momentum_unchanged", "all_batchnorm_eval", "policy_applied_after_each_model_train"):
        require(record.get(key) is True, "BatchNorm eval-buffer/trainable-weight proof is missing")
    return {**record, "training_normalization_uses_original_running_statistics": True,
        "affine_or_backbone_head_parameter_freezing_asserted": False}


def validate_history(control, frozen, protocol, plan, initial_bn_sha):
    sampling = identical_histories(control, frozen, protocol["requested_epochs"], protocol["draws_per_epoch"])
    require(len(plan["epochs"]) == len(frozen) == 6, "The same fixed six epochs are required")
    for left, right, prepared in zip(control, frozen, plan["epochs"]):
        order = prepared["control_order_sha256"]
        for row in (left, right):
            require(row["sampled_row_indices_sha256"] == order, "The original control draw schedule changed")
            fixed = row.get("fixed_sampling", {})
            require(fixed.get("draw_hash_matches_prepared") is True and fixed.get("declared_draw_sha256") == order
                    and type(fixed.get("changed_positions_from_control")) is int and fixed["changed_positions_from_control"] == 0,
                    "A sampling intervention entered the head-LR follow-up")
            audit = row.get("retention_distillation", {})
            require(type(audit.get("weight")) in (int, float) and audit["weight"] == 4.
                    and type(audit.get("temperature")) in (int, float) and audit["temperature"] == 2., "Fixed weight4/T2 differs")
            raw, weighted = audit.get("unweighted_mean_batch_loss"), audit.get("weighted_mean_batch_loss")
            require(type(raw) in (int, float) and math.isfinite(raw) and raw >= 0
                    and type(weighted) in (int, float) and math.isfinite(weighted)
                    and math.isclose(weighted, 4.*raw, rel_tol=0., abs_tol=1e-12), "Actual weighted teacher loss differs")
            require(type(audit.get("teacher_forward_batches")) is int
                    and audit["teacher_forward_batches"] == row["optimizer_step_diagnostics"]["attempted_batches"]
                    and type(audit.get("contributing_batches")) is int and 0 < audit["contributing_batches"] <= audit["teacher_forward_batches"]
                    and audit.get("excluded_primary_classes") == TARGETS
                    and all(audit.get(key) is True for key in ("teacher_eval", "teacher_frozen", "teacher_state_unchanged", "teacher_gradients_absent")),
                    "Actual unchanged teacher/masking evidence is missing")
            counts = row.get("sampled_photo_target_counts", {})
            require(set(counts) == set(CLASSES), "All-seven actual draw label counts are required")
            for point in counts.values():
                require(set(point) == {"positive", "negative", "unknown"}
                        and all(type(value) is int and value >= 0 for value in point.values())
                        and sum(point.values()) == protocol["draws_per_epoch"], "Invalid known/unknown exposure counts")
            known = sum(counts[label]["positive"]+counts[label]["negative"] for label in OTHER_CLASSES)
            require(type(audit.get("known_other_class_entries")) is int and audit["known_other_class_entries"] == known,
                    "Unknown source entries entered teacher supervision")
        require(left["sampled_photo_target_counts"] == right["sampled_photo_target_counts"], "Paired original seven-label exposure differs")
        bn = right.get("batchnorm_training", {})
        require(bn.get("buffer_sha256") == initial_bn_sha and bn.get("buffers_unchanged") is True
                and type(bn.get("actual_eval_layers")) is int and bn["actual_eval_layers"] == 47
                and type(bn.get("affine_trainable_tensors")) is int and bn["affine_trainable_tensors"] == 94
                and type(bn.get("affine_gradient_tensors")) is int and bn["affine_gradient_tensors"] == 94
                and type(bn.get("affine_gradient_nonzero_tensors")) is int and 1 <= bn["affine_gradient_nonzero_tensors"] <= 94
                and bn.get("affine_gradients_finite") is True and bn.get("policy_applied_after_model_train") is True,
                "Actual frozen BN buffers and last-batch learnable affine gradients must be verified each epoch")
    return {**sampling, "all_seven_photo_label_exposure_identical_each_epoch": True,
        "teacher_forward_batches_each_arm": sum(row["retention_distillation"]["teacher_forward_batches"] for row in frozen),
        "known_other_class_entries_each_arm": sum(row["retention_distillation"]["known_other_class_entries"] for row in frozen),
        "same_original_control_arrays": True, "new_completed_training_epochs": 6, "new_control_training_epochs": 0,
        "reused_frozen_batchnorm_weight4_control": True, "original_buffers_verified_all_six_epochs": True,
        "affine_gradient_evidence_scope": "Actual final minibatch of each epoch; not every gradient/update"}


def validate_training(training, protocol, protocol_sha):
    expected = {key: protocol[key] for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
        "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    expected.update(status="complete", actual_epochs=6, imgsz=640, model_variant="low_lr", distillation_weight=4.,
        architecture=protocol["architecture_by_variant"]["low_lr"], initial_weights_sha256=protocol["initial_weights_sha256"],
        source_sha256=protocol["source_sha256"], study_protocol_sha256=protocol_sha,
        core_spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"], spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        auxiliary_manifest_sha256=protocol["auxiliary_manifest_sha256"], sampling_intervention_applied=False,
        paired_draws_key="control", sampler_plan_sha256=protocol["sampler_plan_sha256"],
        private_draw_archive_sha256=protocol["paired_draws_sha256"], distillation_recipe=protocol["distillation_recipe"])
    expected["head_lr_policy"] = protocol["head_lr_policy"]
    for key, value in expected.items():
        require(type(training.get(key)) is type(value) and training[key] == value, f"Actual fixed BN study condition differs: {key}")
    return validate_teacher(training, protocol), validate_batchnorm(training, protocol)


def validate_technical(proof, protocol, protocol_sha, training, protected):
    require(proof.get("schema") == "facility_head_lr_study_verification_v1" and proof.get("status") == "passed",
            "Actual completed BatchNorm verification is required")
    require(proof.get("protocol_sha256") == protocol_sha and proof.get("source_sha256") == protocol["source_sha256"]
            and type(proof.get("runtime_source_count")) is int and proof["runtime_source_count"] == len(protocol["source_sha256"]),
            "Completion proof belongs to another frozen study")
    require(isinstance(proof.get("source_git_commit"), str) and re.fullmatch(r"[a-f0-9]{40}", proof["source_git_commit"]),
            "Actual before-training source commit is missing")
    for key in ("git_blob_bytes_verified", "working_runtime_sources_unchanged", "protected_files_unchanged", "existing_control_preserved"):
        require(proof.get(key) is True, "Actual original/source/control preservation proof is missing")
    require(type(proof.get("actual_completed_training_epochs")) is int and proof["actual_completed_training_epochs"] == 6
            and type(proof.get("reused_control_epochs")) is int and proof["reused_control_epochs"] == 6
            and type(proof.get("new_candidate_count")) is int and proof["new_candidate_count"] == 1
            and proof.get("control_training_repeated") is False and proof.get("verification_training_epochs") == 0,
            "Only one new six-epoch candidate may enter the budget")
    for key in ("source_test_inference_executed", "app_model_promoted", "deployed", "accuracy_measured_by_verifier"):
        require(proof.get(key) is False, "Completion verifier cannot claim TEST, promotion or field accuracy")
    for key in ("additional_expert_confirmed_labels", "label_changes", "new_photo_targets", "new_pixel_targets"):
        require(type(proof.get(key)) is int and proof[key] == 0, "BatchNorm or teacher output cannot create truth")
    require(proof.get("protected_file_sha256") == protected, "Protected prior/original files changed")
    require(proof.get("candidate_original_batchnorm_buffers_preserved_affine_learnable") is True,
            "Actual checkpoint original-BN-buffer/trainable-affine verification is required")
    reused = proof.get("reused_control", {})
    require(reused.get("weights_sha256") == protocol["reused_control_weights_sha256"]
            and reused.get("reused") is True and reused.get("training_repeated") is False
            and reused.get("previous_protocol_sha256") == protocol["previous_protocol_sha256"]
            and reused.get("previous_verification_sha256") == protocol["previous_verification_sha256"]
            and all(reused.get("cpu_reload", {}).get(key) is True for key in ("strict_factory_reload_verified", "all_outputs_finite",
                "public_output_equals_training_photo_output")), "Actual unchanged frozen-BN head2.5e-4 control identity/CPU proof differs")
    sampling = proof.get("actual_sampling_verification", {})
    require(type(sampling.get("epochs_compared")) is int and sampling["epochs_compared"] == 6
            and type(sampling.get("draws_per_epoch_each_arm")) is int and sampling["draws_per_epoch_each_arm"] == 14248
            and type(sampling.get("changed_positions")) is int and sampling["changed_positions"] == 0
            and sampling.get("teacher_forward_batches_by_variant") == {"control": 10686, "low_lr": 10686}
            and all(sampling.get(key) is True for key in ("both_actual_draw_orders_match_fixed_original_control_array",
                "actual_ordered_row_index_hashes_identical_each_epoch", "actual_domain_full_crop_joint_counts_identical_each_epoch",
                "actual_all_seven_photo_target_exposure_identical_each_epoch", "teacher_eval_frozen_no_grad_verified_both_arms")),
            "Actual unchanged draw/known-teacher verification is missing")
    tests = proof.get("tests", {})
    require(type(tests.get("tests_run")) is int and tests["tests_run"] > 0 and tests["tests_run"] == tests.get("expected_tests_collected")
            and all(type(tests.get(key)) is int and tests[key] == 0 for key in ("failures", "errors", "skipped"))
            and tests.get("source_sha256") == protocol["source_sha256"], "Actual complete focused tests must pass")
    data = proof.get("prepared_data_integrity", {})
    require(data.get("status") == "passed" and type(data.get("input_files_checked")) is int and data["input_files_checked"] > 0
            and data.get("all_original_input_sha_size_mtime_preserved") is True
            and data.get("original_train_images_masks_annotations_verified") is True
            and data.get("individual_input_paths_published") is False, "Actual original TRAIN input-byte proof is missing")
    points = proof.get("experiments", [])
    require(isinstance(points, list) and len(points) == 1, "Only the fresh frozen-BN candidate needs new checkpoint proof")
    point = points[0]
    require(point.get("variant") == "low_lr" and point.get("weights_sha256") == training["weights_sha256"]
            and type(point.get("actual_epochs")) is int and point["actual_epochs"] == 6
            and point.get("strict_state_inventory_verified") is True and point.get("state_tensor_count") == 324
            and type(point.get("new_state_tensor_count")) is int and point["new_state_tensor_count"] == 0
            and point.get("teacher_state_preservation") == training["teacher_state_preservation"]
            and point.get("batchnorm_state_preservation") == training["batchnorm_state_preservation"]
            and point.get("actual_checkpoint_bn_buffers_and_configuration_verified") is True
            and point.get("actual_lower_head_learning_rate_curve_verified") is True,
            "Actual candidate/teacher/BatchNorm state proof differs")
    require(all(point.get("cpu_reload", {}).get(key) is True for key in ("strict_factory_reload_verified", "all_outputs_finite",
        "public_output_equals_training_photo_output")), "Actual frozen-BN candidate CPU reload must pass")
    return {"status": "passed", "source_git_commit": proof["source_git_commit"], "runtime_source_count": proof["runtime_source_count"],
        "tests_run": tests["tests_run"], "actual_completed_training_epochs": 6}


def render(result):
    entries, change, retention = result["experiments"], result["comparisons"], result["retention_gate"]
    lr = result["learning_rate_verification"]
    lines = ["# 비backbone 초기 학습률 감소 후속 실험 결과", "",
        "이전 frozen-BN의 반복 source-VAL 결과를 확인한 뒤, 비backbone 초기 학습률만 줄이는 후보 하나를 선택했다. 이번 head LR1e-4 후보만 원래0773에서 새로6epoch 학습하고, 완료된 frozen-BN head LR2.5e-4 대조6epoch를 그대로 재사용했다. 새 대조 학습은0epoch이며 이전6epoch를 새 비용에 더하지 않았다.",
        "head 그룹은 기존 optimizer의 모든 비backbone student 파라미터다. 초기 LR만0.00025→0.0001로 낮추고 backbone LR0.00004, AdamW weight decay0.0002, CosineAnnealingLR T_max6·eta_min0.000005를 유지했다. 초기 head 비율은0.4지만 공통 최소값 때문에 전체 곡선이0.4배가 되는 실험은 아니다.",
        "두 군 모두 BatchNorm47개의 원래 running 통계·141개 버퍼를 유지한다. 학습 정규화 정책은 같고 γ/β94개 affine, backbone·head 가중치가 계속 학습 가능하다. teacher weight4/T2, 원래 주석·추출 배열·640 입력·80×80 마스크·증강 난수·기본 손실 가중치를 유지했다.",
        "원래 알려진 다른5종에 frozen0773 teacher의 Bernoulli KL을 적용하고 균열·박락·미확인 출처 항목은 증류 손실에서 제외했다. teacher 예측은 정규화 신호이며 새 정답이 아니다. 이전 관찰은 후보 선택의 이유이며 낮은 LR이 오류 원인이나 개선을 보장한다는 증거가 아니다.", "",
        "![Head learning rate comparison](facility-head-lr-study-comparison.png)", "",
        "| 모델 | 이번 새 epoch | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한5% 기준 |", "|---|---:|---:|---:|---:|---|"]
    for index, entry in enumerate(entries):
        lines.append(f'| {entry["title"]} | {6 if index == 2 else 0} | {entry["actual_epochs"]} | {entry["best_epoch"]} | {100*entry["worst_error"]:.2f}% | {"통과" if entry["target_passed"] else "미달"} |')
    lines += ["", f'head1e-4−초기 최대 오류 {change["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, head1e-4−재사용 head2.5e-4 대조 {change["maximum_error_treatment_minus_control_pp"]:+.2f}pp. 양수는 악화다.',
        "최대값은 균열·박락 × DACL710/Dam424/CODEBRIM611 × FNR/FPR의12개 비율 중 최대이며 전체 사진 오답 비율·앱 정확도가 아니다.", "",
        "## 변경한 학습률과 실제 관측", "",
        "| scheduler step | 대조 head LR(기존 소스로 재구성) | 새 head LR(실제 관측) | 동일 backbone LR |", "|---:|---:|---:|---:|"]
    curves = lr["step0_to6_curves"]
    for step in range(7):
        lines.append(f'| {step} | {curves["control_head"][step]:.9g} | {curves["candidate_head"][step]:.9g} | {curves["backbone"][step]:.9g} |')
    lines += ["", "step0~5의 새 LR은 각 epoch 시작 전 실제 optimizer에서 읽었고 step6은 마지막 scheduler.step 후 실제 그룹 LR이다. 기존 대조 RAW history에 LR 관측값이 없으므로 대조 곡선은 보존된 프로토콜·metadata·scheduler 소스로 계산했으며 과거 실제 관측으로 표시하지 않았다. 두 곡선은 마지막에 동일 floor5e-6에 도달한다.", "",
        "## 유지 기준과 AP", "",
        f'고정 유지 후보 기준: **{"통과" if retention["retention_candidate_nominated"] else "미달"}**. 다른 알려진 항목별 AP 하락이 초기 대비0.02 이하, DACL 철근 노출 AP 회복이 이번 frozen-BN head2.5e-4 대조 대비0.02 이상, 최대 균열·박락 오류 악화가 초기·이번 대조 각각 대비2pp 이하여야 한다. 이번 대조가 이미 높은 AP여도 회복 기준을 제외하거나 완화하지 않았다.', "",
        "| 출처 | 다른 항목 | 초기 AP | head2.5e-4 AP | head1e-4 AP | 새−초기 | 새−대조 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in retention["per_source_other_class_ap"]:
        lines.append(f'| {row["domain"]} | {LABELS[row["class"]]} | {row["reference_ap"]:.4f} | {row["control_ap"]:.4f} | {row["treatment_ap"]:.4f} | {row["treatment_minus_reference_ap"]:+.4f} | {row["treatment_minus_control_ap"]:+.4f} |')
    iron = retention["dacl_exposed_rebar"]
    lines += ["", f'DACL 철근 노출 AP: 초기 **{iron["reference_ap"]:.4f}**, 재사용 head2.5e-4 **{iron["control_ap"]:.4f}**, 새 head1e-4 **{iron["treatment_ap"]:.4f}**. 새−대조 회복 {iron["treatment_minus_control_ap"]:+.4f}.',
        "AP는 확률 순위 지표이며 정답률·오탐률이 아니다. 다른5종 알려진 조합8개×3모델=24개 AP, 전체7종 알려진 조합14개×3모델=42개 AP를 동일 정답·양성 분모에서 확인했다. 미확인 항목에0점 AP·정상 정답을 만들지 않았다.", "",
        "## 기존 연구 기준과 작은 손상", "",
        f'기존 연구 후보 기준: **{"통과" if result["research_gate"]["research_candidate_nominated"] else "미달"}**. 초기0773·이번 head2.5e-4 대조 각각 대비 최대 오류0.5pp 이상 개선, 각 target 오류 악화2pp 이하, 다른 알려진 AP 하락0.02 이하, 작은 FN합계2건 이상 감소·항목별 FNR 악화2pp 이하를 그대로 요구했다. 유지·연구·엄격한5%는 독립 기준이다.', "",
        "| 작은 손상 | 초기 FN/양성 | head2.5e-4 FN/양성 | head1e-4 FN/양성 |", "|---|---:|---:|---:|"]
    for task in TARGETS:
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{entry["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"]}/{entry["small_dacl_polygon_area_below_one_percent"][task]["positive_photos"]}' for entry in entries) + ' |')
    lines += ["", "분모는 균열93·박락105의198개 항목·사진 사례다. 동일 사진이 두 항목에 들어갈 수 있어 고유 사진198장·물리적 손상 크기로 설명하지 않는다.", "",
        "## 출처별 관측 오류", "",
        "Wilson95% 구간은 고정 예측·독립 사진 가정의 기술 통계다. 반복 VAL epoch·임계값·후속 LR 선택을 보정한 현장 보장·모델 간 유의성 검정이 아니다.", "",
        "| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | FP/음성 | 오탐률 |", "|---|---|---|---:|---:|---:|---:|"]
    titles = {entry["run"]: entry["title"] for entry in entries}
    for row in result["error_rows"]:
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% |')
    resource = entries[2]["resources"]; steps = resource["optimizer_step_diagnostics"]
    lines += ["", "## 이번 비용과 보존", "",
        f'이번 새 학습6epoch·대조군 재학습0epoch다. 학습·epoch 검증 {resource["elapsed_training_minutes"]:.2f}분, allocated peak {resource["peak_cuda_allocated_bytes"]/1024**3:.3f}GiB, 실제 update {steps["actual_optimizer_steps"]}, AMP skip {steps["amp_skipped_steps"]}회를 기록했다.',
        f'학습 전 commit의 {result["technical_verification"]["runtime_source_count"]}개 소스, 실제 코드 테스트 {result["technical_verification"]["tests_run"]}개와 새·재사용 가중치 CPU 재로딩·finite7/80/19 계약을 확인했다. 테스트 수는 정확도 사례 수가 아니다. 원래 TRAIN 입력 SHA·size·mtime, 모든7종 draw 노출·손실 가중치, teacher state·gradient 부재 및 두 군의 원래 BN141 버퍼를 유지했다.',
        "새 LR은 실제 optimizer 관측으로 고정 곡선과 비교했다. 각 epoch 마지막 minibatch의 BN affine94개 gradient 존재·유한성·일부 nonzero를 확인했으며 모든 minibatch의 gradient나 모든 실제 update를 검증했다는 뜻은 아니다. AP loader·캐시 검증 뒤 정확한 정수값 float 양성 개수만 int로 표시하며 원래 점수·정답·AP·임계값은 그대로 유지했다.", "",
        "## 적용 상태와 범위", "",
        "보류 TEST 추론·자동 배포·앱 모델 승격은 하지 않았다. 기본 facility-validation-v2와 프로필 SHA를 유지했다. 새 독립 사진·라벨 변경·전문가 확정 라벨·새 사진/픽셀 정답은0개다.",
        "한 seed, 공개 교량·댐 콘크리트, 반복 source-VAL에 따른 LR 선택과 사진 존재 분류 연구다. 공장 사진·정밀 위치·구조 안전·미래 현장 오차는 측정하지 않았다. 독립 평가·현장5% 미만 보장으로 설명하지 않으며 최종 판단은 점검자가 한다.", "",
        "[고정 LR 정책](facility-head-lr-study-protocol.json), [실측 집계](facility-head-lr-study-comparison.json), [기술 검증](facility-head-lr-study-verification.json), [재사용 frozen-BN 결과](FACILITY_BATCHNORM_STUDY_RESULTS_KO.md)", ""]
    return "\n".join(lines)


def main():
    from scripts.facility_head_lr_study import validate_protocol, protected_hashes
    from scripts.verify_facility_head_lr import validate_test_results
    require(not any((ROOT/path).exists() for path in (OUTPUT_PATH, MARKDOWN_PATH)), "Preserve completed BN reporting evidence")
    protocol = validate_protocol(read(ROOT/PROTOCOL_PATH), ROOT); protocol_sha = sha(ROOT/PROTOCOL_PATH)
    protected = protected_hashes(ROOT)
    require(not any((ROOT/"reports"/f"{name}-target-test.json").exists() for name in RUNS), "Source-VAL study refuses TEST inference records")
    loaded = [normalized_load_run(name) for name in RUNS]
    trainings, entries = [row[0] for row in loaded], [row[1] for row in loaded]
    teacher, bn = validate_training(trainings[2], protocol, protocol_sha)
    require(trainings[0]["weights_sha256"] == protocol["initial_weights_sha256"], "Original0773 reference differs")
    histories = [read(ROOT/"runs"/name/"history.json") for name in RUNS[1:]]
    plan_path = ROOT/"reports/facility-spalling-sampler-dry-run.json"
    require(sha(plan_path) == protocol["sampler_plan_sha256"], "Original control sampling plan changed")
    sampling = validate_history(*histories, protocol, read(plan_path), bn["initial_buffers_sha256"])
    learning_rates = learning_rate_measurements(histories[1], trainings[2], trainings[1], protocol)
    truth, normalization = [], []
    for entry, loaded_row in zip(entries, loaded):
        run = ROOT/"runs"/entry["run"]; digests = {}; converted = 0
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run/f"validation-{domain}.json"), read(run/f"target-validation-{domain}-grid1.json"), entry, domain)
            wrapped, count = normalize_ap_positive_support({"ranking_ap": {domain: points}})
            entry["ranking_ap"][domain] = wrapped["ranking_ap"][domain]; converted += count; digests[domain] = digest
        truth.append(digests); normalization.append({"run": entry["run"], "loader_support_fields_normalized": loaded_row[2], "verified_cache_support_fields_normalized": converted})
    require(truth[0] == truth[1] == truth[2], "Original full VAL truth/order differs")
    prior_path = ROOT/PRIOR_COMPARISON
    require(sha(prior_path) == protocol["previous_comparison_sha256"], "Previous final weight4 comparison changed")
    prior = read(prior_path)
    for old, entry in zip((prior["experiments"][0], prior["experiments"][2]), entries[:2]):
        require(all(old.get(key) == entry.get(key) for key in ("run", "weights_sha256", "per_class", "worst_error", "target_passed", "ranking_ap", "small_dacl_polygon_area_below_one_percent")),
                "Reused original/frozen-BN head2.5e-4 metrics changed")
    for index, (entry, training, title) in enumerate(zip(entries, trainings,
            ("추가 학습 전 ROI 모델", "재사용 head LR2.5e-4 대조군", "후속 head LR1e-4 보강군"))):
        entry.update(title=title, imgsz=training["imgsz"], newly_trained_this_followup=index == 2)
        if index:
            entry["resources"] = {key: training[key] for key in ("elapsed_training_minutes", "peak_cuda_allocated_bytes", "optimizer_step_diagnostics")}
        if index == 2:
            entry.update(teacher_verification=teacher, batchnorm_verification=bn)
    summary = measurements(entries, protocol)
    steps = trainings[2]["optimizer_step_diagnostics"]
    require(all(type(steps.get(key)) is int and steps[key] == sum(row["optimizer_step_diagnostics"][key] for row in histories[1])
                for key in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")), "New candidate optimizer summary differs")
    require(type(trainings[2].get("elapsed_training_minutes")) in (int, float) and math.isfinite(trainings[2]["elapsed_training_minutes"])
            and trainings[2]["elapsed_training_minutes"] > 0 and type(trainings[2].get("peak_cuda_allocated_bytes")) is int
            and trainings[2]["peak_cuda_allocated_bytes"] > 0, "Actual resource measurements are missing")
    proof_path = ROOT/VERIFICATION_PATH; proof = read(proof_path)
    technical = validate_technical(proof, protocol, protocol_sha, trainings[2], protected)
    validate_learning_rate_proof(proof.get("learning_rate_verification", {}), learning_rates)
    order = [row["sampled_row_indices_sha256"] for row in histories[1]]
    require(proof["actual_sampling_verification"]["ordered_row_hashes_by_variant"] == {"control": order, "low_lr": order}
            and proof["actual_sampling_verification"]["known_other_class_entries_each_arm"] == sampling["known_other_class_entries_each_arm"],
            "Completion sampler/known-mask totals differ from saved histories")
    require(proof.get("preflight_sha256") == sha(ROOT/PREFLIGHT_PATH) and proof.get("source_before_training_sha256") == sha(ROOT/SOURCE_RECORD_PATH)
            and proof.get("test_results_sha256") == sha(ROOT/TEST_RECORD_PATH), "Actual completion evidence changed")
    preflight = read(ROOT/PREFLIGHT_PATH)
    require(proof["prepared_data_integrity"]["ledger_sha256"] == preflight["protected_input_ledger_sha256"]
            == sha(ROOT/preflight["protected_input_ledger_path"]), "Original input ledger changed")
    require(validate_test_results(read(ROOT/TEST_RECORD_PATH), ROOT) == proof["tests"], "Executed tests differ from completion proof")
    result = {"schema": "facility_head_lr_study_comparison_v1", "protocol_sha256": protocol_sha, "experiments": entries,
        **summary, "actual_sampling_verification": sampling, "original_validation_truth_sha256": truth[0],
        "learning_rate_verification": learning_rates,
        "technical_verification": technical, "technical_verification_sha256": sha(proof_path), "prepared_data_integrity": proof["prepared_data_integrity"],
        "app_profile_sha256": sha(ROOT/APP_PROFILE), "previous_comparison_sha256": protocol["previous_comparison_sha256"],
        "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0, "reused_control_training_epochs": 6,
        "previous_training_epochs_counted_again": False, "head_lr_chosen_after_previous_source_val_results": True,
        "repeated_source_val_adaptation": True, "training_normalization_changed": False, "teacher_predictions_are_new_truth": False,
        "deployed": False, "app_model_promoted": False, "source_test_inference_executed": False,
        "additional_expert_confirmed_labels": 0, "label_changes": 0, "new_independent_photos": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
        "reporting_provenance": {"normalization": "finite exact integral positive_photos floats only after original provenance validation", "normalizations_by_run": normalization,
            "reporter_source_sha256": sha(Path(__file__))}, "scope": "One head-LR source-VAL-adaptive follow-up; both arms frozen-BN weight4; not independent factory accuracy"}
    require(protected_hashes(ROOT) == protected, "Reporting changed original/previous/profile bytes")
    with (ROOT/OUTPUT_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with (ROOT/MARKDOWN_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render(result))
    print(json.dumps({"actual_new_epochs": 6, "new_control_epochs": 0, "maximum_validation_error": [entry["worst_error"] for entry in entries],
        "retention_candidate_nominated": result["retention_gate"]["retention_candidate_nominated"], "research_candidate_nominated": result["research_gate"]["research_candidate_nominated"],
        "strict_target_passed": [entry["target_passed"] for entry in entries], "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
