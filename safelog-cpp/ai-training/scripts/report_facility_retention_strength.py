"""Report the one-candidate weight4 follow-up using an unchanged prior control.

Saved source-VAL predictions are verified; nothing here performs inference,
edits labels, promotes an app model or treats a teacher prediction as truth.
"""
from __future__ import annotations

from copy import deepcopy
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
from scripts.report_facility_retention import (OTHER_CLASSES, RETENTION_GATE, normalized_load_run,
    retention_measurements, validate_teacher)
from scripts.report_facility_resolution import validate_histories as identical_histories

RUNS = ("facility-presence-target-roi-control", "facility-presence-target-retention-control",
        "facility-presence-target-retention-distill", "facility-presence-target-retention-strong")
PROTOCOL_PATH = "reports/facility-retention-strength-study-protocol.json"
VERIFICATION_PATH = "reports/facility-retention-strength-study-verification.json"
PREFLIGHT_PATH = "runs/facility-retention-strength-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-retention-strength-source-before-training.json"
TEST_RECORD_PATH = "runs/facility-retention-strength-test-results.json"
OUTPUT_PATH = "reports/facility-retention-strength-study-comparison.json"
MARKDOWN_PATH = "reports/FACILITY_RETENTION_STRENGTH_STUDY_RESULTS_KO.md"
APP_PROFILE = "reports/facility-inference-profile.json"
PREVIOUS_COMPARISON_PATH = "reports/facility-retention-study-comparison.json"


def measurements(entries, protocol):
    """Keep original gates while counting four unique models only once."""
    require(isinstance(entries, list) and len(entries) == 4
            and [entry.get("run") for entry in entries] == list(RUNS), "Exactly four named measured models are required")
    primary = validate_measurements([entries[0], entries[1], entries[3]], protocol)
    previous_protocol = deepcopy(protocol); previous_protocol["treatment"] = RUNS[2]
    previous = validate_measurements(entries[:3], previous_protocol)
    require(primary["verified_known_class_ap_measurements"] == previous["verified_known_class_ap_measurements"] == 42,
            "All original known source labels must be measured")
    strong_retention = retention_measurements([entries[0], entries[1], entries[3]], protocol["retention_candidate_gate"])
    previous_retention = retention_measurements(entries[:3], protocol["retention_candidate_gate"])
    other = []
    for old, new in zip(previous_retention["per_source_other_class_ap"], strong_retention["per_source_other_class_ap"]):
        require((old["domain"], old["class"], old["known_photos"], old["positive_photos"])
                == (new["domain"], new["class"], new["known_photos"], new["positive_photos"]),
                "Four-model AP source/support differs")
        other.append({"domain": new["domain"], "class": new["class"], "known_photos": new["known_photos"],
            "positive_photos": new["positive_photos"], "reference_ap": new["reference_ap"],
            "control_ap": new["control_ap"], "weight1_ap": old["treatment_ap"], "weight4_ap": new["treatment_ap"],
            "weight4_minus_weight1_ap": new["treatment_ap"]-old["treatment_ap"]})
    require(len(other) == 8, "Eight known other-class combinations per model are required")
    worst = [previous_retention["worst_other_class_ap_decline_by_model"][0],
        previous_retention["worst_other_class_ap_decline_by_model"][1],
        previous_retention["worst_other_class_ap_decline_by_model"][2],
        strong_retention["worst_other_class_ap_decline_by_model"][2]]
    increment = {"maximum_error_weight4_minus_weight1_pp": 100*(entries[3]["worst_error"]-entries[2]["worst_error"]),
        "dacl_exposed_rebar_weight4_minus_weight1_ap": strong_retention["dacl_exposed_rebar"]["treatment_ap"]
            -previous_retention["dacl_exposed_rebar"]["treatment_ap"],
        "small_false_negative_cases_weight4_minus_weight1": sum(entries[3]["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"]
            -entries[2]["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"] for task in TARGETS),
        "descriptive_only_no_additional_promotion_gate": True}
    # The original descriptive helper deliberately accepts exactly three
    # models. Preserve its validated rows and append only the fourth model;
    # neither relax its contract nor count shared reference/control rows twice.
    shared_previous = [row for row in previous["error_rows"] if row["run"] in RUNS[:2]]
    shared_primary = [row for row in primary["error_rows"] if row["run"] in RUNS[:2]]
    require(shared_previous == shared_primary, "Shared measured reference/control descriptive rows differ")
    all_rows = previous["error_rows"] + [row for row in primary["error_rows"] if row["run"] == RUNS[3]]
    require(len(all_rows) == 24, "Exactly24 unique model/source/target descriptive rows are required")
    return {**primary, "error_rows": all_rows, "verified_known_class_ap_measurements": 56,
        "verified_known_other_class_ap_measurements": 32, "known_combinations_each_model": {"all_seven": 14, "other_five": 8},
        "retention_gate": strong_retention, "previous_weight1_retention_gate": previous_retention,
        "previous_weight1_research_gate": previous["research_gate"], "four_model_other_class_ap": other,
        "worst_other_class_ap_decline_by_model": worst, "increment_vs_weight1": increment}


def validate_histories(control, strong, protocol, plan):
    """Reuse the fixed control; require the same original draws and weight4."""
    same = identical_histories(control, strong, protocol["requested_epochs"], protocol["draws_per_epoch"])
    require(len(plan["epochs"]) == len(strong) == 6, "The fixed six-epoch schedules are required")
    for left, right, point in zip(control, strong, plan["epochs"]):
        order = point["control_order_sha256"]
        for row, weight in ((left, 0.), (right, 4.)):
            require(row["sampled_row_indices_sha256"] == order, "Original control order differs")
            fixed = row.get("fixed_sampling", {})
            require(fixed.get("declared_draw_sha256") == order and fixed.get("draw_hash_matches_prepared") is True
                    and type(fixed.get("changed_positions_from_control")) is int and fixed["changed_positions_from_control"] == 0,
                    "A sampler intervention entered the strength follow-up")
            audit = row.get("retention_distillation", {})
            require(type(audit.get("weight")) in (int, float) and audit["weight"] == weight
                    and type(audit.get("temperature")) in (int, float) and audit["temperature"] == 2., "Fixed weight or T differs")
            raw, weighted = audit.get("unweighted_mean_batch_loss"), audit.get("weighted_mean_batch_loss")
            require(type(raw) in (int, float) and math.isfinite(raw) and raw >= 0
                    and type(weighted) in (int, float) and math.isfinite(weighted)
                    and math.isclose(weighted, weight*raw, rel_tol=0., abs_tol=1e-12), "Measured weighted teacher loss differs")
            require(type(audit.get("teacher_forward_batches")) is int
                    and audit["teacher_forward_batches"] == row["optimizer_step_diagnostics"]["attempted_batches"]
                    and type(audit.get("contributing_batches")) is int
                    and 0 < audit["contributing_batches"] <= audit["teacher_forward_batches"]
                    and audit.get("excluded_primary_classes") == TARGETS
                    and all(audit.get(key) is True for key in ("teacher_eval", "teacher_frozen", "teacher_gradients_absent", "teacher_state_unchanged")),
                    "Actual fixed teacher/masking proof is required")
            counts = row.get("sampled_photo_target_counts", {})
            require(set(counts) == set(CLASSES), "All-seven target exposure counts are required")
            for count in counts.values():
                require(set(count) == {"positive", "negative", "unknown"}
                        and all(type(value) is int and value >= 0 for value in count.values())
                        and sum(count.values()) == protocol["draws_per_epoch"], "Invalid actual target exposure")
            known = sum(counts[label]["positive"]+counts[label]["negative"] for label in OTHER_CLASSES)
            require(type(audit.get("known_other_class_entries")) is int and audit["known_other_class_entries"] == known,
                    "Unknown source entries entered teacher supervision")
        require(left["sampled_photo_target_counts"] == right["sampled_photo_target_counts"]
                and left["retention_distillation"]["contributing_batches"] == right["retention_distillation"]["contributing_batches"],
                "Matched all-seven labels/masking differ")
    return {**same, "both_orders_match_original_control_arrays": True,
        "actual_all_seven_photo_target_exposure_identical_each_epoch": True,
        "teacher_forward_batches_each_arm": sum(row["retention_distillation"]["teacher_forward_batches"] for row in strong),
        "known_other_class_entries_each_arm": sum(row["retention_distillation"]["known_other_class_entries"] for row in strong),
        "control_reused_from_completed_prior_study": True, "new_completed_training_epochs": 6,
        "new_control_training_epochs": 0, "teacher_predictions_are_new_truth": False}


def validate_training(training, protocol, protocol_sha):
    expected = {key: protocol[key] for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
        "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    expected.update(status="complete", actual_epochs=6, imgsz=640, model_variant="strong", distillation_weight=4.,
        architecture=protocol["architecture_by_variant"]["strong"],
        initial_weights_sha256=protocol["initial_weights_sha256"], source_sha256=protocol["source_sha256"],
        study_protocol_sha256=protocol_sha, core_spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"], auxiliary_manifest_sha256=protocol["auxiliary_manifest_sha256"],
        sampling_intervention_applied=False, paired_draws_key="control", sampler_plan_sha256=protocol["sampler_plan_sha256"],
        private_draw_archive_sha256=protocol["paired_draws_sha256"], distillation_recipe=protocol["distillation_recipe"])
    for key, value in expected.items():
        require(type(training.get(key)) is type(value) and training[key] == value, f"Actual fixed strength condition differs: {key}")
    return validate_teacher(training, protocol)


def validate_technical(proof, protocol, protocol_sha, training, protected):
    require(proof.get("schema") == "facility_retention_strength_study_verification_v1" and proof.get("status") == "passed",
            "Actual completed strength verification is required")
    require(proof.get("protocol_sha256") == protocol_sha and proof.get("source_sha256") == protocol["source_sha256"]
            and type(proof.get("runtime_source_count")) is int and proof["runtime_source_count"] == len(protocol["source_sha256"]),
            "Technical proof belongs to another frozen follow-up")
    require(isinstance(proof.get("source_git_commit"), str) and re.fullmatch(r"[a-f0-9]{40}", proof["source_git_commit"]),
            "Before-training Git proof is missing")
    for key in ("git_blob_bytes_verified", "working_runtime_sources_unchanged", "protected_files_unchanged"):
        require(proof.get(key) is True, "Actual frozen source/protection proof is missing")
    require(type(proof.get("actual_completed_training_epochs")) is int and proof["actual_completed_training_epochs"] == 6
            and type(proof.get("verification_training_epochs")) is int and proof["verification_training_epochs"] == 0,
            "Only the new six strength epochs can enter follow-up cost")
    require(type(proof.get("reused_control_epochs")) is int and proof["reused_control_epochs"] == 6
            and type(proof.get("new_candidate_count")) is int and proof["new_candidate_count"] == 1
            and proof.get("control_training_repeated") is False and proof.get("existing_control_preserved") is True,
            "The unchanged completed control must be reused once")
    for key in ("source_test_inference_executed", "app_model_promoted", "deployed", "accuracy_measured_by_verifier"):
        require(proof.get(key) is False, "Technical proof cannot claim TEST, app promotion or accuracy")
    for key in ("additional_expert_confirmed_labels", "label_changes", "new_photo_targets", "new_pixel_targets"):
        require(type(proof.get(key)) is int and proof[key] == 0, "Teacher predictions cannot create truth")
    require(proof.get("protected_file_sha256") == protected, "Protected previous/original files changed")
    require(proof.get("both_teachers_unchanged_eval_frozen_no_grad") is True, "Both teacher states must remain frozen")
    sampling = proof.get("actual_sampling_verification", {})
    require(type(sampling.get("epochs_compared")) is int and sampling["epochs_compared"] == 6
            and type(sampling.get("draws_per_epoch_each_arm")) is int and sampling["draws_per_epoch_each_arm"] == 14248
            and type(sampling.get("changed_positions")) is int and sampling["changed_positions"] == 0
            and sampling.get("teacher_forward_batches_by_variant") == {"control": 10686, "strong": 10686}
            and all(sampling.get(key) is True for key in ("both_actual_draw_orders_match_fixed_original_control_array",
                "actual_ordered_row_index_hashes_identical_each_epoch", "actual_domain_full_crop_joint_counts_identical_each_epoch",
                "actual_all_seven_photo_target_exposure_identical_each_epoch", "teacher_eval_frozen_no_grad_verified_both_arms")),
            "Actual six-epoch original-draw/masking verification is missing")
    reused = proof.get("reused_control", {})
    require(reused.get("variant") == "control" and reused.get("reused") is True and reused.get("training_repeated") is False
            and reused.get("weights_sha256") == protocol["reused_control_weights_sha256"]
            and reused.get("previous_protocol_sha256") == protocol["previous_protocol_sha256"]
            and reused.get("previous_verification_sha256") == protocol["previous_verification_sha256"]
            and type(reused.get("actual_epochs")) is int and reused["actual_epochs"] == 6
            and all(reused.get("cpu_reload", {}).get(key) is True for key in ("strict_factory_reload_verified", "all_outputs_finite",
                "public_output_equals_training_photo_output")), "Actual unchanged prior-control CPU proof differs")
    tests = proof.get("tests", {})
    require(type(tests.get("tests_run")) is int and tests["tests_run"] > 0 and tests["tests_run"] == tests.get("expected_tests_collected")
            and all(type(tests.get(key)) is int and tests[key] == 0 for key in ("failures", "errors", "skipped"))
            and tests.get("source_sha256") == protocol["source_sha256"], "All focused tests must actually execute and pass")
    for path, expected in tests.get("test_source_sha256", {}).items():
        require(sha(ROOT / path) == expected, "Executed focused test source changed")
    data = proof.get("prepared_data_integrity", {})
    require(data.get("status") == "passed" and data.get("all_original_input_sha_size_mtime_preserved") is True
            and data.get("original_train_images_masks_annotations_verified") is True
            and type(data.get("input_files_checked")) is int and data["input_files_checked"] > 0
            and data.get("individual_input_paths_published") is False, "Actual original input-byte proof is missing")
    points = proof.get("experiments", [])
    require(isinstance(points, list) and len(points) == 1, "Only the fresh strength checkpoint must be verified")
    point = points[0]
    require(point.get("variant") == "strong" and point.get("weights_sha256") == training["weights_sha256"]
            and type(point.get("actual_epochs")) is int and point["actual_epochs"] == 6
            and point.get("strict_state_inventory_verified") is True and point.get("state_tensor_count") == 324
            and type(point.get("new_state_tensor_count")) is int and point["new_state_tensor_count"] == 0
            and point.get("teacher_state_preservation") == training["teacher_state_preservation"],
            "New checkpoint or actual fixed teacher identity differs")
    require(all(point.get("cpu_reload", {}).get(key) is True for key in ("strict_factory_reload_verified", "all_outputs_finite",
            "public_output_equals_training_photo_output")), "Actual strength checkpoint CPU reload must pass")
    return {"status": "passed", "source_git_commit": proof["source_git_commit"], "runtime_source_count": proof["runtime_source_count"],
        "tests_run": tests["tests_run"], "actual_completed_training_epochs": 6}


def render(result):
    entries, measured, retention = result["experiments"], result["comparisons"], result["retention_gate"]
    lines = ["# 항목 유지 증류 강도4 후속 학습 결과", "",
        "이전 weight1의 source-VAL 결과를 확인한 뒤, 동일 조건의 weight4 후보 하나를 선택했다. 이번 후보의 조건은 새 학습 전에 고정했지만 앞선 VAL에 따른 적응적 후속 실험이므로 독립 확인 실험은 아니다.",
        "weight4만 원래0773 초기 모델에서 새로6epoch 학습했다. 완료된 weight0 대조군6epoch와 이전 weight1 6epoch는 그대로 재사용했다. 새 대조군을 재학습하거나 이전12epoch를 이번 비용에 다시 더하지 않았다.",
        "원래 control 추출 배열·동일 증강 난수·원본 주석·640 입력·80×80 마스크·기존7종 출력·19종 보조 태그를 유지했다. 0773 eval·frozen teacher의 온도2 Bernoulli KL을 원래 알려진 다른5종에만 적용하며 weight만1→4로 바꿨다. 균열·박락과 미확인 출처 항목은 손실에서 제외했다. teacher 예측은 soft 정규화 신호이고 새 정답이 아니다.", "",
        "![Retention strength follow-up](facility-retention-strength-study-comparison.png)", "",
        "| 모델 | 이번 새 epoch | 기존/이번 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한5% 기준 |",
        "|---|---:|---:|---:|---:|---|"]
    for index, entry in enumerate(entries):
        lines.append(f'| {entry["title"]} | {6 if index == 3 else 0} | {entry["actual_epochs"]} | {entry["best_epoch"]} | {100*entry["worst_error"]:.2f}% | {"통과" if entry["target_passed"] else "미달"} |')
    lines += ["", f'weight4−초기 최대 오류 {measured["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, weight4−재사용 대조 {measured["maximum_error_treatment_minus_control_pp"]:+.2f}pp, weight4−이전 weight1 {result["increment_vs_weight1"]["maximum_error_weight4_minus_weight1_pp"]:+.2f}pp. 양수는 악화다.',
        "최대값은 균열·박락 × DACL710/Dam424/CODEBRIM611 × FNR/FPR의12개 비율 중 최대다. 전체 사진 오답 비율이나 앱 정확도가 아니다.", "",
        "## 원래 유지 기준과 다른 항목 AP", "",
        f'고정된 유지 후보 기준: **{"통과" if retention["retention_candidate_nominated"] else "미달"}**. 다른 알려진 항목 각각의 AP 하락이 초기 대비0.02 이하, DACL 철근 노출 AP 회복이 재사용 weight0 대비0.02 이상, 최대 균열·박락 오류 악화가 초기·재사용 대조 각각 대비2pp 이하여야 한다. 기준을 완화하지 않았다.', "",
        "| 출처 | 다른 항목 | 초기 AP | 대조0 AP | 이전1 AP | 후속4 AP | 4−1 AP |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["four_model_other_class_ap"]:
        lines.append(f'| {row["domain"]} | {LABELS[row["class"]]} | {row["reference_ap"]:.4f} | {row["control_ap"]:.4f} | {row["weight1_ap"]:.4f} | {row["weight4_ap"]:.4f} | {row["weight4_minus_weight1_ap"]:+.4f} |')
    iron = next(row for row in result["four_model_other_class_ap"] if row["domain"] == "dacl" and row["class"] == "exposed_rebar")
    lines += ["", f'DACL 철근 노출 AP: 초기 **{iron["reference_ap"]:.4f}**, 대조0 **{iron["control_ap"]:.4f}**, 이전1 **{iron["weight1_ap"]:.4f}**, 후속4 **{iron["weight4_ap"]:.4f}**. 후속4−이전1 {iron["weight4_minus_weight1_ap"]:+.4f}; 후속4−대조0 {iron["weight4_ap"]-iron["control_ap"]:+.4f}.', "",
        "| 모델 | 다른 알려진 항목 최대 AP 하락(초기 대비) |", "|---|---:|"]
    titles = {entry["run"]: entry["title"] for entry in entries}
    for row in result["worst_other_class_ap_decline_by_model"]:
        lines.append(f'| {titles[row["run"]]} | {row["maximum_other_ap_drop_vs_reference"]:.4f} |')
    lines += ["", "AP는 확률 순위 지표이며 정답률·오탐률이 아니다. 다른5종의 알려진 출처 조합은 DACL5·Dam0·CODEBRIM3=8개로 네 모델32개 AP를 비교했다. 전체7종의 알려진 조합은14개로 네 모델56개 AP를 확인했다. 미확인 항목에0점 AP·정상 정답을 만들지 않았다.", "",
        "## 기존 연구 기준과 작은 손상", "",
        f'기존 연구 후보 기준: **{"통과" if result["research_gate"]["research_candidate_nominated"] else "미달"}**. 원래 초기0773·재사용 weight0 대조 각각 대비 최대 오류0.5pp 이상 개선, 각 target 오류 악화2pp 이하, 다른 알려진 AP 하락0.02 이하, 작은 사례 FN합계2건 이상 감소·항목별 FNR 악화2pp 이하를 그대로 계산했다.',
        "이전 weight1과의 추가 차이는 기술 통계이며 새 승격 기준을 만들지 않았다. 유지 기준·기존 연구 기준·엄격한5% 통과를 서로 대체하지 않는다.", "",
        "| 작은 손상 | 초기 FN/양성 | 대조0 FN/양성 | 이전1 FN/양성 | 후속4 FN/양성 |", "|---|---:|---:|---:|---:|"]
    for task in TARGETS:
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{entry["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"]}/{entry["small_dacl_polygon_area_below_one_percent"][task]["positive_photos"]}' for entry in entries) + ' |')
    lines += ["", "분모는 균열93·박락105의198개 항목·사진 사례다. 같은 사진이 두 항목에 들어갈 수 있고 고유 사진198장·실제 손상 크기를 뜻하지 않는다.", "",
        "## 출처별 관측 오류", "",
        "Wilson95% 구간은 고정 예측·독립 사진 가정의 기술 통계다. 반복 사용한 VAL의 epoch·임계값 선택과 후속 강도 선택을 보정한 독립 현장 보장·모델 간 유의성 검정이 아니다.", "",
        "| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | FP/음성 | 오탐률 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in result["error_rows"]:
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% |')
    resource = entries[3]["resources"]; steps = resource["optimizer_step_diagnostics"]
    lines += ["", "## 이번 실제 비용과 보존", "",
        f'새 weight4 학습·epoch 검증 {resource["elapsed_training_minutes"]:.2f}분, allocated peak {resource["peak_cuda_allocated_bytes"]/1024**3:.3f}GiB, 실제 update {steps["actual_optimizer_steps"]}, AMP skip {steps["amp_skipped_steps"]}, teacher forward {entries[3]["teacher_verification"]["actual_teacher_forward_batches"]}회다. 이번 새 완료 epoch는6, 새 대조군 학습은0이다.',
        f'학습 전 commit의 {result["technical_verification"]["runtime_source_count"]}개 소스와 실제 코드 테스트 {result["technical_verification"]["tests_run"]}개를 확인했다. 테스트 수는 정확도 사례 수가 아니다. 새 가중치를 CPU에서 엄격하게 재로딩하고 finite7/80/19 계약을 검증했다.',
        "원래 full TRAIN14,248·전체26,289행의 이미지·마스크·known·주석·auxiliary 태그·기본 추출 가중치·손실 양성 가중치를 유지했다. 재사용 대조·이전 증류 가중치·기존61개 소스 바이트와 teacher 초기/최종 state SHA·eval·gradient 부재를 보존했다. 양쪽 추출 순서와 모든7종 정답 노출은 같다.",
        "원래 loader·캐시 검증 뒤 유한·비음수·known 분모 이내의 정확한 정수값 float AP 양성 개수만 int로 표시했다. bool·소수·NaN·무한대는 거부하며 점수·정답·AP·임계값·기존 기준을 바꾸지 않았다.", "",
        "## 적용 상태와 한계", "",
        "보류 TEST 추론·자동 배포·앱 모델 승격을 하지 않았다. 앱 기본 `facility-validation-v2`와 프로필 SHA를 유지했다. 새 독립 사진·원본 라벨 변경·전문가 확정 라벨·새 사진/픽셀 정답은0개다.",
        "공개 교량·댐 콘크리트 자료, 한 seed, 반복 사용한 source-VAL의 사진 존재 분류 연구다. 공장 현장 사진·정밀 위치·구조 안전·미래 현장 오차율은 측정하지 않았다. 반복 VAL 적응 후 결과를 독립 평가나 현장5% 미만 보장으로 설명하지 않는다. 최종 안전 판단은 점검자가 한다.", "",
        "[고정 후속 조건](facility-retention-strength-study-protocol.json), [실측 집계](facility-retention-strength-study-comparison.json), [기술 검증](facility-retention-strength-study-verification.json)", ""]
    return "\n".join(lines)


def main():
    from scripts.facility_retention_strength_study import validate_protocol, protected_hashes
    from scripts.verify_facility_retention_strength import validate_test_results
    require(not any((ROOT/path).exists() for path in (OUTPUT_PATH, MARKDOWN_PATH)), "Preserve completed follow-up evidence")
    protocol = validate_protocol(read(ROOT/PROTOCOL_PATH), ROOT); protocol_sha = sha(ROOT/PROTOCOL_PATH)
    protected = protected_hashes(ROOT)
    require(protocol["reference"] == RUNS[0] and protocol["control"] == RUNS[1] and protocol["treatment"] == RUNS[3],
            "Fresh strength/reference/reused-control identity differs")
    require(not any((ROOT/"reports"/f"{name}-target-test.json").exists() for name in RUNS), "Source-VAL follow-up refuses TEST records")
    loaded = [normalized_load_run(name) for name in RUNS]
    trainings, entries = [row[0] for row in loaded], [row[1] for row in loaded]
    teacher = validate_training(trainings[3], protocol, protocol_sha)
    require(trainings[0]["weights_sha256"] == protocol["initial_weights_sha256"], "Original0773 initializer differs")
    control_history = read(ROOT/"runs"/RUNS[1]/"history.json"); strong_history = read(ROOT/"runs"/RUNS[3]/"history.json")
    plan = read(ROOT/"reports/facility-spalling-sampler-dry-run.json")
    require(sha(ROOT/"reports/facility-spalling-sampler-dry-run.json") == protocol["sampler_plan_sha256"], "Original control plan changed")
    sampling = validate_histories(control_history, strong_history, protocol, plan)
    truth, normalization = [], []
    for entry, loaded_row in zip(entries, loaded):
        run = ROOT/"runs"/entry["run"]; digests = {}; converted = 0
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run/f"validation-{domain}.json"),
                read(run/f"target-validation-{domain}-grid1.json"), entry, domain)
            wrapped, count = normalize_ap_positive_support({"ranking_ap": {domain: points}})
            entry["ranking_ap"][domain] = wrapped["ranking_ap"][domain]; converted += count; digests[domain] = digest
        truth.append(digests)
        normalization.append({"run": entry["run"], "loader_support_fields_normalized": loaded_row[2], "verified_cache_support_fields_normalized": converted})
    require(all(digest == truth[0] for digest in truth), "Four-model VAL truth/order changed")
    require(sha(ROOT/PREVIOUS_COMPARISON_PATH) == protocol["previous_comparison_sha256"], "Previous final comparison changed")
    prior_result = read(ROOT/PREVIOUS_COMPARISON_PATH)
    require(prior_result.get("technical_verification_sha256") == protocol["previous_verification_sha256"]
            and prior_result.get("original_validation_truth_sha256") == truth[0], "Previous completed evaluation proof differs")
    for old, entry in zip(prior_result["experiments"], entries[:3]):
        require(all(old.get(key) == entry.get(key) for key in ("run", "weights_sha256", "per_class", "worst_error", "target_passed",
            "ranking_ap", "small_dacl_polygon_area_below_one_percent")), "Reused model measured metrics differ from immutable final comparison")
    for index, (entry, training, title) in enumerate(zip(entries, trainings,
            ("추가 학습 전 ROI 모델", "재사용 weight0 대조군", "이전 weight1 증류군", "후속 weight4 증류군"))):
        entry.update(title=title, imgsz=training["imgsz"], newly_trained_this_followup=index == 3)
        if index:
            entry["resources"] = {key: training[key] for key in ("elapsed_training_minutes", "peak_cuda_allocated_bytes", "optimizer_step_diagnostics")}
        if index == 3:
            entry["teacher_verification"] = teacher
    summary = measurements(entries, protocol)
    steps = trainings[3]["optimizer_step_diagnostics"]
    require(all(type(steps.get(key)) is int and steps[key] == sum(row["optimizer_step_diagnostics"][key] for row in strong_history)
                for key in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")), "New strength optimizer summary differs")
    require(type(trainings[3].get("elapsed_training_minutes")) in (int, float) and math.isfinite(trainings[3]["elapsed_training_minutes"])
            and trainings[3]["elapsed_training_minutes"] > 0 and type(trainings[3].get("peak_cuda_allocated_bytes")) is int
            and trainings[3]["peak_cuda_allocated_bytes"] > 0, "Actual strength resource measurements are missing")
    proof_path = ROOT/VERIFICATION_PATH; proof = read(proof_path)
    technical = validate_technical(proof, protocol, protocol_sha, trainings[3], protected)
    order = [row["sampled_row_indices_sha256"] for row in strong_history]
    require(proof["actual_sampling_verification"]["ordered_row_hashes_by_variant"] == {"control": order, "strong": order}
            and proof["actual_sampling_verification"]["known_other_class_entries_each_arm"] == sampling["known_other_class_entries_each_arm"],
            "Actual completion sampler/masking totals differ from saved histories")
    require(proof.get("preflight_sha256") == sha(ROOT/PREFLIGHT_PATH)
            and proof.get("source_before_training_sha256") == sha(ROOT/SOURCE_RECORD_PATH)
            and proof.get("test_results_sha256") == sha(ROOT/TEST_RECORD_PATH), "Actual completion evidence changed")
    preflight = read(ROOT/PREFLIGHT_PATH)
    require(proof["prepared_data_integrity"]["ledger_sha256"] == preflight["protected_input_ledger_sha256"]
            == sha(ROOT/preflight["protected_input_ledger_path"]), "Original private input ledger changed")
    require(validate_test_results(read(ROOT/TEST_RECORD_PATH), ROOT) == proof["tests"], "Executed tests differ from completion proof")
    result = {"schema": "facility_retention_strength_study_comparison_v1", "protocol_sha256": protocol_sha,
        "experiments": entries, **summary, "actual_sampling_verification": sampling, "original_validation_truth_sha256": truth[0],
        "technical_verification": technical, "technical_verification_sha256": sha(proof_path),
        "previous_comparison_sha256": protocol["previous_comparison_sha256"],
        "previous_verification_sha256": protocol["previous_verification_sha256"],
        "prepared_data_integrity": proof["prepared_data_integrity"], "app_profile_sha256": sha(ROOT/APP_PROFILE),
        "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0, "reused_control_training_epochs": 6,
        "previous_pair_completed_epochs_counted_again": False, "weight_chosen_after_previous_source_val_results": True,
        "repeated_source_val_adaptation": True, "teacher_predictions_are_new_truth": False,
        "deployed": False, "app_model_promoted": False, "source_test_inference_executed": False,
        "additional_expert_confirmed_labels": 0, "label_changes": 0, "new_independent_photos": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
        "reporting_provenance": {"normalization": "finite exact integral positive_photos floats only after original provenance validation",
            "normalizations_by_run": normalization, "reporter_source_sha256": sha(Path(__file__))},
        "scope": "One weight4 source-VAL-adaptive follow-up; reused unchanged control; not independent field accuracy or localization"}
    require(protected_hashes(ROOT) == protected, "Reporting changed protected original/previous/app inputs")
    with (ROOT/OUTPUT_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with (ROOT/MARKDOWN_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render(result))
    print(json.dumps({"actual_new_epochs": 6, "new_control_epochs": 0,
        "maximum_validation_error": [entry["worst_error"] for entry in entries],
        "retention_candidate_nominated": result["retention_gate"]["retention_candidate_nominated"],
        "research_candidate_nominated": result["research_gate"]["research_candidate_nominated"],
        "strict_target_passed": [entry["target_passed"] for entry in entries], "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
