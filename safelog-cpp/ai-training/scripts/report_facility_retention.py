"""Publish actual source-VAL results for the fixed other-class retention pair.

Teacher predictions are a regularizer, never new source truth. Only aggregate
measurements and hashes are published; there is no inference or app promotion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import CLASSES, DOMAINS, LABELS, TARGETS, load_run, read, require, sha
from scripts.report_facility_discrimination import KNOWN_INDICES, verify_validation_cache
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_native_roi_results import normalize_ap_positive_support
from scripts.report_facility_resolution import validate_histories as identical_histories

PROTOCOL_PATH = "reports/facility-retention-study-protocol.json"
VERIFICATION_PATH = "reports/facility-retention-study-verification.json"
PREFLIGHT_PATH = "runs/facility-retention-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-retention-source-before-training.json"
TEST_RECORD_PATH = "runs/facility-retention-test-results.json"
OUTPUT_PATH = "reports/facility-retention-study-comparison.json"
MARKDOWN_PATH = "reports/FACILITY_RETENTION_STUDY_RESULTS_KO.md"
APP_PROFILE = "reports/facility-inference-profile.json"
VARIANTS = ("control", "distill")
OTHER_CLASSES = tuple(CLASSES[2:])
RETENTION_GATE = {
    "maximum_known_other_class_ap_drop_vs_reference": .02,
    "minimum_dacl_exposed_rebar_ap_recovery_vs_control": .02,
    "maximum_primary_worst_error_regression_vs_reference_and_control": .02,
}


def normalized_load_run(name, loader=load_run):
    """Preserve all original loader checks; adapt only exact integral AP counts."""
    training, entry = loader(name)
    normalized, count = normalize_ap_positive_support(entry)
    return training, normalized, count


def retention_measurements(entries, declared):
    """Apply the independent predeclared retention gate to validated aggregates."""
    require(declared == RETENTION_GATE, "Declared retention gate changed")
    require(isinstance(entries, list) and len(entries) == 3, "Reference and matched pair are required")
    require(all(type(entry.get("worst_error")) in (int, float) and math.isfinite(entry["worst_error"])
                and 0 <= entry["worst_error"] <= 1 for entry in entries), "Invalid primary maximum error")
    rows = []
    for domain in DOMAINS:
        for k, label in enumerate(CLASSES):
            if label not in OTHER_CLASSES or k not in KNOWN_INDICES[domain]:
                continue
            points = [entry["ranking_ap"][domain][label] for entry in entries]
            require(len({(p["known_photos"], p["positive_photos"]) for p in points}) == 1,
                    "Retention AP support changed across models")
            require(all(p.get("status") == "measured_on_asserted_source_labels" for p in points),
                    "Originally known retention AP status changed")
            values = [p["ap"] for p in points]
            require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
                    "Only originally known source-class AP can enter retention")
            change_ref, change_control = values[2]-values[0], values[2]-values[1]
            rows.append({"domain": domain, "class": label,
                "known_photos": points[0]["known_photos"], "positive_photos": points[0]["positive_photos"],
                "reference_ap": values[0], "control_ap": values[1], "treatment_ap": values[2],
                "treatment_minus_reference_ap": change_ref, "treatment_minus_control_ap": change_control,
                "retained_vs_reference": change_ref >= -declared["maximum_known_other_class_ap_drop_vs_reference"]-1e-12})
    require(len(rows) == 8, "Eight known other-class source combinations must be measured per model")
    exposed = next(row for row in rows if row["domain"] == "dacl" and row["class"] == "exposed_rebar")
    recovery = exposed["treatment_minus_control_ap"]
    primary = {name: entries[2]["worst_error"]-entries[i]["worst_error"]
               for i, name in enumerate(("reference", "control"))}
    primary_passed = all(v <= declared["maximum_primary_worst_error_regression_vs_reference_and_control"]+1e-12
                         for v in primary.values())
    other_passed = all(row["retained_vs_reference"] for row in rows)
    recovery_passed = recovery >= declared["minimum_dacl_exposed_rebar_ap_recovery_vs_control"]-1e-12
    drops = []
    for index, name in enumerate(("reference", "control", "treatment")):
        field = ("reference_ap", "control_ap", "treatment_ap")[index]
        worst = min(rows, key=lambda row: row[field]-row["reference_ap"])
        drops.append({"run": entries[index]["run"], "domain": worst["domain"], "class": worst["class"],
            "minimum_other_ap_change_vs_reference": worst[field]-worst["reference_ap"],
            "maximum_other_ap_drop_vs_reference": max(0., worst["reference_ap"]-worst[field])})
    return {"declared": dict(declared), "other_class_ap_measurements": 24,
        "known_source_class_combinations_each_model": 8, "per_source_other_class_ap": rows,
        "dacl_exposed_rebar": {key: exposed[key] for key in ("known_photos", "positive_photos",
            "reference_ap", "control_ap", "treatment_ap", "treatment_minus_reference_ap", "treatment_minus_control_ap")},
        "worst_other_class_ap_decline_by_model": drops,
        "primary_maximum_error_changes": primary,
        "all_known_other_class_reference_retention_passed": other_passed,
        "dacl_exposed_rebar_recovery_vs_control_passed": recovery_passed,
        "primary_maximum_error_regression_guard_passed": primary_passed,
        "retention_candidate_nominated": other_passed and recovery_passed and primary_passed,
        "independent_of_research_gate_and_strict_five_percent": True}


def validate_histories(control, distill, protocol, plan):
    sampling = identical_histories(control, distill, protocol["requested_epochs"], protocol["draws_per_epoch"])
    for epoch, (left, right, prepared) in enumerate(zip(control, distill, plan["epochs"]), 1):
        order = prepared["control_order_sha256"]
        for variant, row in zip(VARIANTS, (left, right)):
            require(row["sampled_row_indices_sha256"] == order, "Both arms must use original control arrays")
            fixed = row.get("fixed_sampling", {})
            require(fixed.get("draw_hash_matches_prepared") is True
                    and fixed.get("declared_draw_sha256") == order
                    and type(fixed.get("changed_positions_from_control")) is int
                    and fixed["changed_positions_from_control"] == 0, "A sampling intervention entered retention")
            audit = row.get("retention_distillation", {})
            weight = 0. if variant == "control" else 1.
            require(type(audit.get("weight")) in (int, float) and audit["weight"] == weight
                    and type(audit.get("temperature")) in (int, float) and audit["temperature"] == 2.,
                    "Teacher regularization weight or temperature changed")
            raw, weighted = audit.get("unweighted_mean_batch_loss"), audit.get("weighted_mean_batch_loss")
            require(type(raw) in (int, float) and math.isfinite(raw) and raw >= 0
                    and type(weighted) in (int, float) and math.isfinite(weighted) and weighted >= 0
                    and math.isclose(weighted, weight*raw, rel_tol=0., abs_tol=1e-12),
                    "Actual distillation loss is missing or differs from declared weight")
            attempted = row["optimizer_step_diagnostics"]["attempted_batches"]
            require(type(audit.get("teacher_forward_batches")) is int and audit["teacher_forward_batches"] == attempted
                    and type(audit.get("known_other_class_entries")) is int and audit["known_other_class_entries"] > 0
                    and type(audit.get("contributing_batches")) is int
                    and 0 < audit["contributing_batches"] <= attempted
                    and audit.get("excluded_primary_classes") == TARGETS
                    and all(audit.get(k) is True for k in ("teacher_eval", "teacher_frozen", "teacher_state_unchanged", "teacher_gradients_absent")),
                    "Actual frozen teacher forward/masking proof differs")
        require(left["retention_distillation"]["known_other_class_entries"] == right["retention_distillation"]["known_other_class_entries"]
                and left["retention_distillation"]["contributing_batches"] == right["retention_distillation"]["contributing_batches"],
                "Known-source teacher masking differs between matched draws")
        for row in (left, right):
            counts = row.get("sampled_photo_target_counts", {})
            require(set(counts) == set(CLASSES), "All-seven actual target exposure counts are required")
            for label, point in counts.items():
                require(set(point) == {"positive", "negative", "unknown"}
                        and all(type(value) is int and value >= 0 for value in point.values())
                        and sum(point.values()) == protocol["draws_per_epoch"], "Invalid all-seven target exposure")
            require(sum(counts[label]["positive"] + counts[label]["negative"] for label in OTHER_CLASSES)
                    == row["retention_distillation"]["known_other_class_entries"], "Unknown teacher entries were not excluded")
        require(left["sampled_photo_target_counts"] == right["sampled_photo_target_counts"],
                "Matched all-seven target exposure differs")
    return {**sampling, "both_actual_orders_match_original_control_arrays": True,
        "sampling_intervention_applied": False,
        "actual_all_seven_photo_target_exposure_identical_each_epoch": True,
        "actual_teacher_forward_batches_each_arm": sum(row["retention_distillation"]["teacher_forward_batches"] for row in control),
        "actual_known_other_class_entries_each_arm": sum(row["retention_distillation"]["known_other_class_entries"] for row in control),
        "teacher_unknown_source_classes_used_as_truth": False}


def validate_teacher(training, protocol):
    proof = training.get("teacher_state_preservation", {})
    initial, final = proof.get("initial_state_sha256"), proof.get("final_state_sha256")
    require(isinstance(initial, str) and re.fullmatch(r"[a-f0-9]{64}", initial) and final == initial
            and initial == protocol["teacher_state_sha256"],
            "Frozen teacher state hashes differ")
    require(proof.get("initial_weights_sha256") == protocol["initial_weights_sha256"]
            and all(proof.get(k) is True for k in ("unchanged", "eval_mode", "all_parameters_frozen",
                "no_parameter_gradients", "teacher_forward_both_arms")), "Original eval-only teacher proof missing")
    require(type(proof.get("student_parameter_count")) is int and proof["student_parameter_count"] == 3244151
            and type(proof.get("teacher_parameter_count")) is int and proof["teacher_parameter_count"] == 3244151
            and type(proof.get("teacher_state_tensor_count")) is int and proof["teacher_state_tensor_count"] == 324,
            "Teacher/student architecture inventory changed")
    require(type(training.get("actual_teacher_forward_batches")) is int and training["actual_teacher_forward_batches"] == 10686,
            "Both arms must run the fixed teacher for every actual minibatch")
    return {"teacher_weights_sha256": protocol["initial_weights_sha256"], "teacher_state_sha256": initial,
        "actual_teacher_forward_batches": training["actual_teacher_forward_batches"],
        "unchanged_eval_frozen_no_gradients": True, "teacher_predictions_are_new_truth": False}


def validate_pair(trainings, protocol, protocol_sha):
    require(len(trainings) == 3 and trainings[0]["weights_sha256"] == protocol["initial_weights_sha256"],
            "Original 0773 reference weights differ")
    common = {key: protocol[key] for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch",
        "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    common.update(status="complete", actual_epochs=protocol["requested_epochs"], imgsz=640,
        initial_weights_sha256=protocol["initial_weights_sha256"],
        core_spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        auxiliary_manifest_sha256=protocol["auxiliary_manifest_sha256"], source_sha256=protocol["source_sha256"],
        study_protocol_sha256=protocol_sha, distillation_recipe=protocol["distillation_recipe"],
        sampling_intervention_applied=False, paired_draws_key="control",
        sampler_plan_sha256=protocol["sampler_plan_sha256"], private_draw_archive_sha256=protocol["paired_draws_sha256"])
    teachers = []
    for variant, training in zip(VARIANTS, trainings[1:]):
        expected = {**common, "model_variant": variant, "architecture": protocol["architecture_by_variant"][variant],
                    "distillation_weight": 0. if variant == "control" else 1.}
        for key, value in expected.items():
            require(type(training.get(key)) is type(value) and training[key] == value,
                    f"Actual fixed retention condition differs: {key}")
        teachers.append(validate_teacher(training, protocol))
    for key in ("classes", "split_sha256", "core_spatial_manifest_sha256", "spatial_manifest_sha256",
        "auxiliary_manifest_sha256", "expected_sampling", "expected_label_sampling", "photo_positive_weights",
        "pixel_positive_weights", "auxiliary_positive_weights", "additional_validation", "additional_test", "paired_label_preservation"):
        require(key in trainings[1] and trainings[1][key] == trainings[2].get(key), "Original paired source/loss/label condition differs")
    require(teachers[0] == teachers[1], "Matched arms used different frozen teachers")
    return teachers


def validate_technical_proof(proof, protocol, protocol_sha, trainings, protected):
    require(proof.get("schema") == "facility_retention_study_verification_v1" and proof.get("status") == "passed",
            "Actual completed retention verification is required")
    require(proof.get("protocol_sha256") == protocol_sha and proof.get("source_sha256") == protocol["source_sha256"]
            and type(proof.get("runtime_source_count")) is int and proof["runtime_source_count"] == len(protocol["source_sha256"]),
            "Technical proof belongs to another frozen study")
    require(isinstance(proof.get("source_git_commit"), str) and re.fullmatch(r"[a-f0-9]{40}", proof["source_git_commit"]),
            "Before-training commit proof is missing")
    for key in ("git_blob_bytes_verified", "working_runtime_sources_unchanged", "protected_files_unchanged"):
        require(proof.get(key) is True, "Technical byte-preservation proof is missing")
    require(type(proof.get("actual_completed_training_epochs")) is int and proof["actual_completed_training_epochs"] == 12
            and type(proof.get("verification_training_epochs")) is int and proof["verification_training_epochs"] == 0,
            "Actual completed-epoch proof differs")
    for key in ("source_test_inference_executed", "app_model_promoted", "deployed", "accuracy_measured_by_verifier"):
        require(proof.get(key) is False, "Technical verifier cannot claim TEST, promotion or accuracy")
    for key in ("additional_expert_confirmed_labels", "label_changes", "new_photo_targets", "new_pixel_targets"):
        require(type(proof.get(key)) is int and proof[key] == 0, "Teacher predictions cannot create new truth")
    require(proof.get("protected_file_sha256") == protected, "Protected data/model/profile proof differs")
    tests = proof.get("tests", {})
    require(type(tests.get("tests_run")) is int and tests["tests_run"] > 0 and tests["tests_run"] == tests.get("expected_tests_collected")
            and all(type(tests.get(k)) is int and tests[k] == 0 for k in ("failures", "errors", "skipped"))
            and tests.get("source_sha256") == protocol["source_sha256"], "Actual complete focused tests must pass")
    for path, expected in tests.get("test_source_sha256", {}).items():
        require(sha(ROOT / path) == expected, "Executed focused test bytes changed")
    data = proof.get("prepared_data_integrity", {})
    require(data.get("status") == "passed" and type(data.get("input_files_checked")) is int and data["input_files_checked"] > 0
            and data.get("all_original_input_sha_size_mtime_preserved") is True
            and data.get("original_train_images_masks_annotations_verified") is True
            and data.get("individual_input_paths_published") is False, "Actual original TRAIN input-byte proof is required")
    require(isinstance(proof.get("experiments"), list) and len(proof["experiments"]) == 2,
            "Both completed checkpoint proofs are required")
    sampling = proof.get("actual_sampling_verification", {})
    require(type(sampling.get("epochs_compared")) is int and sampling["epochs_compared"] == 6
            and type(sampling.get("draws_per_epoch_each_arm")) is int and sampling["draws_per_epoch_each_arm"] == 14248
            and type(sampling.get("changed_positions")) is int and sampling["changed_positions"] == 0
            and sampling.get("teacher_forward_batches_by_variant") == {"control": 10686, "distill": 10686}
            and all(sampling.get(key) is True for key in (
                "both_actual_draw_orders_match_fixed_original_control_array",
                "actual_ordered_row_index_hashes_identical_each_epoch",
                "actual_domain_full_crop_joint_counts_identical_each_epoch",
                "actual_all_seven_photo_target_exposure_identical_each_epoch",
                "teacher_eval_frozen_no_grad_verified_both_arms")), "Actual fixed draw/teacher completion proof differs")
    for variant, training, point in zip(VARIANTS, trainings[1:], proof["experiments"]):
        expected = {"variant": variant, "weights_sha256": training["weights_sha256"], "actual_epochs": 6,
            "imgsz": 640, "architecture": protocol["architecture_by_variant"][variant],
            "strict_state_inventory_verified": True, "new_state_tensor_count": 0, "state_tensor_count": 324}
        for key, value in expected.items():
            require(type(point.get(key)) is type(value) and point[key] == value, "Completed checkpoint identity/inventory differs")
        require(all(point.get("cpu_reload", {}).get(k) is True for k in ("strict_factory_reload_verified", "all_outputs_finite",
            "public_output_equals_training_photo_output")), "Actual CPU checkpoint reload must pass")
        require(point.get("teacher_state_preservation") == training.get("teacher_state_preservation"),
                "Actual completed teacher preservation differs from run metadata")
    return {"status": "passed", "source_git_commit": proof["source_git_commit"], "runtime_source_count": proof["runtime_source_count"],
        "tests_run": tests["tests_run"], "actual_completed_training_epochs": proof["actual_completed_training_epochs"]}


def render(result):
    entries, change, proof, retention = result["experiments"], result["comparisons"], result["technical_verification"], result["retention_gate"]
    lines = ["# 시설 항목 성능 유지 증류 대조 학습 결과", "",
        "같은 0773 초기 가중치에서 대조군·증류군을 각각6epoch, 총12epoch 실제 추가 학습했다. 두 군 모두 원래 control 추출 배열·같은 증강 난수·640 입력·80×80 마스크·기존7종 출력·19종 보조 태그를 사용했다. 이전 하위유형 보강 추출 배열은 사용하지 않았다.",
        "원래 0773 모델을 eval·파라미터 고정 상태의 teacher로 두고, TRAIN에서 원래 정답이 알려진 다른5종(녹 흔적·철근 노출·젖은 표면·백화·공동)에만 온도2 Bernoulli KL 정규화를 적용했다. 대조군 weight0·증류군 weight1이며 teacher forward는 양쪽 모두 수행했다. 균열·박락과 미확인 출처 항목은 증류 손실에서 제외했다.",
        "teacher 예측은 원래 기능을 유지하도록 돕는 soft 정규화 신호다. 새 ground truth·전문가 판단·새 라벨로 취급하지 않았다. teacher forward 추가 비용은 실측 학습 시간에 포함되며 배포 학생 모델의 파라미터 수는 늘리지 않았다.", "",
        "![Retention paired results](facility-retention-study-comparison.png)", "",
        "| 모델 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |", "|---|---:|---:|---:|---|"]
    for e in entries:
        lines.append(f'| {e["title"]} | {e["actual_epochs"]} | {e["best_epoch"]} | {100*e["worst_error"]:.2f}% | {"통과" if e["target_passed"] else "미달"} |')
    lines += ["", f'증류군−초기 모델 최대 오류 {change["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, 증류군−새 대조군 {change["maximum_error_treatment_minus_control_pp"]:+.2f}pp. 양수는 악화다.',
        "최대값은 균열·박락 × DACL710/Dam424/CODEBRIM611 × FNR/FPR의12개 비율 중 최대이며 전체 사진 오답 비율·앱 정확도가 아니다.", "",
        "## 항목 유지 기준과 관측 AP", "",
        f'사전 선언한 유지 후보 기준: **{"통과" if retention["retention_candidate_nominated"] else "미달"}**. 알려진 다른5종의 각 출처별 AP가 초기 모델보다0.02를 초과해 하락하지 않고, DACL 철근 노출 AP가 새 대조군보다0.02 이상 회복하며, 최대 균열·박락 오류가 초기 모델·새 대조군 각각보다2pp를 초과해 악화하지 않아야 한다.',
        "이 기준은 기존 연구 후보 기준·엄격한5% 기준과 별도로 계산했다. AP는 확률 순위 지표이며 정답률·오탐률과 같은 지표가 아니다.", "",
        "| 출처 | 다른 항목 | 초기 AP | 새 대조 AP | 증류 AP | 증류−초기 | 증류−대조 | 초기 대비 유지 |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in retention["per_source_other_class_ap"]:
        lines.append(f'| {row["domain"]} | {LABELS[row["class"]]} | {row["reference_ap"]:.4f} | {row["control_ap"]:.4f} | {row["treatment_ap"]:.4f} | {row["treatment_minus_reference_ap"]:+.4f} | {row["treatment_minus_control_ap"]:+.4f} | {"통과" if row["retained_vs_reference"] else "미달"} |')
    iron = retention["dacl_exposed_rebar"]
    lines += ["", f'DACL 철근 노출 AP: 초기 **{iron["reference_ap"]:.4f}**, 새 대조 **{iron["control_ap"]:.4f}**, 증류 **{iron["treatment_ap"]:.4f}**. 증류−새 대조 회복 {iron["treatment_minus_control_ap"]:+.4f}; 기준 {"통과" if retention["dacl_exposed_rebar_recovery_vs_control_passed"] else "미달"}.', "",
        "| 모델 | 다른 알려진 항목의 최대 AP 하락(초기 대비) | 해당 출처·항목 |", "|---|---:|---|"]
    titles = {e["run"]: e["title"] for e in entries}
    for row in retention["worst_other_class_ap_decline_by_model"]:
        lines.append(f'| {titles[row["run"]]} | {row["maximum_other_ap_drop_vs_reference"]:.4f} | {row["domain"]}·{LABELS[row["class"]]} |')
    lines += ["", "다른5종 중 알려진 출처 조합은 DACL5·Dam0·CODEBRIM3의8개이며3모델 총24개 AP를 비교했다. 전체7종의 알려진 AP는14조합×3모델=42개다. 미확인 항목에0점 AP나 정상 정답을 만들지 않았다. 대조군에도 추가 학습과 선택의 영향이 있으므로 이전 변화 전체를 하위유형 추출만의 원인으로 단정하지 않는다.", "",
        "## 기존 연구 후보 기준과 작은 손상", "",
        f'기존 연구 후보 기준: **{"통과" if result["research_gate"]["research_candidate_nominated"] else "미달"}**. 초기 모델·새 대조군 각각 대비 최대 오류 최소0.5pp 개선, 각 target 오류 악화2pp 이하, 다른 알려진 항목 AP 하락0.02 이하 및 작은 사례 FN 합계 최소2건 감소·항목별 FNR 악화2pp 이하를 그대로 요구했다.', "",
        "| 작은 손상 | 초기 FN/양성 | 새 대조 FN/양성 | 증류 FN/양성 |", "|---|---:|---:|---:|"]
    for task in TARGETS:
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{e["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"]}/{e["small_dacl_polygon_area_below_one_percent"][task]["positive_photos"]}' for e in entries) + ' |')
    lines += ["", "분모는 균열93·박락105의198개 항목·사진 사례다. 같은 사진이 두 항목에 들어갈 수 있으며 고유 사진198장·실제 물리적 크기를 뜻하지 않는다.", "",
        "## 출처별 관측 오류", "",
        "Wilson 양측95%는 고정 예측과 독립 사진 가정 아래 기술 통계다. 반복한 VAL epoch·임계값 선택과 파생 crop 상관을 보정한 현장 보장·모델 간 유의성 검정이 아니다.", "",
        "| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | Wilson95 | FP/음성 | 오탐률 | Wilson95 |",
        "|---|---|---|---:|---:|---|---:|---:|---|"]
    interval = lambda values: f'{100*values[0]:.2f}%–{100*values[1]:.2f}%'
    for row in result["error_rows"]:
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {interval(row["fnr_wilson95_descriptive"])} | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% | {interval(row["fpr_wilson95_descriptive"])} |')
    lines += ["", "## 실제 자원·자료·teacher 보존", "",
        "| 모델 | 학습·epoch 검증 시간 | allocated peak | 실제 update | AMP skip | teacher forward |",
        "|---|---:|---:|---:|---:|---:|"]
    for e in entries[1:]:
        resource, teacher = e["resources"], e["teacher_verification"]
        steps = resource["optimizer_step_diagnostics"]
        lines.append(f'| {e["title"]} | {resource["elapsed_training_minutes"]:.2f}분 | {resource["peak_cuda_allocated_bytes"]/1024**3:.3f} GiB | {steps["actual_optimizer_steps"]} | {steps["amp_skipped_steps"]} | {teacher["actual_teacher_forward_batches"]} |')
    lines += ["", f'원래 full TRAIN14,248·전체26,289행 및 두 군6×14,248개 draw 순서·출처·full/crop·모든7종 정답 노출을 확인했다. 원래 이미지·마스크·known·auxiliary 태그·기본 추출 가중치·pixel/photo/auxiliary 손실 가중치를 유지했다. teacher의 초기·최종 state SHA, eval 상태와 gradient 부재를 실제 기록했다.',
        f'학습 전 commit의 {proof["runtime_source_count"]}개 소스 바이트, 코드 테스트 {proof["tests_run"]}개 통과, 실제 두 가중치 CPU 엄격 재로딩·finite7/80/19 계약을 확인했다. 테스트 수는 정확도 평가 사례 수가 아니다. 원본 TRAIN 입력 SHA·size·mtime와 앱 프로필 SHA를 유지했다.',
        "AP loader와 캐시 검증 후 유한·비음수·known 분모 이내의 정확한 정수값 float 양성 개수만 int로 표시했다. bool·소수·NaN·무한대는 거부하며 원래 점수·정답·AP·임계값·미탐·오탐·기존 후보 계산을 바꾸지 않았다.",
        "시간은 데이터 읽기·teacher forward·epoch 검증·캐시 영향이 포함된다. AMP overflow로 실제 optimizer update 수는 달라질 수 있어 건너뛴 update를 별도 집계했다. 모바일 추론 시간은 측정하지 않았다.", "",
        "## 적용 상태와 범위", "",
        "보류 TEST 추론·자동 배포·앱 모델 승격은 실행하지 않았다. 기본 `facility-validation-v2`와 프로필 SHA를 유지했다. 새 독립 사진·원본 라벨 변경·전문가 확정 라벨·새 사진/픽셀 정답은0개다.",
        "한 seed와 반복 사용한 공개 교량·댐 콘크리트 source-VAL의 사진 존재 분류 연구다. 공장 사진·정밀 위치·시설 정상/안전 판정·미래 현장 오차율은 측정하지 않았다. 이 결과로 현장5% 미만이나 구조 안전·법적 점검 완료를 보장하지 않으며 최종 판단은 점검자가 한다.", "",
        "[사전 고정 조건](facility-retention-study-protocol.json), [실측 집계](facility-retention-study-comparison.json), [기술 검증](facility-retention-study-verification.json)", ""]
    return "\n".join(lines)


def main():
    from scripts.facility_retention_study import validate_protocol, protected_hashes
    from scripts.verify_facility_retention import validate_test_results
    require(not any((ROOT / p).exists() for p in (OUTPUT_PATH, MARKDOWN_PATH)), "Preserve completed report evidence")
    protocol = validate_protocol(read(ROOT / PROTOCOL_PATH), ROOT); protocol_sha = sha(ROOT / PROTOCOL_PATH)
    protected = protected_hashes(ROOT)
    plan_path = ROOT / "reports/facility-spalling-sampler-dry-run.json"
    require(sha(plan_path) == protocol["sampler_plan_sha256"], "Original control sampling plan changed")
    plan = read(plan_path); names = [protocol["reference"], protocol["control"], protocol["treatment"]]
    require(not any((ROOT / "reports" / f"{name}-target-test.json").exists() for name in names), "Source-VAL study refuses TEST records")
    loaded = [normalized_load_run(name) for name in names]
    trainings, entries = [row[0] for row in loaded], [row[1] for row in loaded]
    teachers = validate_pair(trainings, protocol, protocol_sha)
    histories = [read(ROOT / "runs" / name / "history.json") for name in names[1:]]
    sampling = validate_histories(*histories, protocol, plan)
    truth, normalization = [], []
    for entry, loaded_row in zip(entries, loaded):
        run = ROOT / "runs" / entry["run"]; digests = {}; converted = 0
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run / f"validation-{domain}.json"),
                read(run / f"target-validation-{domain}-grid1.json"), entry, domain)
            wrapped, count = normalize_ap_positive_support({"ranking_ap": {domain: points}})
            entry["ranking_ap"][domain] = wrapped["ranking_ap"][domain]; converted += count; digests[domain] = digest
        truth.append(digests)
        normalization.append({"run": entry["run"], "loader_support_fields_normalized": loaded_row[2],
                              "verified_cache_support_fields_normalized": converted})
    require(truth[0] == truth[1] == truth[2], "Original VAL truth/order changed across models")
    for index, (entry, training, title) in enumerate(zip(entries, trainings,
            ("추가 학습 전 ROI 모델", "증류 weight0 대조군", "항목 유지 증류 weight1 보강군"))):
        entry.update(title=title, imgsz=training["imgsz"])
        if index:
            steps = training["optimizer_step_diagnostics"]
            require(all(type(steps.get(k)) is int and steps[k] == sum(h["optimizer_step_diagnostics"][k]
                for h in histories[index-1]) for k in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")),
                "Actual optimizer summary differs from histories")
            require(type(training.get("elapsed_training_minutes")) in (int, float)
                and math.isfinite(training["elapsed_training_minutes"]) and training["elapsed_training_minutes"] > 0
                and type(training.get("peak_cuda_allocated_bytes")) is int and training["peak_cuda_allocated_bytes"] > 0,
                "Actual training resources are missing")
            entry["resources"] = {k: training[k] for k in ("elapsed_training_minutes", "peak_cuda_allocated_bytes", "optimizer_step_diagnostics")}
            entry["teacher_verification"] = teachers[index-1]
    measured = validate_measurements(entries, protocol)
    retention = retention_measurements(entries, protocol["retention_candidate_gate"])
    proof_path = ROOT / VERIFICATION_PATH; proof = read(proof_path)
    technical = validate_technical_proof(proof, protocol, protocol_sha, trainings, protected)
    ordered = [row["sampled_row_indices_sha256"] for row in histories[0]]
    require(proof["actual_sampling_verification"]["ordered_row_hashes_by_variant"] == {variant: ordered for variant in VARIANTS}
            and proof["actual_sampling_verification"]["known_other_class_entries_each_arm"]
            == sampling["actual_known_other_class_entries_each_arm"], "Actual technical teacher/draw totals differ from histories")
    require(proof.get("preflight_sha256") == sha(ROOT / PREFLIGHT_PATH)
        and proof.get("source_before_training_sha256") == sha(ROOT / SOURCE_RECORD_PATH)
        and proof.get("test_results_sha256") == sha(ROOT / TEST_RECORD_PATH), "Completion evidence bytes changed")
    preflight = read(ROOT / PREFLIGHT_PATH)
    require(proof["prepared_data_integrity"]["ledger_sha256"] == preflight["protected_input_ledger_sha256"]
            == sha(ROOT / preflight["protected_input_ledger_path"]), "Original TRAIN private ledger changed")
    require(validate_test_results(read(ROOT / TEST_RECORD_PATH), ROOT) == proof["tests"], "Executed tests differ from completion proof")
    result = {"schema": "facility_retention_study_comparison_v1", "protocol_sha256": protocol_sha,
        "experiments": entries, **measured, "retention_gate": retention, "actual_sampling_verification": sampling,
        "original_validation_truth_sha256": truth[0], "app_profile_sha256": sha(ROOT / APP_PROFILE),
        "prepared_data_integrity": proof["prepared_data_integrity"], "technical_verification": technical,
        "technical_verification_sha256": sha(proof_path), "sampler_plan_sha256": protocol["sampler_plan_sha256"],
        "private_draw_archive_sha256": protocol["paired_draws_sha256"], "actual_new_completed_training_epochs": 12,
        "deployed": False, "app_model_promoted": False, "source_test_inference_executed": False,
        "additional_expert_confirmed_labels": 0, "label_changes": 0, "new_independent_photos": 0,
        "new_photo_targets": 0, "new_pixel_targets": 0, "teacher_predictions_are_new_truth": False,
        "reporting_provenance": {"normalization": "finite exact integral positive_photos floats only; original known support bounds",
            "normalizations_by_run": normalization, "original_metrics_labels_thresholds_research_gate_unchanged": True,
            "reporter_source_sha256": sha(Path(__file__))},
        "scope": "Repeated original public-source full-photo VAL presence; frozen other-class teacher regularizer; not factory accuracy or fine localization"}
    require(protected_hashes(ROOT) == protected, "Reporting changed protected original/app/protocol files")
    with (ROOT / OUTPUT_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with (ROOT / MARKDOWN_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render(result))
    print(json.dumps({"actual_new_epochs": 12, "maximum_validation_error": [e["worst_error"] for e in entries],
        "retention_candidate_nominated": retention["retention_candidate_nominated"],
        "research_candidate_nominated": result["research_gate"]["research_candidate_nominated"],
        "strict_target_passed": [e["target_passed"] for e in entries], "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
