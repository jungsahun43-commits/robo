"""Publish aggregate VAL results for the frozen native-detail-pixel pair.

Saved VAL scores are checked; this reporter never performs model inference,
edits labels, opens held-out images, or promotes an application model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_native_roi_study import (
    AUDIT, MANIFESTS, PROTOCOL, protected_hashes, validate_protocol,
)
from scripts.report_facility_small_region import (
    CLASSES, COUNTS, DOMAINS, LABELS, TARGETS, comparisons, load_run, number,
)
from scripts.report_facility_discrimination import (
    KNOWN_INDICES, error_rows, nonnegative_integer, verify_validation_cache,
)
from scripts.report_facility_detail import GATE, detail_research_gate, validate_small_area
from scripts.report_facility_resolution import validate_histories
from scripts.verify_facility_native_roi import (
    APP_PROFILE, LEDGER_PATH, OUTPUT_PATH as VERIFICATION_PATH, PREFLIGHT_PATH,
    SOURCE_RECORD_PATH, TEST_SOURCES, hash_map, read, require, sha, validate_test_counts, validate_test_results,
)

OUTPUT_PATH = "reports/facility-native-roi-study-comparison.json"
MARKDOWN_PATH = "reports/FACILITY_NATIVE_ROI_STUDY_RESULTS_KO.md"
TEST_RECORD_PATH = "runs/facility-native-roi-test-results.json"


def validate_technical_proof(proof, protocol, protocol_sha, trainings, protected):
    require(proof.get("schema") == "facility_native_roi_study_verification_v1"
            and proof.get("status") == "passed", "Completed technical verification is required")
    require(proof.get("protocol_sha256") == protocol_sha
            and proof.get("source_sha256") == protocol["source_sha256"]
            and type(proof.get("runtime_source_count")) is int
            and proof["runtime_source_count"] == len(protocol["source_sha256"]),
            "Technical proof belongs to another frozen study")
    require(isinstance(proof.get("source_git_commit"), str)
            and re.fullmatch(r"[a-f0-9]{40}", proof["source_git_commit"]),
            "Technical proof does not identify its before-training commit")
    for key in ("git_blob_bytes_verified", "working_runtime_sources_unchanged", "protected_files_unchanged"):
        require(proof.get(key) is True, "Technical source/protection proof is missing")
    require(type(proof.get("actual_completed_training_epochs")) is int
            and proof["actual_completed_training_epochs"] == 2*protocol["requested_epochs"]
            and type(proof.get("verification_training_epochs")) is int
            and proof["verification_training_epochs"] == 0, "Actual epoch proof differs")
    for key in ("source_test_inference_executed", "app_model_promoted", "deployed", "accuracy_measured_by_verifier"):
        require(proof.get(key) is False, "Technical proof cannot claim deployment, held-out inference, or accuracy")
    for key in ("additional_expert_confirmed_labels", "label_changes", "new_photo_targets", "new_pixel_targets"):
        require(type(proof.get(key)) is int and proof[key] == 0, "Technical proof cannot assert new truth")
    require(proof.get("protected_file_sha256") == protected, "Protected data/app/protocol proof differs")
    tests = proof.get("tests", {})
    validate_test_counts(tests)
    require(tests.get("source_sha256") == protocol["source_sha256"],
            "Actual focused tests must pass against the same runtime bytes")
    hash_map(tests.get("test_source_sha256"), TEST_SOURCES)
    data = proof.get("prepared_data_integrity", {})
    require(data.get("status") == "passed" and all(data.get(k) is True for k in (
        "full_rows_unchanged", "labels_and_masks_unchanged", "row_order_unchanged", "derived_png_hashes_verified"))
        and type(data.get("changed_rows")) is int and data["changed_rows"] > 0
        and type(data.get("derived_pngs")) is int and data["derived_pngs"] == 2*data["changed_rows"],
        "Actual paired PNG/full/label/mask integrity proof is missing")
    require(isinstance(proof.get("experiments"), list) and len(proof["experiments"]) == 2
            and len(trainings) == 3, "Both completed real checkpoints require proof")
    for variant, training, entry in zip(("control", "native"), trainings[1:], proof["experiments"]):
        expected = {"variant": variant, "weights_sha256": training["weights_sha256"],
                    "actual_epochs": protocol["requested_epochs"], "imgsz": 640,
                    "architecture": protocol["architecture_by_variant"][variant],
                    "strict_state_inventory_verified": True, "new_state_tensor_count": 0}
        for key, value in expected.items():
            require(type(entry.get(key)) is type(value) and entry[key] == value,
                    "Actual checkpoint identity or parameter inventory differs")
        reload = entry.get("cpu_reload", {})
        require(all(reload.get(k) is True for k in ("strict_factory_reload_verified", "all_outputs_finite",
                    "public_output_equals_training_photo_output")), "Actual CPU checkpoint reload must pass")
    return {"status": "passed", "source_git_commit": proof["source_git_commit"],
            "runtime_source_count": proof["runtime_source_count"], "tests_run": tests["tests_run"],
            "actual_completed_training_epochs": proof["actual_completed_training_epochs"]}


def validate_measurements(entries, protocol):
    """Reject altered counts/support/rates/gates before rendering aggregates."""
    require(isinstance(entries, list) and len(entries) == 3
            and [e.get("run") for e in entries] == [protocol["reference"], protocol["control"], protocol["treatment"]],
            "The original reference and both fixed native ROI runs are required")
    require(protocol.get("research_candidate_gate") == GATE, "Declared research gate changed")
    ap_count = 0
    for entry in entries:
        require(entry.get("selected_grid") == 1 and type(entry.get("selected_grid")) is int
                and entry.get("configured_grids") == [1] and entry.get("test_executed") is False
                and entry.get("test_result") is None, "Comparison must use original full-photo VAL only")
        per_class = entry.get("per_class", {})
        require(set(per_class) == set(TARGETS), "Both concrete targets are required")
        measured = []
        for task in TARGETS:
            require(set(per_class[task].get("domains", {})) == set(DOMAINS), "The three original VAL sources are required")
            number(per_class[task].get("threshold"), "Target threshold")
            for domain, point in per_class[task]["domains"].items():
                counts = {k: nonnegative_integer(point.get(k), "Invalid measured confusion count/support")
                          for k in ("tp", "fn", "fp", "tn", "positive_photos", "negative_photos")}
                pos, neg = counts["positive_photos"], counts["negative_photos"]
                require(pos > 0 and neg > 0 and pos+neg == COUNTS[domain]
                        and counts["tp"]+counts["fn"] == pos and counts["tn"]+counts["fp"] == neg,
                        "Confusion counts differ from original VAL support")
                for rate, numerator, denominator in (("fnr", counts["fn"], pos), ("fpr", counts["fp"], neg)):
                    value = number(point.get(rate), "Measured target error rate")
                    require(math.isclose(value, numerator/denominator, rel_tol=0., abs_tol=1e-12),
                            "Reported target rate differs from actual confusion counts")
                    measured.append(value)
        worst = number(entry.get("worst_error"), "Maximum VAL target error")
        require(math.isclose(worst, max(measured), rel_tol=0., abs_tol=1e-12)
                and type(entry.get("target_passed")) is bool
                and entry["target_passed"] == all(v < .05 for v in measured),
                "Headline maximum/strict five-percent flag differs from the 12 measured rates")
        validate_small_area(entry)
        ranking = entry.get("ranking_ap", {})
        require(set(ranking) == set(DOMAINS), "Ranking AP sources are missing")
        for domain in DOMAINS:
            require(set(ranking[domain]) == set(CLASSES), "All seven AP known/unknown states are required")
            for k, task in enumerate(CLASSES):
                point = ranking[domain][task]
                known = nonnegative_integer(point.get("known_photos"), "Invalid AP known support")
                positive = nonnegative_integer(point.get("positive_photos"), "Invalid AP positive support")
                if k in KNOWN_INDICES[domain]:
                    require(known == COUNTS[domain] and positive <= known
                            and point.get("status") == "measured_on_asserted_source_labels",
                            "Measured AP source label support differs")
                    number(point.get("ap"), "Measured known-class AP")
                    if task in TARGETS:
                        require(positive == per_class[task]["domains"][domain]["positive_photos"],
                                "Target AP support differs from measured confusion support")
                    ap_count += 1
                else:
                    require(point.get("ap") is None and known == positive == 0
                            and point.get("status") == "not_measured_source_labels_unknown",
                            "Unknown source class cannot acquire AP, known labels, or positives")
    require(ap_count == 42, "Exactly 42 known source-class AP measurements are required")
    # This additionally checks identical support in every comparison.
    change = comparisons(*entries)
    return {"comparisons": change, "research_gate": detail_research_gate(*entries, protocol["research_candidate_gate"]),
            "error_rows": error_rows(entries), "verified_known_class_ap_measurements": ap_count}


def validate_preparation_resources(audit, data_proof):
    elapsed = audit.get("preparation_elapsed_minutes")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
            "Actual data preparation time is missing")
    workers = audit.get("worker_count")
    require(type(workers) is int and 1 <= workers <= 4, "Actual bounded preparation worker count is missing")
    sizes = audit.get("derived_png_bytes_by_variant", {})
    require(set(sizes) == {"control", "native"}
            and all(type(v) is int and v > 0 for v in sizes.values()),
            "Actual encoded PNG bytes by variant are missing")
    total = audit.get("derived_png_bytes_total")
    require(type(total) is int and total == sum(sizes.values()) == data_proof.get("output_png_bytes")
            and sizes == data_proof.get("output_png_bytes_by_variant"),
            "Prepared PNG resource total differs from actual verified files")
    resizes = audit.get("control_parent_resizes")
    require(type(resizes) is int and resizes == data_proof.get("changed_dacl_parents"),
            "Cached control parent resize count differs from affected parents")
    return {"preparation_elapsed_minutes": elapsed, "worker_count": workers,
            "control_parent_resizes": resizes, "derived_png_bytes_by_variant": dict(sizes),
            "derived_png_bytes_total": total, "included_in_training_elapsed_minutes": False}


def render(result):
    entries = result["experiments"]; change = result["comparisons"]
    data = result["prepared_data_integrity"]; proof = result["technical_verification"]
    lines = ["# 원본 세부 ROI 정보 보강 대조 실험 결과", "",
        "같은 0773 초기 가중치에서 processed 대조군과 native 보강군을 각 6epoch, 총 12epoch 실제 학습했다. 두 군 모두 기존 AUX 구조·640 입력·80×80 마스크·top32·7종 출력·19종 보조 태그를 유지한다.",
        "같은 EXIF 보정 native RGB를 사용한다. 대조군은 기존 processed 부모 크기로 LANCZOS 축소한 다음 원래 crop box를 읽고, 보강군은 같은 영역의 반열린 floating native box를 직접 읽는다. 둘 다 동일한 640×640 PNG로 저장하며 새 JPEG 압축은 넣지 않았다.",
        "두 신규 군의 640 PNG는 과거 최대512 JPEG crop과 다르다. 새 대조군은 역사적 JPEG 픽셀 재현이 아니며, 초기 모델 비교에는 이 전처리 차이와 추가 학습이 함께 포함된다. native 정보 효과의 직접 대조는 새 대조군과 보강군 사이에서 해석한다.", "",
        f"기하 조건만으로 기존 DACL TRAIN 부모 {data['changed_dacl_parents']:,}개의 세부 crop {data['changed_rows']:,}행을 선택했다. 양성·음성·미확인 crop을 모두 유지했고 실제 PNG {data['derived_pngs']:,}개를 검증했다. full 행·다른 출처·정답·마스크·행 순서·추출 가중치는 보존한다. crop의 19종 보조 태그는 계속 미확인이다.", "",
        "![Native ROI paired results](facility-native-roi-study-comparison.png)", "",
        "| 모델 | 입력 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |",
        "|---|---:|---:|---:|---:|---|"]
    for entry in entries:
        lines.append(f'| {entry["title"]} | {entry["imgsz"]} | {entry["actual_epochs"]} | {entry["best_epoch"]} | {100*entry["worst_error"]:.2f}% | {"통과" if entry["target_passed"] else "미달"} |')
    lines += ["", f'보강군−초기 모델 최대 오류 {change["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, 보강군−새 대조군 {change["maximum_error_treatment_minus_control_pp"]:+.2f}pp. 양수는 악화다.',
        "최대값은 균열·박락 × DACL710/Dam424/CODEBRIM611 × FNR/FPR의 12개 비율 중 최대다. 전체 사진 오답 비율이나 앱 정확도가 아니다.", "",
        "## 사전 선언한 연구 후보 기준", "",
        f'연구 후보 기준: **{"통과" if result["research_gate"]["research_candidate_nominated"] else "미달"}**. 초기 모델과 새 대조군 각각 대비 최대 오류 최소0.5pp 개선, 각 target 비율 악화2pp 이하, 다른 알려진 항목 AP 하락0.02 이하를 요구했다.',
        "작은 DACL 항목·사진 양성 사례의 FN 합계도 각각 최소2건 줄고 항목별 FNR 악화가2pp 이하여야 한다. 이는 연구 후보 조건이며 엄격한5%·현장 검증·배포 승인과 별도다.", "",
        "| 작은 손상 | 초기 FN/양성 | 새 대조 FN/양성 | native FN/양성 |", "|---|---:|---:|---:|"]
    for task in TARGETS:
        values = [entry["small_dacl_polygon_area_below_one_percent"][task] for entry in entries]
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{v["false_negatives"]}/{v["positive_photos"]}' for v in values) + ' |')
    lines += ["", "분모는 균열93·박락105로 고정된 198개 항목·사진 사례다. 같은 사진이 두 항목에 들어갈 수 있으며 고유 사진198장이나 물리적 크기 정답을 뜻하지 않는다.", "",
        "## 출처별 관측 오류와 기술 통계", "",
        "Wilson 양측95%는 고정 예측과 독립 사진 가정의 기술 통계다. 같은 VAL에서 반복한 epoch·임계값 선택과 같은 부모의 파생 crop 상관 때문에 확인적 현장 보장이나 모델 간 유의성 검정으로 해석하지 않는다.", "",
        "| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | Wilson95 | FP/음성 | 오탐률 | Wilson95 |",
        "|---|---|---|---:|---:|---|---:|---:|---|"]
    titles = {e["run"]: e["title"] for e in entries}
    interval = lambda values: f'{100*values[0]:.2f}%–{100*values[1]:.2f}%'
    for row in result["error_rows"]:
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {interval(row["fnr_wilson95_descriptive"])} | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% | {interval(row["fpr_wilson95_descriptive"])} |')
    lines += ["", "AP는 확률 순위 지표다. 세 모델의 알려진 출처·항목 AP42개(DACL7/Dam2/CODEBRIM5)를 원래 VAL 캐시로 재계산하고 동일 정답·양성 분모를 확인했다. 미확인 항목에0점 AP를 만들지 않는다.", "",
        "## 자원·보존·재현 확인", "",
        "| 모델 | 학습·epoch 검증 시간 | allocated peak | 실제 update | AMP skip |", "|---|---:|---:|---:|---:|"]
    for entry in entries[1:]:
        resources = entry["resources"]; steps = resources["optimizer_step_diagnostics"]
        lines.append(f'| {entry["title"]} | {resources["elapsed_training_minutes"]:.2f}분 | {resources["peak_cuda_allocated_bytes"]/1024**3:.3f} GiB | {steps["actual_optimizer_steps"]} | {steps["amp_skipped_steps"]} |')
    lines += ["", f'새 쌍의 완료 학습은 {result["actual_new_completed_training_epochs"]}epoch다. 자료 준비·preflight의 버려지는 AMP2회·검증 CPU forward는 완료 학습 epoch에 더하지 않는다. 다른 실험과 실행 중단의 완료 epoch는 누적 보고에서 별도 집계한다.',
        "실제6개 epoch의 추출 행 순서 SHA·출처·full/crop·target 조합 수를 대조했다. preflight에서는 실제 TRAIN8행의 같은 난수 증강으로 라벨·마스크·known·보조 태그·행 순서를 확인하고 군별 AMP1회와 CPU 재로딩을 실행했다. 이미지 tensor 자체는 바뀐 crop에서 같을 필요가 없다.",
        f'학습 전 commit의 {proof["runtime_source_count"]}개 소스 바이트와 실제 {proof["tests_run"]}개 코드 테스트 통과를 확인했다. 실제 완료 가중치 두 개는 원래 TRAIN 사진으로 CPU 재로딩·finite7/80/19 출력 계약을 확인했다. 코드 테스트 수는 정확도 평가 수가 아니다.',
        "모든 기록된 원본 TRAIN 입력의 SHA·size·mtime와 생성 PNG·재사용 마스크 바이트를 검사했다. 기하학적으로 같은 ROI라도 coarse80 격자의 원래 주석 오차는 남으며 새 native 정밀 위치 정답으로 바꾸지 않았다. 시간에는 자료 읽기·epoch 검증·캐시 영향이 포함되며 동일 epoch가 동일 시간·모바일 비용을 뜻하지 않는다.", "",
        "## 적용 상태와 범위", "",
        "보류 TEST 추론은 실행하지 않았다. 앱 기본 `facility-validation-v2`와 프로필 SHA를 유지하고 자동 배포·승격을 하지 않았다. 전문가 확정 라벨·원본 정답 변경·새 독립 사진은0개다.",
        "기존 공개 교량·댐 콘크리트 자료의 한 seed 반복 VAL 연구다. TRAIN 세부 crop만 native 정보로 보강하며 VAL은 계속 원래 processed full 사진640 입력이다. native full 현장 입력·공장 사진·시설 정상/안전 판정·정밀 검출 성능은 측정하지 않았다.",
        "원본이 작으면 새 세부 정보를 만들 수 없다. DACL19종 원본 주석은 이미 사용하던 보조 태그이며 새 자료 수집이 아니다. DamSegment Non-Crack의 기존 항목별 음성 변환과 미확인 태그 범위는 [정답 범위](FACILITY_LABEL_SCOPE_KO.md)에 따른다.",
        "반복 검증 결과로 미래 현장의 오차율5% 미만을 보장하지 않는다. 검출 없음으로 구조 안전이나 법적 점검 완료를 확정하지 않으며 최종 판단은 점검자가 한다. 공개 보고에는 개별 사진·주석·box·검수 내용·개인 절대 경로를 넣지 않았다.", "",
        "[사전 조건](facility-native-roi-study-protocol.json), [자료 보존 집계](facility-native-roi-data-audit.json), [실측 집계](facility-native-roi-study-comparison.json), [기술 검증](facility-native-roi-study-verification.json)", ""]
    resources = result.get("data_preparation_resources")
    if resources is not None:
        lines += ["## 자료 준비 실측 비용", "",
            f'자료 준비 {resources["preparation_elapsed_minutes"]:.2f}분, workers {resources["worker_count"]}, 생성 PNG 총 {resources["derived_png_bytes_total"]/1024**3:.3f} GiB를 기록했다. 대조 부모 축소는 부모별 캐시로 {resources["control_parent_resizes"]:,}회 실행했다.',
            "PNG 크기는 실제 압축 파일 합계다. 원본 자료·NPZ 마스크·모델·중간 캐시를 포함하는 전체 디스크 용량이 아니며, 준비 시간은 GPU 학습 시간과 따로 집계했다.", ""]
    return "\n".join(lines)


def main():
    require(not any((ROOT / path).exists() for path in (OUTPUT_PATH, MARKDOWN_PATH)), "Preserve existing study results")
    protocol = validate_protocol(read(ROOT / PROTOCOL), ROOT); protocol_sha = sha(ROOT / PROTOCOL)
    names = [protocol["reference"], protocol["control"], protocol["treatment"]]
    require(not any((ROOT / "reports" / f"{name}-target-test.json").exists() for name in names),
            "No-test study refuses held-out inference records")
    loaded = [load_run(name) for name in names]; trainings = [r[0] for r in loaded]; entries = [r[1] for r in loaded]
    require(trainings[0]["weights_sha256"] == protocol["initial_weights_sha256"], "Reference weights changed")
    for variant, training in zip(("control", "native"), trainings[1:]):
        fixed = {"status": "complete", "actual_epochs": protocol["requested_epochs"],
            "architecture": protocol["architecture_by_variant"][variant], "imgsz": 640, "model_variant": variant,
            "initial_weights_sha256": protocol["initial_weights_sha256"],
            "core_spatial_manifest_sha256": protocol["core_spatial_manifest_sha256"],
            "spatial_manifest_sha256": protocol["paired_manifest_sha256"][variant],
            "spatial_manifest_path": MANIFESTS[variant], "auxiliary_manifest_sha256": protocol["auxiliary_manifest_sha256"],
            "source_sha256": protocol["source_sha256"], "study_protocol_sha256": protocol_sha,
            "native_roi_recipe": protocol["native_roi_recipe"], "paired_data_audit_sha256": protocol["paired_data_audit_sha256"]}
        for key in ("seed", "requested_epochs", "patience", "batch_size", "draws_per_epoch", "backbone_lr",
                    "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions"):
            fixed[key] = protocol[key]
        for key, expected in fixed.items():
            require(type(training.get(key)) is type(expected) and training[key] == expected,
                    f"Actual native ROI condition differs: {key}")
    for key in ("classes", "split_sha256", "core_spatial_manifest_sha256", "auxiliary_manifest_sha256",
                "expected_sampling", "expected_label_sampling", "photo_positive_weights", "pixel_positive_weights",
                "auxiliary_positive_weights", "additional_validation", "additional_test", "paired_label_preservation"):
        require(key in trainings[1] and trainings[1][key] == trainings[2].get(key), f"Paired source/loss differs: {key}")
    histories = [read(ROOT / "runs" / name / "history.json") for name in names[1:]]
    sampling = validate_histories(*histories, protocol["requested_epochs"], protocol["draws_per_epoch"])
    truth = []
    for entry in entries:
        run = ROOT / "runs" / entry["run"]; digests = {}
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run / f"validation-{domain}.json"),
                    read(run / f"target-validation-{domain}-grid1.json"), entry, domain)
            entry["ranking_ap"][domain] = points; digests[domain] = digest
        truth.append(digests)
    require(truth[0] == truth[1] == truth[2], "Original VAL truth/order differs across the three models")
    for index, (entry, training, title) in enumerate(zip(entries, trainings,
                ("추가 학습 전 ROI 모델", "processed640 PNG 대조군", "native640 PNG 보강군"))):
        entry.update(title=title, imgsz=training["imgsz"])
        if index:
            summary = training["optimizer_step_diagnostics"]
            require(all(type(summary.get(key)) is int and summary[key] == sum(h["optimizer_step_diagnostics"][key]
                        for h in histories[index-1]) for key in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")),
                    "Training optimizer summary differs from actual epochs")
            require(type(training.get("elapsed_training_minutes")) in (int, float)
                    and math.isfinite(training["elapsed_training_minutes"]) and training["elapsed_training_minutes"] > 0
                    and type(training.get("peak_cuda_allocated_bytes")) is int and training["peak_cuda_allocated_bytes"] > 0,
                    "Actual resource measurements missing")
            entry["resources"] = {k: training[k] for k in ("elapsed_training_minutes", "peak_cuda_allocated_bytes", "optimizer_step_diagnostics")}
    measured = validate_measurements(entries, protocol)
    protected = protected_hashes(ROOT); proof_path = ROOT / VERIFICATION_PATH; proof = read(proof_path)
    technical = validate_technical_proof(proof, protocol, protocol_sha, trainings, protected)
    require(proof.get("preflight_sha256") == sha(ROOT / PREFLIGHT_PATH)
            and proof.get("source_before_training_sha256") == sha(ROOT / SOURCE_RECORD_PATH)
            and proof.get("test_results_sha256") == sha(ROOT / TEST_RECORD_PATH)
            and proof["prepared_data_integrity"].get("ledger_sha256") == sha(ROOT / LEDGER_PATH)
            and proof["prepared_data_integrity"].get("data_audit_sha256") == sha(ROOT / AUDIT),
            "Technical proof input bytes changed")
    require(validate_test_results(read(ROOT / TEST_RECORD_PATH), ROOT) == proof["tests"],
            "Actual executed test record differs from completion proof")
    preparation_resources = validate_preparation_resources(read(ROOT / AUDIT), proof["prepared_data_integrity"])
    result = {"schema": "facility_native_roi_study_comparison_v1", "protocol_sha256": protocol_sha,
        "experiments": entries, **measured, "actual_sampling_verification": sampling,
        "original_validation_truth_sha256": truth[0], "app_profile_sha256": sha(ROOT / APP_PROFILE),
        "prepared_data_integrity": proof["prepared_data_integrity"], "technical_verification": technical,
        "data_preparation_resources": preparation_resources,
        "technical_verification_sha256": sha(proof_path), "actual_new_completed_training_epochs": 12,
        "deployed": False, "app_model_promoted": False, "source_test_inference_executed": False,
        "additional_expert_confirmed_labels": 0, "label_changes": 0, "new_independent_photos": 0,
        "new_photo_targets": 0, "new_pixel_targets": 0,
        "historical_crop_pixel_replay_asserted": False, "both_new_arms_input_png_size": [640, 640],
        "scope": "Repeated public-source processed full-photo VAL presence; TRAIN native detail pixels only; not factory accuracy or fine localization"}
    require(protected_hashes(ROOT) == protected, "Reporting changed protected original/app/protocol files")
    with (ROOT / OUTPUT_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with (ROOT / MARKDOWN_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render(result))
    print(json.dumps({"actual_new_epochs": result["actual_new_completed_training_epochs"],
        "maximum_validation_error": [e["worst_error"] for e in entries],
        "research_candidate_nominated": result["research_gate"]["research_candidate_nominated"],
        "strict_target_passed": [e["target_passed"] for e in entries], "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
