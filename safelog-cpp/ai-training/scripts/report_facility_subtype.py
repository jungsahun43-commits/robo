"""Report the completed, predeclared TRAIN subtype-sampling pair.

Only saved, provenance-checked source-VAL aggregates are published. This script
does not infer on photographs, change labels, or promote an application model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_facility_small_region import DOMAINS, LABELS, TARGETS, load_run, read, require, sha
from scripts.report_facility_discrimination import verify_validation_cache
from scripts.report_facility_native_roi import validate_measurements
from scripts.report_facility_native_roi_results import normalize_ap_positive_support

PROTOCOL_PATH = "reports/facility-subtype-study-protocol.json"
VERIFICATION_PATH = "reports/facility-subtype-study-verification.json"
PREFLIGHT_PATH = "runs/facility-subtype-preflight.json"
SOURCE_RECORD_PATH = "runs/facility-subtype-source-before-training.json"
TEST_RECORD_PATH = "runs/facility-subtype-test-results.json"
OUTPUT_PATH = "reports/facility-subtype-study-comparison.json"
MARKDOWN_PATH = "reports/FACILITY_SUBTYPE_STUDY_RESULTS_KO.md"
APP_PROFILE = "reports/facility-inference-profile.json"
VARIANTS = ("control", "negative")


def normalized_load_run(name, loader=load_run):
    """Run every frozen loader contract before adapting count representation."""
    training, entry = loader(name)
    normalized, count = normalize_ap_positive_support(entry)
    return training, normalized, count


def validate_histories(control, treatment, protocol, plan):
    """Require the exact prepared schedules, including their intentional changes."""
    epochs, draws = protocol["requested_epochs"], protocol["draws_per_epoch"]
    require(isinstance(control, list) and isinstance(treatment, list)
            and len(control) == len(treatment) == epochs == len(plan["epochs"]),
            "Both complete predeclared histories are required")
    counts_keys = ("sampled_domain_counts", "sampled_row_type_counts", "sampled_full_target_joint_counts")
    actual_rows = []
    for epoch, (left, right, expected) in enumerate(zip(control, treatment, plan["epochs"]), 1):
        require(type(expected.get("epoch")) is int and expected["epoch"] == epoch,
                "Prepared sampler epoch sequence differs")
        for variant, row in zip(VARIANTS, (left, right)):
            require(type(row.get("epoch")) is int and row["epoch"] == epoch,
                    "Actual epoch sequence differs")
            declared_hash = expected["control_order_sha256" if variant == "control" else "treatment_order_sha256"]
            require(isinstance(declared_hash, str) and re.fullmatch(r"[a-f0-9]{64}", declared_hash)
                    and row.get("sampled_row_indices_sha256") == declared_hash,
                    "Actual draw order differs from the prepared schedule")
            for history_key, plan_key in zip(counts_keys, ("domain_counts", "full_crop_counts", "full_target_joint_counts")):
                require(row.get(history_key) == expected[plan_key], "Actual source/full-crop/target counts differ")
            require(set(row["sampled_domain_counts"]) == set(DOMAINS)
                    and set(row["sampled_row_type_counts"]) == {"full", "crop"}
                    and all(type(v) is int and v >= 0 for key in counts_keys[:2] for v in row[key].values())
                    and all(sum(row[key].values()) == draws for key in counts_keys[:2]),
                    "Actual sample category support differs")
            for domain, joints in row["sampled_full_target_joint_counts"].items():
                require(domain in DOMAINS and set(joints) == {"00", "10", "01", "11", "unknown"}
                        and all(type(v) is int and v >= 0 for v in joints.values()) and joints["unknown"] == 0,
                        "Original full target-state support differs")
            require(sum(sum(v.values()) for v in row["sampled_full_target_joint_counts"].values())
                    == row["sampled_row_type_counts"]["full"], "Target-state counts do not cover full rows")
            sampling = row.get("subtype_sampling", {})
            eligible = expected["eligible_draws_control" if variant == "control" else "eligible_draws_treatment"]
            changed = 0 if variant == "control" else expected["changed_positions"]
            require(type(sampling.get("eligible_draws")) is int and sampling["eligible_draws"] == eligible
                    and type(sampling.get("changed_positions_from_control")) is int
                    and sampling["changed_positions_from_control"] == changed
                    and sampling.get("declared_draw_sha256") == declared_hash
                    and sampling.get("draw_hash_matches_prepared") is True,
                    "Actual eligible exposure or prepared-draw proof differs")
            steps = row.get("optimizer_step_diagnostics", {})
            require(all(type(steps.get(k)) is int and steps[k] >= 0 for k in
                        ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps"))
                    and steps["attempted_batches"] == math.ceil(draws / protocol["batch_size"])
                    and steps["actual_optimizer_steps"] + steps["amp_skipped_steps"] == steps["attempted_batches"],
                    "Actual optimizer update evidence is inconsistent")
        require(all(left[key] == right[key] for key in counts_keys), "Paired categorical exposure differs")
        require(left["sampled_row_indices_sha256"] != right["sampled_row_indices_sha256"]
                and expected["changed_positions"] > 0,
                "The prepared subtype intervention was not actually used")
        actual_rows.append({"epoch": epoch, "control_order_sha256": left["sampled_row_indices_sha256"],
            "treatment_order_sha256": right["sampled_row_indices_sha256"],
            "eligible_draws_control": left["subtype_sampling"]["eligible_draws"],
            "eligible_draws_treatment": right["subtype_sampling"]["eligible_draws"],
            "changed_positions": right["subtype_sampling"]["changed_positions_from_control"]})
    totals = {key: sum(row[key] for row in actual_rows) for key in
              ("eligible_draws_control", "eligible_draws_treatment", "changed_positions")}
    require(totals == {"eligible_draws_control": 3539, "eligible_draws_treatment": 4743,
                       "changed_positions": 1204}, "Fixed prepared subtype totals differ")
    return {"epochs_compared": epochs, "draws_per_epoch": draws, **totals, "epochs": actual_rows,
        "actual_order_hashes_match_prepared_each_epoch": True,
        "actual_source_full_crop_two_target_counts_identical_each_epoch": True,
        "actual_ordered_row_indices_identical_asserted": False,
        "other_photo_and_auxiliary_label_exposure_identical_asserted": False}


def validate_pair(trainings, protocol, protocol_sha):
    require(len(trainings) == 3 and trainings[0]["weights_sha256"] == protocol["initial_weights_sha256"],
            "Original 0773 reference weights differ")
    common = {key: protocol[key] for key in ("seed", "requested_epochs", "patience", "batch_size",
        "draws_per_epoch", "backbone_lr", "head_lr", "auxiliary_weight", "loader_randomness", "domain_proportions")}
    common.update(status="complete", actual_epochs=protocol["requested_epochs"], imgsz=640,
        initial_weights_sha256=protocol["initial_weights_sha256"],
        core_spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        spatial_manifest_sha256=protocol["core_spatial_manifest_sha256"],
        auxiliary_manifest_sha256=protocol["auxiliary_manifest_sha256"],
        source_sha256=protocol["source_sha256"], study_protocol_sha256=protocol_sha,
        subtype_sampling_recipe=protocol["subtype_sampling_recipe"], sampler_plan_sha256=protocol["sampler_plan_sha256"],
        private_draw_archive_sha256=protocol["paired_draws_sha256"])
    for variant, training in zip(VARIANTS, trainings[1:]):
        expected = {**common, "model_variant": variant,
                    "architecture": protocol["architecture_by_variant"][variant]}
        for key, value in expected.items():
            require(type(training.get(key)) is type(value) and training[key] == value,
                    f"Actual fixed subtype condition differs: {key}")
    matched = ("classes", "split_sha256", "core_spatial_manifest_sha256", "spatial_manifest_sha256",
        "auxiliary_manifest_sha256", "expected_sampling", "expected_label_sampling", "photo_positive_weights",
        "pixel_positive_weights", "auxiliary_positive_weights", "additional_validation", "additional_test",
        "paired_label_preservation")
    for key in matched:
        require(key in trainings[1] and trainings[1][key] == trainings[2].get(key),
                f"Original paired source/loss/label condition differs: {key}")


def validate_technical_proof(proof, protocol, protocol_sha, trainings, protected):
    require(proof.get("schema") == "facility_subtype_study_verification_v1" and proof.get("status") == "passed",
            "Actual completed subtype verification is required")
    require(proof.get("protocol_sha256") == protocol_sha and proof.get("source_sha256") == protocol["source_sha256"]
            and type(proof.get("runtime_source_count")) is int
            and proof["runtime_source_count"] == len(protocol["source_sha256"]),
            "Technical proof belongs to another frozen study")
    require(isinstance(proof.get("source_git_commit"), str)
            and re.fullmatch(r"[a-f0-9]{40}", proof["source_git_commit"]), "Before-training commit proof missing")
    for key in ("git_blob_bytes_verified", "working_runtime_sources_unchanged", "protected_files_unchanged"):
        require(proof.get(key) is True, "Technical byte-preservation proof is missing")
    require(type(proof.get("actual_completed_training_epochs")) is int
            and proof["actual_completed_training_epochs"] == 2 * protocol["requested_epochs"]
            and type(proof.get("verification_training_epochs")) is int and proof["verification_training_epochs"] == 0,
            "Actual completed-epoch proof differs")
    for key in ("source_test_inference_executed", "app_model_promoted", "deployed", "accuracy_measured_by_verifier"):
        require(proof.get(key) is False, "Technical verification cannot claim TEST, promotion or accuracy")
    for key in ("additional_expert_confirmed_labels", "label_changes", "new_photo_targets", "new_pixel_targets"):
        require(type(proof.get(key)) is int and proof[key] == 0, "No new truth was authorized for this study")
    require(proof.get("protected_file_sha256") == protected, "Protected source/data/profile proof differs")
    tests = proof.get("tests", {})
    require(type(tests.get("tests_run")) is int and tests["tests_run"] > 0
            and tests.get("tests_run") == tests.get("expected_tests_collected")
            and all(type(tests.get(k)) is int and tests[k] == 0 for k in ("failures", "errors", "skipped"))
            and tests.get("source_sha256") == protocol["source_sha256"], "Actual complete focused tests must pass")
    for path, expected in tests.get("test_source_sha256", {}).items():
        require(sha(ROOT / path) == expected, "Executed focused test bytes changed")
    data = proof.get("prepared_data_integrity", {})
    require(data.get("status") == "passed" and type(data.get("input_files_checked")) is int
            and data["input_files_checked"] > 0
            and data.get("all_original_input_sha_size_mtime_preserved") is True
            and data.get("original_train_images_masks_annotations_verified") is True
            and data.get("individual_input_paths_published") is False,
            "Actual original TRAIN input-byte proof is required")
    sampling = proof.get("actual_sampling_verification", {})
    require(type(sampling.get("epochs_compared")) is int
            and sampling["epochs_compared"] == protocol["requested_epochs"]
            and type(sampling.get("draws_per_epoch_each_arm")) is int
            and sampling["draws_per_epoch_each_arm"] == protocol["draws_per_epoch"]
            and type(sampling.get("changed_positions")) is int and sampling["changed_positions"] == 1204
            and sampling.get("eligible_draws_by_variant") == {"control": 3539, "negative": 4743}
            and all(sampling.get(k) is True for k in (
                "both_actual_draw_orders_match_fixed_prepared_arrays",
                "actual_source_full_crop_two_target_strata_identical_at_every_position",
                "actual_domain_full_crop_joint_counts_identical_each_epoch",
                "other_five_photo_and_auxiliary_exposure_expected_to_differ")),
            "Actual fixed paired array/exposure proof differs")
    require(isinstance(proof.get("experiments"), list) and len(proof["experiments"]) == 2,
            "Both completed checkpoint proofs are required")
    for variant, training, point in zip(VARIANTS, trainings[1:], proof["experiments"]):
        expected = {"variant": variant, "weights_sha256": training["weights_sha256"],
            "actual_epochs": protocol["requested_epochs"], "imgsz": 640,
            "architecture": protocol["architecture_by_variant"][variant],
            "strict_state_inventory_verified": True, "new_state_tensor_count": 0,
            "state_tensor_count": 324}
        for key, value in expected.items():
            require(type(point.get(key)) is type(value) and point[key] == value,
                    "Completed checkpoint identity/inventory differs")
        reload = point.get("cpu_reload", {})
        require(all(reload.get(k) is True for k in ("strict_factory_reload_verified", "all_outputs_finite",
                    "public_output_equals_training_photo_output")), "Actual CPU checkpoint reload must pass")
    return {"status": "passed", "source_git_commit": proof["source_git_commit"],
        "runtime_source_count": proof["runtime_source_count"], "tests_run": tests["tests_run"],
        "actual_completed_training_epochs": proof["actual_completed_training_epochs"]}


def render(result):
    entries, change, proof = result["experiments"], result["comparisons"], result["technical_verification"]
    sampling = result["actual_sampling_verification"]
    lines = ["# 박락 음성 하위유형 추출 보강 대조 학습 결과", "",
        "같은 0773 초기 가중치에서 대조군·보강군을 각 6epoch, 총 12epoch 실제 추가 학습했다. 기존 AUX 구조·640 입력·80×80 마스크·7종 출력·19종 보조 태그를 유지했다.",
        "원본 DACL full TRAIN 중 박락 음성이고 Rockpocket·WConccor·Hollowareas·Cavity 중 하나가 주석된 666개 부모만 후보로 삼았다. 이 태그들을 박락으로 재명명하거나 태그 부재를 정상·안전 정답으로 바꾸지 않았다.",
        "Crack/Spalling 00·10 조합 안에서 조건부 가중치 1.5를 사용했다. 별도 seed59 최대 결합으로 원래 eligible 추출을 모두 유지하고, 같은 조합의 noneligible 행 일부만 후보 행으로 교체했다. 전체 가중치를 일괄 1.5배한 실험이 아니므로 실제 노출 증가가 정확히 1.5배는 아니다.", "",
        f'실제 85,488회 추출 중 {sampling["changed_positions"]:,}위치만 달라졌다. 후보 하위유형 노출은 대조 {sampling["eligible_draws_control"]:,}→보강 {sampling["eligible_draws_treatment"]:,}회였다. 각 위치의 출처·full/crop·균열/박락 정답은 동일하다. 다른 5종 사진 항목·19종 보조 태그의 노출은 달라질 수 있어 AP 퇴행 방지를 함께 확인했다.', "",
        "![Subtype sampling paired results](facility-subtype-study-comparison.png)", "",
        "| 모델 | 입력 | 실제 epoch | 선택 epoch | 최대 검증 미탐·오탐 | 엄격한 5% 기준 |",
        "|---|---:|---:|---:|---:|---|"]
    for entry in entries:
        lines.append(f'| {entry["title"]} | {entry["imgsz"]} | {entry["actual_epochs"]} | {entry["best_epoch"]} | {100*entry["worst_error"]:.2f}% | {"통과" if entry["target_passed"] else "미달"} |')
    lines += ["", f'보강군−초기 모델 최대 오류 {change["maximum_error_treatment_minus_initializer_pp"]:+.2f}pp, 보강군−새 대조군 {change["maximum_error_treatment_minus_control_pp"]:+.2f}pp. 양수는 악화다.',
        "최대값은 균열·박락 × DACL710/Dam424/CODEBRIM611 × FNR/FPR의 12개 비율 중 최대다. 전체 사진 오답 비율이나 앱 정확도가 아니다.", "",
        "## 사전 선언한 연구 후보 기준", "",
        f'연구 후보 기준: **{"통과" if result["research_gate"]["research_candidate_nominated"] else "미달"}**. 초기 모델·새 대조군 각각 대비 최대 오류 최소0.5pp 개선, 각 target 비율 악화2pp 이하, 다른 알려진 항목 AP 하락0.02 이하를 요구했다.',
        "작은 DACL 항목·사진 양성 사례 FN 합계도 두 비교 기준 각각보다 최소2건 줄고 항목별 FNR 악화가2pp 이하여야 한다. 연구 후보 조건은 엄격한5%·현장 성능·앱 배포를 뜻하지 않는다.", "",
        "| 작은 손상 | 초기 FN/양성 | 새 대조 FN/양성 | 보강 FN/양성 |", "|---|---:|---:|---:|"]
    for task in TARGETS:
        values = [e["small_dacl_polygon_area_below_one_percent"][task] for e in entries]
        lines.append(f'| {LABELS[task]} | ' + ' | '.join(f'{v["false_negatives"]}/{v["positive_photos"]}' for v in values) + ' |')
    lines += ["", "분모는 균열93·박락105의 198개 항목·사진 사례다. 같은 사진이 두 항목에 포함될 수 있고 고유 사진198장·물리적 손상 크기를 뜻하지 않는다.", "",
        "## 출처별 관측 오류", "",
        "Wilson 양측95% 구간은 고정 예측과 독립 사진 가정 아래 기술 통계다. 같은 VAL에서 반복한 epoch·임계값 선택 및 파생 crop 상관을 보정한 확인적 현장 보장이나 모델 간 유의성 검정이 아니다.", "",
        "| 모델 | 출처 | 항목 | FN/양성 | 미탐률 | Wilson95 | FP/음성 | 오탐률 | Wilson95 |",
        "|---|---|---|---:|---:|---|---:|---:|---|"]
    titles = {e["run"]: e["title"] for e in entries}
    interval = lambda values: f'{100*values[0]:.2f}%–{100*values[1]:.2f}%'
    for row in result["error_rows"]:
        lines.append(f'| {titles[row["run"]]} | {row["domain"]} | {LABELS[row["class"]]} | {row["fn"]}/{row["positive_photos"]} | {100*row["fnr"]:.2f}% | {interval(row["fnr_wilson95_descriptive"])} | {row["fp"]}/{row["negative_photos"]} | {100*row["fpr"]:.2f}% | {interval(row["fpr_wilson95_descriptive"])} |')
    lines += ["", "알려진 출처·항목 AP42개(DACL7/Dam2/CODEBRIM5 ×3모델)를 원래 full VAL 캐시로 재계산하고 정답·순서·양성 분모를 확인했다. 미확인 항목에0점 AP를 만들지 않는다.", "",
        "## 자원·보존·재현 확인", "",
        "| 모델 | 학습·epoch 검증 시간 | allocated peak | 실제 update | AMP skip |", "|---|---:|---:|---:|---:|"]
    for entry in entries[1:]:
        resources = entry["resources"]; steps = resources["optimizer_step_diagnostics"]
        lines.append(f'| {entry["title"]} | {resources["elapsed_training_minutes"]:.2f}분 | {resources["peak_cuda_allocated_bytes"]/1024**3:.3f} GiB | {steps["actual_optimizer_steps"]} | {steps["amp_skipped_steps"]} |')
    lines += ["", f'학습 전 commit의 {proof["runtime_source_count"]}개 실행 소스 바이트와 실제 코드 테스트 {proof["tests_run"]}개 통과를 확인했다. 테스트 수는 정확도 평가 사례 수가 아니다. 실제 완료 가중치 두 개는 CPU에서 엄격하게 재로딩하고 finite7/80/19 출력 계약을 확인했다.',
        "원래 TRAIN full14,248·전체26,289행의 사진·마스크·known·정답·순서·기존 추출 가중치·pixel/photo/auxiliary 손실 양성 가중치를 유지했다. 두 군의 optimizer update 수는 AMP overflow로 달라질 수 있으며 건너뛴 update를 epoch와 별도로 기록했다.",
        "과거 native640 PNG pair를 사용하지 않고 원래 historical core manifest를 사용했다. 원본 이미지·주석·체크포인트의 SHA·size·mtime를 검사했다. 시간에는 자료 읽기·epoch 검증·캐시 영향이 포함되며 모바일 추론 시간은 측정하지 않았다.",
        "원래 AP loader 검증 후 유한·비음수·known 분모 이내의 정확한 정수값 float 양성 개수만 int로 표시했다. bool·소수·NaN·무한대는 거부한다. 점수·정답·AP·미탐·오탐·임계값·후보 기준은 바꾸지 않았다.", "",
        "## 적용 상태와 범위", "",
        "보류 TEST 추론과 자동 배포·모델 승격은 실행하지 않았다. 앱 기본 `facility-validation-v2` 및 프로필 SHA를 유지했다. 새 독립 사진·새 사진/픽셀 정답·전문가 확정 라벨·원본 라벨 수정은0개다.",
        "TRAIN 감사에서 보인 관련 태그 동시 존재는 오탐 원인의 확정이나 라벨 오류 판정이 아니다. 후보666개는 기존 원본 태그로 정했고 사진별 점수·VAL 오답으로 추출 대상을 바꾸지 않았다.",
        "한 seed와 반복 사용한 공개 교량·댐 콘크리트 자료의 source-VAL 사진 존재 분류 연구다. 공장 시설 사진·정밀 위치 검출·시설 정상/안전 판정·실제 현장 오차율은 측정하지 않았다. 반복 VAL 결과로 향후 현장5% 미만을 보장하지 않는다. 최종 안전 판단은 점검자가 한다.", "",
        "[사전 고정 조건](facility-subtype-study-protocol.json), [준비한 추출 집계](facility-spalling-sampler-dry-run.json), [실측 집계](facility-subtype-study-comparison.json), [기술 검증](facility-subtype-study-verification.json)", ""]
    return "\n".join(lines)


def main():
    from scripts.facility_subtype_study import validate_protocol, protected_hashes
    from scripts.verify_facility_subtype import validate_test_results
    require(not any((ROOT / p).exists() for p in (OUTPUT_PATH, MARKDOWN_PATH)), "Preserve completed reporting evidence")
    protocol = validate_protocol(read(ROOT / PROTOCOL_PATH), ROOT); protocol_sha = sha(ROOT / PROTOCOL_PATH)
    protected = protected_hashes(ROOT)
    plan = read(ROOT / "reports/facility-spalling-sampler-dry-run.json")
    require(sha(ROOT / "reports/facility-spalling-sampler-dry-run.json") == protocol["sampler_plan_sha256"],
            "Fixed prepared sampler report changed")
    names = [protocol["reference"], protocol["control"], protocol["treatment"]]
    require(not any((ROOT / "reports" / f"{name}-target-test.json").exists() for name in names),
            "Source-VAL study refuses held-out TEST records")
    loaded = [normalized_load_run(name) for name in names]
    trainings, entries = [r[0] for r in loaded], [r[1] for r in loaded]
    validate_pair(trainings, protocol, protocol_sha)
    histories = [read(ROOT / "runs" / name / "history.json") for name in names[1:]]
    sampling = validate_histories(*histories, protocol, plan)
    truth, normalization = [], []
    for entry, loaded_row in zip(entries, loaded):
        run = ROOT / "runs" / entry["run"]; digests = {}; cache_conversions = 0
        for domain in DOMAINS:
            points, digest = verify_validation_cache(read(run / f"validation-{domain}.json"),
                read(run / f"target-validation-{domain}-grid1.json"), entry, domain)
            wrapped, conversions = normalize_ap_positive_support({"ranking_ap": {domain: points}})
            entry["ranking_ap"][domain] = wrapped["ranking_ap"][domain]
            cache_conversions += conversions; digests[domain] = digest
        truth.append(digests)
        normalization.append({"run": entry["run"], "loader_support_fields_normalized": loaded_row[2],
                              "verified_cache_support_fields_normalized": cache_conversions})
    require(truth[0] == truth[1] == truth[2], "Original VAL truth/order changed across the three models")
    for index, (entry, training, title) in enumerate(zip(entries, trainings,
            ("추가 학습 전 ROI 모델", "원래 추출 대조군", "박락 음성 하위유형 보강군"))):
        entry.update(title=title, imgsz=training["imgsz"])
        if index:
            steps = training["optimizer_step_diagnostics"]
            require(all(type(steps.get(k)) is int and steps[k] == sum(h["optimizer_step_diagnostics"][k]
                for h in histories[index - 1]) for k in ("attempted_batches", "actual_optimizer_steps", "amp_skipped_steps")),
                "Actual resource summary differs from completed histories")
            require(type(training.get("elapsed_training_minutes")) in (int, float)
                and math.isfinite(training["elapsed_training_minutes"]) and training["elapsed_training_minutes"] > 0
                and type(training.get("peak_cuda_allocated_bytes")) is int and training["peak_cuda_allocated_bytes"] > 0,
                "Actual training resource measurements are missing")
            entry["resources"] = {k: training[k] for k in ("elapsed_training_minutes", "peak_cuda_allocated_bytes", "optimizer_step_diagnostics")}
    measured = validate_measurements(entries, protocol)
    proof_path = ROOT / VERIFICATION_PATH; proof = read(proof_path)
    technical = validate_technical_proof(proof, protocol, protocol_sha, trainings, protected)
    require(proof.get("preflight_sha256") == sha(ROOT / PREFLIGHT_PATH)
        and proof.get("source_before_training_sha256") == sha(ROOT / SOURCE_RECORD_PATH)
        and proof.get("test_results_sha256") == sha(ROOT / TEST_RECORD_PATH), "Verification input bytes changed")
    preflight = read(ROOT / PREFLIGHT_PATH)
    require(proof["prepared_data_integrity"]["ledger_sha256"] == preflight["protected_input_ledger_sha256"]
            == sha(ROOT / preflight["protected_input_ledger_path"]), "Original TRAIN private input ledger changed")
    verified_sampling = proof["actual_sampling_verification"]
    require(verified_sampling["ordered_row_hashes_by_variant"] == {
        "control": [row["control_order_sha256"] for row in sampling["epochs"]],
        "negative": [row["treatment_order_sha256"] for row in sampling["epochs"]]},
        "Technical completion proof reports another prepared draw order")
    require(validate_test_results(read(ROOT / TEST_RECORD_PATH), ROOT) == proof["tests"],
            "Executed test evidence differs from completion proof")
    result = {"schema": "facility_subtype_study_comparison_v1", "protocol_sha256": protocol_sha,
        "experiments": entries, **measured, "actual_sampling_verification": sampling,
        "original_validation_truth_sha256": truth[0], "app_profile_sha256": sha(ROOT / APP_PROFILE),
        "prepared_data_integrity": proof["prepared_data_integrity"], "technical_verification": technical,
        "technical_verification_sha256": sha(proof_path), "sampler_plan_sha256": protocol["sampler_plan_sha256"],
        "private_draw_archive_sha256": protocol["paired_draws_sha256"],
        "actual_new_completed_training_epochs": 12, "deployed": False, "app_model_promoted": False,
        "source_test_inference_executed": False, "additional_expert_confirmed_labels": 0,
        "label_changes": 0, "new_independent_photos": 0, "new_photo_targets": 0, "new_pixel_targets": 0,
        "reporting_provenance": {"normalization": "finite exact integral positive_photos floats only; original known support bounds",
            "normalizations_by_run": normalization, "metrics_labels_thresholds_gate_code_unchanged": True,
            "reporter_source_sha256": sha(Path(__file__))},
        "scope": "Repeated original public-source full-photo VAL presence; TRAIN conditional subtype sampling only; not factory accuracy or fine localization"}
    require(protected_hashes(ROOT) == protected, "Reporting changed protected original/app/protocol files")
    with (ROOT / OUTPUT_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    with (ROOT / MARKDOWN_PATH).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render(result))
    print(json.dumps({"actual_new_epochs": 12, "maximum_validation_error": [e["worst_error"] for e in entries],
        "research_candidate_nominated": result["research_gate"]["research_candidate_nominated"],
        "strict_target_passed": [e["target_passed"] for e in entries], "app_promoted": False}))
    return result


if __name__ == "__main__":
    main()
