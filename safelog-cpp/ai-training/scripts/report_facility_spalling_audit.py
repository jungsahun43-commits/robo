"""Publish only aggregate findings from the completed TRAIN spalling audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "reports/FACILITY_SPALLING_TRAIN_AUDIT_KO.md"
    if target.exists():
        raise ValueError("Preserve the previous completed report")
    audit = read("reports/facility-spalling-train-audit.json")
    tag = read("reports/facility-spalling-tag-score-audit.json")
    plan = read("reports/facility-spalling-sampler-dry-run.json")
    if (audit.get("status") != "audited" or tag.get("status") != "audited"
            or audit["case_counts"] != {"all": 130, "errors": 54, "direct_spalling_errors": 49,
                                        "additional_other_task_errors": 5}):
        raise ValueError("Actual completed TRAIN audit inputs are required")
    if (plan.get("schema") != "facility_spalling_sampler_dry_run_v1"
            or plan.get("status") != "prepared_dry_run" or plan.get("eligible_factor") != 1.5
            or plan.get("all_six_actual_control_and_native_order_hashes_match") is not True
            or plan.get("all_six_source_full_crop_crack_spalling_draw_counts_equal") is not True
            or plan.get("every_position_original_two_photo_targets_equal") is not True
            or plan.get("new_training_epochs") != 0):
        raise ValueError("Source-bound sampler dry-run must actually pass before reporting")
    dacl = {(g["domain"], g["outcome"]): g for g in audit["groups"]}
    fn, tp = dacl["dacl", "FN"], dacl["dacl", "TP"]
    groups = tag["disjoint_four_groups"]
    rows = []
    titles = {"spalling_positive_other_four_present": "박락 양성·관련4태그 있음",
              "spalling_positive_other_four_absent": "박락 양성·관련4태그 없음",
              "spalling_negative_other_four_present": "박락 음성·관련4태그 있음",
              "spalling_negative_other_four_absent": "박락 음성·관련4태그 없음"}
    for key, group in groups.items():
        positive = key.startswith("spalling_positive")
        kind = "FN" if positive else "FP"
        errors = group["confusion"][kind]
        rate = group["descriptive_train_fnr" if positive else "descriptive_train_fpr"]
        rows.append(f"| {titles[key]} | {group['photos']} | {kind} {errors} | {100 * rate:.2f}% |")
    lines = ["# 박락 TRAIN 오답 감사와 다음 표본 추출 준비", "",
        "0773 모델의 기존 TRAIN 점수와 원본 주석·80격자 마스크를 읽어 감사했다. 이번에는 새 모델 추론·학습·라벨 수정·전문가 판정을 수행하지 않았다.", "",
        "## 검수 표본 130장의 실제 확인", "",
        "| 출처 | 박락 오탐 FP | 박락 미탐 FN | 박락 오류로 직접 선택 | 다른 과제에서 추가 확인 |",
        "|---|---:|---:|---:|---:|",
        "| DACL | 17 | 12 | 24 | 5 |", "| CODEBRIM | 12 | 12 | 24 | 0 |",
        "| DamSegment | 1 | 0 | 1 | 0 |", "| 합계 | 30 | 24 | 49 | 5 |", "",
        "표본은 기존 seed49의 출처·항목·오류별 선별 묶음이다. 일부는 강한 오답이고 일부는 난수 선택이다. 이 54/130을 전체 TRAIN 또는 현장의 오류율로 해석하지 않는다.", "",
        f"모든 130장의 7종 마스크·known 플래그를 원본 변환 규칙으로 재계산해 저장된 정답과 일치함을 확인했다. {audit['original_inputs_checked']}개 입력의 SHA·크기·mtime를 전후 대조했다. 확인된 사진/픽셀 presence 충돌은 {sum(audit['photo_pixel_presence_conflicts'].values())}건이다.", "",
        "CODEBRIM의 박락 미탐12장은 사진 단위 양성 정답만 있고 위치 주석이 없다. 저장 마스크0·pixel known0은 ‘박락 면적0’이 아니다. 해당 면적을 미확인으로 남겼다. 박락 음성의 pixel known1은 원래 사진 태그에서 유도한 음성 지도이며 전문가의 정상·안전 판정이 아니다.", "",
        "## DACL의 작은 영역과 80격자", "",
        "| 이 선별 표본의 박락 양성 | 사례 수 | 640 주석 면적 1% 미만 | 주석 면적 중앙값 | 80 any 점유율 / 640 면적 중앙값 |",
        "|---|---:|---:|---:|---:|",
        f"| FN | {fn['selected_cases']} | {fn['positive_geometry_small_below_one_percent']} | {100*fn['fine_spalling_area_fraction_positive_only']['median']:.3f}% | {fn['coarse_expansion_ratio_positive_only']['median']:.3f}배 |",
        f"| TP 비교 | {tp['selected_cases']} | {tp['positive_geometry_small_below_one_percent']} | {100*tp['fine_spalling_area_fraction_positive_only']['median']:.3f}% | {tp['coarse_expansion_ratio_positive_only']['median']:.3f}배 |", "",
        "면적은 실제 물리적 크기가 아니라 기존 processed 부모 크기에 주석을 그린 뒤 BOX640으로 만든 raster의 점유율이다. 80지도는 8×8 블록 안에 표시된 픽셀이 하나라도 있으면 양성이다. 점유율 확대는 이 변환의 특성이며 위치 정답 오류나 오답 원인의 입증이 아니다. 작은 영역은 TP에도 있어 이 특징만으로 FN을 설명할 수 없다.", "",
        "DACL 오탐17장 중9장, 미탐12장 중7장, TP15장 중7장에는 아래 관련 태그 중 하나 이상이 함께 있다. 원본 polygon의 실제 raster 겹침도 계산했지만, 태그 공존이나 겹침은 모델 혼동·오라벨 원인의 증명이 아니다.", "",
        "## 전체 DACL TRAIN 6,225장의 저장 점수", "",
        "관련4태그는 출판자의 Rockpocket/WConccor/Hollowareas/Cavity다. Spalling 태그를 대신하거나 원래7종 정답으로 새로 합치지 않았다.", "",
        "| 서로 겹치지 않는 원본 태그 집단 | 사진 | 해당 TRAIN 오류 | 저장 점수의 TRAIN 비율 |",
        "|---|---:|---:|---:|", *rows, "",
        "이 비율은 모델이 학습한 사진을 이미 저장된 고정 점수·임계값으로 분류한 기술 집계다. 검증 성능 향상·현장 정확도·신뢰구간 또는 인과효과가 아니다. 관련4태그별 개별 집계는 겹치므로 서로 더하지 않는다.", "",
        "박락 음성·관련4태그 집단666장 중 Crack/ACrack 태그가 없는421장에는 FP144건(34.20%), 있는245장에는 FP32건(13.06%)이 있었다. 이 하위집단을 정상 시설로 바꾸거나 새 정답으로 간주하지 않는다.", "",
        "## 다음 실험 후보의 모의 표본 추출", "",
        "후보 조건을 하나로 좁혔다. 원래 박락 음성이면서 관련4태그가 있는 기존 DACL 전체 사진666장만 같은 출처·full/crop·균열/박락 조합 안에서 1.5배 가중 후보로 삼는다. 개별 오답 점수로 새 라벨을 고르지 않는다.", "",
        "전역1.5배 sampler가 아니라 층별 조건부 비중 보강이다. seed56의 기존 표본 순서를 대조군으로 재현하고 별도 seed59로 일부 음성 표본만 같은 층의 후보로 대체한다. 대조군의 원래 후보 표본은 그대로 둔다. 다른 출처·crop·박락 양성 표본은 바꾸지 않는다.", "",
        "6회×14,248개의 실제 모의 추출에서 기존 대조 기록의 표본 순서 SHA와 출처·full/crop·균열/박락 조합을 대조했다. 실제 검증 결과와 노출량은 [모의 실행 집계](facility-spalling-sampler-dry-run.json)를 참고한다. 모의 실행은 GPU 학습 epoch에 포함하지 않는다.", "",
        f"총85,488위치 중 {plan['total_changed_positions']:,}곳만 같은 층의 음성 후보로 바뀌었다. 후보 추출은 {plan['total_eligible_draws_control']:,}→{plan['total_eligible_draws_treatment']:,}회다. 이는 층별 정규화 뒤 관측한 노출량이며 ‘실제 추출이 정확히1.5배’라는 주장이 아니다.", "",
        "공개 전 독립 코드 검토에서 극단적인 확률에서의 뺄셈 오차를 발견해 동치인 안정식으로 보완했다. 초기 모의 실행 증거를 로컬에 보존하고 추가 경계 테스트를 포함한6개 테스트를 다시 실행했다. 수정 전후 양군 전체 표본 배열이 정확히 같음을 확인했으며 원래 학습 소스·정답·모델은 바꾸지 않았다. 자세한 경위는 모의 집계의 `numerical_repair_note`에 기록했다.", "",
        "후속 학습 후보는 같은0773 초기 모델·기존 구조·640입력·기존7종/19종·정답·손실 가중치를 사용하는 대조6회/보강6회의 한 쌍이다. 새 학습은 아직 시작하지 않았다. 나머지5종·보조 태그 노출은 달라질 수 있으므로 알려진 항목 AP 회귀와 균열/박락 미탐·오탐, 작은 결함 누락을 모두 평가해야 한다.", "",
        "연구 후보 기준은 기존 실험의 최대오류0.5pp 개선·항목오류 악화2pp 이하·기타 AP 하락0.02 이하·작은 FN 최소2건 감소 조건을 유지한다. 엄격한5% 미만 목표 및 앱 적용은 별도다. 새 실험의 학습 전 소스/조건 선언을 마친 뒤 고정 예산으로 실행한다.", "",
        "## 화면과 재현", "",
        "현재 PC의 로컬 화면: `runs/facility-spalling-train-audit/SPALLING-AUDIT.html`. 기본 FP/FN54장, 출처·오류 필터, TP/TN 비교와 기존 검수 의견 화면 링크를 제공한다. 계측과 모델 점수는 판정·전문가 의견과 별도다.", "",
        "원본 사진·개별 case·마스크·상세 기록·표본 index NPZ·브라우저 확인 이미지는 ignored runs에 둔다. GitHub clone만으로 이 화면의 자료가 생기지는 않는다. 원본 이용 조건과 로컬 자료·모델·기존 검수 묶음을 함께 준비해야 한다.", "",
        "```powershell", "./.venv/Scripts/python.exe scripts/test_facility_spalling_audit.py",
        "./.venv/Scripts/python.exe scripts/audit_facility_spalling_train.py",
        "./.venv/Scripts/python.exe scripts/audit_facility_spalling_tag_scores.py",
        "./.venv/Scripts/python.exe scripts/prepare_facility_spalling_sampler.py", "```", "",
        "기존 결과가 있으면 덮어쓰지 않고 종료한다. 상세 실행 조건·추가 테스트 기록은 각 코드와 아래 집계에 남겼다. 이전 source30·학습 전 테스트·프로토콜·가중치·앱 프로필은 보존했다.", "",
        "- [130장 계측 집계](facility-spalling-train-audit.json) · [관련14개 테스트](facility-spalling-train-audit-tests.json)",
        "- [전체 TRAIN 태그 집계](facility-spalling-tag-score-audit.json) · [관련6개 테스트](facility-spalling-tag-score-tests.json)",
        "- [표본 추출 모의 집계](facility-spalling-sampler-dry-run.json) · [표본 추출 테스트](facility-spalling-sampler-tests.json)",
        "- [실제 브라우저 확인](facility-spalling-audit-ui-verification.json)", "",
        "기존 최대 source VAL 미탐·오탐22.28%, 5% 미만 목표 미달, 공장 현장 정확도 미측정이라는 상태를 유지한다. 이번 감사는 성능 개선 실측이 아니다.", ""]
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    sidecar = {"schema": "facility_spalling_audit_report_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
        "input_sha256": {p: sha(ROOT / p) for p in ("reports/facility-spalling-train-audit.json",
            "reports/facility-spalling-tag-score-audit.json", "reports/facility-spalling-sampler-dry-run.json")},
        "report_sha256": sha(target), "new_training_epochs": 0, "accuracy_measured": False}
    with (ROOT / "reports/facility-spalling-audit-report-provenance.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(sidecar, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"status": "reported", "selected_case_errors": 54, "full_dacl_train": 6225, "new_training_epochs": 0}))


if __name__ == "__main__":
    main()
