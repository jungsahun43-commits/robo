"""Write the real-data experiment record from completed training and frozen tests."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_facilities import save


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def percent(value): return "미정" if value is None else f"{value * 100:.1f}%"


def table(report):
    rows = ["| 항목 | 기존 미탐률 | 후보 미탐률 | 기존 오탐률 | 후보 오탐률 |", "|---|---:|---:|---:|---:|"]
    labels = {"concrete_crack": "균열", "concrete_spalling": "박락", "rust_stain": "녹 얼룩",
              "exposed_rebar": "철근 노출", "wet_surface": "물기", "efflorescence": "백화", "surface_cavity": "공동"}
    for label, item in report["per_class"].items():
        old, new = item["round1"], item["round4"]
        rows.append(f"| {labels[label]} | {percent(old['miss_fraction'])} | {percent(new['miss_fraction'])} | {percent(old['false_positive_rate'])} | {percent(new['false_positive_rate'])} |")
    return "\n".join(rows)


def main():
    reports = ROOT / "reports"
    run = ROOT / "runs/facility-presence-damsegment"
    training, history = read(run / "TRAINING.json"), read(run / "history.json")
    audit = read(reports / "facility-round4-data-audit.json")
    validation = read(reports / "facility-round4-validation.json")
    extra = read(reports / "facility-round4-external-test.json")
    decision = read(reports / "facility-round4-release.json")
    verification = read(reports / "facility-round4-verification.json")
    save(reports / "facility-round4-training.json", training)
    save(reports / "facility-round4-training-history.json", history)
    best = max(history, key=lambda row: row["val_macro_average_precision"])
    old_training = read(reports / "facility-presence-training.json")
    adopted = decision["status"] == "adopted_round4"
    selection = "、".join(label for label, item in validation["per_class"].items() if item["eligible"]) or "없음"
    test_path = reports / "facility-round4-test.json"
    dacl_table = table(read(test_path)) if test_path.exists() else "검증에서 새 항목이 선택되지 않아 DACL 시험으로 새 후보를 고르지 않았다. 기존 1차 모델을 유지한다."
    status = "4차 후보를 채택했다." if adopted else "4차 후보는 채택하지 않았다. 기본 모델과 기존 전달 ZIP의 가중치는 유지한다."
    rows = ["| 항목 | 기존 분류기 AP | 새 분류기 AP |", "|---|---:|---:|"]
    for label, entry in extra["per_class"].items():
        ap = entry["classifier_ranking_ap"]
        rows.append(f"| {label} | {percent(ap['facility-presence'])} | {percent(ap['facility-presence-damsegment'])} |")
    diagnostics = ["| 항목 | 검증에서 고정한 임계값 | 새 모델 결합 미탐률 | 새 모델 결합 오탐률 |", "|---|---:|---:|---:|"]
    for label, entry in extra["per_class"].items():
        d = entry.get("unreleased_new_model_diagnostic")
        if d:
            diagnostics.append(f"| {label} | {d['threshold_from_dacl_validation']} | {percent(d['metrics']['miss_fraction'])} | {percent(d['metrics']['false_positive_rate'])} |")
    val_rows = ["| 항목 | 기존 검증 미탐률 | 오탐 상한 내 새 모델 미탐률 | 채택 가능 |", "|---|---:|---:|---|"]
    for label, entry in validation["per_class"].items():
        d = entry["candidate_under_fpr_cap"]
        rate = percent(d["metrics"]["miss_fraction"]) if d else "해당 없음"
        val_rows.append(f"| {label} | {percent(entry['round1']['miss_fraction'])} | {rate} | {'예' if entry['eligible'] else '아니오'} |")
    text = f"""# 4차 실제 댐 표면 보강 결과

## 결론

{status}
현재 기본 프로필: **{decision['active_version']}**. 검증에서 새 모델 적용 후보로 고른 항목: {selection}.
새 모델의 학습 완료, 모델 교체 기준 통과, 현장 성능 확인은 서로 다른 상태이다.

## 실제로 다운로드·사용한 자료

[DamSegment V1](https://data.mendeley.com/datasets/z5z6gtt5t4/1), DOI 10.17632/z5z6gtt5t4.1,
Gharehbaghi, Bennett, Lequesne, Zhao, Li (2025). 데이터는 CC BY 4.0이며 출처와 변경 사항을 표시한다.
[저자 논문](https://doi.org/10.1016/j.dib.2026.112671)의 스마트폰 촬영 댐 표면에서 생성한 패치이다.
원래의 비공개 고해상도 현장 사진을 다운로드한 것으로 주장하지 않는다.

- 공식 세 ZIP 총 {sum(f['bytes'] for f in audit['verified_files']):,}바이트. 각 파일의 저자 SHA256을 검증했다.
- RGB 파일 5,000개 중 동일한 탐지/분할 패치 1,500개는 한 번만 계산했다.
- 분류 Crack 폴더 1,000개는 균열/박락을 섞은 양성이라 균열 단독 정답으로 쓰지 않았다.
- 두 항목 정답을 확인한 고유 패치 {audit['unique_labeled_patches']:,}개: 학습 {audit['split_counts']['train']:,}, 추가 시험 {audit['split_counts']['test']:,}.
- 학습 정답: 균열 양성/음성 {audit['per_split_labels']['train']['concrete_crack']['positive']}/{audit['per_split_labels']['train']['concrete_crack']['negative']},
  박락 양성/음성 {audit['per_split_labels']['train']['concrete_spalling']['positive']}/{audit['per_split_labels']['train']['concrete_spalling']['negative']}.
- Non-Crack은 균열·박락의 음성으로만 사용했다. 물기·백화·공동·철근·녹 얼룩은 -1(미확인)이며 학습 손실에서 제외했다.
- 실제 RGB 마스크는 빨강=균열0, 파랑=박락1이다. 논문 예시 팔레트와 반대이며 실제 JSON/YOLO/마스크 1,500세트를 확인했다.
- 학습 전 JPG로 재인코딩(quality95)했다. 원본 픽셀 중복 및 기존 DACL 7,910장과의 정확한 픽셀 겹침을 검사했다.
- 원본 사진·패치는 Git/모델 ZIP에 포함하지 않았다. 약 7.9GB CODEBRIM 자료는 다운로드를 중단한 부분 파일만 있어 이번 학습에 사용하지 않았다.

같은 패치 ID와 dHash 거리≤6의 연결 그룹을 통째로 분리했다. 학습/시험 그룹·픽셀 겹침은 0이다.
**원본 촬영 ID가 공개되지 않아 인접 패치나 같은 촬영 장면의 모든 누수를 증명하지는 못한다.**
이 시험은 단일 댐의 보류 패치 평가이며 다른 시설의 현장 검증이 아니다. 비·건조 조건 자료가 있어도 개별 물기 정답은 제공되지 않아 물기 학습 자료로 세지 않았다.

## 학습

기존 DACL 학습 {training['train_images']:,}장 + 실제 추가 패치 {training['supplemental_images']:,}개를 사용했다.
EfficientNet-B0, 1차 가중치에서 시작, 512px, batch12, 추가 자료 손실0.3, seed42,
최대16회 중 {training['actual_epochs']}회 수행. 기존 검증 710장의 macro AP로 epoch {best['epoch']}을 골랐다.
검증 macro AP: 기존 {old_training['best_val_macro_average_precision'] * 100:.2f} → {training['best_val_macro_average_precision'] * 100:.2f}.
AP는 점수 순위 지표이며 정확도/현장 오차율이 아니다. 학습 자료와 입력 해상도가 함께 바뀌었으므로 데이터 추가만의 인과 효과로 해석하지 않는다.

## 고정 설정의 시험 결과

검증에서 두 항목별 오탐 증가 없이 미탐을 2%p 이상 줄인 경우에만 새 모델을 선택했다.
임계값을 고정한 뒤 시험했으며 시험 결과를 보고 재조정하지 않았다.
교체는 DACL 항목별 FP/FN 비증가 및 하나 이상 감소, 추가 패치 시험의 항목별 FP/FN 비증가를 모두 요구했다.

{'\n'.join(val_rows)}

균열 검증의 미탐 감소는 약 0.5%p(1개 사진)로 사전에 정한 2%p에 미달했다.
박락은 오탐 증가 없이 미탐을 줄이지 못했다. 추가 댐 시험에서 균열 개선이 커도 이 기준을 사후 변경하지 않았다.

### 기존 DACL 시험 975장

{dacl_table}

반복 확인한 기존 시험 분할로, 독립적인 현장 성능을 의미하지 않는다.

### 추가 보류 댐 패치 {extra['images']}개: 실제 적용 후보와 기존 모델 비교

{table(extra)}

양성/음성 사진을 분모로 한 존재 여부의 미탐/오탐 비율이다. 픽셀 분할 또는 박스 위치 정확도가 아니다.
학습하지 않은 다섯 항목은 평가하지 않았다. 선택되지 않은 새 분류기 항목은 후보에서도 기존 분류기로 유지한다.

### 분류기 자체의 순위 AP: 별도 진단

{'\n'.join(rows)}

{'\n'.join(diagnostics)}

위 추가 진단은 실제 배포하지 않은 새 분류기와 기존 검출기의 조합이다. 검증에서 후보의 오탐 상한을 만족한
진단 임계값을 시험 전에 고정했다. 채택 기준을 통과하지 않은 항목을 앱에 적용한 것으로 해석하지 않는다.

이 표는 새 분류기의 진단 결과이며 앱 적용 후보 전체의 오탐/미탐 표와 구분한다.
추가 시험은 체크포인트나 임계값 선택에 사용하지 않았다.

## 실제 파일 검증

CPU PyTorch/ONNX 최대 확률 차이 {verification['classifier_cpu_onnx_parity']['max_probability_difference']:.8f},
동적 batch 출력 {verification['classifier_cpu_onnx_parity']['dynamic_batch_shape']} 확인.
best.pt SHA256: `{verification['model_sha256']}`
best.onnx SHA256: `{verification['onnx_sha256']}`
학습으로 변경된 가중치 텐서 {verification['changed_weight_tensors']}개를 확인했다.
기존 앱의 세 API, C++ 인터페이스와 위치 없는 사진 의견의 box=null 형식을 유지한다.
이번 작업의 CPU 모델 실행 검증은 Android 실기기 동작 시험을 의미하지 않는다.

## 재현 순서

ai-training에서 설치·가상환경 활성화 후 실행한다. 기존 DACL 자료와 1차 가중치가 필요하다.
이미 학습된 같은 이름의 실험을 덮어쓰지 않는다.

```powershell
python scripts/download_damsegment.py
python -m scripts.prepare_damsegment
python scripts/train_facility_presence.py --name facility-presence-damsegment --initial runs/facility-presence/best.pt --extra-manifest data/damsegment-training/train.json --extra-weight 0.3 --imgsz 512 --batch 12 --epochs 16 --patience 5 --device cuda
python scripts/refine_damsegment.py select
python scripts/refine_damsegment.py test
python scripts/refine_damsegment.py external
python scripts/refine_damsegment.py export
python scripts/verify_damsegment.py
python scripts/refine_damsegment.py release
python scripts/report_damsegment.py
```

공식 다운로드·라벨 감사는 datasets/damsegment_source.json과 facility-round4-data-audit.json,
선택 기준은 FACILITY_ROUND4_PLAN_KO.md, 평가·판단은 facility-round4-*.json에 기록했다.
기존 DACL 가중치의 CC BY-NC 4.0 조건은 이 파생 모델에도 유지한다.
"""
    (reports / "FACILITY_ROUND4_KO.md").write_text(text, encoding="utf-8")
    print(f"Report ready: {reports / 'FACILITY_ROUND4_KO.md'}")


if __name__ == "__main__": main()
