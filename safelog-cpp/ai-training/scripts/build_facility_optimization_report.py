from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from build_facility_report import LABELS_KO, percentage

ROOT = Path(__file__).resolve().parents[1]


def main():
    validation = json.loads((ROOT / "reports/facility-optimization-validation.json").read_text(encoding="utf-8"))
    test = json.loads((ROOT / "reports/facility-optimization-test.json").read_text(encoding="utf-8"))
    lines = ["# SafeLog 시설 성능 보강 — 1차 실험", "",
        "## 이번 보강의 결론", "",
        "추가 학습과 1280px 입력은 검증 영역 AP를 높이지 못해 기존 시설 가중치·960px 입력을 유지했다.",
        "항목별 탐지 기준 조정과 사진 전체 분류 모델을 결합했다. 결함 위치를 찾는 능력과 사진의 항목 존재 여부를 구분해 평가한다.",
        "시험 사진에서는 파임·백화·박리 등 사진 미탐이 줄었지만 오탐은 늘었다. 물기는 개선되지 않았다. 구체적인 비율은 아래 최종 표에 모두 공개한다.",
        "현장용 성능 검증을 마친 모델은 아니다. 영역 미탐은 여전히 높고 실제 대상 시설 사진 검증이 필요하다.", "",
        "## 실험 방법", "",
        "기존 시설 best.pt를 출발점으로 낮은 학습률(AdamW 0.0001), mosaic 제거, 약한 확대·색상 증강으로 추가 학습했다.",
        "가중치 2개(기존/추가 학습) × 입력 960/1280px를 검증 분할에서 비교했다. 검증 영역 mAP50-95가 가장 높은 조합을 선택했다.",
        "항목별 신뢰도 기준은 검증 사진 F2를 최대화하되 사진 오탐률이 기존보다 2%p 이상, 사진·영역 정밀도가 3%p 이상 나빠지지 않는 후보에서 골랐다.",
        "조건을 만족하는 후보가 없으면 선택된 모델의 0.25를 유지한다. 이때 기존 모델 대비 제약 만족을 보장하지 않으며 아래 실측값을 읽어야 한다.",
        "정상 사진이 없는 부식 검증셋에서는 임계값을 조정하지 않았다. 신뢰도 점수는 정답일 확률로 보정한 값이 아니다.", "",
        "## 검증 비교 및 추가 학습", "",
        "| 대상 | 후보 가중치 | 입력 | 검증 mAP50 | 검증 mAP50-95 | 선택 |",
        "|---|---|---:|---:|---:|---|"]
    for exp in validation["experiments"]:
        for candidate in exp["candidates"]:
            selected = candidate == exp["selected"]
            lines.append(f"| {exp['model']} | {candidate['run']} | {candidate['imgsz']} | {percentage(candidate['map50'])} | {percentage(candidate['map50_95'])} | {'사용' if selected else ''} |")
    lines += ["", "추가 학습의 epoch 수는 초기 학습의 epoch 수와 별개다.", ""]
    for baseline in ("facility-dacl", "facility-corrosion"):
        with (ROOT / f"runs/{baseline}-refined/results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        lines.append(f"- {baseline}-refined: 실제 {len(rows)} epoch, 이 실험의 best epoch {best['epoch']}.")
    lines += ["", "## 고정 설정으로 평가한 시험 결과", "",
              "가중치·해상도·임계값을 먼저 확정하고 시험 분할을 평가했다. 기존 라운드에서 이미 확인한 동일 시험 분할의 재평가이므로 새 외부 검증은 아니다.",
              "아래 AP는 의미 분할을 박스로 변환한 영역 검출 지표다. 임계값 변경은 AP 상승으로 표현하지 않는다.", "",
              "| 대상 | 기존 시험 mAP50 | 보강 시험 mAP50 | 기존 시험 mAP50-95 | 보강 시험 mAP50-95 |",
              "|---|---:|---:|---:|---:|"]
    for exp in test["experiments"]:
        old = json.loads((ROOT / f"runs/evaluation-{exp['model'].removesuffix('-optimized')}.json").read_text())
        new = json.loads((ROOT / f"runs/evaluation-{exp['model']}.json").read_text())
        lines.append(f"| {exp['model']} | {percentage(old['map50'])} | {percentage(new['map50'])} | {percentage(old['map50_95'])} | {percentage(new['map50_95'])} |")
    lines += ["", "### 사진별 해당 항목의 존재 여부", "",
              "사진에 해당 결함 종류가 있는지 집계한다. 박스 위치는 확인하지 않으므로 영역 결과와 함께 읽는다.",
              "사진 미탐률=FN/(TP+FN), 사진 오탐률=FP/(FP+TN). 전체 현장 안전 판정 정확도가 아니다.", "",
              "| 항목 | 적용 기준 | 기존 사진 미탐률 | 보강 사진 미탐률 | 기존 사진 오탐률 | 보강 사진 오탐률 | 양성/음성 사진 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for exp in test["experiments"]:
        for label, new in exp["optimized"].items():
            old = exp["baseline"][label]["photo"]
            p = new["photo"]
            lines.append(f"| {LABELS_KO[label]} | {new['threshold']:.2f} | {percentage(old['miss_fraction'])} | {percentage(p['miss_fraction'])} | {percentage(old['false_positive_rate'])} | {percentage(p['false_positive_rate'])} | {p['tp']+p['fn']}/{p['fp']+p['tn']} |")
    lines += ["", "### 영역 위치를 포함한 결과", "",
              "같은 클래스 박스 IoU≥0.5로 일대일 매칭했다. 오탐 비중은 FP/(TP+FP)이며 사진 오탐률과 다르다.", "",
              "| 항목 | 기존 영역 미탐 | 보강 영역 미탐 | 기존 영역 오탐 비중 | 보강 영역 오탐 비중 | 보강 TP/FP/FN |",
              "|---|---:|---:|---:|---:|---:|"]
    for exp in test["experiments"]:
        for label, new in exp["optimized"].items():
            old = exp["baseline"][label]["region"]
            r = new["region"]
            lines.append(f"| {LABELS_KO[label]} | {percentage(old['miss_fraction'])} | {percentage(r['miss_fraction'])} | {percentage(old['false_discovery_fraction'])} | {percentage(r['false_discovery_fraction'])} | {r['tp']}/{r['fp']}/{r['fn']} |")
    presence_test_path = ROOT / "reports/facility-presence-test.json"
    if presence_test_path.exists():
        presence = json.loads(presence_test_path.read_text(encoding="utf-8"))
        training = json.loads((ROOT / "runs/facility-presence/TRAINING.json").read_text())
        lines += ["", "## 사진 전체 분류로 보강한 최종 결과", "",
                  "EfficientNet-B0의 ImageNet 사전 학습 가중치에서 출발해 동일 train 6,225장으로 시설 항목 7개를 동시에 분류하도록 학습했다.",
                  "사진 전체를 384×384로 변환하며 결함이 잘리는 crop은 사용하지 않는다. 박스 검출에 없는 항목만 추가 의견으로 제안한다.",
                  f"추가 학습 {training['actual_epochs']} epoch, 검증 사진 분류 macro average precision {percentage(training['best_val_macro_average_precision'])}. 이 수치는 박스 mAP50이나 안전 판정 정확도가 아니다.",
                  "분류 기준도 검증에서만 선택했다. 위 표의 보강 박스 검출 대비 항목별 사진 오탐률 +2%p, 정밀도 -3%p를 제한으로 두고 결합 F2를 최대화했다. 1.00은 해당 분류 항목을 비활성화한다.",
                  "검출 기준 조정과 분류 결합은 각각 검증 제약을 사용한다. 최초 모델 대비 최종 오탐 증가가 2%p 이내라는 의미는 아니며 시험 결과의 실제 변화는 아래 표를 확인한다.",
                  "한 사진에 여러 항목이 있을 수 있으며 아래 양성/음성은 항목별 값이다. 원래 시험 975장을 재사용한 비교로, 새 현장 검증은 아니다.", "",
                  "| 항목 | 분류 기준 | 최초 모델 사진 미탐률 | 최종 사진 미탐률 | 최초 사진 오탐률 | 최종 사진 오탐률 | 분류가 추가로 찾은 양성/음성 사진 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        dacl_test = next(exp for exp in test["experiments"] if exp["model"] == "facility-dacl-optimized")
        for label, item in presence["per_class"].items():
            old, new = dacl_test["baseline"][label]["photo"], item["combined"]
            lines.append(f"| {LABELS_KO[label]} | {item['threshold']:.2f} | {percentage(old['miss_fraction'])} | {percentage(new['miss_fraction'])} | {percentage(old['false_positive_rate'])} | {percentage(new['false_positive_rate'])} | {item['additional_positive_photos']}/{item['additional_negative_photos']} |")
        lines += ["", "추가 의견은 `detections[].box=null`, `evidence_scope=photo_presence`로 반환한다. 해당 의견에 사각형을 그리면 안 된다.",
                  "분류 보강은 영역 검출 성능을 높인 결과가 아니다. 위 영역 미탐·오탐은 그대로이며 사람의 위치 확인이 필요하다.",
                  "[사전 학습 모델 출처: PyTorch 공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)", ""]
    lines += ["", "## 적용 및 남은 한계", "",
              "- 앱 요청·응답 형식은 유지한다. ./start_ai_server.ps1 -FacilitiesOnly가 검증 설정을 읽어 optimized 모델을 사용한다.",
              "- 가중치 SHA256과 설정이 다르면 분석 API가 503을 반환해 잘못된 조합을 막는다.",
              "- 기본 시설 모델로 비교하려면 -BaselineFacilities를 추가한다. PPE·화재 모델은 이전 설정을 사용한다.",
              "- 입력 해상도가 높아지면 처리 시간이 증가할 수 있다. 실제 스마트폰/네트워크 속도는 별도 측정이 필요하다.",
              "- 데이터가 교량 표면 중심이고 장면 ID가 없어 장면 단위 독립성을 보장하지 못한다. 시설 전체 종류에 대한 일반화를 확인한 결과가 아니다.",
              "- 부식 시험 사진은 15장이고 모두 양성이다. 정상 금속 사진에 대한 오탐률은 정의할 수 없다.",
              "- 영역 박스는 원본 의미 분할의 겹친 폴리곤/마스크에서 파생했다. 구조 강도·균열 깊이·배관 누수는 판정하지 않는다.",
              "- 검증에서 둔 오탐 제한은 시험/실제 현장에서 보장되는 제한이 아니다. 실제 결과는 위 표를 확인한다.",
              "- AI는 점검 제안이며 사람의 최종 확인이 필요하다. 사라진 탐지만으로 조치 완료를 확정하지 않는다.", "",
              "## 재현 순서", "", "```powershell",
              "# 이미 원본 데이터와 기준 모델이 준비된 ai-training 폴더에서 실행",
              "./.venv/Scripts/python.exe scripts/refine_facilities.py",
              "./.venv/Scripts/python.exe scripts/finish_facility_optimization.py",
              "# 기존 추가 학습이 완료되어 검증만 다시 수행할 때는 --evaluate-only 사용",
              "```", "", "영역 모델 선택의 원자료는 facility-optimization-validation.json, 시험 집계는 facility-optimization-test.json이다.",
              "사진 전체 분류의 검증 기준과 시험 결과는 facility-presence-validation.json / facility-presence-test.json이다.", ""]
    (ROOT / "reports/FACILITY_OPTIMIZATION_KO.md").write_text("\n".join(lines), encoding="utf-8")
    if presence_test_path.exists():
        for source, target in (("TRAINING.json", "facility-presence-training.json"), ("history.json", "facility-presence-training-history.json")):
            shutil.copyfile(ROOT / "runs/facility-presence" / source, ROOT / "reports" / target)
        from plot_facility_improvement import main as plot_improvement
        plot_improvement()


if __name__ == "__main__":
    main()
