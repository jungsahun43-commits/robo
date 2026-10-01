from __future__ import annotations

import csv
from collections import Counter
import json
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (("facility-dacl", "dacl10k"), ("facility-corrosion", "ostrava-corrosion"))
LABELS_KO = {
    "concrete_crack": "표면 균열", "concrete_spalling": "콘크리트 박리", "rust_stain": "녹 얼룩",
    "exposed_rebar": "철근 노출", "wet_surface": "젖은 표면", "efflorescence": "백화",
    "surface_cavity": "표면 공동·파임", "metal_corrosion": "금속 부식",
}


def percentage(value):
    return "정의 불가" if value is None else f"{value*100:.1f}%"


def class_presence(fixed: dict) -> dict:
    """Class presence per photo; deliberately does NOT check box localization."""
    result = {}
    for label in fixed["per_class"]:
        counts = Counter()
        for image in fixed["per_image"]:
            regions = image["counts"][label]
            target = regions["tp"] + regions["fn"] > 0
            prediction = regions["tp"] + regions["fp"] > 0
            key = "tp" if target and prediction else "fn" if target else "fp" if prediction else "tn"
            counts[key] += 1
        tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
        result[label] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "positive_images": tp + fn, "negative_images": fp + tn,
                         "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                         "miss_fraction": fn / (tp + fn) if tp + fn else None,
                         "definition": "class presence only; a wrongly localized box can count as a positive photo"}
    return result


def main() -> int:
    experiments = []
    sources = json.loads((ROOT / "datasets/facility_sources.json").read_text(encoding="utf-8"))
    lines = ["# SafeLog 시설 결함 모델 학습 결과", "",
             "사진에서 보이는 시설 결함 영역을 찾는 YOLO11s 모델 2개를 추가했다. 기존 PPE·화재 모델은 유지했다.",
             "두 시설 데이터셋의 전체 사진 합계는 8,015장이다. train/val/test는 아래에 구분한다.", "",
             "## 학습 및 평가", "",
             "최대 변 1,280px인 학습 복사본에서 입력 960px, 배치 8, seed 42로 학습했다.",
             "best.pt 선택과 조기 종료는 검증 분할에서만 수행했다. 시험 사진은 가중치 갱신이나 후보 선택에 사용하지 않았다.",
             "아래 AP는 원본의 의미 분할 마스크·폴리곤을 영역 박스로 변환한 자체 평가다. 원본 논문의 mIoU와 비교할 수 없다.", "",
             "| 모델 | train | val | test | 실제 epoch | 시험 Precision | 시험 Recall | 시험 mAP50 | 시험 mAP50-95 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run, dataset in EXPERIMENTS:
        preparation = json.loads((ROOT / f"data/{dataset}-yolo/PREPARATION.json").read_text(encoding="utf-8"))
        metrics = json.loads((ROOT / f"runs/evaluation-{run}.json").read_text(encoding="utf-8"))
        fixed = json.loads((ROOT / f"runs/fixed-{run}-test.json").read_text(encoding="utf-8"))
        with (ROOT / f"runs/{run}/results.csv").open(encoding="utf-8") as handle:
            epochs = list(csv.DictReader(handle))
        args = yaml.safe_load((ROOT / f"runs/{run}/args.yaml").read_text(encoding="utf-8"))
        best = max(epochs, key=lambda row: float(row["metrics/mAP50-95(B)"]))
        experiment = {"name": run, "source": sources[dataset], "preparation": preparation,
                      "training": {"base_model": args["model"], "epochs_completed": len(epochs),
                                   "requested_epochs": args["epochs"], "best_validation_epoch": int(best["epoch"]),
                                   "best_validation_map50_95": float(best["metrics/mAP50-95(B)"]),
                                   "imgsz": args["imgsz"], "batch": args["batch"], "seed": args["seed"]},
                      "test": {k: v for k, v in metrics.items() if k not in {"weights", "data"}},
                      "fixed_threshold_test": {k: v for k, v in fixed.items() if k != "per_image"},
                      "image_class_presence_test": class_presence(fixed)}
        experiments.append(experiment)
        counts = preparation["split_counts"]
        lines.append(f"| {run} | {counts['train']:,} | {counts['val']:,} | {counts['test']:,} | {len(epochs)} | "
                     + " | ".join(percentage(metrics[key]) for key in ("precision", "recall", "map50", "map50_95")) + " |")
    lines += ["", "## 앱과 같은 임계값에서의 오탐·미탐", "",
              "신뢰도 0.25, 입력 960px, 최대 검출 300개, 같은 클래스의 영역 IoU≥0.5, 일대일 매칭을 사용했다.",
              "`오탐 비중=FP/(TP+FP)`은 예측한 영역 중 정답과 매칭되지 않은 비율이다. TN 기반의 FPR은 아니다.",
              "`미탐 비중=FN/(TP+FN)`은 정답 영역을 놓친 비율이다. 사진 전체의 안전/불안전 판정 정확도로 해석하지 않는다.",
              "위 표의 Precision/Recall은 AP 계산 과정의 operating point라 아래 고정 임계값 결과와 다를 수 있다.", "",
              "| 항목 | 시험 정답 영역 | TP | FP | FN | 오탐 비중 | 미탐 비중 | AP50 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for experiment in experiments:
        for label, item in experiment["fixed_threshold_test"]["per_class"].items():
            ap = experiment["test"]["per_class"].get(label, {}).get("map50")
            lines.append(f"| {LABELS_KO[label]} | {item['ground_truth_regions']} | {item['tp']} | {item['fp']} | {item['fn']} | "
                         f"{percentage(item['false_discovery_fraction'])} | {percentage(item['miss_fraction'])} | {percentage(ap)} |")
    lines += ["", "## 성능 해석", "",
              "현재 결과는 시설 AI의 첫 기준 모델이며 현장 점검을 대신할 성능으로 판단하지 않는다. mAP50은 정확도 백분율이 아니다.",
              "신뢰도 0.25에서 특히 젖은 표면·표면 공동의 영역 미탐이 높다. 자동 완료 판정에 사용하지 않고 사람이 점검표로 확인해야 한다.",
              "균열·박리·철근 노출도 영역 미탐이 많아 계속 보강해야 한다. 금속 부식은 시험 사진 15장뿐이고 정상 금속 사진이 없어 사진 단위 오탐률을 측정하지 못했다.",
              "다음 보강은 실제 대상 시설의 정상/손상 사진을 함께 확보하고, 작은 손상 확대·타일 분석과 원본 의미 분할 마스크에 맞는 모델을 검토하는 방향이다. 이 개선을 이미 완료한 것으로 기록하지 않는다.", "",
              "## 사진별 해당 항목의 존재 여부", "",
              "같은 시험 결과에서 해당 종류의 정답 영역이 하나라도 있으면 양성 사진, 예측이 하나라도 있으면 검출 사진으로 집계했다.",
              "이 지표는 박스 위치를 확인하지 않으며, 위치가 틀린 박스도 양성으로 집계될 수 있다. 위 영역 지표를 함께 읽어야 한다.",
              "`사진 오탐률=FP/(FP+TN)`, `사진 미탐률=FN/(TP+FN)`이다. 해당 항목의 음성/양성 사진이 없으면 비율을 정의할 수 없다.",
              "각 항목의 존재 여부만 평가했으며 사진 전체의 안전/불안전 판정 지표가 아니다.", "",
              "| 항목 | 해당 항목 양성 사진 | 음성 사진 | FP 사진 | FN 사진 | 사진 오탐률 | 사진 미탐률 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for experiment in experiments:
        for label, item in experiment["image_class_presence_test"].items():
            lines.append(f"| {LABELS_KO[label]} | {item['positive_images']} | {item['negative_images']} | {item['fp']} | {item['fn']} | "
                         f"{percentage(item['false_positive_rate'])} | {percentage(item['miss_fraction'])} |")
    lines += ["", "## 자료와 변환 방식", "",
              "- dacl10k는 교량 콘크리트 중심 데이터다. 원본 train 6,935장을 파일명 SHA256 기준으로 train/val에 나누고, 원본 validation 975장은 시험용으로 보관했다. 공개 정답이 없는 testdev는 학습·평가에서 제외했다.",
              "- dacl10k의 Crack/Alligator Crack(ACrack)은 학습 내부 키 concrete_crack으로 합쳤다. API는 이를 surface_crack(표면 균열 의심)으로 변환하며 콘크리트라는 재질을 단정하지 않는다. 재질 분류기를 학습한 것은 아니다.",
              "- 나머지는 Spalling/Rust/ExposedRebars/Wetspot/Efflorescence/Cavity를 사용했다. 다른 손상과 시설 종류는 학습 범위 밖이다.",
              "- 폴리곤의 최소·최대 좌표를 영역 박스로 변환하고, 좌표를 이미지 경계로 제한했다. 같은 클래스의 동일 박스만 중복 제거했다. 겹친 의미 분할 영역이 실제 개별 손상 개수라는 뜻은 아니다.",
              "- 이미지의 EXIF 회전을 적용하고 주석과 사진 크기를 대조한 다음 좌표를 변환했다. 크기가 뒤집혀 있던 원본 10장의 회전 문제를 확인해 보정했다.",
              "- Ostrava는 산업 시설 금속 부식 105장이다. 공식 train 75 / validation 15 / test 15를 유지했다. 마스크 >0의 8방향 연결 영역을 박스로 변환했고 원본 64px 미만의 작은 마스크 조각은 제외했다.",
              "- 각 자료의 분할 사이에서 파일 SHA256과 디코딩 픽셀 SHA256이 같은 중복 사진은 없음을 확인했다. 교량·촬영 현장 ID를 통한 장면 분리는 확보하지 못했으므로 새로운 현장 성능은 별도로 측정해야 한다.",
              "- tunnel/wall water leakage (Mendeley xz2nykszbs/1)는 조사했지만 원본 사진 다운로드에 Baidu 계정이 필요해 제외했다. 주석 파일만 받은 것을 학습 자료로 계산하지 않았다.", "",
              "## 사용할 수 있는 범위", "",
              "- 균열·박리·녹 얼룩·철근 노출·젖은 표면·백화·표면 공동·금속 부식의 **점검 제안**이다. 검출 신뢰도는 실제 위험 확률이나 구조 안전 확률이 아니다.",
              "- 젖은 표면 검출로 배관 누수를 확정하거나, 부식 사진으로 손실 두께를 계산하거나, 균열 사진으로 붕괴 위험을 확정하지 않는다.",
              "- 위험 등급과 개선 문장은 규칙으로 매핑한다. 위험도 모델이나 보고서 생성 LLM을 새로 학습한 것은 아니다.",
              "- 시설이 발견되지 않아도 안전하다는 판단을 하지 않는다. 조치 후 미탐지는 실제 해결의 증거가 아니므로 점검자의 최종 확인을 유지한다.",
              "- 원본의 객체별 영역이 작거나 세로·가로로 긴 경우 박스 IoU 평가가 까다롭다. 그래도 위 미탐을 숨기거나 임의의 현장 정확도로 바꾸지 않는다.",
              "- 원본은 같은 위치에 여러 손상이 겹치는 의미 분할 자료다. 폴리곤을 개별 검출 박스로 바꾼 학습은 원본 과제를 완전히 재현하지 않으며, 겹친 손상과 얇은 균열에 한계가 있다.",
              "- 산업 시설 부식 시험은 15장뿐이며, 교량 데이터가 실내 시설·공장 전체를 대표하지 않는다. 새로운 현장 사진을 별도 보관하고 추가 검증해야 한다.", "",
              "## 앱 연결", "", "```powershell", "./start_ai_server.ps1 -FacilitiesOnly", "```", "",
              "시설 모델 2개 + 화재·연기 모델로 실행된다. PPE도 포함하려면 `-Facilities`, 모든 보조 PPE까지 포함하려면 `-AllModels`를 쓴다.",
              "`GET /health`에서 모델 목록과 입력 크기를 확인한다. C++ 앱의 기존 세 API 요청·필수 응답 필드는 유지했다.",
              "각 검출은 모델명·클래스·신뢰도·박스를 반환한다. Android APK에서의 촬영/전송은 역할 4와 통합 후 실제 휴대폰으로 확인해야 한다.", "",
              "## 출처 및 이용 조건", "",
              "- [dacl10k toolkit](https://github.com/phiyodr/dacl10k-toolkit) — CC BY-NC 4.0, Flotzinger/Rösch/Braml (2023). 연구·대회용 비상업 조건을 확인하고 상용 도입 전에 권리 검토가 필요하다.",
              "- [Corrosion in Industrial Complexes in Ostrava](https://zenodo.org/records/11235637) — CC BY 4.0, Frič (2024). 출처 표시 조건을 유지한다.", ""]
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    environment = {name: version(name) for name in ("torch", "torchvision", "ultralytics", "opencv-python", "pyyaml")}
    (reports / "facility-training-results.json").write_text(json.dumps({"environment": environment, "experiments": experiments}, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "FACILITY_TRAINING_RESULTS_KO.md").write_text("\n".join(lines), encoding="utf-8")
    for _, dataset in EXPERIMENTS:
        text = (ROOT / f"data/{dataset}-yolo/PREPARATION.json").read_text(encoding="utf-8")
        (reports / f"preparation-{dataset}.json").write_text(text, encoding="utf-8")
    print("Facility training report generated from completed training and held-out evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
