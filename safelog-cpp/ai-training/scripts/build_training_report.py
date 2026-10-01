from __future__ import annotations

import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("ppe-baseline", "Construction-PPE", "construction-ppe", 1416),
    ("fire-smoke", "Indoor Fire Smoke", "indoor-fire-smoke", 5000),
    ("chvg-ppe", "CHVG", "chvg-yolo", 1699),
    ("sh17-ppe", "SH17", "sh17-1280", 8099),
    ("ppe-sh17-transfer", "Construction-PPE (SH17 전이)", "construction-ppe", 1416),
)


def main() -> int:
    reports = []
    selection = json.loads((ROOT / "reports" / "selected-models.json").read_text(encoding="utf-8"))
    for run, title, dataset, count in DATASETS:
        metrics = json.loads((ROOT / "runs" / f"evaluation-{run}.json").read_text(encoding="utf-8"))
        with (ROOT / "runs" / run / "results.csv").open(encoding="utf-8") as stream:
            epochs = list(csv.DictReader(stream))
        args = yaml.safe_load((ROOT / "runs" / run / "args.yaml").read_text(encoding="utf-8"))
        audit = json.loads((ROOT / "reports" / f"dataset-audit-{dataset}.json").read_text(encoding="utf-8"))
        split = metrics.get("split", "test")
        support = {name: audit["splits"][split]["instances"].get(class_id, 0)
                   for class_id, name in audit["classes"].items()}
        reports.append({
            "run": run, "dataset": title, "image_count": count,
            "split_image_counts": {name: details["images"] for name, details in audit["splits"].items()},
            "epochs_completed": len(epochs),
            "base_model": args["model"], "imgsz": args["imgsz"], "batch": args["batch"], "seed": args["seed"],
            "evaluation_split": split, "evaluation_images": audit["splits"][split]["images"],
            "ground_truth_instances": support,
            "metrics": {key: metrics[key] for key in ("precision", "recall", "map50", "map50_95")},
            "per_class": metrics.get("per_class", {}),
        })
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    environment = {name: version(name) for name in ("torch", "torchvision", "ultralytics", "fastapi", "onnx")}
    environment["python"] = platform.python_version()
    environment["gpu"] = "NVIDIA GeForce RTX 4070 SUPER 12GB"
    (output / "training-results.json").write_text(json.dumps({"environment": environment, "selection": selection, "experiments": reports},
                                                           ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# SafeLog 학습 결과", "",
        "네 데이터셋 전체는 16,214장이고 실제 train 분할 합계는 12,492장이다. val/test 사진은 학습 입력에서 제외했다.",
        "데이터셋 간 중복을 제거한 고유 이미지 수는 아니다.",
        "PPE 전이 모델은 SH17 학습 결과를 출발점으로 같은 Construction-PPE 데이터를 다시 학습한 추가 실험이다.", "",
        "| 데이터셋 | 전체 이미지 | 모델 | 실제 epoch | 평가 분할 | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        m = report["metrics"]
        lines.append(f"| {report['dataset']} | {report['image_count']:,} | {report['base_model']} | {report['epochs_completed']} | "
                     f"{report['evaluation_split']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['map50']:.3f} | {m['map50_95']:.3f} |")
    lines += [
        "", "모델마다 데이터셋·클래스·분할이 다르므로 표의 수치만으로 모델의 우열을 비교하지 않는다.",
        "Precision·Recall은 Ultralytics 평가기가 산출한 값이다. 서버의 기본 탐지 임계값 0.25에서 측정한 현장 성능과 동일하지 않다.",
        "SH17은 공식 val 1,620장을 학습 중 모델 선택과 최종 평가에 함께 사용했다. 독립 test 결과와 구분한다.",
        "PPE와 화재는 공식 분할, CHVG는 원본 이름을 묶은 seed 없는 SHA256 결정 분할을 사용했다.",
        "SH17 원본 8,099장은 보관했으며 학습용 사본의 최대 변은 1280, 모든 모델의 입력 크기는 640이다.",
        "SH17 클래스 번호는 저자 YAML과 모든 이미지의 VOC 객체 수를 대조했다.", "",
        "## 앱에 연결할 PPE 모델 선택", "",
        f"선택 모델: `{selection['primary_ppe']}`. Construction-PPE의 val에서 명시적 미착용 네 클래스의 AP50 평균으로 선택했다.",
        "대상 클래스는 no_helmet, no_goggle, no_gloves, no_boots다. test 결과는 선택 점수에 사용하지 않았다.", "",
        "no_boots의 val 정답 객체는 4개뿐이므로 선택 점수의 불확실성이 크다. 추가 현장 사진으로 확인해야 한다.", "",
        "| 후보 모델 | val 미착용 AP50 평균 |", "|---|---:|",
        *[f"| {name} | {score:.4f} |" for name, score in selection["scores"].items()], "",
        "## 실행 환경", "", f"```json\n{json.dumps(environment, ensure_ascii=False, indent=2)}\n```", "",
        "## 탐지 클래스별 결과", "",
    ]
    for report in reports:
        lines += [f"### {report['dataset']}", "", f"평가 이미지: {report['evaluation_images']:,}장 ({report['evaluation_split']}).", "",
                  "| 클래스 | 정답 객체 수 | Precision | Recall | mAP50 |", "|---|---:|---:|---:|---:|"]
        for name, metrics in report["per_class"].items():
            lines.append(f"| {name} | {report['ground_truth_instances'].get(name, 0)} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['map50']:.3f} |")
        lines.append("")
    lines += [
        "## 앱에서 사용하는 범위", "",
        "- Construction-PPE의 명시적 미착용 클래스와 화재·연기를 위험 제안에 사용한다.",
        "- CHVG와 SH17의 보호구·사람·신체 부품 탐지는 근거로 기록한다. 부품 미탐지로 미착용을 확정하지 않는다.",
        "- 위험도와 조치 문장은 규칙 템플릿이며 별도 언어 모델 학습 결과가 아니다.",
        "- 조치 전후 사진에서 위험이 사라진 결과도 최종 현장 확인을 대체하지 않는다.",
        "- 통로 장애물·노출 전선은 이번 네 데이터셋으로 학습하지 않았다. 정책에 있는 해당 항목은 향후 전용 모델용이다.",
        "- 미착용 클래스별 recall을 확인해 현장 적용 범위를 정한다. 안전관리자의 검토는 계속 필요하다.",
        "- 실내 화재 사진에서의 test 수치는 실제 공장 사진에서의 오탐·미탐 성능을 검증한 결과가 아니다.",
        "", "## 데이터 출처", "",
        "- [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/) — AGPL-3.0",
        "- [Indoor Fire Smoke](https://zenodo.org/records/15826133) — CC BY 4.0",
        "- [CHVG](https://figshare.com/articles/dataset/chvg_dataset/19625166) — CC BY 4.0",
        "- [SH17](https://github.com/ahmadmughees/SH17dataset) — CC BY-NC-SA 4.0",
        "",
    ]
    (output / "TRAINING_RESULTS_KO.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"결과 보고서: {output / 'TRAINING_RESULTS_KO.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
