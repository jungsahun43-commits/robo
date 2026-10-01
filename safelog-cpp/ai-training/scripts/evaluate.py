from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".config" / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(root / ".config" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".config" / "matplotlib"))
    for variable in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR"):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="SafeLog YOLO 모델 평가")
    parser.add_argument("weights", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.weights.exists() or not args.data.exists():
        raise SystemExit("모델 가중치와 데이터 YAML 경로를 확인하세요.")

    from ultralytics import YOLO

    run_name = args.weights.resolve().parents[1].name
    options = {
        "data": str(args.data.resolve()), "split": args.split, "plots": True,
        "project": str(root / "runs" / "evaluations"), "name": f"{run_name}-{args.split}",
        "exist_ok": True, "workers": 0, "imgsz": args.imgsz,
        "batch": args.batch,
    }
    if args.device is not None:
        options["device"] = args.device
    metrics = YOLO(str(args.weights.resolve())).val(**options)
    report = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {
            str(metrics.names[int(class_id)]): {
                "precision": float(precision), "recall": float(recall),
                "map50": float(map50), "map50_95": float(map95),
            }
            for class_id, precision, recall, map50, map95 in zip(
                metrics.box.ap_class_index, metrics.box.p, metrics.box.r,
                metrics.box.ap50, metrics.box.ap,
            )
        },
    }
    output = args.output or root / "runs" / f"evaluation-{run_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
