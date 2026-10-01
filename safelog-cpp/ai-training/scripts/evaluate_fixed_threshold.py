"""Count region TP/FP/FN at the SAME confidence threshold as the app server.

IoU>=0.5, one-to-one same-class matches, highest IoU first. 1-precision is
false discovery among predicted boxes, not a true-negative based false positive rate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".config" / "ultralytics"))


def match_counts(predictions: np.ndarray, targets: np.ndarray, iou_threshold: float = 0.5) -> tuple[int, int, int]:
    if len(predictions) == 0 or len(targets) == 0:
        return 0, len(predictions), len(targets)
    low = np.maximum(predictions[:, None, :2], targets[None, :, :2])
    high = np.minimum(predictions[:, None, 2:], targets[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    pred_area = (predictions[:, 2:] - predictions[:, :2]).prod(axis=1)
    target_area = (targets[:, 2:] - targets[:, :2]).prod(axis=1)
    ious = intersection / np.maximum(pred_area[:, None] + target_area[None, :] - intersection, 1e-9)
    indices = np.argwhere(ious >= iou_threshold)
    indices = sorted(indices.tolist(), key=lambda pair: ious[pair[0], pair[1]], reverse=True)
    used_pred, used_target = set(), set()
    for p, t in indices:
        if p not in used_pred and t not in used_target:
            used_pred.add(p)
            used_target.add(t)
    return len(used_pred), len(predictions) - len(used_pred), len(targets) - len(used_target)


def rates(counts: Counter) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "ground_truth_regions": tp + fn,
            "precision": precision, "recall": recall,
            "false_discovery_fraction": 1 - precision if precision is not None else None,
            "miss_fraction": 1 - recall if recall is not None else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    from ultralytics import YOLO
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    base = Path(config["path"])
    names = config["names"]
    counts = {name: Counter() for name in names.values()}
    model = YOLO(str(args.weights.resolve()))
    per_image = []
    for result in model.predict(source=str(base / config[args.split]), stream=True, conf=args.conf,
                                imgsz=args.imgsz, device=args.device, max_det=300, verbose=False):
        image_path = Path(result.path)
        label_path = base / "labels" / args.split / f"{image_path.stem}.txt"
        labels = np.array([[float(v) for v in line.split()] for line in label_path.read_text().splitlines()], dtype=float).reshape(-1, 5)
        h, w = result.orig_shape
        gt_boxes = np.empty((len(labels), 4))
        gt_boxes[:, :2] = (labels[:, 1:3] - labels[:, 3:5] / 2) * [w, h]
        gt_boxes[:, 2:] = (labels[:, 1:3] + labels[:, 3:5] / 2) * [w, h]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy()
        image_counts = {}
        for class_id, name in names.items():
            tp, fp, fn = match_counts(pred_boxes[pred_classes == class_id], gt_boxes[labels[:, 0] == class_id])
            item = Counter(tp=tp, fp=fp, fn=fn)
            counts[name].update(item)
            image_counts[name] = dict(item)
        per_image.append({"image": image_path.name, "counts": image_counts})
        if len(per_image) % 100 == 0:
            print(f"Fixed threshold evaluation: {len(per_image)} images", flush=True)
    report = {"name": args.weights.resolve().parents[1].name, "split": args.split,
              "confidence_threshold": args.conf, "iou_threshold": 0.5, "imgsz": args.imgsz,
              "max_det": 300, "images": len(per_image), "per_class": {k: rates(v) for k, v in counts.items()},
              "micro": rates(sum(counts.values(), Counter())),
              "limitations": "Derived region boxes; 1-precision is false discovery, not FPR. Test scenes may overlap source training scenes.",
              "per_image": per_image}
    output = ROOT / "runs" / f"fixed-{report['name']}-{args.split}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
