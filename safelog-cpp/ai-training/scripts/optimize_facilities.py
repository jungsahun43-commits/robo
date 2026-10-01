"""Select weights/resolution, calibrate thresholds on val, then evaluate frozen test.

Uses the same NMS, maximum detection count, resolution and threshold filtering as
the HTTP server. Photo-presence and region-localization scores are kept separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".config/ultralytics"))
from scripts.evaluate_fixed_threshold import match_counts, rates

EXPERIMENTS = (("facility-dacl", "dacl10k"), ("facility-corrosion", "ostrava-corrosion"))


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def proposals(run: str, dataset: str, split: str, size: int, device: str) -> dict:
    from ultralytics import YOLO
    weights = ROOT / f"runs/{run}/weights/best.pt"
    target = ROOT / f"runs/proposals-{run}-{split}-{size}.json"
    signature = sha(weights)
    config = yaml.safe_load((ROOT / f"data/{dataset}-yolo/data.yaml").read_text(encoding="utf-8"))
    base = Path(config["path"])
    label_digest = hashlib.sha256()
    for label in sorted((base / "labels" / split).glob("*.txt")):
        label_digest.update(label.name.encode())
        label_digest.update(label.read_bytes())
    data_signature = label_digest.hexdigest()
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["weights_sha256"] == signature and old.get("labels_sha256") == data_signature:
            return old
    images = []
    for result in YOLO(str(weights)).predict(source=str(base / config[split]), conf=.01, iou=.7,
                                            max_det=300, imgsz=size, device=device, verbose=False, stream=True):
        path = Path(result.path)
        labels = np.array([[float(v) for v in line.split()] for line in
                           (base / f"labels/{split}/{path.stem}.txt").read_text().splitlines()]).reshape(-1, 5)
        h, w = result.orig_shape
        boxes = np.empty((len(labels), 4))
        boxes[:, :2] = (labels[:, 1:3] - labels[:, 3:5] / 2) * [w, h]
        boxes[:, 2:] = (labels[:, 1:3] + labels[:, 3:5] / 2) * [w, h]
        images.append({"image": path.name, "target_classes": labels[:, 0].astype(int).tolist(),
                       "target_boxes": boxes.tolist(), "classes": result.boxes.cls.int().cpu().tolist(),
                       "confidence": result.boxes.conf.cpu().tolist(), "boxes": result.boxes.xyxy.cpu().tolist()})
        if len(images) % 100 == 0:
            print(f"{run} {split} {size}: {len(images)}", flush=True)
    expected = json.loads((base / "PREPARATION.json").read_text())["split_counts"][split]
    if len(images) != expected:
        raise ValueError(f"Incomplete {split} evaluation: {len(images)} != {expected}")
    report = {"run": run, "split": split, "imgsz": size, "weights_sha256": signature, "labels_sha256": data_signature,
              "names": list(config["names"].values()), "images": images,
              "prediction_floor": .01, "iou": .7, "max_det": 300}
    save(target, report)
    return report


def measure(cache: dict, class_id: int, threshold: float) -> dict:
    photo, region = Counter(), Counter()
    for item in cache["images"]:
        gt = np.array(item["target_boxes"]).reshape(-1, 4)[np.array(item["target_classes"]) == class_id]
        mask = (np.array(item["classes"]) == class_id) & (np.array(item["confidence"]) >= threshold)
        pred = np.array(item["boxes"]).reshape(-1, 4)[mask]
        photo["tp" if len(gt) and len(pred) else "fn" if len(gt) else "fp" if len(pred) else "tn"] += 1
        tp, fp, fn = match_counts(pred, gt)
        region.update(tp=tp, fp=fp, fn=fn)
    tp, fp, fn, tn = (photo[k] for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"threshold": threshold, "photo": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "precision": precision, "recall": recall,
             "miss_fraction": 1 - recall if tp + fn else None,
             "false_positive_rate": fp / (fp + tn) if fp + tn else None,
             "f2": 5 * precision * recall / (4 * precision + recall) if precision or recall else 0.},
            "region": rates(region)}


def choose_threshold(cache: dict, class_id: int, baseline: dict) -> tuple[dict, dict]:
    default = measure(cache, class_id, .25)
    # No negative photos: do not tune photo alarm rates on an all-positive split.
    if baseline["photo"]["false_positive_rate"] is None:
        return default, {"reason": "no negative validation photos; keep 0.25"}
    limits = {"photo_fpr_max": baseline["photo"]["false_positive_rate"] + .02,
              "photo_precision_min": max(0., baseline["photo"]["precision"] - .03),
              "region_precision_min": max(0., (baseline["region"]["precision"] or 0.) - .03)}
    candidates = []
    for threshold in sorted(set([.25] + [round(v / 100, 2) for v in range(5, 81)])):
        item = measure(cache, class_id, threshold)
        p = item["photo"]
        if (p["false_positive_rate"] <= limits["photo_fpr_max"] and
                p["precision"] >= limits["photo_precision_min"] and
                (item["region"]["precision"] or 0.) >= limits["region_precision_min"]):
            candidates.append(item)
    if not candidates:
        return default, {"limits": limits, "reason": "no eligible threshold; retain selected model at 0.25"}
    best = max(candidates, key=lambda x: (x["photo"]["f2"], x["region"]["recall"] or 0., x["threshold"]))
    return best, {"limits": limits, "eligible_thresholds": len(candidates), "reason": "max validation photo F2 under alarm/precision limits"}


def evaluate_profile(cache: dict, entry: dict) -> dict:
    return {label: measure(cache, i, float(entry["thresholds"][label])) for i, label in enumerate(cache["names"])}


def select(device: str):
    from ultralytics import YOLO
    profile = {"version": "facility-validation-v2", "selection_split": "val",
               "weight_resolution_criterion": "maximum validation region mAP50-95; baseline wins exact ties",
               "threshold_criterion": "photo F2; FPR <= baseline + 0.02, photo/region precision >= baseline - 0.03",
               "models": {}}
    comparisons = []
    for baseline, dataset in EXPERIMENTS:
        candidates = []
        for run in (baseline, baseline + "-refined"):
            for size in (960, 1280):
                suffix = "" if size == 960 else f"-{size}"
                path = ROOT / f"runs/evaluation-{run}-val{suffix}.json"
                if not path.exists():
                    metrics = YOLO(str(ROOT / f"runs/{run}/weights/best.pt")).val(
                        data=str(ROOT / f"data/{dataset}-yolo/data.yaml"), split="val", imgsz=size,
                        device=device, workers=0, plots=False, verbose=False)
                    save(path, {"split": "val", "imgsz": size, "map50": float(metrics.box.map50),
                                "map50_95": float(metrics.box.map)})
                metrics = json.loads(path.read_text(encoding="utf-8"))
                if metrics["split"] != "val":
                    raise ValueError("Only validation metrics may select models")
                candidates.append({"run": run, "imgsz": size, "map50": metrics["map50"], "map50_95": metrics["map50_95"]})
        selected = max(candidates, key=lambda x: x["map50_95"])
        original_cache = proposals(baseline, dataset, "val", 960, device)
        chosen_cache = proposals(selected["run"], dataset, "val", selected["imgsz"], device)
        optimized = baseline + "-optimized"
        entry = {"source_run": selected["run"], "dataset": dataset, "imgsz": selected["imgsz"],
                 "weights_sha256": chosen_cache["weights_sha256"], "thresholds": {}}
        comparison = {"model": optimized, "candidates": candidates, "selected": selected, "validation": {}}
        for i, label in enumerate(chosen_cache["names"]):
            old = measure(original_cache, i, .25)
            new, audit = choose_threshold(chosen_cache, i, old)
            entry["thresholds"][label] = new["threshold"]
            comparison["validation"][label] = {"baseline": old, "optimized": new, "calibration": audit}
        # Use new stable model IDs without overwriting baseline checkpoints/exports.
        target = ROOT / f"runs/{optimized}/weights/best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f"runs/{selected['run']}/weights/best.pt", target)
        save(target.parents[1] / "SELECTION.json", {"selection_split": "val", "source": selected, "profile": entry})
        profile["models"][optimized] = entry
        comparisons.append(comparison)
        print(f"Selected {optimized}: {selected}; thresholds={entry['thresholds']}", flush=True)
    save(ROOT / "reports/facility-inference-profile.json", profile)
    save(ROOT / "reports/facility-optimization-validation.json", {"profile": profile, "experiments": comparisons})


def test(device: str):
    from safelog_ai.facility_profile import load_profile, verify_weights
    profile_path = ROOT / "reports/facility-inference-profile.json"
    profile = load_profile(profile_path)
    if not profile["models"]:
        raise ValueError("Select/freeze a profile on validation first")
    frozen = sha(profile_path)
    reports = []
    for name, entry in profile["models"].items():
        verify_weights(ROOT / f"runs/{name}/weights/best.pt", entry)
        cache = proposals(name, entry["dataset"], "test", entry["imgsz"], device)
        old = proposals(name.removesuffix("-optimized"), entry["dataset"], "test", 960, device)
        reports.append({"model": name, "images": len(cache["images"]),
                        "baseline": {label: measure(old, i, .25) for i, label in enumerate(old["names"])},
                        "optimized": evaluate_profile(cache, entry)})
    if sha(profile_path) != frozen:
        raise ValueError("The frozen profile changed during test evaluation")
    save(ROOT / "reports/facility-optimization-test.json", {"profile_sha256": frozen, "split": "test", "experiments": reports,
         "limitation": "Same source holdout already evaluated in the baseline round. No external field validation; no scene grouping IDs. Photo presence ignores localization. Corrosion has no negative photos."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    select(args.device) if args.stage == "select" else test(args.device)


if __name__ == "__main__":
    main()
