"""Calibrate a complementary photo classifier on validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_facility_presence import Photos, scores
from scripts.optimize_facilities import proposals, sha, save
from safelog_ai.presence_classifier import PresenceClassifier


def photo_rates(target: np.ndarray, prediction: np.ndarray) -> dict:
    tp, fp, fn, tn = (int(x.sum()) for x in (target & prediction, ~target & prediction, target & ~prediction, ~target & ~prediction))
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "miss_fraction": 1 - recall if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "f2": 5 * precision * recall / (4 * precision + recall) if precision or recall else 0.}


def presence_scores(split: str, device: str, run: str = "facility-presence") -> dict:
    weights = ROOT / "runs" / run / "best.pt"
    target_path = ROOT / (f"runs/presence-scores-{split}.json" if run == "facility-presence" else f"runs/presence-scores-{run}-{split}.json")
    signature = sha(weights)
    if target_path.exists():
        old = json.loads(target_path.read_text())
        if old["weights_sha256"] == signature:
            return old
    classifier = PresenceClassifier(weights, device)
    dataset = Photos(split, classifier.imgsz)
    loader = DataLoader(dataset, batch_size=24, num_workers=4)
    target, probability, names = scores(classifier.model, loader, classifier.device)
    if len(dataset) != {"val": 710, "test": 975}[split]:
        raise ValueError("Incomplete facility photo evaluation")
    result = {"weights_sha256": signature, "split": split, "imgsz": classifier.imgsz, "classes": classifier.classes,
              "images": names, "targets": target.astype(int).tolist(), "probabilities": probability.tolist()}
    save(target_path, result)
    return result


def baseline_presence(cache: dict, label_id: int, threshold: float, names: list[str]):
    by_name = {item["image"]: item for item in cache["images"]}
    return np.array([any(c == label_id and p >= threshold for c,p in zip(by_name[name]["classes"], by_name[name]["confidence"])) for name in names], dtype=bool)


def select(device: str):
    path = ROOT / "reports/facility-inference-profile.json"
    profile = json.loads(path.read_text())
    dacl = profile["models"]["facility-dacl-optimized"]
    classifier = presence_scores("val", device)
    cache = proposals(dacl["source_run"], "dacl10k", "val", dacl["imgsz"], "0" if device == "cuda" else device)
    targets, probability = np.array(classifier["targets"], dtype=bool), np.array(classifier["probabilities"])
    entry = {"model": "facility-presence", "architecture": "efficientnet_b0_multilabel_v1",
             "weights_sha256": classifier["weights_sha256"], "imgsz": classifier["imgsz"], "thresholds": {},
             "evidence_scope": "photo_presence_without_location", "selection_split": "val"}
    report = {}
    for i, label in enumerate(classifier["classes"]):
        base = baseline_presence(cache, i, dacl["thresholds"][label], classifier["images"])
        old = photo_rates(targets[:, i], base)
        candidates = []
        for threshold in [round(v / 100, 2) for v in range(10, 101)]:
            item = photo_rates(targets[:, i], base | ((probability[:, i] >= threshold) & (threshold < 1)))
            if (item["false_positive_rate"] <= old["false_positive_rate"] + .02 and
                    item["precision"] >= old["precision"] - .03):
                candidates.append((threshold, item))
        # 1.0 disables the classifier head; ties prefer higher threshold/fewer added opinions.
        threshold, new = max(candidates, key=lambda x: (x[1]["f2"], x[0]))
        entry["thresholds"][label] = threshold
        report[label] = {"detector_only": old, "combined": new, "threshold": threshold,
                         "photo_fpr_max": old["false_positive_rate"] + .02,
                         "photo_precision_min": max(0., old["precision"] - .03)}
    profile["photo_classifier"] = entry
    save(path, profile)
    save(ROOT / "reports/facility-presence-validation.json", {"selection_split": "val", "profile": entry, "per_class": report})
    print(json.dumps(report, indent=2), flush=True)


def test(device: str):
    path = ROOT / "reports/facility-inference-profile.json"
    frozen = sha(path)
    profile = json.loads(path.read_text())
    entry, dacl = profile["photo_classifier"], profile["models"]["facility-dacl-optimized"]
    classifier = presence_scores("test", device)
    if entry["weights_sha256"] != classifier["weights_sha256"]:
        raise ValueError("Classifier/profile mismatch")
    cache = proposals("facility-dacl-optimized", "dacl10k", "test", dacl["imgsz"], "0" if device == "cuda" else device)
    targets, probability = np.array(classifier["targets"], dtype=bool), np.array(classifier["probabilities"])
    report = {}
    for i, label in enumerate(classifier["classes"]):
        base = baseline_presence(cache, i, dacl["thresholds"][label], classifier["images"])
        classifier_prediction = (probability[:, i] >= entry["thresholds"][label]) & (entry["thresholds"][label] < 1)
        added = classifier_prediction & ~base
        report[label] = {"detector_only": photo_rates(targets[:, i], base),
                         "combined": photo_rates(targets[:, i], base | added),
                         "classifier_only": photo_rates(targets[:, i], classifier_prediction),
                         "additional_positive_photos": int((added & targets[:, i]).sum()),
                         "additional_negative_photos": int((added & ~targets[:, i]).sum()),
                         "threshold": entry["thresholds"][label]}
    if sha(path) != frozen:
        raise ValueError("Test changed the frozen profile")
    save(ROOT / "reports/facility-presence-test.json", {"profile_sha256": frozen, "split": "test", "images": len(classifier["images"]),
         "per_class": report, "limitation": "Class presence only, no defect localization. Existing source holdout, not external field safety validation."})
    print(json.dumps(report, indent=2), flush=True)


def export(run: str = "facility-presence"):
    classifier = PresenceClassifier(ROOT / "runs" / run / "best.pt", "cpu")
    target = ROOT / "runs" / run / "best.onnx"
    torch.onnx.export(classifier.model, torch.zeros(1, 3, classifier.imgsz, classifier.imgsz), str(target),
                      input_names=["images"], output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(target))
    # ONNX logits require sigmoid and the same full-photo resize/normalization as PyTorch.
    save(ROOT / "runs" / run / "EXPORT.json", {"input": f"RGB NCHW float32, full-photo resize to {classifier.imgsz}, /255, ImageNet mean/std",
         "output": "7 logits; apply sigmoid and validation-selected thresholds; no boxes",
         "classes": classifier.classes, "onnx_sha256": sha(target)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test", "export"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"select": select, "test": test}.get(args.stage, lambda _: export())(args.device)


if __name__ == "__main__":
    main()
