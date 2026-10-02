"""Round 4: select on real validation; test and release a frozen separate candidate."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_facilities import proposals, sha, save
from scripts.optimize_facility_presence import presence_scores, baseline_presence, photo_rates, export
from safelog_ai.facility_profile import photo_entries
from safelog_ai.presence_classifier import PresenceClassifier
from scripts.train_facility_presence import scores, average_precision

RUN = "facility-presence-damsegment"
OLD = ROOT / "reports/facility-inference-profile-round1.json"
CANDIDATE = ROOT / "reports/facility-inference-profile-round4-candidate.json"
PREFIX = ROOT / "reports"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def choose(target, base, old_score, old_cutoff, new_score):
    original = photo_rates(target, base | ((old_score >= old_cutoff) & (old_cutoff < 1)))
    choices, capped = [], []
    for value in range(5, 100):
        cutoff = value / 100
        metrics = photo_rates(target, base | (new_score >= cutoff))
        if metrics["false_positive_rate"] <= original["false_positive_rate"] + 1e-12:
            capped.append((metrics["miss_fraction"] + metrics["false_positive_rate"], -cutoff, cutoff, metrics))
        if (metrics["false_positive_rate"] <= original["false_positive_rate"] + 1e-12 and
            original["miss_fraction"] - metrics["miss_fraction"] >= .02 - 1e-12):
            choices.append((metrics["miss_fraction"] + metrics["false_positive_rate"], -cutoff, cutoff, metrics))
    selected = min(choices) if choices else None
    diagnostic = min(capped) if capped else None
    return {"round1": original, "eligible": selected is not None,
            "candidate_under_fpr_cap": {"threshold": diagnostic[2], "metrics": diagnostic[3]} if diagnostic else None,
            "threshold": selected[2] if selected else old_cutoff,
            "selected": selected[3] if selected else original}


def select(device):
    training = read(ROOT / "runs" / RUN / "TRAINING.json")
    if training["status"] != "complete" or training["supplemental_images"] <= 0:
        raise ValueError("Complete the supplemental training first")
    old_profile = read(OLD)
    old = presence_scores("val", device)
    new = presence_scores("val", device, RUN)
    assert old["images"] == new["images"] and old["classes"] == new["classes"] and old["targets"] == new["targets"]
    dacl = old_profile["models"]["facility-dacl-optimized"]
    detector = proposals(dacl["source_run"], "dacl10k", "val", dacl["imgsz"], "0" if device == "cuda" else device)
    targets = np.array(old["targets"], dtype=bool)
    original_entry = deepcopy(old_profile["photo_classifier"])
    new_entry = {**deepcopy(original_entry), "model": RUN, "imgsz": new["imgsz"], "weights_sha256": new["weights_sha256"],
                 "thresholds": {label: 1. for label in old["classes"]}}
    selected = {}
    for label in ("concrete_crack", "concrete_spalling"):
        i = old["classes"].index(label)
        base = baseline_presence(detector, i, dacl["thresholds"][label], old["images"])
        item = choose(targets[:, i], base, np.array(old["probabilities"])[:, i], original_entry["thresholds"][label], np.array(new["probabilities"])[:, i])
        selected[label] = item
        if item["eligible"]:
            original_entry["thresholds"][label] = 1.
            new_entry["thresholds"][label] = item["threshold"]
    candidate = deepcopy(old_profile)
    candidate.update(version="facility-validation-v5", feedback_round=4, evaluation_report="facility-round4-test.json",
                     photo_classifier=original_entry, photo_classifiers=[original_entry])
    if any(item["eligible"] for item in selected.values()): candidate["photo_classifiers"].append(new_entry)
    save(CANDIDATE, candidate)
    save(PREFIX / "facility-round4-validation.json", {"split": "val", "images": len(old["images"]),
         "profile_sha256": sha(CANDIDATE), "per_class": selected, "eligible_heads": sum(item["eligible"] for item in selected.values()),
         "criterion": "FPR does not increase and FNR falls by at least 2 percentage points; crack/spalling heads only"})
    print(json.dumps(selected, indent=2), flush=True)


def test(device):
    validation = read(PREFIX / "facility-round4-validation.json")
    if validation["profile_sha256"] != sha(CANDIDATE):
        raise ValueError("Candidate changed after validation selection")
    if not validation["eligible_heads"]:
        print("Validation did not qualify a new head; no new test selection or default change", flush=True)
        return
    frozen = sha(CANDIDATE)
    candidate, previous = read(CANDIDATE), read(OLD)
    old = presence_scores("test", device)
    caches = {entry["model"]: presence_scores("test", device, entry["model"]) for entry in photo_entries(candidate)}
    for entry in photo_entries(candidate):
        cache = caches[entry["model"]]
        assert cache["weights_sha256"] == entry["weights_sha256"] and cache["images"] == old["images"] and cache["targets"] == old["targets"] and cache["classes"] == old["classes"]
    dacl = candidate["models"]["facility-dacl-optimized"]
    detector = proposals("facility-dacl-optimized", "dacl10k", "test", dacl["imgsz"], "0" if device == "cuda" else device)
    target = np.array(old["targets"], dtype=bool)
    result = {}
    for i, label in enumerate(old["classes"]):
        base = baseline_presence(detector, i, dacl["thresholds"][label], old["images"])
        cutoff = previous["photo_classifier"]["thresholds"][label]
        original = base | ((np.array(old["probabilities"])[:, i] >= cutoff) & (cutoff < 1))
        current = base.copy()
        for entry in photo_entries(candidate):
            cutoff = entry["thresholds"][label]
            current |= (np.array(caches[entry["model"]]["probabilities"])[:, i] >= cutoff) & (cutoff < 1)
        result[label] = {"round1": photo_rates(target[:, i], original), "round4": photo_rates(target[:, i], current),
                         "recovered_positives": int((target[:, i] & ~original & current).sum()),
                         "lost_positives": int((target[:, i] & original & ~current).sum()),
                         "removed_false_positives": int((~target[:, i] & original & ~current).sum()),
                         "new_false_positives": int((~target[:, i] & ~original & current).sum())}
    assert sha(CANDIDATE) == frozen
    save(PREFIX / "facility-round4-test.json", {"split": "test", "images": len(old["images"]), "profile_sha256": frozen, "per_class": result,
        "limitation": "Previously observed source holdout used for release acceptance; not independent field validation. Photo presence only."})
    print(json.dumps(result, indent=2), flush=True)


def acceptable(items):
    safe = all(item["round4"][key] <= item["round1"][key] for item in items for key in ("fp", "fn"))
    better = any(item["round4"][key] < item["round1"][key] for item in items for key in ("fp", "fn"))
    return safe and better


class DamPhotos(Dataset):
    def __init__(self, manifest, transform):
        self.items, self.transform = manifest["items"], transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        item = self.items[i]
        with Image.open(ROOT / item["image"]) as image:
            inputs = self.transform(image.convert("RGB"))
        return inputs, torch.tensor(item["targets"]), item["image"]


def external(device):
    """Evaluate reserved patches only AFTER threshold/profile freezing."""
    torch.set_num_threads(4)
    manifest_path = ROOT / "data/damsegment-training/test.json"
    manifest = read(manifest_path)
    validation = read(PREFIX / "facility-round4-validation.json")
    frozen = sha(CANDIDATE)
    if validation["profile_sha256"] != frozen: raise ValueError("Candidate not frozen")
    candidate, previous = read(CANDIDATE), read(OLD)
    all_scores = {}
    for name in ("facility-presence", RUN):
        weights = ROOT / "runs" / name / "best.pt"
        path = ROOT / "runs" / f"damsegment-scores-{name}.json"
        cached = read(path) if path.exists() else None
        if cached and cached["weights_sha256"] == sha(weights) and cached["manifest_sha256"] == sha(manifest_path):
            result = cached
        else:
            classifier = PresenceClassifier(weights, device)
            if classifier.classes != manifest["classes"]: raise ValueError("External class order mismatch")
            dataset = DamPhotos(manifest, classifier.transform)
            loader = DataLoader(dataset, batch_size=24, num_workers=4)
            targets, probability, names = scores(classifier.model, loader, classifier.device)
            result = {"weights_sha256": sha(weights), "manifest_sha256": sha(manifest_path),
                      "images": names, "targets": targets.tolist(), "probabilities": probability.tolist()}
            save(path, result)
        all_scores[name] = result
    old_scores = all_scores["facility-presence"]
    for value in all_scores.values():
        if value["images"] != old_scores["images"] or value["targets"] != old_scores["targets"]:
            raise ValueError("External ordering differs")
    detector_entry = previous["models"]["facility-dacl-optimized"]
    weights = ROOT / "runs/facility-dacl-optimized/weights/best.pt"
    detector_path = ROOT / "runs/damsegment-detector-proposals.json"
    cached = read(detector_path) if detector_path.exists() else None
    signature = {"weights_sha256": sha(weights), "manifest_sha256": sha(manifest_path),
                 "imgsz": detector_entry["imgsz"], "confidence_floor": .05}
    if cached and cached["signature"] == signature: detector = cached["images"]
    else:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        detector = []
        # Ultralytics treats an in-memory path list as one batch even with batch=.
        # Bound the list itself; never warm up on all held-out photos at once.
        for start in range(0, len(manifest["items"]), 8):
            results = model.predict([str(ROOT / item["image"]) for item in manifest["items"][start:start + 8]],
                        imgsz=detector_entry["imgsz"], conf=.05, batch=8,
                        device="0" if device == "cuda" else device, stream=True, verbose=False)
            detector.extend({"classes": r.boxes.cls.cpu().int().tolist(), "confidence": r.boxes.conf.cpu().tolist()} for r in results)
        save(detector_path, {"signature": signature, "images": detector})
    if len(detector) != len(manifest["items"]): raise ValueError("Incomplete external detector results")
    result = {}
    target = np.array(old_scores["targets"])
    for i, label in enumerate(manifest["classes"]):
        known = target[:, i] >= 0
        if not known.any(): continue
        base = np.array([any(c == i and p >= detector_entry["thresholds"][label]
                       for c, p in zip(row["classes"], row["confidence"])) for row in detector])
        cutoff = previous["photo_classifier"]["thresholds"][label]
        old = base | ((np.array(old_scores["probabilities"])[:, i] >= cutoff) & (cutoff < 1))
        current = base.copy()
        for entry in photo_entries(candidate):
            if all_scores[entry["model"]]["weights_sha256"] != entry["weights_sha256"]:
                raise ValueError("External model/profile mismatch")
            cutoff = entry["thresholds"][label]
            current |= (np.array(all_scores[entry["model"]]["probabilities"])[:, i] >= cutoff) & (cutoff < 1)
        result[label] = {"round1": photo_rates(target[known, i].astype(bool), old[known]),
                         "round4": photo_rates(target[known, i].astype(bool), current[known]),
                         "classifier_ranking_ap": {name: average_precision(target[known, i], np.array(cache["probabilities"])[known, i])
                                                   for name, cache in all_scores.items()}}
        # A separately labelled diagnostic operating point, frozen on validation.
        # It never enables a head that failed the predeclared release criterion.
        diagnostic = validation["per_class"][label]["candidate_under_fpr_cap"]
        if diagnostic:
            cutoff = diagnostic["threshold"]
            hypothetical = base | (np.array(all_scores[RUN]["probabilities"])[:, i] >= cutoff)
            result[label]["unreleased_new_model_diagnostic"] = {
                "threshold_from_dacl_validation": cutoff,
                "metrics": photo_rates(target[known, i].astype(bool), hypothetical[known]),
                "scope": "Diagnostic only; this head was not necessarily eligible or deployed"}
    if sha(CANDIDATE) != frozen: raise ValueError("External test changed candidate")
    save(PREFIX / "facility-round4-external-test.json", {"split": "reserved_damsegment_patches",
         "images": len(manifest["items"]), "profile_sha256": frozen, "manifest_sha256": sha(manifest_path), "per_class": result,
         "classifier_weights_sha256": {name: item["weights_sha256"] for name, item in all_scores.items()},
         "limitation": manifest["audit"]["limitation"], "threshold_selection": "DACL validation only; no DamSegment test calibration"})
    print(json.dumps(result, indent=2), flush=True)


def release():
    validation = read(PREFIX / "facility-round4-validation.json")
    if validation["profile_sha256"] != sha(CANDIDATE):
        raise ValueError("Candidate changed after validation selection")
    active = PREFIX / "facility-inference-profile.json"
    if sha(active) != sha(OLD) and sha(active) != sha(CANDIDATE):
        raise ValueError("Default profile changed independently; release aborted")
    decision = {"status": "kept_round1", "active_version": "facility-validation-v2", "candidate_sha256": sha(CANDIDATE),
                "criterion": "Validation FPR nonincrease and FNR reduction >=2pp per selected head; DACL FP/FN nonincrease and some improvement; reserved-patch FP/FN nonincrease",
                "validation_eligible_heads": validation["eligible_heads"],
                "reason": "No new classifier head qualified on validation"}
    extra = read(PREFIX / "facility-round4-external-test.json")
    if extra["profile_sha256"] != sha(CANDIDATE): raise ValueError("External report/profile mismatch")
    if extra["classifier_weights_sha256"][RUN] != sha(ROOT / "runs" / RUN / "best.pt"):
        raise ValueError("External report and trained model differ")
    decision["reserved_patch_test_checked"] = True
    if validation["eligible_heads"]:
        report = read(PREFIX / "facility-round4-test.json")
        if report["profile_sha256"] != sha(CANDIDATE): raise ValueError("Candidate profile and test report differ")
        items = list(report["per_class"].values())
        external_safe = all(item["round4"][key] <= item["round1"][key]
                            for item in extra["per_class"].values() for key in ("fp", "fn"))
        adopt = acceptable(items) and external_safe
        decision.update(status="adopted_round4" if adopt else "kept_round1",
                        active_version="facility-validation-v5" if adopt else "facility-validation-v2",
                        reason="Both per-label test gates passed" if adopt else "DACL or reserved-patch per-label test gate failed",
                        external_no_regression=external_safe,
                        label_photo_counts={f"{name}_{key}": sum(item[name][key] for item in items)
                                            for name in ("round1", "round4") for key in ("fp", "fn")})
        if adopt:
            if not (ROOT / "runs" / RUN / "best.onnx").is_file(): raise ValueError("Export ONNX before adopting")
            active.write_bytes(CANDIDATE.read_bytes())
        else: active.write_bytes(OLD.read_bytes())
    else: active.write_bytes(OLD.read_bytes())
    save(PREFIX / "facility-round4-release.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test", "external", "export", "release"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "select": select(args.device)
    elif args.stage == "test": test(args.device)
    elif args.stage == "external": external(args.device)
    elif args.stage == "export": export(RUN)
    else: release()


if __name__ == "__main__": main()
