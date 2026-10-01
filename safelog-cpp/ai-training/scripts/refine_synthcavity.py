"""Round 3: select on real validation; test and release a frozen separate candidate."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_facilities import proposals, sha, save
from scripts.optimize_facility_presence import presence_scores, baseline_presence, photo_rates, export
from safelog_ai.facility_profile import photo_entries

RUN = "facility-presence-synthetic"
OLD = ROOT / "reports/facility-inference-profile-round1.json"
CANDIDATE = ROOT / "reports/facility-inference-profile-round3-candidate.json"
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
    for label in ("wet_surface", "surface_cavity"):
        i = old["classes"].index(label)
        base = baseline_presence(detector, i, dacl["thresholds"][label], old["images"])
        item = choose(targets[:, i], base, np.array(old["probabilities"])[:, i], original_entry["thresholds"][label], np.array(new["probabilities"])[:, i])
        selected[label] = item
        if item["eligible"]:
            original_entry["thresholds"][label] = 1.
            new_entry["thresholds"][label] = item["threshold"]
    candidate = deepcopy(old_profile)
    candidate.update(version="facility-validation-v4", feedback_round=3, evaluation_report="facility-round3-test.json",
                     photo_classifier=original_entry, photo_classifiers=[original_entry])
    if any(item["eligible"] for item in selected.values()): candidate["photo_classifiers"].append(new_entry)
    save(CANDIDATE, candidate)
    save(PREFIX / "facility-round3-validation.json", {"split": "val", "images": len(old["images"]),
         "profile_sha256": sha(CANDIDATE), "per_class": selected, "eligible_heads": sum(item["eligible"] for item in selected.values()),
         "criterion": "FPR does not increase and FNR falls by at least 2 percentage points; wet/cavity heads only"})
    print(json.dumps(selected, indent=2), flush=True)


def test(device):
    validation = read(PREFIX / "facility-round3-validation.json")
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
        result[label] = {"round1": photo_rates(target[:, i], original), "round3": photo_rates(target[:, i], current),
                         "recovered_positives": int((target[:, i] & ~original & current).sum()),
                         "lost_positives": int((target[:, i] & original & ~current).sum()),
                         "removed_false_positives": int((~target[:, i] & original & ~current).sum()),
                         "new_false_positives": int((~target[:, i] & ~original & current).sum())}
    assert sha(CANDIDATE) == frozen
    save(PREFIX / "facility-round3-test.json", {"split": "test", "images": len(old["images"]), "profile_sha256": frozen, "per_class": result,
        "limitation": "Previously observed source holdout used for release acceptance; not independent field validation. Photo presence only."})
    print(json.dumps(result, indent=2), flush=True)


def acceptable(items):
    safe = all(item["round3"][key] <= item["round1"][key] for item in items for key in ("fp", "fn"))
    better = any(item["round3"][key] < item["round1"][key] for item in items for key in ("fp", "fn"))
    return safe and better


def release():
    validation = read(PREFIX / "facility-round3-validation.json")
    if validation["profile_sha256"] != sha(CANDIDATE):
        raise ValueError("Candidate changed after validation selection")
    active = PREFIX / "facility-inference-profile.json"
    if sha(active) != sha(OLD) and sha(active) != sha(CANDIDATE):
        raise ValueError("Default profile changed independently; release aborted")
    decision = {"status": "kept_round1", "active_version": "facility-validation-v2", "candidate_sha256": sha(CANDIDATE),
                "criterion": "Every label's FP and FN must not increase; at least one must decrease",
                "reason": "No new classifier head qualified on validation"}
    if validation["eligible_heads"]:
        report = read(PREFIX / "facility-round3-test.json")
        if report["profile_sha256"] != sha(CANDIDATE): raise ValueError("Candidate profile and test report differ")
        items = list(report["per_class"].values())
        adopt = acceptable(items)
        decision.update(status="adopted_round3" if adopt else "kept_round1",
                        active_version="facility-validation-v4" if adopt else "facility-validation-v2",
                        reason="Per-label test acceptance passed" if adopt else "Per-label test acceptance failed",
                        label_photo_counts={f"{name}_{key}": sum(item[name][key] for item in items)
                                            for name in ("round1", "round3") for key in ("fp", "fn")})
        if adopt:
            if not (ROOT / "runs" / RUN / "best.onnx").is_file(): raise ValueError("Export ONNX before adopting")
            active.write_bytes(CANDIDATE.read_bytes())
        else: active.write_bytes(OLD.read_bytes())
    else: active.write_bytes(OLD.read_bytes())
    save(PREFIX / "facility-round3-release.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test", "export", "release"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "select": select(args.device)
    elif args.stage == "test": test(args.device)
    elif args.stage == "export": export(RUN)
    else: release()


if __name__ == "__main__": main()
