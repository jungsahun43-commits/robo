"""Second feedback round: keep or replace each photo head using validation only."""
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
from scripts.optimize_facility_presence import baseline_presence, photo_rates, presence_scores, export
from safelog_ai.facility_profile import photo_entries

NEW_RUN = "facility-presence-refined"
ROUND1 = ROOT / "reports/facility-inference-profile-round1.json"
PROFILE = ROOT / "reports/facility-inference-profile.json"


def pick_head(target, base, old_score, old_cutoff, new_score):
    old_prediction = base | ((old_score >= old_cutoff) & (old_cutoff < 1))
    old = photo_rates(target, old_prediction)
    best = {"source": "facility-presence", "threshold": old_cutoff, "metrics": old}
    cost = old["miss_fraction"] + old["false_positive_rate"]
    for cutoff in [round(v / 100, 2) for v in range(5, 101)]:
        metrics = photo_rates(target, base | ((new_score >= cutoff) & (cutoff < 1)))
        # Per-head Pareto guard: no validation recall/FPR regression against round 1.
        if (metrics["miss_fraction"] <= old["miss_fraction"] + 1e-12 and
                metrics["false_positive_rate"] <= old["false_positive_rate"] + 1e-12):
            candidate_cost = metrics["miss_fraction"] + metrics["false_positive_rate"]
            if candidate_cost < cost - 1e-12 or (best["source"] == NEW_RUN and abs(candidate_cost - cost) < 1e-12 and cutoff > best["threshold"]):
                best = {"source": NEW_RUN, "threshold": cutoff, "metrics": metrics}
                cost = candidate_cost
    return {"round1": old, "selected": best, "criterion": "minimize validation FNR+FPR with neither increasing; keep old head on model ties"}


def select(device):
    if json.loads((ROOT / f"runs/{NEW_RUN}/TRAINING.json").read_text())["status"] != "complete":
        raise ValueError("Complete the new classifier training first")
    old_profile = json.loads(ROUND1.read_text())
    dacl = old_profile["models"]["facility-dacl-optimized"]
    old, new = presence_scores("val", device), presence_scores("val", device, NEW_RUN)
    assert old["classes"] == new["classes"] and old["images"] == new["images"] and old["targets"] == new["targets"]
    cache = proposals(dacl["source_run"], "dacl10k", "val", dacl["imgsz"], "0" if device == "cuda" else device)
    target = np.array(old["targets"], dtype=bool)
    old_score, new_score = np.array(old["probabilities"]), np.array(new["probabilities"])
    choices = {}
    old_entry = deepcopy(old_profile["photo_classifier"])
    new_entry = {**deepcopy(old_entry), "model": NEW_RUN, "weights_sha256": new["weights_sha256"], "imgsz": new["imgsz"]}
    old_entry["thresholds"] = {label: 1. for label in old["classes"]}
    new_entry["thresholds"] = {label: 1. for label in old["classes"]}
    for i, label in enumerate(old["classes"]):
        base = baseline_presence(cache, i, dacl["thresholds"][label], old["images"])
        choice = pick_head(target[:, i], base, old_score[:, i], old_profile["photo_classifier"]["thresholds"][label], new_score[:, i])
        choices[label] = choice
        entry = new_entry if choice["selected"]["source"] == NEW_RUN else old_entry
        entry["thresholds"][label] = choice["selected"]["threshold"]
    profile = deepcopy(old_profile)
    profile["version"] = "facility-validation-v3"
    profile["photo_classifier"] = old_entry  # Single-model compatibility; latest server reads the plural entries.
    profile["photo_classifiers"] = [e for e in (old_entry, new_entry) if any(v < 1 for v in e["thresholds"].values())]
    profile["feedback_criterion"] = "per-label validation Pareto: FNR and FPR cannot increase against round 1; never select on test"
    save(ROOT / "reports/facility-feedback-validation.json", {"split": "val", "per_class": choices, "profile": profile,
         "limitation": "Repeated use of the same internal validation can overfit; no deployment error-rate guarantee."})
    save(PROFILE, profile)
    print(json.dumps(choices, indent=2), flush=True)


def test(device):
    frozen = sha(PROFILE)
    profile, previous = json.loads(PROFILE.read_text()), json.loads(ROUND1.read_text())
    dacl = profile["models"]["facility-dacl-optimized"]
    score_files = {entry["model"]: presence_scores("test", device, entry["model"]) for entry in photo_entries(profile)}
    old = presence_scores("test", device)
    targets = np.array(old["targets"], dtype=bool)
    for entry in photo_entries(profile):
        cache = score_files[entry["model"]]
        assert cache["weights_sha256"] == entry["weights_sha256"] and cache["images"] == old["images"] and cache["classes"] == old["classes"]
    detector = proposals("facility-dacl-optimized", "dacl10k", "test", dacl["imgsz"], "0" if device == "cuda" else device)
    result = {}
    for i, label in enumerate(old["classes"]):
        base = baseline_presence(detector, i, dacl["thresholds"][label], old["images"])
        old_cutoff = previous["photo_classifier"]["thresholds"][label]
        original = base | ((np.array(old["probabilities"])[:, i] >= old_cutoff) & (old_cutoff < 1))
        prediction = base.copy()
        for entry in photo_entries(profile):
            cutoff = entry["thresholds"][label]
            prediction |= (np.array(score_files[entry["model"]]["probabilities"])[:, i] >= cutoff) & (cutoff < 1)
        active = next((entry for entry in photo_entries(profile) if entry["thresholds"][label] < 1), None)
        result[label] = {"round1": photo_rates(targets[:, i], original), "round2": photo_rates(targets[:, i], prediction),
                         "selected_source": active["model"] if active else "detector only", "threshold": active["thresholds"][label] if active else 1.,
                         "removed_false_positives": int((~targets[:, i] & original & ~prediction).sum()),
                         "new_false_positives": int((~targets[:, i] & ~original & prediction).sum()),
                         "recovered_positives": int((targets[:, i] & ~original & prediction).sum()),
                         "lost_positives": int((targets[:, i] & original & ~prediction).sum())}
    assert sha(PROFILE) == frozen
    save(ROOT / "reports/facility-feedback-test.json", {"split": "test", "images": len(old["images"]), "profile_sha256": frozen,
        "per_class": result, "limitation": "Same previously observed source holdout; not external field validation. Photo presence only, not location/safety accuracy."})
    print(json.dumps(result, indent=2), flush=True)


def release():
    """Whole-candidate acceptance gate; do not retune heads from observed test errors."""
    report = json.loads((ROOT / "reports/facility-feedback-test.json").read_text())
    candidate_path = ROOT / "reports/facility-inference-profile-round2-candidate.json"
    matched = next((path for path in (PROFILE, candidate_path)
                    if path.is_file() and sha(path) == report["profile_sha256"]), None)
    if matched is None:
        raise ValueError("No candidate profile matches the frozen test report; release aborted")
    candidate_bytes = matched.read_bytes()
    profile = json.loads(candidate_bytes)
    if profile.get("version") != "facility-validation-v3":
        raise ValueError("The feedback release requires the round 2 candidate profile")
    old_fp = sum(item["round1"]["fp"] for item in report["per_class"].values())
    new_fp = sum(item["round2"]["fp"] for item in report["per_class"].values())
    old_fn = sum(item["round1"]["fn"] for item in report["per_class"].values())
    new_fn = sum(item["round2"]["fn"] for item in report["per_class"].values())
    adopt = new_fp <= old_fp and new_fn <= old_fn and (new_fp < old_fp or new_fn < old_fn)
    candidate_path.write_bytes(candidate_bytes)
    if adopt:
        PROFILE.write_bytes(candidate_path.read_bytes())
    else:
        PROFILE.write_bytes(ROUND1.read_bytes())
    decision = {"status": "adopted_round2" if adopt else "kept_round1", "active_version": json.loads(PROFILE.read_text())["version"],
                "criterion": "whole candidate must not increase aggregate label-photo FP or FN, and must reduce at least one",
                "label_photo_counts": {"round1_fp": old_fp, "round2_fp": new_fp, "round1_fn": old_fn, "round2_fn": new_fn},
                "limitation": "This source holdout was used for release acceptance, so it is not independent model-selection evidence. Require a new external evaluation before field-performance claims."}
    save(ROOT / "reports/facility-feedback-release.json", decision)
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test", "export", "release"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "select": select(args.device)
    elif args.stage == "test": test(args.device)
    elif args.stage == "release": release()
    else: export(NEW_RUN)


if __name__ == "__main__":
    main()
