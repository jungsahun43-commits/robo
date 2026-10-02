"""Freeze validation thresholds/views, then test exactly that configuration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.presence_classifier import PresenceClassifier
from scripts.train_facility_target import dacl_items, read, save, sha, TARGETS
from scripts.facility_error_target import rates, operating_point, under_target


def views(image, grid):
    result = [image]
    if grid > 1:
        width, height = image.size
        for y in range(grid):
            for x in range(grid):
                result.append(image.crop((x * width // grid, y * height // grid,
                                          (x + 1) * width // grid, (y + 1) * height // grid)))
    return result


class ViewPhotos(Dataset):
    def __init__(self, items, transform, grid): self.items, self.transform, self.grid = items, transform, grid
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        with Image.open(ROOT / self.items[i]["image"]) as handle:
            image = handle.convert("RGB")
            inputs = torch.stack([self.transform(value) for value in views(image, self.grid)])
        return inputs


@torch.inference_mode()
def probabilities(weights, items, grid, device):
    classifier = PresenceClassifier(weights, device)
    loader = DataLoader(ViewPhotos(items, classifier.transform, grid), batch_size=2 if grid > 1 else 16,
                        num_workers=4)
    scores = []
    for inputs in loader:
        b, k, c, h, w = inputs.shape
        chunks = []
        flattened = inputs.view(b*k, c, h, w)
        for start in range(0, len(flattened), 16):
            chunks.append(classifier.model(flattened[start:start+16].to(device)).sigmoid().cpu())
        scores.append(torch.cat(chunks).view(b, k, -1).amax(1).numpy())
    return np.concatenate(scores), classifier.classes


def cached(weights, items, grid, output, device):
    signature = {"weights_sha256": sha(weights), "images_sha256": __import__("hashlib").sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(), "grid": grid}
    if output.exists():
        value = read(output)
        if value["signature"] == signature: return np.array(value["probabilities"])
    probability, classes = probabilities(weights, items, grid, device)
    save(output, {"signature": signature, "classes": classes, "probabilities": probability.tolist()})
    return probability


def select(name, device):
    run = ROOT / "runs" / name
    training, split = read(run / "TRAINING.json"), read(run / "SPLIT.json")
    if training["status"] != "complete": raise ValueError("Complete training before freezing selection")
    weights = run / "best.pt"
    if training["weights_sha256"] != sha(weights): raise ValueError("Trained weights changed")
    dacl, classes = dacl_items("val", training["imgsz"])
    items = {"dacl": dacl, "damsegment": split["val"]}
    attempts = []
    for grid in (1, 2, 3):
        predictions = {domain: cached(weights, records, grid, run / f"target-validation-{domain}-grid{grid}.json", device)
                       for domain, records in items.items()}
        points = {label: operating_point({domain: (np.array([item["targets"][classes.index(label)] for item in records], dtype=bool),
                                  predictions[domain][:, classes.index(label)]) for domain, records in items.items()}) for label in TARGETS}
        worst = max(p["worst_error"] for p in points.values())
        result = {"grid": grid, "views": 1 if grid == 1 else 1+grid**2, "per_class": points,
                  "worst_error": worst, "target_passed": all(p["target_passed"] for p in points.values())}
        attempts.append(result)
        print(json.dumps(result, indent=2), flush=True)
    chosen = min(attempts, key=lambda row: (row["worst_error"], sum(p["worst_error"] for p in row["per_class"].values()), row["views"]))
    result = {"run": name, "weights_sha256": sha(weights), "selection_split": "val", "classes": classes,
              "validation_counts": {k: len(v) for k,v in items.items()}, "attempts": attempts, "selected": chosen,
              "criterion": "Per target class FNR AND FPR strictly below .05 in BOTH validation domains; no test selection",
              "limitation": "Both are source validation domains; dam scene IDs unavailable; no independent field guarantee"}
    save(run / "TARGET-SELECTION.json", result)
    save(ROOT / "reports" / f"{name}-target-validation.json", result)


def test(name, device):
    run = ROOT / "runs" / name
    selection = read(run / "TARGET-SELECTION.json")
    if not selection["selected"]["target_passed"]:
        raise ValueError("Validation target not achieved; keep tests outside repeat training selection")
    weights = run / "best.pt"
    if selection["weights_sha256"] != sha(weights): raise ValueError("Frozen weights changed")
    frozen = sha(run / "TARGET-SELECTION.json")
    training = read(run / "TRAINING.json")
    dacl, classes = dacl_items("test", training["imgsz"])
    items = {"dacl": dacl, "damsegment": read(ROOT / "data/damsegment-training/test.json")["items"]}
    result = {}
    for domain, records in items.items():
        probabilities_ = cached(weights, records, selection["selected"]["grid"], run / f"target-test-{domain}.json", device)
        result[domain] = {label: rates(np.array([item["targets"][classes.index(label)] for item in records], dtype=bool),
                    probabilities_[:, classes.index(label)] >= selection["selected"]["per_class"][label]["threshold"]) for label in TARGETS}
    if frozen != sha(run / "TARGET-SELECTION.json"): raise ValueError("Test changed frozen selection")
    report = {"run": name, "selection_sha256": frozen, "weights_sha256": sha(weights), "split": "test", "domains": result,
              "target_passed": under_target([row for domain in result.values() for row in domain.values()]),
              "limitation": "Previously inspected source holdouts used only for release acceptance, not independent field validation"}
    save(ROOT / "reports" / f"{name}-target-test.json", report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("select", "test")); p.add_argument("--name", default="facility-presence-target-v2s")
    p.add_argument("--device", default="cuda"); args=p.parse_args()
    torch.set_num_threads(4)
    (select if args.stage == "select" else test)(args.name, args.device)


if __name__ == "__main__": main()
