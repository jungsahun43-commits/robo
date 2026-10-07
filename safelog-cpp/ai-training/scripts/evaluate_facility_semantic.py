"""Evaluate only the fixed semantic candidate on existing whole-photo VAL."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.research_presence import ResearchPresenceClassifier as PresenceClassifier
from scripts.evaluate_facility_target import ViewPhotos
from scripts.train_facility_target import dacl_items,read,save,sha,TARGETS
from scripts.facility_error_target import operating_point
from scripts.facility_semantic_study import PROTOCOL,RUN_NAME,validate_protocol,require


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


def select(name, device, grids=(1, 2, 3)):
    run = ROOT / "runs" / name
    training, split = read(run / "TRAINING.json"), read(run / "SPLIT.json")
    if training["status"] != "complete": raise ValueError("Complete training before freezing selection")
    weights = run / "best.pt"
    if training["weights_sha256"] != sha(weights): raise ValueError("Trained weights changed")
    dacl, classes = dacl_items("val", training["imgsz"])
    items = {"dacl": dacl, "damsegment": split["val"]}
    if training.get("additional_validation"):
        source = training["additional_validation"]
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]: raise ValueError("Additional validation manifest changed")
        data = read(path)
        if data["split"] != "val" or data["classes"] != classes: raise ValueError("Additional validation mismatch")
        items["codebrim"] = data["items"]
    attempts = []
    for grid in grids:
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
    result = {"run": name, "weights_sha256": sha(weights), "selection_split": "val", "classes": classes, "configured_grids": list(grids),
              "validation_counts": {k: len(v) for k,v in items.items()}, "attempts": attempts, "selected": chosen,
              "criterion": "Per target class FNR AND FPR strictly below .05 in EVERY recorded validation domain; no test selection",
              "limitation": "All recorded domains are source validation domains; dam scene IDs unavailable; no independent field guarantee"}
    save(run / "TARGET-SELECTION.json", result)
    save(ROOT / "reports" / f"{name}-target-validation.json", result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('select',))
    parser.add_argument('--name',default=RUN_NAME)
    parser.add_argument('--grids',default='1')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    validate_protocol(read(ROOT/PROTOCOL),ROOT)
    require(args.name==RUN_NAME and args.grids=='1','Only the predeclared grid1 semantic candidate may be evaluated')
    require(not (ROOT/'runs'/RUN_NAME/'TARGET-SELECTION.json').exists(),'Preserve completed VAL selection evidence')
    select(RUN_NAME,args.device,(1,))


if __name__=='__main__':main()
