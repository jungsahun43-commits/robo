"""Larger partially-labelled model selected by two-domain validation errors."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".config/torch"))
from safelog_ai.presence_classifier import build_model, image_transform, MEAN, STD
from scripts.train_facility_presence import Photos, average_precision
from scripts.facility_error_target import operating_point

ARCH = "efficientnet_v2_s_multilabel_v1"
TARGETS = ("concrete_crack", "concrete_spalling")
DOMAINS = {"dacl": 0, "damsegment": 1, "codebrim": 2}


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def save(path, value): path.write_text(json.dumps(value, indent=2), encoding="utf-8")
def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def split_supplemental():
    source = ROOT / "data/damsegment-training/train.json"
    manifest = read(source)
    items = manifest["items"]
    for item in items:
        digest = hashlib.sha256(("facility-target-seed43:" + item["group_id"]).encode()).hexdigest()
        item["target_split"] = "val" if int(digest[:8], 16) % 5 == 0 else "train"
    train, val = ([item for item in items if item["target_split"] == split] for split in ("train", "val"))
    if {i["group_id"] for i in train} & {i["group_id"] for i in val}: raise ValueError("Group leakage")
    test = read(ROOT / "data/damsegment-training/test.json")["items"]
    if {i["group_id"] for i in items} & {i["group_id"] for i in test}: raise ValueError("Original test leakage")
    result = {"classes": manifest["classes"], "train": train, "val": val, "source_sha256": sha(source),
              "counts": {"train": len(train), "val": len(val), "reserved_test": len(test)},
              "policy": "Only round4 TRAIN redivided by fixed group hash; round4 model must not initialize this run"}
    return result


class FacilityPhotos(Dataset):
    def __init__(self, items, size, training=False):
        self.items, self.transform = items, image_transform(size, training)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        item = self.items[i]
        with Image.open(ROOT / item["image"]) as image: inputs = self.transform(image.convert("RGB"))
        targets = torch.tensor(item["targets"], dtype=torch.float32)
        return inputs, targets.clamp_min(0), (targets >= 0).float(), DOMAINS[item.get("domain", "damsegment")]


def dacl_items(split, size):
    dataset = Photos(split, size)
    return [{"image": path.relative_to(ROOT).as_posix(), "targets": target.int().tolist(), "domain": "dacl"}
            for path, target in zip(dataset.paths, dataset.targets)], dataset.classes


def masked_focal(logits, labels, known, positive_weights, gamma=1.5, emphasis=None):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probability = logits.sigmoid()
    correct = torch.where(labels > 0, probability, 1 - probability)
    weights = positive_weights[None, :] if positive_weights.ndim == 1 else positive_weights
    balanced = torch.where(labels > 0, weights, 1.)
    pointwise = bce * (1 - correct).pow(gamma) * balanced * known
    if emphasis is not None: pointwise = pointwise * emphasis[None, :]
    return (pointwise.sum(1) / known.sum(1).clamp_min(1)).mean()


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); target, score = [], []
    for image, labels, known, *_ in loader:
        score.append(model(image.to(device)).sigmoid().cpu().numpy())
        target.append(torch.where(known > 0, labels, -1).numpy())
    return np.concatenate(target), np.concatenate(score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="facility-presence-target-v2s")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--domain-balance", action="store_true")
    parser.add_argument("--details", type=Path)
    parser.add_argument("--detail-exposure", type=float, default=.5)
    parser.add_argument("--domain-proportions", help="Ordered dacl,damsegment[,codebrim] exposure fractions")
    parser.add_argument("--supplement", type=Path, help="Verified CODEBRIM training manifest; official validation/test remain reserved")
    parser.add_argument("--target-emphasis", type=float, default=1.)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.gamma < 0 or not 0 < args.detail_exposure < 1 or args.target_emphasis <= 0:
        raise ValueError("Invalid run options")
    if args.domain_proportions and not args.domain_balance: raise ValueError("Domain fractions require domain balancing")
    output = ROOT / "runs" / args.name
    if (output / "best.pt").exists(): raise ValueError("Preserve the existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    split = split_supplemental()
    save(output / "SPLIT.json", split)
    original, classes = dacl_items("train", args.imgsz)
    if classes != split["classes"]: raise ValueError("Class order mismatch")
    train = original + [{**item, "domain": "damsegment"} for item in split["train"]]
    full_count = len(train)
    additional = None
    if args.supplement:
        additional = read(args.supplement)
        if additional["split"] != "train" or additional["classes"] != classes or additional["audit"]["status"] != "prepared":
            raise ValueError("Supplement is not audited matching training data")
        if not all(i["split"] == "train" and i["domain"] == "codebrim" for i in additional["items"]):
            raise ValueError("Supplement contains validation/test or other domain")
        train += additional["items"]
        full_count = len(train)
    if args.details:
        details = read(args.details)
        if details["split"] != "train" or details["classes"] != classes: raise ValueError("Detail manifest mismatch")
        allowed = {item["image"] for item in train}
        if any(item["parent_image"] not in allowed or item["parent_split"] != "train" for item in details["items"]):
            raise ValueError("Detail crop parent not in this training split")
        train += details["items"]
    val, _ = dacl_items("val", args.imgsz)
    loaders = {"dacl": DataLoader(FacilityPhotos(val, args.imgsz), batch_size=16, num_workers=4, persistent_workers=True),
               "damsegment": DataLoader(FacilityPhotos(split["val"], args.imgsz), batch_size=16, num_workers=4, persistent_workers=True)}
    supplemental_validation = supplemental_test = None
    if additional:
        supplemental_validation = args.supplement.parent / "val.json"
        supplemental_test = args.supplement.parent / "test.json"
        sets = [additional, read(supplemental_validation), read(supplemental_test)]
        for expected, data in zip(("train", "val", "test"), sets):
            if data["classes"] != classes or data["split"] != expected: raise ValueError("Supplement split/classes mismatch")
        groups = [{i["group_id"] for i in data["items"]} for data in sets]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i+1,3)): raise ValueError("Supplement parent group leakage")
        loaders["codebrim"] = DataLoader(FacilityPhotos(sets[1]["items"], args.imgsz), batch_size=16, num_workers=4, persistent_workers=True)
    target = torch.tensor([item["targets"] for item in train])
    positives, negatives = (target == 1).sum(0), (target == 0).sum(0)
    positive_weights = (negatives / positives.clamp_min(1)).clamp(1, 6)
    # Rare task positives get more exposure, bounded at 3x. No test errors enter this sampler.
    sample_weights = torch.ones(len(train), dtype=torch.float64)
    for label in TARGETS:
        i = classes.index(label)
        value = min(3., max(1., len(train) / (2 * max(1, int(positives[i])))))
        sample_weights = torch.maximum(sample_weights, torch.where(target[:, i] == 1, value, 1.).double())
    if args.details:
        sample_weights[full_count:] *= full_count / max(1, len(train)-full_count) * args.detail_exposure / (1-args.detail_exposure)
    domains = torch.tensor([DOMAINS[item["domain"]] for item in train])
    active_domains = sorted(domains.unique().tolist())
    exposure = [float(x) for x in args.domain_proportions.split(",")] if args.domain_proportions else [1/len(active_domains)]*len(active_domains)
    if len(exposure) != len(active_domains) or any(x <= 0 for x in exposure) or abs(sum(exposure)-1) > .000001:
        raise ValueError("Invalid domain exposure fractions")
    if args.domain_balance:
        # Equal aggregate domain exposure. Positive BCE weights correct each
        # domain's weighted training distribution rather than mixing its prior.
        for domain in active_domains:
            mask = domains == domain
            sample_weights[mask] *= exposure[domain] / sample_weights[mask].sum()
        per_domain = []
        for domain in active_domains:
            weights = sample_weights * (domains == domain)
            pos = ((target == 1) * weights[:, None]).sum(0)
            neg = ((target == 0) * weights[:, None]).sum(0)
            per_domain.append((neg / pos.clamp_min(1e-8)).clamp(.2, 6).float())
        positive_weights = torch.stack(per_domain)
    emphasis = torch.ones(len(classes))
    for label in TARGETS: emphasis[classes.index(label)] = args.target_emphasis
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sample_weights, full_count, replacement=True, generator=generator)
    loader = DataLoader(FacilityPhotos(train, args.imgsz, True), batch_size=args.batch, sampler=sampler,
                         num_workers=4, persistent_workers=True, pin_memory=True)
    device = torch.device("cuda")
    model = build_model(len(classes), pretrained=args.initial is None, architecture=ARCH).to(device)
    if args.initial:
        initial = torch.load(args.initial, map_location="cpu", weights_only=True)
        if initial["architecture"] != ARCH or initial["classes"] != classes or initial["split_sha256"] != sha(output / "SPLIT.json"):
            raise ValueError("Initializer architecture/classes/split mismatch")
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW([{"params": model.features.parameters(), "lr": .00008},
                                   {"params": model.classifier.parameters(), "lr": .0005}], weight_decay=.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.000005)
    scaler = torch.amp.GradScaler("cuda")
    history, best_key, bad_epochs = [], None, 0
    started = time.perf_counter()
    training = {"status": "running", "architecture": ARCH, "classes": classes, "imgsz": args.imgsz,
                "train_dacl": len(original), "train_damsegment": len(split["train"]), "val_dacl": len(val), "val_damsegment": len(split["val"]),
                "reserved_dam_test": split["counts"]["reserved_test"], "requested_epochs": args.epochs,
                "seed": args.seed, "gamma": args.gamma, "positive_weights": positive_weights.tolist(),
                "initial": str(args.initial) if args.initial else "Official ImageNet V2-S; never trained on these dam patches",
                "split_sha256": sha(output / "SPLIT.json"), "selection": "minimax FNR/FPR on both validation domains, test never selected"}
    training.update(domain_balance=args.domain_balance, detail_crops=len(train)-full_count,
                    detail_manifest_sha256=sha(args.details) if args.details else None, target_emphasis=args.target_emphasis,
                    draws_per_epoch=full_count, detail_exposure=args.detail_exposure if args.details else 0,
                    domain_proportions=exposure if args.domain_balance else None,
                    train_codebrim=len(additional["items"]) if additional else 0,
                    supplement_sha256=sha(args.supplement) if additional else None,
                    additional_validation={"path": supplemental_validation.relative_to(ROOT).as_posix(), "sha256": sha(supplemental_validation)} if additional else None,
                    additional_test={"path": supplemental_test.relative_to(ROOT).as_posix(), "sha256": sha(supplemental_test)} if additional else None,
                    validation_domains=list(loaders))
    training["selection"] = "minimax FNR/FPR in every recorded validation domain; test never selected"
    save(output / "TRAINING.json", training)
    for epoch in range(1, args.epochs + 1):
        freeze = epoch == 1 and args.initial is None
        for parameter in model.features.parameters(): parameter.requires_grad = not freeze
        model.train()
        if freeze: model.features.eval()
        total = 0.
        for image, labels, known, domain in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                balance = positive_weights[domain] if args.domain_balance else positive_weights
                loss = masked_focal(model(image.to(device)), labels.to(device), known.to(device), balance.to(device), args.gamma, emphasis.to(device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item() * len(image)
        predictions = {name: predict(model, value, device) for name, value in loaders.items()}
        points = {label: operating_point({domain: (t[:, classes.index(label)].astype(bool), p[:, classes.index(label)])
                                          for domain, (t, p) in predictions.items()}) for label in TARGETS}
        per_ap = {domain: {label: average_precision(t[:, i], p[:, i]) for i, label in enumerate(classes) if (t[:, i] >= 0).all()}
                  for domain, (t, p) in predictions.items()}
        worst = max(point["worst_error"] for point in points.values())
        sum_errors = sum(item[key] for point in points.values() for item in point["domains"].values() for key in ("fnr", "fpr"))
        row = {"epoch": epoch, "elapsed_minutes": (time.perf_counter() - started) / 60,
               "train_loss": total / full_count, "worst_target_error": worst, "sum_target_errors": sum_errors,
               "target_passed": all(point["target_passed"] for point in points.values()), "operating_points": points, "ranking_ap": per_ap}
        history.append(row)
        print(json.dumps({k: row[k] for k in ("epoch", "elapsed_minutes", "train_loss", "worst_target_error", "target_passed")}), flush=True)
        key = (worst, sum_errors)
        if best_key is None or key < best_key:
            best_key, bad_epochs = key, 0
            torch.save({"architecture": ARCH, "state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
                        "classes": classes, "imgsz": args.imgsz, "mean": MEAN, "std": STD, "epoch": epoch,
                        "selection_split": "val", "split_sha256": training["split_sha256"], "worst_target_error": worst}, output / "best.pt")
            save(output / "VALIDATION.json", row)
            for domain, (t, p) in predictions.items():
                save(output / f"validation-{domain}.json", {"split": "val", "targets": t.tolist(), "probabilities": p.tolist(),
                                                            "weights_sha256": sha(output / "best.pt")})
        else: bad_epochs += 1
        save(output / "history.json", history)
        scheduler.step()
        if bad_epochs >= args.patience: break
    training.update(status="complete", actual_epochs=len(history), best_worst_target_error=best_key[0], weights_sha256=sha(output / "best.pt"))
    save(output / "TRAINING.json", training)
    print(json.dumps(training, indent=2), flush=True)


if __name__ == "__main__": main()
