"""Learn all seven facility labels per photo, using ONLY the existing train split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".config/torch"))
from safelog_ai.presence_classifier import build_model, image_transform, MEAN, STD


class Photos(Dataset):
    def __init__(self, split: str, size: int, training: bool = False):
        self.base = ROOT / "data/dacl10k-yolo"
        self.classes = list(yaml.safe_load((self.base / "data.yaml").read_text())["names"].values())
        self.paths = sorted((self.base / "images" / split).glob("*.jpg"))
        self.targets = []
        for path in self.paths:
            target = torch.zeros(len(self.classes))
            for line in (self.base / "labels" / split / f"{path.stem}.txt").read_text().splitlines():
                target[int(line.split()[0])] = 1
            self.targets.append(target)
        self.transform = image_transform(size, training)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            data = self.transform(image.convert("RGB"))
        return data, self.targets[i], self.paths[i].name


def average_precision(target: np.ndarray, scores: np.ndarray) -> float | None:
    """Non-interpolated binary AP, grouped at tied score boundaries."""
    positives = target.sum()
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    target, scores = target[order], scores[order]
    ends = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tp = target.cumsum()[ends]
    recall, precision = tp / positives, tp / (ends + 1)
    return float((np.diff(np.r_[0., recall]) * precision).sum())


@torch.inference_mode()
def scores(model, loader, device):
    model.eval()
    targets, probabilities, names = [], [], []
    for image, target, name in loader:
        probabilities.append(model(image.to(device)).sigmoid().cpu().numpy())
        targets.append(target.numpy())
        names.extend(name)
    return np.concatenate(targets), np.concatenate(probabilities), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default="facility-presence")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--hard-negatives", action="store_true", help="오답 점수가 높은 음성 항목의 손실을 강화")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise SystemExit("Experiment name must be a single directory name")
    output = ROOT / "runs" / args.name
    if (output / "best.pt").exists():
        raise SystemExit("Existing photo classifier experiment; preserve it before starting another run")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    train, val = Photos("train", args.imgsz, True), Photos("val", args.imgsz)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4,
                              pin_memory=device.type == "cuda", persistent_workers=True, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=4, persistent_workers=True)
    model = build_model(len(train.classes), pretrained=args.initial is None).to(device)
    if args.initial:
        initial = torch.load(args.initial, map_location="cpu", weights_only=True)
        if initial["classes"] != train.classes:
            raise ValueError("Initial classifier class order mismatch")
        model.load_state_dict(initial["state_dict"])
    positives = torch.stack(train.targets).sum(0)
    weights = ((len(train) - positives) / positives.clamp_min(1)).clamp(1, 6).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW([{"params": model.features.parameters(), "lr": .00002 if args.initial else .00005},
                                   {"params": model.classifier.parameters(), "lr": .0001 if args.initial else .0005}], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.000005)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best, bad_epochs = [], -1., 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for param in model.features.parameters():
            param.requires_grad = epoch > 1 or args.initial is not None
        model.train()
        if epoch == 1 and args.initial is None:
            model.features.eval()  # Frozen features also keep pretrained BatchNorm statistics.
        total = 0.
        for image, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits, labels = model(image.to(device)), target.to(device)
                if args.hard_negatives:
                    # Use train labels only. No validation/test photos enter the optimizer.
                    # Hard negatives receive up to 3x weight; positive targets remain 1x.
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                    emphasis = 1 + 2 * logits.detach().sigmoid() * (1 - labels)
                    loss = (bce * emphasis).mean()
                else:
                    loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total += loss.item() * len(image)
        target, probability, names = scores(model, val_loader, device)
        per_class = {name: average_precision(target[:, i], probability[:, i]) for i, name in enumerate(train.classes)}
        metric = float(np.mean([v for v in per_class.values() if v is not None]))
        row = {"epoch": epoch, "elapsed_minutes": (time.perf_counter() - started) / 60,
               "train_loss": total / len(train), "val_macro_average_precision": metric, "per_class": per_class}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metric > best:
            best, bad_epochs = metric, 0
            torch.save({"architecture": "efficientnet_b0_multilabel_v1", "state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
                        "classes": train.classes, "imgsz": args.imgsz, "mean": MEAN, "std": STD,
                        "epoch": epoch, "selection_split": "val", "val_macro_average_precision": metric}, output / "best.pt")
        else:
            bad_epochs += 1
        (output / "history.json").write_text(json.dumps(history, indent=2))
        scheduler.step()
        if bad_epochs >= args.patience:
            break
    (output / "TRAINING.json").write_text(json.dumps({"status": "complete", "train_images": len(train), "val_images": len(val),
        "seed": 42, "requested_epochs": args.epochs, "actual_epochs": len(history), "imgsz": args.imgsz,
        "best_val_macro_average_precision": best, "label_type": "multi-label photo presence, not defect localization",
        "initial_weights": str(args.initial) if args.initial else None, "hard_negatives": args.hard_negatives,
        "pretrained_weights": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
        "pretrained_documentation": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html",
        "training_positive_photos": dict(zip(train.classes, positives.int().tolist()))}, indent=2))


if __name__ == "__main__":
    main()
