"""Original TRAIN draws with a new, training-only native19 spatial task."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import random
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_rc_positive import INITIAL, INITIAL_SHA, CONTROL, CONTROL_SHA, REFERENCE
from scripts.facility_head_lr_study import prepare_data as prepare_original, RETENTION_GATE
from scripts.report_facility_detail import GATE
from scripts.train_facility_target import DOMAINS, read, sha
from scripts.fetch_rc2119 import write_new
from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.presence_classifier import image_transform
from safelog_ai.dense_auxiliary_classifier import ARCH

NAME = "facility-presence-target-dense-native19"
PROTOCOL = "reports/facility-dense-auxiliary-study-protocol.json"
MANIFEST = "data/facility-dense-auxiliary-training/train.json"
EPOCHS, DRAW_COUNT, SEED = 6, 14248, 56
DENSE_WEIGHT = .1
NEW_SOURCES = (
    "safelog_ai/dense_auxiliary_classifier.py", "scripts/prepare_facility_dense_auxiliary.py",
    "scripts/facility_dense_auxiliary.py", "scripts/train_facility_dense_auxiliary.py",
    "scripts/verify_facility_dense_auxiliary.py", "scripts/report_facility_dense_auxiliary.py",
    "tests/test_dense_auxiliary_classifier.py", "tests/test_dense_auxiliary_data.py",
    "tests/test_dense_auxiliary_pipeline.py",
)


def require(condition, message):
    if not condition: raise ValueError(message)


def dense_fields(row):
    path = (ROOT / row["mask"]).resolve()
    require(path.is_relative_to((ROOT / "data/facility-dense-auxiliary-training").resolve())
            and path.is_file() and not path.is_symlink() and sha(path) == row["mask_sha256"],
            "Dense TRAIN mask changed or escaped")
    with np.load(path, allow_pickle=False) as arrays:
        masks, known = arrays["masks"].copy(), arrays["known"].copy()
    require(masks.shape == (19, 80, 80) and known.shape == (19,)
            and masks.dtype == known.dtype == np.uint8
            and np.isin(masks, (0, 1)).all() and np.isin(known, (0, 1)).all()
            and not masks[known == 0].any() and known.tolist() == row["known"],
            "Dense TRAIN supervision is not independent binary19")
    targets = np.asarray(row["targets"])
    require(targets.shape == (19,) and np.isin(targets, (0, 1)).all()
            and np.array_equal(masks.any((1, 2))[known.astype(bool)], targets[known.astype(bool)].astype(bool)),
            "Known dense channel disagrees with original photo tag")
    return masks, known


def load_data():
    torch.set_num_threads(4)
    data = prepare_original(ROOT, read(ROOT / "reports/facility-head-lr-study-protocol.json"),
                            "low_lr", ROOT / "data/facility-auxiliary-training/train.json")
    manifest = read(ROOT / MANIFEST)
    require(manifest.get("split") == "train" and manifest.get("classes") == list(AUX_CLASSES),
            "Dense task requires the original19 TRAIN ontology")
    rows = manifest["items"]
    require(len(rows) == len(data["auxiliary"]) == 6225 and len({r["image"] for r in rows}) == 6225
            and {r["image"] for r in rows} == set(data["auxiliary"]),
            "Dense task must cover exactly the original full DACL TRAIN photos")
    eligible = {r["image"] for r in data["items"][:data["full_count"]] if r["domain"] == "dacl"}
    require(eligible == set(data["auxiliary"]), "DACL full-photo row boundary changed")
    positive, negative, unknown = np.zeros((3, 19), dtype=np.int64)
    for row in rows:
        require(row["targets"] == data["auxiliary"][row["image"]], "Original19 photo tags changed")
        for key, hashkey in (("image", "image_sha256"), ("annotation", "annotation_sha256")):
            path = (ROOT / row[key]).resolve()
            require(path.is_relative_to((ROOT / "data").resolve()) and sha(path) == row[hashkey],
                    "Dense original source bytes changed")
        masks, known = dense_fields(row)
        cells = masks.sum((1, 2), dtype=np.int64)
        positive += cells * known
        negative += (6400 - cells) * known
        unknown += 6400 * (1 - known.astype(np.int64))
    data["dense"] = {r["image"]: r for r in rows}
    data["dense_weights"] = torch.tensor(np.clip(negative / np.maximum(positive, 1), 1, 20), dtype=torch.float32)
    data["dense_counts"] = {c: {"positive": int(positive[k]), "negative": int(negative[k]), "unknown": int(unknown[k])}
                            for k, c in enumerate(AUX_CLASSES)}
    return data


class DensePhotos(Dataset):
    """One shared flip for RGB, original7 masks, and native19 masks.

The existing dataset's random.random, jitter, transform and row contract are
preserved. Loading native19 arrays does not consume any random numbers.
    """
    def __init__(self, data):
        self.items, self.auxiliary, self.full_count = data["items"], data["auxiliary"], data["full_count"]
        self.dense = data["dense"]
        self.transform, self.jitter = image_transform(640), ColorJitter(.1, .1, .1, .01)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(ROOT / item["image"]) as handle: image = handle.convert("RGB")
        with np.load(ROOT / item["pixel_target"], allow_pickle=False) as arrays:
            masks, pixel_known = arrays["mask"].copy(), arrays["known"].copy()
        row = self.dense.get(item["image"]) if index < self.full_count and item["domain"] == "dacl" else None
        if row is None:
            dense, dense_known = np.zeros((19, 80, 80), np.uint8), np.zeros(19, np.uint8)
        else:
            with np.load(ROOT / row["mask"], allow_pickle=False) as arrays:
                dense, dense_known = arrays["masks"].copy(), arrays["known"].copy()
        if random.random() < .5:
            image, masks, dense = ImageOps.mirror(image), masks[:, :, ::-1].copy(), dense[:, :, ::-1].copy()
        targets = torch.tensor(item["targets"], dtype=torch.float32)
        tags = torch.tensor(self.auxiliary.get(item["image"], [-1] * 19), dtype=torch.float32)
        return (self.transform(self.jitter(image)), targets.clamp_min(0), (targets >= 0).float(),
                DOMAINS[item["domain"]], torch.from_numpy(masks).float(), torch.from_numpy(pixel_known).float(),
                tags.clamp_min(0), (tags >= 0).float(), int(index >= self.full_count), index,
                torch.from_numpy(dense).float(), torch.from_numpy(dense_known).float(), int(row is not None))


def validate_protocol(protocol):
    fixed = {"schema": "facility_dense_auxiliary_study_v1", "run": NAME, "architecture": ARCH,
             "epochs": EPOCHS, "draws_per_epoch": DRAW_COUNT, "seed": SEED,
             "dense_spatial_loss_weight": DENSE_WEIGHT, "head_lr": .0001, "backbone_lr": .00004,
             "distillation_weight": 4., "auxiliary_photo_weight": .5,
             "research_candidate_gate": GATE, "retention_candidate_gate": RETENTION_GATE}
    for key, value in fixed.items(): require(protocol.get(key) == value, "Declared dense condition changed: " + key)
    for field in ("source_sha256", "input_sha256"):
        for path, digest in protocol[field].items():
            require(sha(ROOT / path) == digest, "Declared source/input changed: " + path)
    require(sha(ROOT / INITIAL) == INITIAL_SHA and sha(ROOT / f"runs/{CONTROL}/best.pt") == CONTROL_SHA,
            "Original initialization or historical control changed")
    return protocol


def declare():
    require(not (ROOT / PROTOCOL).exists() and not (ROOT / "runs" / NAME).exists(), "Declare before fresh training")
    previous = read(ROOT / "reports/facility-rc-positive-study-verification.json")
    require(previous["status"] == "passed" and previous["runtime_source_count"] == 123, "Original123 sources required")
    data = load_data()
    sources = {**previous["source_sha256"], **{p: sha(ROOT / p) for p in NEW_SOURCES}}
    paths = (MANIFEST, "reports/facility-dense-auxiliary-data-audit.json", "data/facility-dense-auxiliary-training/input-ledger.json",
             "data/facility-spatial-training/train.json", "data/facility-auxiliary-training/train.json",
             "runs/facility-spalling-sampler-plan/paired-draws.npz", "reports/facility-rc-positive-study-verification.json")
    protocol = {"schema": "facility_dense_auxiliary_study_v1", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "declared_before_training": True, "run": NAME, "reference": REFERENCE, "control": CONTROL,
        "architecture": ARCH, "epochs": EPOCHS, "batch_size": 8, "imgsz": 640, "dense_grid": [80, 80],
        "dense_classes": list(AUX_CLASSES), "seed": SEED, "draws_per_epoch": DRAW_COUNT,
        "dense_spatial_loss_weight": DENSE_WEIGHT, "head_lr": .0001, "backbone_lr": .00004,
        "distillation_weight": 4., "distillation_temperature": 2., "auxiliary_photo_weight": .5,
        "dense_positive_weights": data["dense_weights"].tolist(), "dense_train_pixel_counts": data["dense_counts"],
        "eligible_full_train_photos": 6225, "eligible_draws_by_epoch":
            [sum(data["items"][int(i)]["image"] in data["dense"] for i in draws) for draws in data["epoch_draws"]],
        "source_sha256": sources, "input_sha256": {p: sha(ROOT / p) for p in paths},
        "initial_weights_sha256": INITIAL_SHA, "control_weights_sha256": CONTROL_SHA,
        "original_sampling_and_augmentation_unchanged": True, "historical_control_retrained": False,
        "package_comparison_scope": "One new native19 spatial head+loss package versus completed identical-budget low-head-LR recipe; exploratory reused baseline, not a new randomized causal experiment",
        "dense_unknown_policy": "Independent multilabel19; source class-outside-polygon cells are benchmark negatives only on eligible full DACL TRAIN. Other domains, all crops, malformed/conflicted channels unknown; no safe-site inference",
        "other_settings": "Original324 state transfer; new4 training-only states/63499params, zero projection; unchanged original7 photo/spatial and19photoaux, fixed47BN/learnableaffine, KD4/T2, AdamW WD.0002 Cosine6 eta_min5e-6; same draw arrays and seeds57/58/156-158",
        "checkpoint_policy": "Preserve selected dense328-state checkpoint plus strict derived324-state inference export; verify exactly identical public predictions before old evaluator",
        "epoch_selection": "Original three-source VAL worst primary FNR/FPR then sum12 errors",
        "research_candidate_gate": GATE, "retention_candidate_gate": RETENTION_GATE,
        "source_test_used": False, "app_model_promoted": False}
    validate_protocol(protocol); write_new(ROOT / PROTOCOL, protocol)
    print(json.dumps({"status": "declared", "sources": len(sources), "protocol_sha256": sha(ROOT / PROTOCOL),
                      "eligible_photos": 6225, "new_epochs_planned": 6}), flush=True)


if __name__ == "__main__": declare()
