"""A bounded RC2119 positive-only data/localisation study, with frozen evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_rc2119 import sha, write_new
from scripts.audit_rc2119 import read, SAFETY_CLASSES
from scripts.facility_head_lr_study import prepare_data as prepare_original, RETENTION_GATE
from scripts.report_facility_detail import GATE
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.train_facility_target import DOMAINS

NAME = "facility-presence-target-rc2119-positive-regions"
CONTROL = "facility-presence-target-head-lr-low"
REFERENCE = "facility-presence-target-roi-control"
PROTOCOL = "reports/facility-rc-positive-study-protocol.json"
MANIFEST = "data/rc2119-training/positive-study/train.json"
DRAWS = "data/rc2119-training/positive-study/draws.npz"
INITIAL = "runs/facility-presence-target-roi-control/best.pt"
INITIAL_SHA = "0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a"
CONTROL_SHA = "058e6de039da9bb2b35c78e35c1cecf64099c8ac187597804fce99c8ce5afcb7"
NEW_DOMAINS = {**DOMAINS, "rc2119": 3}
EPOCHS, DRAW_COUNT, REPLACEMENTS, SEED = 6, 14248, 704, 56
NEW_SOURCES = (
    "scripts/fetch_rc2119.py", "scripts/audit_rc2119.py", "scripts/screen_rc2119_crops.py",
    "safelog_ai/positive_region_loss.py", "scripts/facility_rc_positive.py",
    "scripts/train_facility_rc_positive.py", "scripts/verify_facility_rc_positive.py",
    "scripts/report_facility_rc_positive.py", "tests/test_rc2119_fetch.py",
    "tests/test_rc2119_audit.py", "tests/test_rc_positive.py",
)


def fixed_draws(original, core_count, supplemental_count):
    if (original.shape != (EPOCHS, DRAW_COUNT) or original.dtype != np.int64
            or not 1 <= supplemental_count <= 200 or core_count <= int(original.max())
            or int(original.min()) < 0):
        raise ValueError("Unexpected original draw budget or source cardinality")
    rng = np.random.default_rng(61)
    result = original.copy()
    for epoch in range(EPOCHS):
        positions = np.sort(rng.choice(DRAW_COUNT, REPLACEMENTS, replace=False))
        chosen = np.arange(REPLACEMENTS, dtype=np.int64) % supplemental_count
        rng.shuffle(chosen)
        result[epoch, positions] = core_count + chosen
    return result


def select_representatives(rows, excluded, limit=100):
    groups = defaultdict(list)
    for row in rows:
        if not row["excluded_reasons"] and row["image"] not in excluded:
            groups[row["group"]].append(row)
    rng = random.Random(61)
    result, used = [], set()
    for category in ("Concrete spalling", "Crack"):
        candidates = []
        for group, values in sorted(groups.items()):
            if group in used: continue
            eligible = [r for r in values if category in r["native_instances"]]
            if eligible: candidates.append(min(eligible, key=lambda r: r["image"]))
        rng.shuffle(candidates)
        for row in candidates[:limit]:
            result.append({**row, "selection_stratum": category}); used.add(row["group"])
    if not result or not any(r["selection_stratum"] == "Concrete spalling" for r in result):
        raise ValueError("No screened spalling positives remain")
    return result


def prepare():
    torch.set_num_threads(4)
    base = ROOT / "data/rc2119-training"
    output = ROOT / MANIFEST
    if output.exists(): raise ValueError("Preserve prepared source study")
    audit = read(base / "source-audit.json"); crop = read(base / "crop-review.json")
    if crop["summary"]["annotation_audit_sha256"] != sha(base / "source-audit.json"):
        raise ValueError("Crop screen is not bound to the source annotation audit")
    selected = select_representatives(audit["items"], set(crop["excluded_images"]))
    destination = base / "positive-study"
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "masks").mkdir(parents=True, exist_ok=True)
    items = []
    codes = audit["summary"]["native_mask_codes"]
    for index, row in enumerate(selected):
        for key, expected in (("image", "source_image_sha256"), ("mask", "source_mask_sha256"), ("annotation", "source_annotation_sha256")):
            if sha(ROOT / row[key]) != row[expected]: raise ValueError("Publisher selected source changed")
        with Image.open(ROOT / row["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if list(image.size) != row["source_size"]: raise ValueError("Declared EXIF registration changed")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        path = destination / "images" / f"rc-positive-{index:03d}.jpg"
        if path.exists(): raise ValueError("Never overwrite prepared source photos")
        image.save(path, quality=95)
        mask = np.zeros((7, 80, 80), dtype=np.uint8)
        with Image.open(ROOT / row["mask"]) as source:
            labels = np.asarray(source)
        for column, native in enumerate(("Crack", "Concrete spalling")):
            # Positive cell means that at least one author foreground pixel
            # falls in its native-resolution footprint; zero remains unknown.
            binary = torch.from_numpy((labels == codes[native]).astype(np.float32))[None, None]
            mask[column] = F.adaptive_max_pool2d(binary, (80, 80))[0, 0].numpy().astype(np.uint8)
        target = [-1] * 7
        for column, native in enumerate(("Crack", "Concrete spalling")):
            if native in row["native_instances"]: target[column] = 1
        if any(mask[k].any() and target[k] != 1 for k in (0, 1)): raise ValueError("Foreground contradicts author photo label")
        pixel = destination / "masks" / f"rc-positive-{index:03d}.npz"
        if pixel.exists(): raise ValueError("Never overwrite prepared source masks")
        # Standard spatial loss is disabled on every new class/channel. Only
        # the separate foreground-positive loss consumes these asserted cells.
        np.savez_compressed(pixel, mask=mask, known=np.zeros(7, np.uint8))
        items.append({"image": path.relative_to(ROOT).as_posix(), "targets": target, "domain": "rc2119",
            "pixel_target": pixel.relative_to(ROOT).as_posix(), "image_sha256": sha(path), "pixel_target_sha256": sha(pixel),
            "source_image": row["image"], "source_mask": row["mask"], "source_annotation": row["annotation"],
            "source_image_sha256": row["source_image_sha256"], "source_mask_sha256": row["source_mask_sha256"],
            "source_annotation_sha256": row["source_annotation_sha256"], "group_id": row["group"],
            "source_size": row["source_size"], "converted_size": list(image.size), "scene_unknown": True,
            "source_split": "train_only_unsplit", "selection_stratum": row["selection_stratum"],
            "positive_cells": [int(mask[k].sum()) for k in range(7)]})
    public = {"schema": "rc2119_positive_region_preparation_v1", "status": "prepared_for_limited_research",
        "source": audit["summary"]["source"], "license": "CC BY 4.0", "selected_photos": len(items),
        "selection_strata": dict(Counter(r["selection_stratum"] for r in items)),
        "positive_photos": {label: sum(r["targets"][k] == 1 for r in items) for k, label in enumerate(SAFETY_CLASSES)},
        "asserted_positive_cells": {label: sum(r["positive_cells"][k] for r in items) for k, label in enumerate(SAFETY_CLASSES)},
        "known_negative_photos": 0, "background_negative_pixels_created": 0, "new_auxiliary_labels": 0,
        "annotation_audit_sha256": sha(base / "source-audit.json"), "crop_review_sha256": sha(base / "crop-review.json"),
        "policy": "Explicit crack/spalling positives only; all absent classes and background unknown. Original author masks supply foreground presence cells only",
        "scope": "TRAIN-only research supplement; no new independent validation/test set or scene-independence claim",
        "limitations": audit["summary"]["limitations"] + crop["summary"]["limitations"]}
    write_new(output, {"split": "train", "classes": list(SAFETY_CLASSES), "items": items, "audit": public})
    public["manifest_sha256"] = sha(output)
    write_new(ROOT / "reports/facility-rc-positive-data.json", public)
    print(json.dumps({k: v for k, v in public.items() if k != "limitations"}), flush=True)


class StudyPhotos(Dataset):
    def __init__(self, data):
        self.core = ResolutionPhotos(data["items"], data["auxiliary"], data["full_count"], DOMAINS, 640)
        self.supplement = ResolutionPhotos(data["supplement"], {}, len(data["supplement"]), NEW_DOMAINS, 640)
    def __len__(self): return len(self.core) + len(self.supplement)
    def __getitem__(self, index):
        if index < len(self.core):
            row = self.core[index]
            return row + (torch.zeros_like(row[4]),)
        row = self.supplement[index - len(self.core)]
        return row[:9] + (index, row[4])


def validate_positive_targets(target):
    if (not isinstance(target, list) or len(target) != 7 or any(type(v) is not int or v not in (-1, 1) for v in target)
            or target[2:] != [-1] * 5 or 1 not in target[:2]):
        raise ValueError("Only explicit primary positives and integer unknown labels are supported")


def load_data():
    torch.set_num_threads(4)
    old_protocol = read(ROOT / "reports/facility-head-lr-study-protocol.json")
    data = prepare_original(ROOT, old_protocol, "low_lr", ROOT / "data/facility-auxiliary-training/train.json")
    manifest = read(ROOT / MANIFEST)
    if manifest["split"] != "train" or manifest["classes"] != list(SAFETY_CLASSES): raise ValueError("Invalid new TRAIN ontology")
    items = manifest["items"]
    if not 1 <= len(items) <= 200 or len({r["group_id"] for r in items}) != len(items): raise ValueError("Invalid source group representatives")
    for row in items:
        validate_positive_targets(row["targets"])
        if row["domain"] != "rc2119":
            raise ValueError("New source must assert only primary positives, no negatives or other-class targets")
        for key, hash_key in (("image", "image_sha256"), ("pixel_target", "pixel_target_sha256"),
                ("source_image", "source_image_sha256"), ("source_mask", "source_mask_sha256"), ("source_annotation", "source_annotation_sha256")):
            path = (ROOT / row[key]).resolve()
            if not path.is_relative_to((ROOT / "data/rc2119-training").resolve()) or sha(path) != row[hash_key]:
                raise ValueError("New prepared/source bytes changed or path escape")
        with np.load(ROOT / row["pixel_target"], allow_pickle=False) as fields:
            if (fields["mask"].shape != (7, 80, 80) or fields["known"].shape != (7,)
                    or fields["known"].any() or fields["mask"][2:].any()
                    or not np.isin(fields["mask"], (0, 1)).all()): raise ValueError("New source has unsupported spatial supervision")
    data["supplement"] = items
    data["candidate_draws"] = fixed_draws(data["epoch_draws"], len(data["items"]), len(items))
    return data


def validate_protocol(protocol):
    if (protocol.get("schema") != "facility_rc_positive_study_v1" or protocol.get("run") != NAME
            or protocol.get("epochs") != EPOCHS or protocol.get("draws_per_epoch") != DRAW_COUNT
            or protocol.get("replacement_draws_per_epoch") != REPLACEMENTS or protocol.get("seed") != SEED
            or protocol.get("positive_region_loss_weight") != .25 or protocol.get("head_lr") != .0001
            or protocol.get("backbone_lr") != .00004 or protocol.get("distillation_weight") != 4.0
            or protocol.get("research_candidate_gate") != GATE or protocol.get("retention_candidate_gate") != RETENTION_GATE):
        raise ValueError("Predeclared candidate conditions changed")
    for name, digest in protocol["source_sha256"].items():
        if sha(ROOT / name) != digest: raise ValueError("Frozen runtime source changed: " + name)
    for name, digest in protocol["input_sha256"].items():
        if sha(ROOT / name) != digest: raise ValueError("Declared study input changed: " + name)
    if sha(ROOT / INITIAL) != INITIAL_SHA or sha(ROOT / f"runs/{CONTROL}/best.pt") != CONTROL_SHA:
        raise ValueError("Original initializer or historical comparator changed")
    return protocol


def declare():
    path = ROOT / PROTOCOL
    if path.exists() or (ROOT / "runs" / NAME).exists(): raise ValueError("Declare only before fresh training")
    prior = read(ROOT / "reports/facility-semantic-study-verification.json")
    if prior["status"] != "passed" or len(prior["source_sha256"]) != 112: raise ValueError("Verified original source112 required")
    data = load_data()
    np.savez_compressed(ROOT / DRAWS, control=data["epoch_draws"], candidate=data["candidate_draws"])
    inputs = {MANIFEST: sha(ROOT / MANIFEST), DRAWS: sha(ROOT / DRAWS),
        "data/rc2119-training/source-audit.json": sha(ROOT / "data/rc2119-training/source-audit.json"),
        "data/rc2119-training/crop-review.json": sha(ROOT / "data/rc2119-training/crop-review.json"),
        "reports/facility-rc-positive-data.json": sha(ROOT / "reports/facility-rc-positive-data.json"),
        "reports/facility-semantic-study-verification.json": sha(ROOT / "reports/facility-semantic-study-verification.json")}
    sources = {**prior["source_sha256"], **{p: sha(ROOT / p) for p in NEW_SOURCES}}
    protocol = {"schema": "facility_rc_positive_study_v1", "declared_utc": datetime.now(timezone.utc).isoformat(),
        "declared_before_training": True, "run": NAME, "control": CONTROL, "reference": REFERENCE,
        "epochs": EPOCHS, "batch_size": 8, "imgsz": 640, "draws_per_epoch": DRAW_COUNT,
        "replacement_draws_per_epoch": REPLACEMENTS, "replacement_fraction": REPLACEMENTS / DRAW_COUNT,
        "seed": SEED, "positive_region_loss_weight": .25, "head_lr": .0001, "backbone_lr": .00004,
        "distillation_weight": 4.0, "distillation_temperature": 2., "auxiliary_weight": .5,
        "initial_weights_sha256": INITIAL_SHA, "control_weights_sha256": CONTROL_SHA,
        "source_sha256": sources, "input_sha256": inputs, "selected_supplement_photos": len(data["supplement"]),
        "other_settings": "Original324-state auxiliary model, frozen47 BN buffer sets/learnable affine, AdamW WD.0002/Cosine6 eta_min5e-6, native80 maps/top32 pool, teacher other-five only",
        "package_comparison_scope": "One new data+foreground-positive-loss package vs completed historical same-budget base recipe; not an isolated data-only or pixel-loss-only causal comparison",
        "unknown_policy": "Absent source photo classes and all source background pixels remain unknown; original labels are unchanged",
        "epoch_selection": "Original source-VAL worst primary FNR/FPR, then sum of all primary FNR/FPR",
        "strict_target": "Each of12 primary class/domain/FNR-FPR cells strictly below.05",
        "candidate_acceptance": "Previous research/retention/small-region gates remain unchanged; no automatic app-profile promotion",
        "research_candidate_gate": GATE, "retention_candidate_gate": RETENTION_GATE,
        "source_test_used": False, "new_training_epochs_planned": 6, "historical_control_retrained": False}
    validate_protocol(protocol); write_new(path, protocol)
    print(json.dumps({"status": "declared", "sources": len(sources), "protocol_sha256": sha(path),
                      "selected_photos": len(data["supplement"]), "new_epochs_planned": 6}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "declare"))
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else declare)()
