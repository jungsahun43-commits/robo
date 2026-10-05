"""Fixed TRAIN provenance, sampling and coarse supervision for a resolution pair.

The existing photo tags and 80-cell pixel truth remain unchanged. This module
does not create higher resolution targets or select data by held-out errors.
"""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path, PurePosixPath
import re

import torch
from torch.nn import functional as F

from safelog_ai.auxiliary_classifier import AUX_CLASSES, ARCH as AUX_ARCH
from safelog_ai.presence_classifier import image_transform
from scripts.train_facility_context import SpatialPhotos
from scripts.train_facility_target import (DOMAINS, TARGETS, read, sha,
                                           split_supplemental, dacl_items)

CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity"]
RES_ARCH = "lraspp_mobilenet_facility_resolution_grid_v1"
INITIAL_SHA = "0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a"
SOURCE_FILES = (
    "scripts/train_facility_resolution.py", "scripts/facility_resolution_study.py",
    "scripts/train_facility_context.py", "scripts/train_facility_target.py",
    "scripts/train_facility_presence.py", "scripts/facility_error_target.py",
    "safelog_ai/spatial_classifier.py", "safelog_ai/auxiliary_classifier.py",
    "safelog_ai/resolution_classifier.py", "safelog_ai/presence_classifier.py",
    "safelog_ai/context_classifier.py",
)
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class ResolutionPhotos(SpatialPhotos):
    """Keep original row, augmentation, mask and known-label semantics."""
    def __init__(self, items, auxiliary=None, full_count=None, domain_map=None, imgsz=640):
        if type(imgsz) is not int or imgsz not in (640, 960):
            raise ValueError("The declared resolution pair uses only 640 and 960")
        super().__init__(items, auxiliary, full_count, domain_map)
        self.transform = image_transform(imgsz)
        self.imgsz = imgsz


def loss_grid(logits, target_shape):
    """Preserve identical grids, or area-average logits down to existing truth.

    No target is resized. In this study both declared models return 80x80 maps;
    the downsampling path is also checked independently for raw 120x120 maps.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("Spatial logits must be a four-dimensional floating tensor")
    try:
        shape = tuple(target_shape)
    except TypeError as error:
        raise ValueError("Target shape must be a two- or four-dimensional shape") from error
    if len(shape) not in (2, 4) or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("Target shape must contain positive integer dimensions")
    if len(shape) == 4 and tuple(logits.shape[:2]) != shape[:2]:
        raise ValueError("Pixel target batch/class dimensions differ from logits")
    height, width = shape[-2:]
    if height > logits.shape[-2] or width > logits.shape[-1]:
        raise ValueError("Pixel supervision must never upsample logits or invent detail")
    if tuple(logits.shape[-2:]) == (height, width):
        return logits
    return F.interpolate(logits, size=(height, width), mode="area")


def _relative_path(value, root=None, *, data_only=False):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or PurePosixPath(value).is_absolute()
            or (data_only and not value.startswith("data/"))):
        raise ValueError("Expected a relative local provenance path")
    if root is None:
        return value
    root = Path(root).resolve(); path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Provenance path leaves the workspace or is missing")
    if data_only and not path.is_relative_to((root / "data").resolve()):
        raise ValueError("Training record leaves the local data directory")
    return path


def _hash(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Expected a lowercase SHA256 provenance value")


def validate_items(manifest, classes=CLASSES, root=None):
    """Verify existing TRAIN rows and parents without editing their targets."""
    if (not isinstance(manifest, dict) or manifest.get("split") != "train"
            or manifest.get("classes") != list(classes)):
        raise ValueError("Resolution study requires the original seven-class TRAIN manifest")
    items = manifest.get("items"); full = manifest.get("full_count")
    if (not isinstance(items, list) or not items or type(full) is not int
            or not 0 < full <= len(items)):
        raise ValueError("Invalid original full/crop row boundary")
    images, pixels = set(), set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("domain") not in DOMAINS:
            raise ValueError("Unknown original TRAIN source")
        if item.get("split", "train") != "train":
            raise ValueError("Held-out row in the resolution TRAIN manifest")
        for field, seen in (("image", images), ("pixel_target", pixels)):
            value = item.get(field); _relative_path(value, root, data_only=True)
            if value in seen:
                raise ValueError("Duplicate training image or pixel-target path")
            seen.add(value)
        target = item.get("targets")
        if (not isinstance(target, list) or len(target) != len(classes)
                or any(type(value) is not int or value not in (-1, 0, 1) for value in target)):
            raise ValueError("TRAIN targets must preserve seven integer asserted/unknown values")
        if index < full and item.get("parent_image") is not None:
            raise ValueError("Original full TRAIN row must not be a derived child")
    parents = {item["image"]: item for item in items[:full]}
    for item in items[full:]:
        parent = parents.get(item.get("parent_image"))
        if parent is None or item.get("parent_split") != "train":
            raise ValueError("Derived crop has no original TRAIN parent")
        if item["domain"] != parent["domain"]:
            raise ValueError("Derived crop changes its source domain")
    return items


def build_sampling(items, full_count, classes=CLASSES, domain_map=DOMAINS,
                   proportions=(.7, .1, .2)):
    """Reproduce frozen source/full/crop/positive exposure and photo weights."""
    if (type(full_count) is not int or not 0 < full_count <= len(items)
            or len(proportions) != len(domain_map)
            or sorted(domain_map.values()) != list(range(len(domain_map)))
            or any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in proportions)
            or not math.isclose(sum(proportions), 1., abs_tol=1e-12)):
        raise ValueError("Invalid sampling boundary or source proportions")
    if any(not isinstance(item.get("targets"), list) or len(item["targets"]) != len(classes)
           or any(type(v) is not int or v not in (-1, 0, 1) for v in item["targets"]) for item in items):
        raise ValueError("Sampling must preserve integer asserted/unknown targets")
    labels = torch.tensor([item["targets"] for item in items])
    if (labels.shape != (len(items), len(classes))
            or not ((labels == -1) | (labels == 0) | (labels == 1)).all()):
        raise ValueError("Sampling must use unchanged asserted/unknown targets")
    try:
        domains = torch.tensor([domain_map[item["domain"]] for item in items], dtype=torch.long)
    except (KeyError, TypeError) as error:
        raise ValueError("Sampling has an undeclared source") from error
    sampling = torch.ones(len(items), dtype=torch.float64)
    for label in TARGETS:
        k = list(classes).index(label)
        ratio = min(3., max(1., len(items) / (2 * max(1, int((labels[:, k] == 1).sum())))))
        sampling = torch.maximum(sampling, torch.where(labels[:, k] == 1, ratio, 1.).double())
    if full_count < len(items):
        sampling[full_count:] *= full_count / (len(items) - full_count) * .25 / .75
    weights = []
    for domain, exposure in enumerate(proportions):
        chosen = domains == domain
        if not chosen.any():
            raise ValueError("Every declared source must contain TRAIN rows")
        sampling[chosen] *= exposure / sampling[chosen].sum()
        weight = sampling * chosen
        positive = ((labels == 1) * weight[:, None]).sum(0)
        negative = ((labels == 0) * weight[:, None]).sum(0)
        weights.append((negative / positive.clamp_min(1e-8)).clamp(.2, 6).float())
    return {"labels": labels, "domains": domains, "sampling": sampling,
            "photo_weights": torch.stack(weights)}


def supervision_weights(manifest, auxiliary_manifest):
    counts = manifest["audit"]["per_label_pixel_cells"]
    pixel = torch.tensor([min(20., max(1., counts[c]["negative"] / max(1, counts[c]["positive"])))
                          for c in manifest["classes"]], dtype=torch.float32)
    tags = torch.tensor([item["targets"] for item in auxiliary_manifest["items"]])
    if tags.ndim != 2 or tags.shape[1] != len(AUX_CLASSES) or not ((tags == 0) | (tags == 1)).all():
        raise ValueError("Auxiliary weights need the original known 19 source tags")
    auxiliary = ((tags == 0).sum(0) / (tags == 1).sum(0).clamp_min(1)).clamp(.2, 6).float()
    return {"pixel_weights": pixel, "auxiliary_weights": auxiliary}


def validate_protocol(protocol, args, root=None):
    """Verify protocol alone with (protocol, root), or CLI with three arguments.

    The two-argument preflight form returns an unchanged protocol copy. The
    three-argument trainer form also checks the CLI and returns its input/arch.
    """
    cli = args if root is not None else None
    root = Path(root if root is not None else args).resolve()
    if not isinstance(protocol, dict) or protocol.get("schema") != "facility_resolution_study_protocol_v1":
        raise ValueError("Expected the immutable resolution study protocol")
    if protocol.get("declared_before_training") is not True:
        raise ValueError("The resolution pair must be declared before training")
    if (protocol.get("control") != "facility-presence-target-resolution-control"
            or protocol.get("treatment") != "facility-presence-target-resolution-highres"
            or (cli is not None and (args.variant not in ("control", "highres")
            or args.name != protocol["control" if args.variant == "control" else "treatment"]))):
        raise ValueError("Run name or input variant differs from the declared resolution pair")
    fixed = {"reference": "facility-presence-target-roi-control",
             "architecture_by_variant": {"control": AUX_ARCH, "highres": RES_ARCH},
             "imgsz_by_variant": {"control": 640, "highres": 960},
             "classes": CLASSES, "auxiliary_classes": list(AUX_CLASSES),
             "seed": 56, "requested_epochs": 6, "patience": 6, "batch_size": 8,
             "draws_per_epoch": 14248, "backbone_lr": .00004, "head_lr": .00025,
             "auxiliary_weight": .5, "target_ranking_weight": 0.,
             "domain_proportions": [.7, .1, .2],
             "loader_randomness": {"sampler_seed": 56, "training_worker_seed": 57,
                                   "post_model_seed": 58,
                                   "validation_worker_seeds": {d: 156 + k for d, k in DOMAINS.items()}},
             "initial_weights_sha256": INITIAL_SHA}
    for key, expected in fixed.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Declared resolution condition changed: {key}")
    fields = {"seed": "seed", "requested_epochs": "epochs", "patience": "patience",
              "batch_size": "batch", "draws_per_epoch": "draws_per_epoch",
              "backbone_lr": "backbone_lr", "head_lr": "head_lr", "auxiliary_weight": "auxiliary_weight"}
    if cli is not None:
        for key, argument in fields.items():
            if getattr(args, argument, None) != protocol[key]:
                raise ValueError(f"CLI argument differs from the protocol: {argument}")
    for field in ("initial_weights_sha256", "core_spatial_manifest_sha256", "auxiliary_manifest_sha256"):
        _hash(protocol.get(field))
    if cli is None and protocol.get("reference") != "facility-presence-target-roi-control":
        raise ValueError("The original ROI reference changed")
    initial = args.initial if cli is not None else root / "runs" / protocol["reference"] / "best.pt"
    auxiliary = args.auxiliary_manifest if cli is not None else root / "data/facility-auxiliary-training/train.json"
    if not initial or sha(Path(initial)) != protocol["initial_weights_sha256"]:
        raise ValueError("The frozen ROI auxiliary initializer changed")
    if not auxiliary or sha(Path(auxiliary)) != protocol["auxiliary_manifest_sha256"]:
        raise ValueError("The original native auxiliary TRAIN manifest changed")
    sources = protocol.get("source_sha256")
    if not isinstance(sources, dict) or not set(SOURCE_FILES).issubset(sources):
        raise ValueError("Protocol omits a critical runtime source hash")
    for relative, expected in sources.items():
        path = _relative_path(relative, root); _hash(expected)
        if sha(path) != expected:
            raise ValueError(f"Frozen runtime source changed: {relative}")
    if cli is None:
        return deepcopy(protocol)
    return {"imgsz": protocol["imgsz_by_variant"][args.variant],
            "architecture": protocol["architecture_by_variant"][args.variant]}


def prepare_data(root, protocol, auxiliary_manifest_path=None):
    """Reuse fixed source splits, core masks, native tags and sample weights."""
    root = Path(root).resolve()
    core_path = root / "data/facility-spatial-training/train.json"
    if sha(core_path) != protocol["core_spatial_manifest_sha256"]:
        raise ValueError("Original coarse TRAIN supervision changed")
    manifest = read(core_path); items = validate_items(manifest, root=root)
    classes = manifest["classes"]; full = manifest["full_count"]
    if classes != protocol["classes"]:
        raise ValueError("Class order differs from the protocol")
    original, source_classes = dacl_items("train", 640)
    if source_classes != classes:
        raise ValueError("DACL original class order changed")
    split = split_supplemental()
    code_train = read(root / "data/codebrim-training/train.json")
    code_val = read(root / "data/codebrim-training/val.json")
    code_test = read(root / "data/codebrim-training/test.json")
    for expected, data in zip(("train", "val", "test"), (code_train, code_val, code_test)):
        if data["split"] != expected or data["classes"] != classes:
            raise ValueError("CODEBRIM source split or classes changed")
    groups = [{item["group_id"] for item in data["items"]} for data in (code_train, code_val, code_test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("CODEBRIM parent-group leakage")
    base = original + [{**item, "domain": "damsegment"} for item in split["train"]] + code_train["items"]
    if [item["image"] for item in base] != [item["image"] for item in items[:full]]:
        raise ValueError("Original TRAIN split or full-row order changed")
    if any(a["targets"] != b["targets"] or a["domain"] != b["domain"]
           for a, b in zip(base, items[:full])):
        raise ValueError("Original TRAIN asserted/unknown tags or source changed")
    auxiliary_path = Path(auxiliary_manifest_path or root / "data/facility-auxiliary-training/train.json").resolve()
    if not auxiliary_path.is_relative_to(root / "data") or sha(auxiliary_path) != protocol["auxiliary_manifest_sha256"]:
        raise ValueError("Original auxiliary TRAIN path or bytes changed")
    auxiliary_manifest = read(auxiliary_path); aux_items = auxiliary_manifest["items"]
    if (auxiliary_manifest.get("split") != "train"
            or auxiliary_manifest.get("classes") != list(AUX_CLASSES)
            or auxiliary_manifest.get("audit", {}).get("status") != "prepared"
            or len(aux_items) != len(original)
            or len({item["image"] for item in aux_items}) != len(aux_items)
            or {item["image"] for item in aux_items} != {item["image"] for item in original}):
        raise ValueError("Native tags must cover exactly the original DACL TRAIN photos")
    for item in aux_items:
        if item.get("split") != "train" or item.get("domain") != "dacl":
            raise ValueError("Held-out or other-source auxiliary row")
        path = _relative_path(item["annotation"], root, data_only=True)
        _hash(item["annotation_sha256"])
        if sha(path) != item["annotation_sha256"]:
            raise ValueError("Original native TRAIN annotation changed")
        if (not isinstance(item.get("targets"), list) or len(item["targets"]) != len(AUX_CLASSES)
                or any(type(v) is not int or v not in (0, 1) for v in item["targets"])):
            raise ValueError("Native auxiliary labels must stay known integer tags")
    auxiliary = {item["image"]: item["targets"] for item in aux_items}
    return {"manifest": manifest, "manifest_path": core_path, "items": items,
            "full_count": full, "classes": classes, "original": original,
            "split": split, "code_train": code_train, "code_val": code_val,
            "auxiliary_manifest": auxiliary_manifest, "auxiliary_manifest_path": auxiliary_path,
            "auxiliary": auxiliary,
            "sampling_data": build_sampling(items, full, classes, DOMAINS, tuple(protocol["domain_proportions"])),
            "supervision_weights": supervision_weights(manifest, auxiliary_manifest)}
