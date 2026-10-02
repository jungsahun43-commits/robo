"""Replace selected TRAIN detail rows with deterministic small-region context crops.

Original full-photo rows and validation/test data are never modified. The derived
images, pixel maps, annotation hashes and detailed manifest stay under ignored data/.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_facility_data import DACL_MAPPING, NAMES
from scripts.prepare_facility_detail import tile_targets
from scripts.train_facility_target import dacl_items, read, save, sha, split_supplemental

RECIPE = "small_source_region_context_v1"
SMALL_FRACTION = .01
MIN_SIDE_FRACTION = .2
MAX_SIDE_FRACTION = 1 / 3
BBOX_CONTEXT_FACTOR = 2.
MIN_POSITIVE_PIXELS = 16


def validate_manifest(manifest, expected_base):
    """Check original TRAIN order/labels and every detail parent's provenance."""
    if manifest.get("split") != "train" or manifest.get("classes") != NAMES:
        raise ValueError("Expected original seven-class TRAIN spatial manifest")
    items = manifest.get("items", [])
    full_count = manifest.get("full_count")
    if full_count != len(expected_base) or len(items) <= full_count:
        raise ValueError("Original full rows/count changed")
    for item, expected in zip(items[:full_count], expected_base):
        for key, value in expected.items():
            if item.get(key) != value:
                raise ValueError("Original full row order/metadata/labels changed")
    parents = {item["image"]: item for item in items[:full_count]}
    if len(parents) != full_count:
        raise ValueError("Duplicate original full image")
    for item in items:
        targets = item.get("targets", [])
        if len(targets) != 7 or any(t not in (-1, 0, 1) for t in targets):
            raise ValueError("Invalid seven-class partial labels")
    for item in items[full_count:]:
        parent = parents.get(item.get("parent_image"))
        if parent is None or item.get("parent_split") != "train":
            raise ValueError("Detail parent outside verified TRAIN")
        if item.get("domain") != parent["domain"] or parent["domain"] == "codebrim":
            raise ValueError("Unsupported detail domain")
        for child, full in zip(item["targets"], parent["targets"]):
            if (full == -1 and child != -1) or (full == 0 and child == 1):
                raise ValueError("Detail invents a parent-unknown/absent class")
        box = item.get("box_in_parent", [])
        if len(box) != 4 or not all(isinstance(v, int) for v in box) or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid original detail box")
    return parents


def context_box(mask):
    """Place a bounded context square on a real positive of the largest component.

    The square is 1/5..1/3 of the parent short side. A small component gets twice
    its bounding-box extent where possible. A very long thin component is sampled
    locally rather than turning the crop back into an entire photograph.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("A nonempty two-dimensional source mask is required")
    height, width = mask.shape
    count, components, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    component = 1 + int(np.argmax(stats[1:count, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(components == component)
    cx, cy = centroids[component]
    closest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    center = (int(xs[closest]), int(ys[closest]))
    short = min(width, height)
    minimum = max(1, math.ceil(short * MIN_SIDE_FRACTION))
    maximum = max(minimum, math.ceil(short * MAX_SIDE_FRACTION))
    extent = max(int(stats[component, cv2.CC_STAT_WIDTH]), int(stats[component, cv2.CC_STAT_HEIGHT]))
    side = min(short, max(minimum, min(maximum, math.ceil(extent * BBOX_CONTEXT_FACTOR))))
    left = min(max(0, center[0] - side // 2), width - side)
    top = min(max(0, center[1] - side // 2), height - side)
    return (left, top, left + side, top + side), center


def crop_targets(masks, box, parent_targets):
    """Keep unasserted classes unknown; do not propagate full-photo positives."""
    targets = tile_targets(masks, box, minimum=MIN_POSITIVE_PIXELS)
    for k, full in enumerate(parent_targets):
        if full == -1:
            targets[k] = -1
        elif full == 0 and targets[k] == 1:
            raise ValueError("Crop positive contradicts an asserted absent parent label")
    return targets


def coarse_targets(masks, box, targets):
    result = np.zeros((7, 80, 80), dtype=np.uint8)
    known = np.array([int(t == 0) for t in targets], dtype=np.uint8)
    conflicts = 0
    for k, mask in enumerate(masks):
        if mask is None or targets[k] < 0:
            continue
        region = Image.fromarray(np.uint8(mask) * 255).crop(box)
        array = np.asarray(region.resize((640, 640), Image.Resampling.BOX)) > 0
        coarse = array.reshape(80, 8, 80, 8).max((1, 3)).astype(np.uint8)
        if bool(coarse.any()) != bool(targets[k]):
            known[k] = 0
            conflicts += 1
            continue
        result[k] = coarse
        known[k] = 1
    return result, known, conflicts


def confined(relative):
    path = (ROOT / str(relative).replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("Source must remain inside the data directory")
    return path


def orientation_geometry(source, document, record, parent_size, maximum_side):
    """Verify the same EXIF transform, decoded hash and resize used by preparation."""
    with Image.open(source) as handle:
        corrected = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = corrected.size
    digest = hashlib.sha256(f"{width}x{height}:".encode() + corrected.tobytes()).hexdigest()
    resized = corrected.copy()
    resized.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
    checks = {
        "annotation_matches_exif_corrected_dimensions": (document["imageWidth"], document["imageHeight"]) == corrected.size,
        "decoded_source_matches_original_record_pixel_sha256": digest == record.get("pixel_sha256"),
        "processed_parent_matches_preparation_thumbnail_dimensions": parent_size == resized.size,
    }
    return {**checks, "verified": all(checks.values())}


def source_masks(parent, records, orientation_checks=None):
    """Read publisher TRAIN truth aligned with the unchanged processed parent."""
    path = confined(parent["image"])
    with Image.open(path) as handle:
        width, height = handle.size
    domain = parent["domain"]
    if domain == "dacl":
        record = records.get(path.stem)
        if record is None or record["split"] != "train":
            raise ValueError("Unverified DACL TRAIN parent")
        source = confined("data/" + record["source"])
        if source.parent.name != "train" or source.parent.parent.name != "images":
            raise ValueError("DACL publisher image is not TRAIN")
        annotation = source.parent.parent.parent / "annotations/train" / f"{source.stem}.json"
        digest = sha(annotation)
        document = read(annotation)
        if sha(annotation) != digest:
            raise ValueError("Source annotation changed during preparation")
        if record.get("exif_orientation_corrected"):
            preparation = read(ROOT / "data/dacl10k-yolo/PREPARATION.json")
            checked = orientation_geometry(source, document, record, (width, height), preparation["max_side"])
            if orientation_checks is not None:
                orientation_checks[parent["image"]] = checked
            if not checked["verified"]:
                return None  # Preserve existing rows; no automatic source geometry repair.
        aw, ah = document["imageWidth"], document["imageHeight"]
        if aw <= 0 or ah <= 0:
            raise ValueError("Invalid publisher image dimensions")
        pil_masks = [Image.new("1", (width, height)) for _ in NAMES]
        for shape in document["shapes"]:
            label = DACL_MAPPING.get(shape["label"])
            if label is None:
                continue
            points = shape["points"]
            if shape.get("shape_type", "polygon") != "polygon" or len(points) < 3 or not all(len(p) == 2 and all(math.isfinite(float(v)) for v in p) for p in points):
                raise ValueError("Invalid publisher polygon")
            adjusted = [(float(x) * width / aw, float(y) * height / ah) for x, y in points]
            ImageDraw.Draw(pil_masks[NAMES.index(label)]).polygon(adjusted, fill=1)
        masks = [np.asarray(mask, dtype=bool) for mask in pil_masks]
    elif domain == "damsegment":
        if parent.get("target_split") != "train" or parent.get("split") != "train":
            raise ValueError("DamSegment parent is not verified TRAIN")
        source = confined(parent["source"])
        if "Damage Detection" not in source.parts:
            return None  # No spatial positives are invented for classification photos.
        level = {"E": "Easy", "M": "Medium", "H": "Hard"}[source.stem[0]]
        annotation = ROOT / "data/damsegment/source/Damage Segmentaion" / level / "Labels/Mask" / f"{source.stem}_mask.png"
        digest = sha(annotation)
        with Image.open(annotation) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        if sha(annotation) != digest or rgb.shape[:2] != (height, width):
            raise ValueError("Dam mask changed or is not aligned with parent")
        masks = [np.all(rgb == color, axis=-1) for color in ((255, 0, 0), (0, 0, 255))] + [None] * 5
    else:
        return None
    for mask, target in zip(masks, parent["targets"]):
        if mask is not None and target >= 0 and bool(mask.any()) != bool(target):
            raise ValueError("Publisher mask/photo-label conflict; do not rewrite source truth")
    return masks, annotation.relative_to(ROOT).as_posix(), digest


def choose_replacement(parent, masks, child_indices, items):
    """At most one positive row per parent, smaller target area first."""
    candidates = []
    for k in (0, 1):
        mask = masks[k]
        if mask is None or parent["targets"][k] != 1:
            continue
        area = int(mask.sum())
        if area < MIN_POSITIVE_PIXELS or area / mask.size > SMALL_FRACTION:
            continue
        positive_children = [index for index in child_indices if items[index]["targets"][k] == 1]
        if positive_children:
            candidates.append((area / mask.size, k, positive_children[0]))
    for fraction, k, index in sorted(candidates):
        box, center = context_box(masks[k])
        old_box = items[index]["box_in_parent"]
        if (box[2] - box[0]) * (box[3] - box[1]) >= (old_box[2] - old_box[0]) * (old_box[3] - old_box[1]):
            continue
        targets = crop_targets(masks, box, parent["targets"])
        if targets[k] == 1:
            return index, k, fraction, box, center, targets
    return None


def main():
    original_path = ROOT / "data/facility-spatial-training/train.json"
    original_sha = sha(original_path)
    manifest = read(original_path)
    original, classes = dacl_items("train", 640)
    split = split_supplemental()
    code = read(ROOT / "data/codebrim-training/train.json")
    if code["split"] != "train" or code["classes"] != classes:
        raise ValueError("CODEBRIM TRAIN contract changed")
    expected = original + [{**item, "domain": "damsegment"} for item in split["train"]] + code["items"]
    parents = validate_manifest(manifest, expected)
    records = {r["stem"]: r for r in read(ROOT / "data/dacl10k-yolo/records.json") if r["split"] == "train"}
    output = ROOT / "data/facility-small-region-training"
    if (output / "train.json").exists():
        raise ValueError("Preserve the existing prepared experiment; output manifest already exists")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    items = [dict(item) for item in manifest["items"]]
    children = defaultdict(list)
    for index, item in enumerate(items[manifest["full_count"]:], manifest["full_count"]):
        children[item["parent_image"]].append(index)
    annotations = {}
    replacements = []
    orientation_checks = {}
    conflicts = 0
    for number, parent in enumerate(parents.values(), 1):
        if parent["image"] not in children or parent["domain"] not in ("dacl", "damsegment"):
            continue
        loaded = source_masks(parent, records, orientation_checks)
        if loaded is None:
            continue
        masks, annotation, annotation_sha = loaded
        annotations[annotation] = annotation_sha
        selected = choose_replacement(parent, masks, children[parent["image"]], items)
        if selected is None:
            continue
        index, k, fraction, box, center, targets = selected
        image_path = output / "images" / f"small-{index:05d}.jpg"
        with Image.open(confined(parent["image"])) as handle:
            crop = handle.convert("RGB").crop(box)
        crop.thumbnail((512, 512), Image.Resampling.LANCZOS)
        crop.save(image_path, quality=95)
        pixel, known, conflict = coarse_targets(masks, box, targets)
        conflicts += conflict
        pixel_path = output / "masks" / f"{index:05d}.npz"
        np.savez_compressed(pixel_path, mask=pixel, known=known)
        old = items[index]
        items[index] = {**old, "image": image_path.relative_to(ROOT).as_posix(),
                        "pixel_target": pixel_path.relative_to(ROOT).as_posix(), "targets": targets,
                        "box_in_parent": list(box), "augmentation": RECIPE,
                        "replacement_of_image": old["image"], "source_annotation": annotation,
                        "source_annotation_sha256": annotation_sha, "small_region_target": classes[k],
                        "source_positive_fraction": fraction, "positive_center_in_parent": list(center)}
        replacements.append({"row": index, "domain": parent["domain"], "target": classes[k],
                             "old_targets": old["targets"], "new_targets": targets})
        if len(replacements) % 100 == 0:
            print(f"Replaced {len(replacements)} TRAIN rows; inspected {number}/{len(parents)} parents", flush=True)
    if not replacements:
        raise ValueError("No eligible small source regions; preserve original manifest")
    if sha(original_path) != original_sha or items[:manifest["full_count"]] != manifest["items"][:manifest["full_count"]]:
        raise ValueError("Original source/full rows changed during preparation")
    counts = np.zeros((7, 2), dtype=np.int64)
    for item in items:
        with np.load(confined(item["pixel_target"])) as handle:
            masks = handle["mask"]
            known = handle["known"]
        if masks.shape != (7, 80, 80) or known.shape != (7,) or not np.isin(masks, (0, 1)).all() or not np.isin(known, (0, 1)).all():
            raise ValueError("Unexpected pixel target contract")
        if any(t < 0 and known[k] for k, t in enumerate(item["targets"])):
            raise ValueError("Unknown photo label acquired pixel supervision")
        positive = (masks * known[:, None, None]).sum((1, 2), dtype=np.int64)
        counts += np.stack((positive, known.astype(np.int64) * 6400 - positive), axis=1)
    annotation_digest = hashlib.sha256(json.dumps(annotations, sort_keys=True).encode()).hexdigest()
    skipped_orientation_parents = [image for image, checked in orientation_checks.items() if not checked["verified"]]
    audit = {**manifest["audit"], "status": "prepared", "recipe": RECIPE,
             "source_manifest_sha256": original_sha, "replaced_rows": len(replacements),
             "unchanged_full_count": manifest["full_count"], "total_rows": len(items),
             "new_independent_photos": 0, "domains": dict(Counter(i["domain"] for i in items)),
             "replacement_domains": dict(Counter(r["domain"] for r in replacements)),
             "replacement_target_classes": dict(Counter(r["target"] for r in replacements)),
             "skipped_orientation_parent_rows": len(skipped_orientation_parents),
             "skipped_orientation_parent_images": skipped_orientation_parents,
             "orientation_geometry_parent_rows_checked": len(orientation_checks),
             "orientation_geometry_parent_rows_verified": sum(checked["verified"] for checked in orientation_checks.values()),
             "orientation_geometry_checks": orientation_checks,
             "orientation_geometry_policy": "EXIF-corrected sources allowed only when annotation dimensions, original decoded source RGB hash and prepared parent thumbnail dimensions match original preparation. Failed checks preserve original rows; no annotation/image repair.",
             "source_annotation_count": len(annotations), "source_annotation_sha256": annotations,
             "source_annotation_set_sha256": annotation_digest,
             "geometry": {"maximum_source_positive_fraction": SMALL_FRACTION,
                          "area_space": "Publisher polygons rasterized in unchanged processed-parent coordinates; aligned native Dam RGB masks. Fraction is an approximate annotated image area, not physical damage size.",
                          "minimum_context_short_side_fraction": MIN_SIDE_FRACTION,
                          "maximum_context_short_side_fraction": MAX_SIDE_FRACTION,
                          "bbox_context_factor": BBOX_CONTEXT_FACTOR, "minimum_positive_source_pixels": MIN_POSITIVE_PIXELS,
                          "center": "Actual positive closest to largest connected-component centroid; deterministic ties",
                          "selection": "Smallest source area class first, first existing target-positive detail row; at most one replacement per parent; strictly smaller crop area"},
             "per_label_pixel_cells": {label: {"positive": int(counts[k, 0]), "negative": int(counts[k, 1])} for k, label in enumerate(classes)},
             "per_label_photo_rows": {label: {str(t): sum(i["targets"][k] == t for i in items) for t in (-1, 0, 1)} for k, label in enumerate(classes)},
             "replacement_photo_pixel_presence_conflicts_masked_unknown": conflicts,
             "source_unknown_policy": "Crop labels recomputed from TRAIN publisher masks; 1..15 source pixels unknown. Parent-unknown remains unknown, parent-absent never positive. Dam only crack/spalling known; CODEBRIM has no generated ROI. No auxiliary tag propagation.",
             "policy": "Original full rows, order, row count and negative detail rows preserved; at most one positive detail row per TRAIN parent replaced. Validation/test untouched. No additional independent photos.",
             "scope": "TRAIN augmentation only; not a new independent dataset, measured localization accuracy or deployment decision",
             "preparation_script_sha256": sha(Path(__file__))}
    save(output / "train.json", {**manifest, "items": items, "audit": audit})
    summary = {key: value for key, value in audit.items() if key not in ("source_annotation_sha256", "skipped_orientation_parent_images", "orientation_geometry_checks")}
    summary["label_change_replacement_rows"] = sum(r["old_targets"] != r["new_targets"] for r in replacements)
    summary["derived_manifest_sha256"] = sha(output / "train.json")
    save(ROOT / "reports/facility-target-small-region-data-audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
