"""Derive independent native19 TRAIN masks without changing original truth.

Only the pinned original DACL full-photo auxiliary manifest is used. Original
images, annotations, photo tags and seven-channel targets remain untouched.
Coarse channel negatives mean absence in the publisher annotation, not safety.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import AUX_CLASSES

SOURCE_MANIFEST = "data/facility-auxiliary-training/train.json"
SOURCE_SHA = "53544bf83e1b32e3a7b8775c5f8348594359bf6e01eade1c8571242e758f4b61"
OUTPUT = "data/facility-dense-auxiliary-training"
PUBLIC_AUDIT = "reports/facility-dense-auxiliary-data-audit.json"
RECIPE = "dacl_v2_full_train_native19_processed_polygon_box640_any80_v1"
EXPECTED_COUNT, WORKERS, GRID = 6225, 4, 80


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_path(root, value, prefix):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
            "Expected a relative POSIX TRAIN input path")
    parts = value.split("/")
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and all(p not in ("", ".", "..") for p in parts)
            and relative.is_relative_to(PurePosixPath(prefix)), "Input leaves its declared TRAIN directory")
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root / prefix),
            "Missing, linked or escaping TRAIN input")
    return path.resolve()


def snapshot(path):
    path = Path(path)
    before = path.stat()
    value = {"sha256": sha(path), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Original input changed while hashing")
    return value


def validate_source_manifest(document, expected_count=EXPECTED_COUNT):
    require(document.get("split") == "train" and document.get("classes") == list(AUX_CLASSES),
            "The original TRAIN native19 vocabulary is required")
    rows = document.get("items")
    require(isinstance(rows, list) and len(rows) == expected_count, "Original full DACL TRAIN count changed")
    images, annotations = set(), set()
    for row in rows:
        require(isinstance(row, dict) and row.get("split") == "train" and row.get("domain") == "dacl"
                and "parent_image" not in row, "Only original full DACL TRAIN rows are eligible")
        targets = row.get("targets")
        require(isinstance(targets, list) and len(targets) == 19
                and all(type(v) is int and v in (0, 1) for v in targets), "Original native19 photo tags must remain asserted integers")
        require(row.get("image") not in images and row.get("annotation") not in annotations,
                "Duplicate full-photo or native annotation reference")
        require(re.fullmatch(r"[a-f0-9]{64}", str(row.get("annotation_sha256", ""))) is not None,
                "Pinned native annotation SHA256 is required")
        images.add(row["image"]); annotations.add(row["annotation"])
    return rows


def native_source(document, annotation, root):
    require(document.get("split") == "train" and document.get("dacl10k_version") == "v2",
            "Only native DACL v2 TRAIN documents are supported")
    relative = document.get("imagePath")
    require(isinstance(relative, str) and re.fullmatch(r"images/train/[A-Za-z0-9_.-]+\.jpg", relative),
            "Native image reference is not a plain author TRAIN JPEG")
    require(document.get("imageName") == PurePosixPath(relative).name
            and PurePosixPath(relative).stem == annotation.stem,
            "Native image and annotation basename registration differs")
    base = annotation.parent.parent.parent
    value = (base.relative_to(root) / relative).as_posix()
    return safe_path(root, value, "data/dacl10k/dacl10k_v2_devphase/images/train")


def registered_native_size(path, document):
    with Image.open(path) as image:
        width, height = image.size
        orientation = image.getexif().get(274, 1)
    require(orientation in (None, 1, 2, 3, 4, 5, 6, 7, 8), "Unsupported native EXIF orientation")
    corrected = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
    require(corrected == (document.get("imageWidth"), document.get("imageHeight")),
            "Publisher polygon dimensions differ from the EXIF-corrected native image")
    return corrected, orientation


def polygon_points(shape, width, height):
    points = shape.get("points")
    if shape.get("shape_type", "polygon") != "polygon" or not isinstance(points, list) or len(points) < 3:
        return None
    parsed = []
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                or not 0 <= point[0] <= width or not 0 <= point[1] <= height):
            return None
        parsed.append((float(point[0]), float(point[1])))
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(parsed, parsed[1:] + parsed[:1]))
    if not math.isfinite(area2) or abs(area2) <= 1e-6:
        return None
    return parsed


def rasterize_dense(document, targets, processed_size):
    """Independent channel unions; any conflict masks that channel unknown.

    Geometry outside the source frame is not clipped into new supervision.
    A malformed polygon makes its whole class unknown, including valid sibling
    polygons. Positive tags and original seven-channel targets are unchanged.
    """
    width, height = document.get("imageWidth"), document.get("imageHeight")
    require(type(width) is int and type(height) is int and min(width, height) > 0,
            "Native annotation dimensions must be positive integers")
    require(isinstance(processed_size, (tuple, list)) and len(processed_size) == 2
            and all(type(v) is int and v > 0 for v in processed_size), "Invalid prepared-parent dimensions")
    require(isinstance(targets, list) and len(targets) == 19
            and all(type(v) is int and v in (0, 1) for v in targets), "Invalid original photo-tag contract")
    shapes = document.get("shapes")
    require(isinstance(shapes, list), "Native document must supply author polygons")
    canvases = [Image.new("L", tuple(processed_size), 0) for _ in AUX_CLASSES]
    known = np.ones(19, dtype=np.uint8)
    reasons = [[] for _ in AUX_CLASSES]
    native_presence = set()
    for shape in shapes:
        require(isinstance(shape, dict) and shape.get("label") in AUX_CLASSES,
                "Unrecognized native polygon class prevents a complete ontology audit")
        column = AUX_CLASSES.index(shape["label"]); native_presence.add(column)
        points = polygon_points(shape, width, height)
        if points is None:
            known[column] = 0
            if "invalid_native_polygon" not in reasons[column]: reasons[column].append("invalid_native_polygon")
            continue
        scaled = [(x * processed_size[0] / width, y * processed_size[1] / height) for x, y in points]
        ImageDraw.Draw(canvases[column]).polygon(scaled, fill=255)
    masks = np.zeros((19, GRID, GRID), dtype=np.uint8)
    for column, canvas in enumerate(canvases):
        fine = np.asarray(canvas.resize((640, 640), Image.Resampling.BOX)) > 0
        coarse = fine.reshape(GRID, 8, GRID, 8).max((1, 3)).astype(np.uint8)
        if int(column in native_presence) != targets[column]:
            known[column] = 0; reasons[column].append("native_presence_photo_tag_conflict")
        if bool(coarse.any()) != bool(targets[column]):
            known[column] = 0; reasons[column].append("raster_presence_photo_tag_conflict")
        if known[column]: masks[column] = coarse
    return masks, known, reasons


def validate_arrays(masks, known, targets):
    require(masks.dtype == np.uint8 and masks.shape == (19, GRID, GRID)
            and known.dtype == np.uint8 and known.shape == (19,)
            and np.isin(masks, (0, 1)).all() and np.isin(known, (0, 1)).all(),
            "Dense native19 arrays must be registered binary uint8 tensors")
    require(not masks[known == 0].any(), "Unknown dense channels must carry no asserted foreground")
    for column, target in enumerate(targets):
        require(target != 0 or not masks[column].any(), "An asserted negative tag acquired foreground")
        require(not known[column] or bool(masks[column].any()) == bool(target),
                "Known dense foreground and original photo tag disagree")


def prepare_row(task):
    index, row, root, output = task
    image = safe_path(root, row["image"], "data/dacl10k-yolo/images/train")
    annotation = safe_path(root, row["annotation"], "data/dacl10k/dacl10k_v2_devphase/annotations/train")
    require(image.stem == "dacl-" + annotation.stem, "Prepared image is not the matching native annotation")
    document = read(annotation)
    native = native_source(document, annotation, root)
    before = {p.relative_to(root).as_posix(): snapshot(p) for p in (image, annotation, native)}
    require(before[row["annotation"]]["sha256"] == row["annotation_sha256"], "Pinned native annotation changed")
    source_size, orientation = registered_native_size(native, document)
    with Image.open(image) as parent:
        require(parent.mode == "RGB", "Prepared parent is not the original RGB JPEG")
        processed_size = parent.size
    masks, known, reasons = rasterize_dense(document, row["targets"], processed_size)
    validate_arrays(masks, known, row["targets"])
    mask_path = output / "masks" / f"native19-{index:05d}.npz"
    with mask_path.open("xb") as stream: np.savez_compressed(stream, masks=masks, known=known)
    result = {"image": row["image"], "image_sha256": before[row["image"]]["sha256"],
        "mask": mask_path.relative_to(root).as_posix(), "mask_sha256": sha(mask_path),
        "known": known.tolist(), "targets": row["targets"][:], "annotation": row["annotation"],
        "annotation_sha256": row["annotation_sha256"], "source": native.relative_to(root).as_posix(),
        "source_sha256": before[native.relative_to(root).as_posix()]["sha256"],
        "source_size": list(source_size), "processed_size": list(processed_size), "source_exif_orientation": orientation,
        "split": "train", "domain": "dacl", "unknown_channel_reasons": reasons}
    positive = masks.sum((1, 2), dtype=np.int64)
    negative = known.astype(np.int64) * GRID * GRID - positive
    unknown = (1 - known.astype(np.int64)) * GRID * GRID
    return result, np.stack((positive, negative, unknown), axis=1), before


def main():
    torch.set_num_threads(4)
    root = ROOT.resolve(); source = root / SOURCE_MANIFEST; output = root / OUTPUT
    require(sha(source) == SOURCE_SHA, "Pinned original full DACL auxiliary manifest changed")
    document = read(source); rows = validate_source_manifest(document)
    require(not output.exists() and not (root / PUBLIC_AUDIT).exists(), "Preserve previous preparation; no outputs may be overwritten")
    output.mkdir(); (output / "masks").mkdir()
    started = time.perf_counter(); items = []; counts = np.zeros((19, 3), dtype=np.int64)
    inputs_before = {SOURCE_MANIFEST: snapshot(source)}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        tasks = ((index, row, root, output) for index, row in enumerate(rows))
        for number, (item, cells, originals) in enumerate(pool.map(prepare_row, tasks), 1):
            items.append(item); counts += cells
            for path, original in originals.items():
                require(path not in inputs_before or inputs_before[path] == original, "Conflicting shared input snapshot")
                inputs_before[path] = original
            if number % 500 == 0: print(f"Dense native19 TRAIN masks prepared {number}/{len(rows)}", flush=True)
    inputs_after = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        paths = sorted(inputs_before)
        for path, value in zip(paths, pool.map(lambda name: snapshot(root / name), paths)):
            inputs_after[path] = value
    require(inputs_after == inputs_before, "Original SHA256, size or mtime changed during dense preparation")
    require([r["image"] for r in items] == [r["image"] for r in rows]
            and [r["targets"] for r in items] == [r["targets"] for r in rows], "Original full-photo order or tags changed")
    ledger_path = output / "input-ledger.json"
    write_new(ledger_path, {"schema": "facility_dense_auxiliary_private_input_ledger_v1", "local_only": True,
        "snapshots_before": inputs_before, "snapshots_after": inputs_after, "all_inputs_preserved": True})
    reasons = Counter(reason for r in items for channel in r["unknown_channel_reasons"] for reason in channel)
    audit = {"schema": "facility_dense_auxiliary_data_audit_v1", "status": "prepared", "recipe": RECIPE,
        "source": "https://github.com/phiyodr/dacl10k-toolkit", "source_version": "DACL v2 development phase",
        "license": "CC BY-NC 4.0", "source_auxiliary_manifest_sha256": SOURCE_SHA,
        "full_train_photos": len(items), "independent_new_photos": 0, "derived_mask_files": len(items),
        "classes": list(AUX_CLASSES), "source_split": "train", "grid_size": GRID,
        "per_label_pixel_cells": {c: dict(zip(("positive", "negative", "unknown"), map(int, counts[k]))) for k, c in enumerate(AUX_CLASSES)},
        "per_label_photo_counts": {c: {"positive": sum(r["targets"][k] == 1 for r in items),
            "negative": sum(r["targets"][k] == 0 for r in items), "spatial_known": sum(r["known"][k] == 1 for r in items),
            "spatial_unknown": sum(r["known"][k] == 0 for r in items)} for k, c in enumerate(AUX_CLASSES)},
        "unknown_channel_reason_counts_nonexclusive": dict(reasons),
        "photos_with_any_unknown_spatial_channel": sum(not all(r["known"]) for r in items),
        "original_input_files_preserved": len(inputs_before), "all_original_input_sha_size_mtime_preserved": True,
        "private_input_ledger_sha256": sha(ledger_path), "preparation_elapsed_minutes": (time.perf_counter() - started) / 60,
        "worker_count": WORKERS, "created_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_script_sha256": sha(Path(__file__)), "heldout_images_or_annotations_opened": False,
        "model_inference_executed": False, "new_training_epochs": 0, "original_photo_targets_changed": False,
        "original_seven_channel_targets_changed": False, "individual_paths_or_annotations_published": False,
        "coverage_policy": "Only6225 original full DACL TRAIN photos. Other sources and all detail crops have19 spatial channels unknown and are absent from this manifest.",
        "pixel_policy": "Independent native-class polygon unions at prepared-parent RGB dimensions; original BOX640 threshold>0 then8x8 any-positive80 cells. Invalid or conflicting whole class channels are zeroed and unknown.",
        "negative_policy": "Only eligible full DACL author annotation coverage defines per-class outside-polygon/absent-class negatives; zero does not assert a normal or structurally safe facility.",
        "limitations": ["Coarse source polygon cells are not exact material boundaries or expert reannotation",
            "Classes can overlap; this is19 independent binary channels, not mutually exclusive segmentation",
            "Hollowareas reflects author inspection markings and does not establish photo-only internal damage diagnosis",
            "Derived spatial supervision adds no independent photos, source validation or field evaluation"]}
    manifest_path = output / "train.json"
    write_new(manifest_path, {"version": 1, "schema": "facility_dense_auxiliary_training_v1", "split": "train",
        "classes": list(AUX_CLASSES), "items": items, "audit": audit})
    public = {**audit, "manifest_sha256": sha(manifest_path)}
    write_new(root / PUBLIC_AUDIT, public)
    print(json.dumps({"status": "prepared", "full_train_photos": len(items), "manifest_sha256": public["manifest_sha256"],
        "photos_with_unknown_channels": audit["photos_with_any_unknown_spatial_channel"],
        "original_inputs_preserved": len(inputs_before), "elapsed_minutes": audit["preparation_elapsed_minutes"]}), flush=True)


if __name__ == "__main__":
    main()
