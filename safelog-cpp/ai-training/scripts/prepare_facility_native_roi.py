"""Re-render existing TRAIN ROIs from verified native sources in a matched pair.

Both arms use the same decoded native RGB, output size and lossless PNG recipe.
Only eligible DACL detail image paths change; original labels/masks/full rows and
the original auxiliary lookup are preserved. This script performs no training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_facility_resolution_sources import (
    AUX_CLASSES, CLASSES, MANIFESTS, native_targets, preserve_inputs,
    select_train_references, snapshot,
)

PAIR_ROOT = "data/facility-native-roi-training"
LEDGER_PATH = "runs/facility-native-roi-data/ledger.json"
REPORT_PATH = "reports/facility-native-roi-data-audit.json"
SOURCE_AUDIT = "reports/facility-resolution-source-audit.json"
SOURCE_INVENTORY = "runs/facility-resolution-source-audit-20261005/individual-inventory.json"
RECIPE = {
    "output_size": [640, 640], "resampling": "LANCZOS",
    "reducing_gap": None, "source_box_interval": "half_open_float_scaled_no_rounding",
    "historical_thumbnail_jpeg_pixel_replay": False,
    "control": "native_RGB_resize_to_processed_dimensions_then_integer_crop",
    "native": "native_RGB_direct_scaled_float_box",
    "png_compress_level": 6, "png_optimize": False,
    "png_metadata": "pixels_only_no_source_metadata",
    "selection": "native_both_axes_at_least_processed_and_one_strictly_larger",
    "new_targets": 0, "auxiliary_crop_tags": "unknown19",
}
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON metadata key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON metadata")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def safe_path(root, relative, *, exists=True, historical=False):
    require(isinstance(relative, str), "Internal path must be a string")
    if historical:
        relative = relative.replace("\\", "/")
    require(relative and "\\" not in relative and ":" not in relative
            and not PurePosixPath(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Expected an internal relative path")
    root = Path(root).resolve(); path = (root / relative).resolve()
    require(path.is_relative_to(root) and (not exists or path.is_file()),
            "Missing input or escaped internal path")
    return path


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixel_sha(image):
    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def eligible(native_size, processed_size):
    for size in (native_size, processed_size):
        require(isinstance(size, (tuple, list)) and len(size) == 2
                and all(type(v) is int and v > 0 for v in size), "Invalid source raster dimensions")
    return all(a >= b for a, b in zip(native_size, processed_size)) and any(
        a > b for a, b in zip(native_size, processed_size))


def validate_box(box, processed_size):
    eligible(processed_size, processed_size)
    require(isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) is int for v in box), "Existing crop box must contain four integers")
    width, height = processed_size
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
            "Existing crop box is outside its original processed parent")
    return tuple(box)


def render_pair(native_rgb, processed_size, box, *, pre_downsampled_rgb=None):
    """Use exact scaled floating coordinates; never round a native ROI box."""
    require(isinstance(native_rgb, Image.Image) and native_rgb.mode == "RGB",
            "Paired rendering requires one corrected RGB source")
    eligible(native_rgb.size, processed_size)  # Validates both sizes; equal sizes allowed for tests.
    require(all(a >= b for a, b in zip(native_rgb.size, processed_size)),
            "Paired ROI source cannot be smaller than its processed parent")
    box = validate_box(box, processed_size)
    sx = native_rgb.width / processed_size[0]; sy = native_rgb.height / processed_size[1]
    native_box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
    require(all(math.isfinite(v) for v in native_box), "Invalid projected crop coordinates")
    size = tuple(RECIPE["output_size"])
    owned = pre_downsampled_rgb is None
    if owned:
        processed = native_rgb.resize(tuple(processed_size), Image.Resampling.LANCZOS, reducing_gap=None)
    else:
        require(isinstance(pre_downsampled_rgb, Image.Image) and pre_downsampled_rgb.mode == "RGB"
                and pre_downsampled_rgb.size == tuple(processed_size), "Cached control parent raster differs")
        processed = pre_downsampled_rgb
    try:
        control = processed.resize(size, Image.Resampling.LANCZOS, box=box, reducing_gap=None)
        native = native_rgb.resize(size, Image.Resampling.LANCZOS, box=native_box, reducing_gap=None)
    finally:
        if owned:
            processed.close()
    return control, native, list(native_box)


def validate_base(base):
    require(isinstance(base, dict) and base.get("split") == "train" and base.get("classes") == CLASSES,
            "Only the original seven-class TRAIN manifest is allowed")
    items = base.get("items"); full = base.get("full_count")
    require(isinstance(items, list) and items and type(full) is int and 0 < full <= len(items),
            "Invalid original full/crop boundary")
    images, masks = set(), set()
    for index, row in enumerate(items):
        require(isinstance(row, dict) and row.get("domain") in ("dacl", "damsegment", "codebrim")
                and row.get("split", "train") == "train", "Foreign or held-out row in TRAIN")
        for key, seen in (("image", images), ("pixel_target", masks)):
            value = row.get(key)
            require(isinstance(value, str) and value.startswith("data/") and "\\" not in value
                    and ":" not in value and all(p not in ("", ".", "..") for p in value.split("/"))
                    and value not in seen, "Invalid or duplicate original image/mask path")
            seen.add(value)
        targets = row.get("targets")
        require(isinstance(targets, list) and len(targets) == 7
                and all(type(v) is int and v in (-1, 0, 1) for v in targets), "Original photo targets changed")
        if index < full:
            require(row.get("parent_image") is None, "Original full row cannot be a derived crop")
    parents = {row["image"]: row for row in items[:full]}
    for row in items[full:]:
        parent = parents.get(row.get("parent_image"))
        require(parent is not None and row.get("parent_split") == "train"
                and row["domain"] == parent["domain"], "Crop lacks its original same-domain TRAIN parent")
        require(row["domain"] != "codebrim", "Original CODEBRIM has no derived detail crops")
    require(isinstance(base.get("audit"), dict) and base["audit"].get("grid_size") == 80
            and set(base["audit"].get("per_label_pixel_cells", {})) == set(CLASSES),
            "Original coarse pixel supervision audit is required")
    return parents


def validate_pair(base, control, native):
    """Pure structural proof: only matched eligible DACL crop image paths differ."""
    parents = validate_base(base); full = base["full_count"]
    for variant, manifest in (("control", control), ("native", native)):
        validate_base(manifest)
        require({k: v for k, v in manifest.items() if k not in ("items", "native_roi")}
                == {k: v for k, v in base.items() if k not in ("items", "native_roi")},
                "Pair changed original manifest metadata, audit, labels or full boundary")
        require(len(manifest["items"]) == len(base["items"]), "Pair changed TRAIN row count")
        metadata = manifest.get("native_roi")
        if metadata is not None:
            require(isinstance(metadata, dict)
                    and set(metadata) == {"schema", "variant", "recipe", "base_manifest_sha256"}
                    and metadata.get("schema") == "facility_native_roi_manifest_v1"
                    and metadata.get("variant") == variant and metadata.get("recipe") == RECIPE
                    and isinstance(metadata.get("base_manifest_sha256"), str)
                    and _SHA.fullmatch(metadata["base_manifest_sha256"]),
                    "Pair rendering recipe metadata differs")
    require(("native_roi" in control) == ("native_roi" in native),
            "Only one paired manifest declares its rendering provenance")
    if "native_roi" in control:
        require(control["native_roi"]["base_manifest_sha256"] == native["native_roi"]["base_manifest_sha256"],
                "Paired manifests declare different original core digests")
    changed, changed_parents = [], set()
    for index, (original, left, right) in enumerate(zip(base["items"], control["items"], native["items"])):
        require(all({k: v for k, v in row.items() if k != "image"}
                    == {k: v for k, v in original.items() if k != "image"} for row in (left, right)),
                "Pair changed original row labels, masks, parent, box or metadata")
        differs = (left["image"] != original["image"], right["image"] != original["image"])
        require(differs[0] == differs[1], "Pair changed different rows")
        if differs[0]:
            require(index >= full and original["domain"] == "dacl"
                    and original.get("parent_split") == "train"
                    and parents[original["parent_image"]]["domain"] == "dacl",
                    "Only existing DACL TRAIN detail crops may change image paths")
            for variant, row in (("control", left), ("native", right)):
                require(row["image"] == f"{PAIR_ROOT}/{variant}/images/roi-{index:05d}.png",
                        "Derived paired image path/order differs")
            changed.append(index); changed_parents.add(original["parent_image"])
        else:
            require(left == right == original, "Unchanged original row differs")
    return {"changed_rows": len(changed), "changed_dacl_parents": len(changed_parents),
            "original_full_rows": full, "train_rows": len(base["items"]),
            "full_rows_unchanged": True, "all_row_metadata_except_image_unchanged": True,
            "labels_and_masks_unchanged": True, "row_order_unchanged": True,
            "crop_auxiliary_targets_remain_unknown19": True}


def validate_audit(audit, inventory, documents, manifest_snapshots, inventory_sha):
    require(audit.get("schema") == "facility_resolution_source_audit_v1"
            and audit.get("status") == "complete"
            and audit.get("all_input_sha256_size_mtime_preserved") is True
            and audit.get("classes") == CLASSES and audit.get("native_classes") == AUX_CLASSES,
            "Completed original TRAIN source audit is required")
    require(inventory.get("schema") == "facility_resolution_source_inventory_v1"
            and inventory.get("local_only") is True and inventory.get("split") == "train"
            and inventory.get("all_inputs_preserved") is True
            and audit.get("private_inventory_sha256") == inventory_sha,
            "Private TRAIN inventory differs from its source audit digest")
    require(audit.get("manifest_sha256") == {name: value["sha256"] for name, value in manifest_snapshots.items()},
            "Original audited manifest bytes changed")
    require(documents["preparation"].get("max_side") == audit.get("preparation_max_side") == 1280,
            "Historical processed source preparation changed")
    count = audit.get("train_photos")
    require(type(count) is int and count > 0 and all(audit.get(key) == count for key in (
        "verified_source_file_hashes", "verified_annotation_hashes", "verified_core7_photo_targets",
        "verified_native19_photo_targets", "verified_exif_corrected_annotation_dimensions")),
        "Original source audit does not cover every selected TRAIN parent")
    references = select_train_references(documents["records"], documents["auxiliary"], documents["spatial"], count)
    rows = inventory.get("rows")
    require(isinstance(rows, list) and len(rows) == count, "Private TRAIN source inventory count differs")
    by_image = {row["processed_image"]: row for row in rows}
    require(len(by_image) == count and set(by_image) == {reference[3]["image"] for reference in references}
            and all(row.get("split") == "train" for row in rows), "Private TRAIN source mapping differs")
    for stem, record, aux, core in references:
        row = by_image[core["image"]]
        require(row.get("stem") == stem and row.get("core_targets") == core["targets"]
                and row.get("auxiliary_targets") == aux["targets"]
                and row.get("native_annotation") == aux["annotation"]
                and row.get("source_image") == "data/" + record["source"].replace("\\", "/"),
                "Private source geometry/tag mapping differs")
    detail = documents["detail"]; spatial = documents["spatial"]
    require(detail.get("split") == "train" and detail.get("classes") == CLASSES
            and [{k: v for k, v in row.items() if k != "pixel_target"} for row in spatial["items"][spatial["full_count"]:]]
            == detail.get("items"), "Original detail crop order/targets/metadata differ")
    return references, by_image


def inspect_and_render(reference, audited, crops, root, preparation_max_side):
    """Decode a parent once, hash it, verify tags/dimensions, render all its crops."""
    stem, record, aux, core = reference
    source = safe_path(root, "data/" + record["source"], historical=True)
    processed = safe_path(root, core["image"]); annotation = safe_path(root, aux["annotation"])
    require(record.get("split") == "train" and source.parent.name == "train"
            and source.parent.parent.name == "images" and processed.parent.name == "train"
            and processed.parent.parent.name == "images" and processed.stem == stem,
            "Only matching original DACL TRAIN sources may render crops")
    require(annotation == (source.parent.parent.parent / "annotations/train" / f"{source.stem}.json").resolve(),
            "Source annotation is outside its matching publisher TRAIN path")
    before = {path.relative_to(root).as_posix(): snapshot(path) for path in (source, processed, annotation)}
    old = {relative.replace("\\", "/"): value for relative, value in audited["input_snapshots_before"].items()}
    preserve_inputs(old, before)
    require(before[source.relative_to(root).as_posix()]["sha256"] == record["file_sha256"]
            and before[annotation.relative_to(root).as_posix()]["sha256"] == aux["annotation_sha256"],
            "Original source or native annotation bytes changed")
    document = read(annotation); aux_targets, core_targets, _ = native_targets(document)
    require(aux_targets == aux["targets"] and core_targets == core["targets"],
            "Original native nineteen/seven-class photo tags changed")
    with Image.open(source) as original:
        orientation = original.getexif().get(274, 1)
        require(orientation in (None, 1, 2, 3, 4, 5, 6, 7, 8), "Unsupported source EXIF orientation")
        require(max(original.size) <= 6000, "Source raster exceeds the audited bounded memory recipe")
        native_rgb = ImageOps.exif_transpose(original).convert("RGB")
    processed_rgb = None
    try:
        require(native_rgb.size == (document["imageWidth"], document["imageHeight"])
                == tuple(audited["source_size"]), "EXIF-corrected source annotation dimensions differ")
        require(pixel_sha(native_rgb) == record.get("pixel_sha256"), "Original decoded RGB source differs")
        with Image.open(processed) as image:
            processed_size = image.size
        require(processed_size == tuple(audited["processed_size"]), "Audited processed raster dimensions differ")
        # Match Pillow.thumbnail's rounding without allocating a second full RGB.
        with Image.new("1", native_rgb.size) as geometry:
            geometry.thumbnail((preparation_max_side, preparation_max_side), Image.Resampling.LANCZOS)
            require(geometry.size == processed_size, "Historical processed thumbnail dimensions differ")
        changed = []; larger = eligible(native_rgb.size, processed_size)
        if larger and crops:
            processed_rgb = native_rgb.resize(processed_size, Image.Resampling.LANCZOS, reducing_gap=None)
        for index, row in crops:
            box = validate_box(row.get("box_in_parent"), processed_size)
            if not larger:
                continue
            left, right, scaled = render_pair(native_rgb, processed_size, box,
                                              pre_downsampled_rgb=processed_rgb)
            try:
                entries = {}
                for variant, image in (("control", left), ("native", right)):
                    relative = f"{PAIR_ROOT}/{variant}/images/roi-{index:05d}.png"
                    destination = safe_path(root, relative, exists=False)
                    require(not destination.exists(), "Derived paired image already exists")
                    image.info.clear()
                    image.save(destination, format="PNG", compress_level=RECIPE["png_compress_level"],
                               optimize=RECIPE["png_optimize"])
                    entries[variant] = {"image": relative, "sha256": file_sha(destination),
                                        "pixel_sha256": pixel_sha(image), "size_bytes": destination.stat().st_size}
                changed.append({"row_index": index, "parent_image": core["image"],
                    "original_image": row["image"], "pixel_target": row["pixel_target"],
                    "pixel_target_sha256": file_sha(safe_path(root, row["pixel_target"])),
                    "box_in_parent": list(box), "native_box": scaled,
                    "source_image": source.relative_to(root).as_posix(),
                    "source_size": list(native_rgb.size), "processed_size": list(processed_size),
                    **entries})
            finally:
                left.close(); right.close()
        return {"parent_image": core["image"], "eligible": larger,
                "source_size": list(native_rgb.size), "processed_size": list(processed_size),
                "exif_corrected": orientation not in (None, 1),
                "control_parent_resizes": int(processed_rgb is not None),
                "input_snapshots_before": before, "changed_rows": changed}
    finally:
        if processed_rgb is not None:
            processed_rgb.close()
        native_rgb.close()


def public_summary(base, pair_proof, parents, changed, input_count, manifest_sha, ledger_sha,
                   source_audit_sha, source_inventory_sha):
    # Whitelist aggregate fields: never publish row indices, source photo paths or boxes.
    selected = [base["items"][row["row_index"]] for row in changed]
    return {"schema": "facility_native_roi_data_audit_v1", "status": "prepared", "split": "train",
            "recipe": deepcopy(RECIPE), **pair_proof,
            "audited_dacl_train_parents": len(parents), "eligible_larger_dacl_parents": sum(p["eligible"] for p in parents),
            "exif_corrected_sources": sum(p["exif_corrected"] for p in parents),
            "verified_decoded_native_rgb_hashes": len(parents), "verified_source_annotations": len(parents),
            "changed_original_target_counts": {label: {
                "positive": sum(row["targets"][k] == 1 for row in selected),
                "negative": sum(row["targets"][k] == 0 for row in selected),
                "unknown": sum(row["targets"][k] == -1 for row in selected)} for k, label in enumerate(CLASSES)},
            "derived_manifest_sha256": manifest_sha, "private_ledger_sha256": ledger_sha,
            "source_audit_sha256": source_audit_sha, "source_inventory_sha256": source_inventory_sha,
            "input_files_checked": input_count, "all_input_sha256_size_mtime_preserved": True,
            "derived_pngs": 2 * len(changed), "new_independent_photos": 0, "new_photo_targets": 0,
            "new_pixel_targets": 0, "expert_confirmed_labels": 0, "label_changes": 0,
            "training_performed_by_preparation": 0, "accuracy_measured": False,
            "heldout_photos_opened": 0, "individual_paths_or_boxes_published": False,
            "limitations": ["Original masks/photo tags/ROI boxes remain unchanged; native crops are not new truth.",
                "Coarse 80-cell masks are reused, not replaced with higher precision localization labels.",
                "Both arms replace eligible historical JPEG crops with identical 640-square PNG recipes.",
                "The control decodes the same native source then downscales, avoiding a different JPEG compression path.",
                "The control is a new global-downsample-before-ROI counterfactual, not a pixel replay of historical thumbnail/JPEG95 preprocessing.",
                "Only larger original DACL TRAIN parents are eligible; low resolution originals remain unchanged.",
                "Existing public bridge imagery does not establish factory accuracy or a five percent error guarantee."]}


def _write_json(path, document):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(root=ROOT, workers=4):
    started = time.perf_counter()
    root = Path(root).resolve()
    require(type(workers) is int and 1 <= workers <= 4, "Use one to four bounded parent workers")
    output = safe_path(root, PAIR_ROOT, exists=False); ledger_path = safe_path(root, LEDGER_PATH, exists=False)
    report_path = safe_path(root, REPORT_PATH, exists=False)
    require(not any(path.exists() for path in (output, ledger_path, report_path)),
            "Preserve existing preparation outputs; never overwrite evidence")
    paths = {name: safe_path(root, relative) for name, relative in MANIFESTS.items()}
    manifest_snapshots = {name: snapshot(path) for name, path in paths.items()}
    documents = {name: read(path) for name, path in paths.items()}
    audit_path = safe_path(root, SOURCE_AUDIT); inventory_path = safe_path(root, SOURCE_INVENTORY)
    audit_snapshot, inventory_snapshot = snapshot(audit_path), snapshot(inventory_path)
    audit, inventory = read(audit_path), read(inventory_path)
    base = documents["spatial"]; validate_base(base)
    references, audited = validate_audit(audit, inventory, documents, manifest_snapshots, inventory_snapshot["sha256"])
    before = {path.relative_to(root).as_posix(): manifest_snapshots[name] for name, path in paths.items()}
    before.update({SOURCE_AUDIT: audit_snapshot, SOURCE_INVENTORY: inventory_snapshot})
    # Hash all original referenced images and masks, including unchanged full rows.
    original_paths = {row[key] for row in base["items"] for key in ("image", "pixel_target")}
    for index, relative in enumerate(sorted(original_paths), 1):
        before[relative] = snapshot(safe_path(root, relative))
        if index % 5000 == 0:
            print(json.dumps({"phase": "input_hash_before", "checked": index, "total": len(original_paths)}), flush=True)
    groups = {reference[3]["image"]: [] for reference in references}
    for index, row in enumerate(base["items"][base["full_count"]:], base["full_count"]):
        if row["domain"] == "dacl":
            groups[row["parent_image"]].append((index, row))
    output.mkdir(parents=True, exist_ok=False)
    for variant in ("control", "native"):
        (output / variant / "images").mkdir(parents=True, exist_ok=False)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = pool.map(lambda ref: inspect_and_render(ref, audited[ref[3]["image"]],
                         groups[ref[3]["image"]], root, documents["preparation"]["max_side"]), references)
        for index, result in enumerate(tasks, 1):
            for relative, evidence in result["input_snapshots_before"].items():
                require(relative not in before or before[relative] == evidence, "Input changed before parent decoding")
                before[relative] = evidence
            results.append(result)
            if index % 250 == 0:
                print(json.dumps({"phase": "native_parent_pairs", "audited_train_parents": index,
                                  "total": len(references)}), flush=True)
    changed = sorted((row for parent in results for row in parent["changed_rows"]), key=lambda row: row["row_index"])
    require(changed, "No eligible existing native detail crop pairs")
    manifests = {variant: deepcopy(base) for variant in ("control", "native")}
    for variant, manifest in manifests.items():
        for row in changed:
            manifest["items"][row["row_index"]]["image"] = row[variant]["image"]
        manifest["native_roi"] = {"schema": "facility_native_roi_manifest_v1", "variant": variant,
                                  "recipe": deepcopy(RECIPE), "base_manifest_sha256": manifest_snapshots["spatial"]["sha256"]}
    pair_proof = validate_pair(base, manifests["control"], manifests["native"])
    after = {}
    for index, relative in enumerate(before, 1):
        after[relative] = snapshot(safe_path(root, relative))
        if index % 5000 == 0:
            print(json.dumps({"phase": "input_hash_after", "checked": index, "total": len(before)}), flush=True)
    preserve_inputs(before, after)
    for row in changed:
        for variant in ("control", "native"):
            require(file_sha(safe_path(root, row[variant]["image"])) == row[variant]["sha256"],
                    "Derived PNG changed during preparation")
    for variant, manifest in manifests.items():
        _write_json(output / variant / "train.json", manifest)
    manifest_sha = {variant: file_sha(output / variant / "train.json") for variant in manifests}
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    ledger = {"schema": "facility_native_roi_ledger_v1", "local_only": True, "split": "train",
              "created_utc": created, "recipe": deepcopy(RECIPE), "changed_rows": changed,
              "parents": [{k: v for k, v in result.items() if k != "changed_rows"} for result in results],
              "base_manifest_sha256": manifest_snapshots["spatial"]["sha256"],
              "derived_manifest_sha256": manifest_sha, "source_audit_sha256": audit_snapshot["sha256"],
              "source_inventory_sha256": inventory_snapshot["sha256"],
              "input_snapshots_before": before, "input_snapshots_after": after, "all_inputs_preserved": True}
    ledger_path.parent.mkdir(parents=True, exist_ok=True); _write_json(ledger_path, ledger)
    summary = public_summary(base, pair_proof, results, changed, len(before), manifest_sha,
                             file_sha(ledger_path), audit_snapshot["sha256"], inventory_snapshot["sha256"])
    derived_bytes = {variant: sum(row[variant]["size_bytes"] for row in changed) for variant in manifests}
    summary.update(created_utc=created, preparation_script_sha256=file_sha(Path(__file__)),
                   preparation_elapsed_minutes=(time.perf_counter() - started) / 60,
                   worker_count=workers, control_parent_resizes=sum(p["control_parent_resizes"] for p in results),
                   derived_png_bytes_by_variant=derived_bytes, derived_png_bytes_total=sum(derived_bytes.values()))
    report_path.parent.mkdir(parents=True, exist_ok=True); _write_json(report_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    result = run(ROOT, parser.parse_args(argv).workers)
    print(json.dumps({key: result[key] for key in ("status", "changed_rows", "changed_dacl_parents",
                      "input_files_checked", "all_input_sha256_size_mtime_preserved", "private_ledger_sha256")}))


if __name__ == "__main__":
    main()
