"""Audit referenced DACL TRAIN source geometry without feeding native images to training.

Only the existing TRAIN manifests select photos and annotations. Public output is
aggregate-only; file paths and per-photo evidence stay in an ignored local ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
           "wet_surface", "efflorescence", "surface_cavity"]
AUX_CLASSES = ["ACrack", "Bearing", "Cavity", "Crack", "Drainage", "EJoint",
               "Efflorescence", "ExposedRebars", "Graffiti", "Hollowareas",
               "JTape", "PEquipment", "Restformwork", "Rockpocket", "Rust",
               "Spalling", "WConccor", "Weathering", "Wetspot"]
MANIFESTS = {
    "records": "data/dacl10k-yolo/records.json",
    "preparation": "data/dacl10k-yolo/PREPARATION.json",
    "spatial": "data/facility-spatial-training/train.json",
    "detail": "data/facility-detail-training/manifest.json",
    "auxiliary": "data/facility-auxiliary-training/train.json",
}
EXPECTED_TRAIN_PHOTOS = 6225
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(path):
    path = Path(path)
    before = path.stat()
    result = {"sha256": digest(path), "size_bytes": before.st_size,
              "mtime_ns": before.st_mtime_ns}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Input changed while hashing")
    return result


def preserve_inputs(before, after):
    require(before == after, "Input content, size or mtime changed during the audit")


def confined(path, allowed):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(Path(allowed).resolve()), "Path escapes the allowed directory")
    return resolved


def corrected_dimensions(raw_size, orientation):
    require(orientation in (None, 1, 2, 3, 4, 5, 6, 7, 8), "Unsupported EXIF orientation")
    return tuple(raw_size[::-1] if orientation in (5, 6, 7, 8) else raw_size)


def native_targets(document):
    width, height = document.get("imageWidth"), document.get("imageHeight")
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            "Invalid native annotation dimensions")
    shapes = document.get("shapes")
    require(isinstance(shapes, list), "Missing publisher polygons")
    counts = Counter()
    for shape in shapes:
        require(isinstance(shape, dict) and shape.get("label") in AUX_CLASSES,
                "Unrecognized native label")
        points = shape.get("points")
        require(shape.get("shape_type", "polygon") == "polygon"
                and isinstance(points, list) and len(points) >= 3,
                "Invalid native polygon")
        require(all(isinstance(point, (list, tuple)) and len(point) == 2
                    and all(not isinstance(value, bool) and isinstance(value, (int, float))
                            and math.isfinite(value) for value in point) for point in points),
                "Nonfinite or malformed polygon point")
        counts[shape["label"]] += 1
    labels = set(counts)
    auxiliary = [int(label in labels) for label in AUX_CLASSES]
    core = [int(bool(labels & {"Crack", "ACrack"}))]
    core += [int(label in labels) for label in
             ("Spalling", "Rust", "ExposedRebars", "Wetspot", "Efflorescence", "Cavity")]
    return auxiliary, core, dict(counts)


def select_train_references(records, auxiliary, spatial, expected_count=EXPECTED_TRAIN_PHOTOS):
    require(isinstance(records, list), "Invalid source records")
    selected = [row for row in records if row.get("split") == "train"]
    require(len(selected) == expected_count, "DACL TRAIN source count changed")
    require(auxiliary.get("split") == "train" and auxiliary.get("classes") == AUX_CLASSES,
            "Auxiliary TRAIN class contract changed")
    require(spatial.get("split") == "train" and spatial.get("classes") == CLASSES,
            "Spatial TRAIN class contract changed")
    full_count = spatial.get("full_count")
    require(type(full_count) is int and 0 < full_count <= len(spatial.get("items", [])),
            "Invalid original full-row boundary")
    auxiliary_rows = auxiliary.get("items", [])
    core_rows = [row for row in spatial["items"][:full_count] if row.get("domain") == "dacl"]
    require(len(auxiliary_rows) == len(core_rows) == expected_count, "DACL TRAIN mapping count changed")
    require(all(row.get("split") == "train" and row.get("domain") == "dacl"
                for row in auxiliary_rows), "Held-out or foreign auxiliary row")
    source_map = {row["stem"]: row for row in selected}
    auxiliary_map = {Path(row["image"]).stem: row for row in auxiliary_rows}
    core_map = {Path(row["image"]).stem: row for row in core_rows}
    require(len(source_map) == len(auxiliary_map) == len(core_map) == expected_count,
            "Duplicate TRAIN reference")
    require(set(source_map) == set(auxiliary_map) == set(core_map), "TRAIN reference sets disagree")
    require([row["image"] for row in auxiliary_rows] == [row["image"] for row in core_rows],
            "Auxiliary original full-photo order changed")
    for stem in source_map:
        require(auxiliary_map[stem]["image"] == core_map[stem]["image"], "Processed image mapping changed")
        for row, classes in ((auxiliary_map[stem], AUX_CLASSES), (core_map[stem], CLASSES)):
            targets = row.get("targets")
            require(isinstance(targets, list) and len(targets) == len(classes)
                    and all(type(value) is int and value in (0, 1) for value in targets),
                    "DACL original targets must remain complete integer zero/one labels")
    return [(stem, source_map[stem], auxiliary_map[stem], core_map[stem])
            for stem in sorted(source_map)]


def inspect_reference(reference, root):
    stem, record, auxiliary, core = reference
    root = Path(root)
    source = confined(root / "data" / record["source"], root / "data")
    processed = confined(root / core["image"], root / "data")
    annotation = confined(root / auxiliary["annotation"], root / "data")
    require(record["split"] == "train" and source.parent.name == "train"
            and source.parent.parent.name == "images", "Source is outside publisher TRAIN")
    require(processed.parent.name == "train" and processed.parent.parent.name == "images"
            and processed.stem == stem, "Processed image is outside mapped TRAIN")
    expected_annotation = source.parent.parent.parent / "annotations" / "train" / f"{source.stem}.json"
    require(annotation == expected_annotation.resolve(), "Annotation is not the matching publisher TRAIN file")
    before = {str(path.relative_to(root)): snapshot(path) for path in (source, processed, annotation)}
    require(_SHA.fullmatch(record.get("file_sha256", "")) is not None
            and before[str(source.relative_to(root))]["sha256"] == record["file_sha256"],
            "Native source file differs from the verified preparation record")
    require(_SHA.fullmatch(auxiliary.get("annotation_sha256", "")) is not None
            and before[str(annotation.relative_to(root))]["sha256"] == auxiliary["annotation_sha256"],
            "Native annotation differs from the fixed auxiliary manifest")
    with Image.open(source) as image:
        raw_size = image.size
        orientation = image.getexif().get(274, 1)
        source_size = corrected_dimensions(raw_size, orientation)
        corrected_pixel_hash_verified = False
        if orientation not in (None, 1):
            corrected = ImageOps.exif_transpose(image).convert("RGB")
            require(corrected.size == source_size, "EXIF dimension correction disagrees")
            pixel_hash = hashlib.sha256(f"{corrected.width}x{corrected.height}:".encode()
                                        + corrected.tobytes()).hexdigest()
            require(pixel_hash == record.get("pixel_sha256"), "EXIF-corrected source pixels changed")
            corrected_pixel_hash_verified = True
    with Image.open(processed) as image:
        processed_size = image.size
    document = json.loads(annotation.read_text(encoding="utf-8"))
    auxiliary_targets, core_targets, counts = native_targets(document)
    require(source_size == (document["imageWidth"], document["imageHeight"]),
            "Native annotation and EXIF-corrected photo dimensions differ")
    require(auxiliary_targets == auxiliary["targets"], "Original native nineteen-class photo tags changed")
    require(core_targets == core["targets"], "Original seven-class photo targets changed")
    return {
        "stem": stem, "split": "train", "source_image": source.relative_to(root).as_posix(),
        "processed_image": processed.relative_to(root).as_posix(),
        "native_annotation": annotation.relative_to(root).as_posix(),
        "raw_source_size": list(raw_size), "source_size": list(source_size),
        "processed_size": list(processed_size), "exif_orientation": orientation,
        "exif_corrected_pixel_hash_verified": corrected_pixel_hash_verified,
        "native_label_polygon_counts": counts, "core_targets": core_targets,
        "auxiliary_targets": auxiliary_targets, "input_snapshots_before": before,
    }


def statistics_for(values):
    values = sorted(values)
    require(bool(values), "Empty size inventory")
    return {"min": values[0], "median": statistics.median(values),
            "p10": values[int((len(values)-1)*.1)], "p90": values[int((len(values)-1)*.9)],
            "max": values[-1]}


def public_summary(rows, manifest_snapshots, spatial, detail, preparation):
    photo_counts, polygon_counts = Counter(), Counter()
    for row in rows:
        photo_counts.update(row["native_label_polygon_counts"].keys())
        polygon_counts.update(row["native_label_polygon_counts"])
    source_bytes = sum(row["input_snapshots_before"][str(Path(row["source_image"]))]["size_bytes"] for row in rows)
    processed_bytes = sum(row["input_snapshots_before"][str(Path(row["processed_image"]))]["size_bytes"] for row in rows)
    source_short = [min(row["source_size"]) for row in rows]
    return {
        "schema": "facility_resolution_source_audit_v1", "status": "complete",
        "scope": "Referenced DACL TRAIN files only; source geometry availability, not accuracy or new training data",
        "train_photos": len(rows), "unique_source_file_references": len({row["source_image"] for row in rows}),
        "classes": CLASSES, "native_classes": AUX_CLASSES,
        "manifest_sha256": {name: value["sha256"] for name, value in manifest_snapshots.items()},
        "preparation_max_side": preparation["max_side"],
        "source_long_side": statistics_for([max(row["source_size"]) for row in rows]),
        "source_short_side": statistics_for(source_short),
        "processed_long_side": statistics_for([max(row["processed_size"]) for row in rows]),
        "processed_short_side": statistics_for([min(row["processed_size"]) for row in rows]),
        "source_larger_than_processed": sum(max(row["source_size"]) > max(row["processed_size"]) for row in rows),
        "source_long_side_gt1280": sum(max(row["source_size"]) > 1280 for row in rows),
        "source_short_side_at_least640": sum(value >= 640 for value in source_short),
        "source_short_side_at_least960": sum(value >= 960 for value in source_short),
        "source_short_side_at_least1024": sum(value >= 1024 for value in source_short),
        "source_bytes_total": source_bytes, "processed_bytes_total": processed_bytes,
        "native_polygon_count": sum(polygon_counts.values()),
        "native_label_photo_counts": dict(sorted(photo_counts.items())),
        "native_label_polygon_counts": dict(sorted(polygon_counts.items())),
        "crack_or_acrack_positive_photos": sum(bool(set(row["native_label_polygon_counts"]) & {"Crack", "ACrack"}) for row in rows),
        "spalling_positive_photos": photo_counts["Spalling"],
        "verified_source_file_hashes": len(rows), "verified_annotation_hashes": len(rows),
        "verified_exif_corrected_annotation_dimensions": len(rows),
        "verified_native19_photo_targets": len(rows), "verified_core7_photo_targets": len(rows),
        "exif_corrected_sources": sum(row["exif_orientation"] not in (None, 1) for row in rows),
        "decoded_pixel_hash_checked_for_exif_sources": sum(row["exif_corrected_pixel_hash_verified"] for row in rows),
        "spatial_context": {"train_rows": len(spatial["items"]), "full_rows": spatial["full_count"],
                            "dacl_full_rows": len(rows), "pixel_grid": spatial["audit"]["grid_size"],
                            "dacl_detail_rows": sum(row["domain"] == "dacl" for row in detail["items"])},
        "current_resolution_comparison": {"control_input": 640, "candidate_input": 960,
            "input_photos": "Existing processed photos in both arms; this audit does not feed native sources",
            "native_polygons_already_used_in_existing_supervision": True,
            "new_native_source_feeding_performed": False},
        "expert_confirmed_labels": 0, "label_changes": 0, "training_performed_by_audit": 0,
        "individual_paths_or_annotations_published": False,
        "limitations": [
            "Original source file availability does not imply that the 640 versus 960 experiment reads those native files.",
            "Existing DACL native polygons and nineteen-class tags were already used; they are not newly labelled data.",
            "Native Crack plus ACrack maps to the historical concrete_crack key; source material and structural safety are not inferred.",
            "Size and polygon counts are not independent factory photos, expert validation or a 5 percent error guarantee.",
            "Low-resolution originals remain present; larger input tensors cannot create missing image detail.",
            "All source byte hashes and annotation hashes are checked; decoded RGB hashes are checked for the EXIF-transformed sources only.",
            "VAL/TEST record metadata can exist in the records manifest; no heldout photo, annotation or per-photo prediction file is opened.",
        ],
    }


def run(root, report_path, inventory_path, workers=4):
    root = Path(root).resolve()
    report_path = confined(report_path, root / "reports")
    inventory_path = confined(inventory_path, root / "runs")
    require(not report_path.exists() and not inventory_path.exists(), "Preserve existing outputs; choose new paths")
    require(type(workers) is int and 1 <= workers <= 8, "Workers must be between one and eight")
    manifest_paths = {name: root / relative for name, relative in MANIFESTS.items()}
    manifest_snapshots = {name: snapshot(path) for name, path in manifest_paths.items()}
    documents = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in manifest_paths.items()}
    require(documents["preparation"].get("max_side") == 1280, "Historical preparation size changed")
    require(documents["detail"].get("split") == "train" and documents["detail"].get("classes") == CLASSES,
            "Detail TRAIN class contract changed")
    references = select_train_references(documents["records"], documents["auxiliary"], documents["spatial"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda reference: inspect_reference(reference, root), references))
    input_before = {str(path.relative_to(root)): manifest_snapshots[name] for name, path in manifest_paths.items()}
    for row in rows:
        for relative, evidence in row["input_snapshots_before"].items():
            require(relative not in input_before, "Duplicate physical input reference")
            input_before[relative] = evidence
    input_after = {relative: snapshot(root / relative) for relative in input_before}
    preserve_inputs(input_before, input_after)
    created_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    inventory = {"schema": "facility_resolution_source_inventory_v1", "created_utc": created_utc,
                 "local_only": True, "split": "train", "rows": rows,
                 "input_snapshots_before": input_before,
                 "input_snapshots_after": input_after, "all_inputs_preserved": True}
    encoded_inventory = (json.dumps(inventory, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    summary = public_summary(rows, manifest_snapshots, documents["spatial"], documents["detail"], documents["preparation"])
    summary.update(created_utc=created_utc, audit_script_sha256=digest(Path(__file__)),
                   private_inventory_sha256=hashlib.sha256(encoded_inventory).hexdigest(),
                   input_files_checked=len(input_before), all_input_sha256_size_mtime_preserved=True)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("xb") as stream:
        stream.write(encoded_inventory)
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/facility-resolution-source-audit.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / "runs/facility-resolution-source-audit-20261005/individual-inventory.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(ROOT, args.report, args.inventory, args.workers)
    print(json.dumps({key: result[key] for key in
                     ("status", "train_photos", "source_larger_than_processed", "native_polygon_count",
                      "input_files_checked", "all_input_sha256_size_mtime_preserved", "private_inventory_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
