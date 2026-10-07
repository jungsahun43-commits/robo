"""Audit RC2119 publisher annotations and conservative whole-photo overlap.

No heldout labels/predictions are read. Output is an acquisition/annotation
audit, NOT a train manifest and NOT proof of scene or crop independence.
Absent author polygons remain unknown SafeLog photo targets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_rc2119 import PINNED, SOURCE_URL, sha, verified_archive, write_new

LABELS = ("Crack", "Concrete spalling", "Rebar exposure", "Rebar corrosion", "Concrete crushing")
SAFETY_CLASSES = ("concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar",
                  "wet_surface", "efflorescence", "surface_cavity")
POSITIVE_MAPPING = {"Crack": 0, "Concrete spalling": 1, "Rebar exposure": 3}
CACHE_SHA = "21013f0c4e7a105e6dfeecca5ca5380d3dc16eb812368359b5a468e01408ca61"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(image):
    tiny = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes(), "big")


def positive_targets(labels):
    if not set(labels) <= set(LABELS):
        raise ValueError("Unknown native damage class")
    result = [-1] * 7
    for name in labels:
        if name in POSITIVE_MAPPING:
            result[POSITIVE_MAPPING[name]] = 1
    return result


def validate_annotation(document, image_size):
    width, height = image_size
    if (document.get("imageWidth") != width or document.get("imageHeight") != height
            or not isinstance(document.get("shapes"), list) or not document["shapes"]):
        raise ValueError("Annotation/image dimensions disagree or no damage annotations")
    for shape in document["shapes"]:
        points = shape.get("points")
        if shape.get("label") not in LABELS or shape.get("shape_type") != "polygon" or not isinstance(points, list) or len(points) < 3:
            raise ValueError("Unexpected annotation label or non-polygon geometry")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                    or not 0 <= point[0] <= width or not 0 <= point[1] <= height):
                raise ValueError("Nonfinite/out-of-frame author polygon")
    return document["shapes"]


def inspect(task):
    image_path, annotation_path, mask_path, ledgers = task
    for kind, path in (("image", image_path), ("json", annotation_path), ("mask", mask_path)):
        record = ledgers[kind].get(path.name)
        if record is None or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Extracted publisher file differs from its pinned archive ledger")
    with Image.open(image_path) as source:
        source.verify()
    with Image.open(image_path) as source:
        raw_size = source.size
        orientation = source.getexif().get(274, 1)
        image = ImageOps.exif_transpose(source).convert("RGB")
        size = image.size
        pixels = hashlib.sha256(f"{size[0]}x{size[1]}:".encode() + image.tobytes()).hexdigest()
        dh = fingerprint(image)
    document = read(annotation_path)
    try:
        validate_annotation(document, size)
        geometry_issue = None
    except ValueError as error:
        # Complete the audit and quarantine the row; never silently drop it or
        # clip invalid publisher geometry into a supposedly valid train label.
        geometry_issue = str(error)
    shapes = document.get("shapes", [])
    if not isinstance(shapes, list) or any(s.get("label") not in LABELS for s in shapes):
        raise ValueError("Unsupported author labels prevent a complete ontology audit")
    with Image.open(mask_path) as source:
        if source.mode != "L":
            raise ValueError("Unexpected author mask mode; numeric vocabulary not established")
        mask_size = source.size
        mask = np.asarray(source)
        # PIL gives (count, code); store the inverse, never infer RGB from prose.
        mask_counts = {str(code): int(count) for count, code in source.getcolors(maxcolors=256)}
    labels = Counter(shape["label"] for shape in shapes)
    return {"image": image_path.relative_to(ROOT).as_posix(), "annotation": annotation_path.relative_to(ROOT).as_posix(),
            "mask": mask_path.relative_to(ROOT).as_posix(), "source_image_sha256": sha(image_path),
            "source_annotation_sha256": sha(annotation_path), "source_mask_sha256": sha(mask_path),
            "pixel_sha256": pixels, "dhash": dh, "source_size": list(size), "raw_size": list(raw_size),
            "mask_size": list(mask_size), "orientation": orientation,
            "mask_registration_issue": None if mask_size == size else "Mask/image dimensions differ after declared EXIF transform",
            "native_instances": dict(labels), "native_mask_pixels": mask_counts,
            "targets": positive_targets(labels), "annotation_flags": document.get("flags"),
            "annotation_geometry_issue": geometry_issue,
            "source_path_matches_basename": document.get("imagePath") == image_path.name}


def derive_codes(rows):
    """Numeric mask IDs require unanimous paired single-class file evidence."""
    evidence = defaultdict(Counter)
    for row in rows:
        if len(row["native_instances"]) == 1:
            nonzero = [int(k) for k, value in row["native_mask_pixels"].items() if k != "0" and value > 0]
            if len(nonzero) != 1:
                raise ValueError("Single-class JSON does not match a single non-background mask code")
            evidence[next(iter(row["native_instances"]))][nonzero[0]] += 1
    if set(evidence) != set(LABELS) or any(len(codes) != 1 for codes in evidence.values()):
        raise ValueError("Cannot establish all five numeric author mask codes unambiguously")
    codes = {name: next(iter(values)) for name, values in evidence.items()}
    if len(set(codes.values())) != 5 or set(codes.values()) != set(range(1, 6)):
        raise ValueError("Author mask vocabulary is not five distinct codes plus background0")
    for row in rows:
        actual = {int(k) for k in row["native_mask_pixels"] if k != "0"}
        expected = {codes[name] for name in row["native_instances"]}
        if not actual <= expected:
            raise ValueError("Mask has a damage code absent from its paired JSON")
    return codes, {name: {str(code): n for code, n in values.items()} for name, values in evidence.items()}


def polygon_check(task):
    row, codes = task
    if row["annotation_geometry_issue"] or row["mask_registration_issue"]:
        return {"image": row["image"], "different_pixels": None, "total_pixels": None,
                "skipped_reason": row["annotation_geometry_issue"] or row["mask_registration_issue"]}
    doc = read(ROOT / row["annotation"])
    width, height = row["source_size"]
    rendered = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(rendered)
    for shape in doc["shapes"]:
        draw.polygon([tuple(p) for p in shape["points"]], fill=codes[shape["label"]])
    with Image.open(ROOT / row["mask"]) as source:
        different = int(np.count_nonzero(np.asarray(rendered) != np.asarray(source)))
    return {"image": row["image"], "different_pixels": different, "total_pixels": width * height}


def existing_inventory():
    """Read cached photo identities, never use old labels or heldout scores."""
    cache_path = ROOT / "data/convid-training/existing-fingerprints.json"
    if sha(cache_path) != CACHE_SHA:
        raise ValueError("Pinned existing-source fingerprint cache changed")
    cache = read(cache_path)
    for name, digest in cache["metadata_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Old photo inventory metadata changed")
    paths = list(cache["image_sizes"])
    if len(paths) != len(cache["dhash"]) or any((ROOT / p).stat().st_size != cache["image_sizes"][p] for p in paths):
        raise ValueError("Old cached photo paths/byte sizes changed")
    pixels = []
    for row in read(ROOT / "data/dacl10k-yolo/records.json"):
        pixels.append(row["pixel_sha256"])
    for source, splits in (("damsegment", ("train", "test")), ("codebrim", ("train", "val", "test"))):
        for split in splits:
            pixels.extend(i["pixel_sha256"] for i in read(ROOT / f"data/{source}-training/{split}.json")["items"])
    pixels.extend(i["pixel_sha256"] for i in read(ROOT / "data/s2ds/pixel-audit.json")["items"])
    pixels.extend(i["pixel_sha256"] for i in read(ROOT / "data/ostrava-corrosion-yolo/records.json"))
    if len(pixels) != len(paths):
        raise ValueError("Native decoded hashes and cached photo inventory disagree")
    extra_path = ROOT / "data/convid-training/source-audit.json"
    extras = read(extra_path)["items"]
    pixels.extend(i["pixel_sha256"] for i in extras)
    hashes = cache["dhash"] + [i["dhash"] for i in extras]
    return set(pixels), np.asarray(hashes, dtype=np.uint64), {
        "original_cached_photos": len(paths), "convid_originals_including_exclusions": len(extras),
        "total_compared_photo_records": len(hashes), "cache_sha256": CACHE_SHA,
        "convid_audit_sha256": sha(extra_path), "metadata_sha256": cache["metadata_sha256"],
        "heldout_labels_or_predictions_used": False,
        "scope": "Native decoded RGB SHA + prepared full-photo dHash; all cached source splits plus all200 ConViD originals"}


def overlap(rows, old_pixels, old_hashes, distance=6):
    hashes = np.asarray([r["dhash"] for r in rows], dtype=np.uint64)
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    seen_pixels = {}
    for i in range(len(rows)):
        if rows[i]["pixel_sha256"] in seen_pixels:
            parents[find(i)] = find(seen_pixels[rows[i]["pixel_sha256"]])
        else:
            seen_pixels[rows[i]["pixel_sha256"]] = i
        for j in np.flatnonzero(np.bitwise_count(hashes[:i] ^ hashes[i]) <= distance):
            parents[find(i)] = find(int(j))
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    results = []
    for indices in groups.values():
        exact = any(rows[i]["pixel_sha256"] in old_pixels for i in indices)
        near = any((np.bitwise_count(old_hashes ^ hashes[i]) <= distance).any() for i in indices)
        reasons = []
        if exact: reasons.append("existing_exact_native_pixels")
        if near: reasons.append("existing_near_whole_photo_candidate")
        if any(rows[i].get("annotation_geometry_issue") for i in indices): reasons.append("annotation_geometry_review")
        if any(rows[i].get("mask_registration_issue") for i in indices): reasons.append("mask_registration_review")
        key = min(rows[i]["pixel_sha256"] for i in indices)
        for i in indices:
            results.append({"image": rows[i]["image"], "group": "rc2119-near:" + key,
                            "group_members": len(indices), "excluded_reasons": reasons})
    return sorted(results, key=lambda r: r["image"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    base = ROOT / "data/rc2119-training"
    output = base / "source-audit.json"
    if output.exists():
        raise ValueError("Preserve completed audit; never overwrite it")
    ledgers = {}
    for name, pin in PINNED.items():
        verified_archive(base / "archives" / name, pin)
        ledger = read(base / "metadata" / (pin[3] + "-extracted.json"))
        if ledger["archive_sha256"] != pin[2]: raise ValueError("Extraction archive identity changed")
        ledgers[pin[3]] = {r["path"]: r for r in ledger["files"]}
    image_paths = sorted((base / "source/image").glob("*.jpg"))
    json_paths = sorted((base / "source/json").glob("*.json"))
    mask_paths = sorted((base / "source/mask").glob("*.png"))
    if not (len(image_paths) == len(json_paths) == len(mask_paths) == 2119
            and {p.stem for p in image_paths} == {p.stem for p in json_paths} == {p.stem for p in mask_paths}):
        raise ValueError("Publisher three-way image/JSON/mask registration is incomplete")
    tasks = [(p, base / "source/json" / (p.stem + ".json"), base / "source/mask" / (p.stem + ".png"), ledgers) for p in image_paths]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(inspect, tasks), 1):
            rows.append(row)
            if i % 250 == 0: print(f"Verified publisher pairs {i}/2119", flush=True)
    codes, evidence = derive_codes(rows)
    checks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, check in enumerate(pool.map(polygon_check, ((r, codes) for r in rows)), 1):
            checks.append(check)
            if i % 500 == 0: print(f"Checked native polygon/mask registration {i}/2119", flush=True)
    old_pixels, old_hashes, inventory_proof = existing_inventory()
    reviews = overlap(rows, old_pixels, old_hashes)
    lookup = {r["image"]: r for r in reviews}
    rows = [{**r, **{k: v for k, v in lookup[r["image"]].items() if k != "image"}} for r in rows]
    instances = Counter(); photos = Counter()
    for row in rows:
        instances.update(row["native_instances"]); photos.update(row["native_instances"].keys())
    reasons = Counter(reason for row in rows for reason in row["excluded_reasons"])
    remaining = [r for r in rows if not r["excluded_reasons"]]
    public = {"schema": "rc2119_annotation_overlap_audit_v1", "source": SOURCE_URL,
        "source_files_sha256": {name: pin[2] for name, pin in PINNED.items()},
        "photo_pairs_verified": len(rows), "actual_native_instances": dict(instances), "actual_positive_photos": dict(photos),
        "native_mask_mode": "L", "background_code": 0, "native_mask_codes": codes,
        "mask_code_evidence_single_class_photo_counts": evidence,
        "annotation_geometry_review_photos": sum(r["annotation_geometry_issue"] is not None for r in rows),
        "mask_registration_review_photos": sum(r["mask_registration_issue"] is not None for r in rows),
        "declared_exif_orientation_corrected_photos": sum(r["orientation"] not in (1, None) for r in rows),
        "polygon_check": {"compared_photos": sum(r["different_pixels"] is not None for r in checks),
                          "skipped_geometry_invalid_photos": sum(r["different_pixels"] is None for r in checks),
                          "exactly_equal_photos": sum(r["different_pixels"] == 0 for r in checks),
                          "different_pixels": sum(r["different_pixels"] or 0 for r in checks)},
        "publisher_crack_instance_count": 1883, "archive_crack_instance_count": instances["Crack"],
        "inventory": inventory_proof, "overlap_excluded_photos": len(rows) - len(remaining),
        "overlap_reason_photo_counts_nonexclusive": dict(reasons),
        "remaining_whole_photo_screen_candidates": len(remaining),
        "remaining_spalling_positive_photos": sum("Concrete spalling" in r["native_instances"] for r in remaining),
        "within_source_near_groups": len({r["group"] for r in rows}),
        "label_policy": "Only explicit crack/spalling/rebar-exposure positives mapped; absent polygons unknown; rebar corrosion/crushing not mapped to rust/cavity/spalling",
        "known_spalling_negative_photos_created": 0, "train_manifest_created": False, "new_training_epochs": 0,
        "eligibility": "Pending crop/parent overlap and upstream provenance/annotation-coverage review; remaining candidates are not approved training photos",
        "limitations": ["Whole-photo dHash<=6 and native pixel SHA cannot prove crop, rotation or scene independence",
                        "Author background mask does not independently verify exhaustive annotation of every damage instance",
                        "No scene/group/split/upstream source mapping is provided by these three archives",
                        "Research dataset labels do not assert real-world structural safety"]}
    write_new(output, {"summary": public, "items": rows, "polygon_checks": checks})
    public["private_source_audit_sha256"] = sha(output)
    write_new(ROOT / "reports/facility-rc2119-audit.json", public)
    print(json.dumps({k: v for k, v in public.items() if k not in ("inventory", "limitations", "mask_code_evidence_single_class_photo_counts")}), flush=True)


if __name__ == "__main__":
    main()
