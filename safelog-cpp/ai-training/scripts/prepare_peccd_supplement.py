"""Audited TRAIN-only PECCD photo labels; native boxes never become pixel masks.

Requires the complete pinned publisher RAR and a verified download sidecar. No
numeric class order is inferred from the order of the website's prose. Detailed
paths/labels/group reviews remain in ignored data/peccd-training/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_damsegment import dhash
from scripts.prepare_facility_data import NAMES
from scripts.train_facility_target import read, save, sha

SOURCE_URL = "https://data.mendeley.com/datasets/w7549ryvx2/1"
ARCHIVE_SHA256 = "9acc37537c2a618c22bbda55d1353823d4e2dd82d98ae729d0c30315c72388ec"
ARCHIVE_BYTES = 3767446444
ARCHIVE_PREFIX = "SyrianPostEarthquakeCrackDataset"
RECIPE = "peccd_native_photo_train_only_v1"
MAX_SELECTED = 500
SEED = 52
NEAR_DISTANCE = 6
CANONICAL_CLASSES = {"deepcrack", "simplecrack", "multibranchedcrack", "spalling", "scaling", "holes"}
CRACK_CLASSES = {"deepcrack", "simplecrack", "multibranchedcrack"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DERIVED_NAME = re.compile(r"(?:^|[._ -])(?:aug(?:mented)?|crop(?:ped)?|flip(?:ped)?|rot(?:ated|ation)?|tile|patch)(?:[._ -]|\d|$)", re.I)


def canonical_class(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def class_order(names):
    normalized = [canonical_class(name) for name in names]
    if len(normalized) != 6 or set(normalized) != CANONICAL_CLASSES:
        raise ValueError("Publisher class order must contain exactly the six evidenced PECCD classes")
    return normalized


def validate_download(archive, sidecar):
    metadata = read(sidecar)
    if metadata.get("verified") is not True or archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Complete verified publisher archive required; partial archives are never extracted")
    digest = sha(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError("Publisher archive SHA256 mismatch")
    return {"archive_sha256": digest, "archive_bytes": archive.stat().st_size,
            "download_sidecar_sha256": sha(sidecar)}


def safe_member(name):
    """Portable archive paths: one expected prefix, plain basenames, no traversal."""
    value = name.replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]
    pieces = value.rstrip("/").split("/")
    if (not pieces or pieces[0] != ARCHIVE_PREFIX or any(p in ("", ".", "..") for p in pieces)
            or any(not re.fullmatch(r"[A-Za-z0-9 ._()-]+", p) for p in pieces)):
        raise ValueError("Archive member is outside the publisher prefix or uses an unsafe basename")
    return PurePosixPath(*pieces).as_posix()


def archive_members(archive, tar):
    names = subprocess.run([tar, "-tf", str(archive)], check=True, capture_output=True, text=True).stdout.splitlines()
    verbose = subprocess.run([tar, "-tvf", str(archive)], check=True, capture_output=True, text=True).stdout.splitlines()
    if not names or len(names) != len(verbose):
        raise ValueError("Archive listing failed or contains ambiguous embedded newlines")
    rows = []
    for name, detail in zip(names, verbose):
        if not detail or detail[0] not in ("-", "d"):
            raise ValueError("Archive links/devices/nonregular entries are not allowed")
        rows.append({"path": safe_member(name), "directory": detail[0] == "d"})
    files = [row["path"] for row in rows if not row["directory"]]
    if len(set(files)) != len(files):
        raise ValueError("Duplicate archive file names")
    if len({path.casefold() for path in files}) != len(files):
        raise ValueError("Case-insensitive archive path collision on Windows")
    return rows


def extract_verified(archive, rows, destination, tar):
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.resolve().is_relative_to((ROOT / "data/peccd-training").resolve()):
        raise ValueError("Extraction destination must remain inside ignored PECCD data")
    # Every entry was checked before extraction. Consuming every entry allows the
    # publisher RAR's native integrity checks to finish; zero exit is required.
    subprocess.run([tar, "-xf", str(archive), "-C", str(destination), "--no-same-owner", "--no-same-permissions"],
                   check=True, capture_output=True, text=True)
    expected = {row["path"] for row in rows if not row["directory"]}
    actual = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("Unexpected extracted symlink")
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
    if actual != expected:
        raise ValueError("Extracted regular-file inventory differs from the fully verified archive")
    return {"full_extraction_returncode": 0, "regular_files": len(expected),
            "member_list_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "integrity_scope": "Pinned full archive SHA256 and successful full bsdtar extraction; all regular-file inventory checked"}


def author_classes(source, member=None):
    recognized = {"classes.txt", "classes.names", "obj.names", "data.yaml", "dataset.yaml"}
    candidates = [source / safe_member(member)] if member else sorted(p for p in source.rglob("*") if p.is_file() and p.name.lower() in recognized)
    found = []
    for path in candidates:
        if not path.is_file() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Author class evidence is not inside the verified archive")
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() in (".yaml", ".yml"):
            document = yaml.safe_load(text)
            names = document.get("names") if isinstance(document, dict) else None
            if isinstance(names, dict):
                if set(int(k) for k in names) != set(range(6)):
                    raise ValueError("Publisher YAML class IDs must be contiguous 0..5")
                names = [names[k] if k in names else names[str(k)] for k in range(6)]
        else:
            names = [line.strip() for line in text.splitlines() if line.strip()]
        if not isinstance(names, list):
            raise ValueError("Author class file does not supply a numeric class order")
        found.append((class_order(names), names, path))
    if not found or any(item[0] != found[0][0] for item in found):
        raise ValueError("Missing/conflicting author numeric class order; inspect publisher evidence, never guess IDs from prose")
    order, native_names, evidence = found[0]
    return order, {"native_classes": native_names, "archive_member": evidence.relative_to(source).as_posix(),
                   "evidence_sha256": sha(evidence), "evidence_kind": "publisher archive numeric class-list file",
                   "class_order_not_inferred_from_website": True}


def parse_labels(text, order):
    class_order(order)
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Native YOLO row {number} is not class+four normalized coordinates")
        if not re.fullmatch(r"\d+", values[0]):
            raise ValueError("Native YOLO class ID must be an integer")
        category = int(values[0])
        if not 0 <= category < 6:
            raise ValueError("Native class ID is outside the evidenced six-class vocabulary")
        x, y, width, height = map(float, values[1:])
        if (not all(math.isfinite(v) for v in (x, y, width, height)) or not 0 < width <= 1 or not 0 < height <= 1
                or x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6
                or y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6):
            raise ValueError("Invalid/nonfinite/out-of-image native normalized box")
        boxes.append([category, x, y, width, height])
    present = {order[box[0]] for box in boxes}
    return [int(bool(present & CRACK_CLASSES)), int("spalling" in present)] + [-1] * 5, boxes


def source_row(task):
    image_path, label_path, source, order = task
    text = label_path.read_text(encoding="utf-8-sig")
    targets, boxes = parse_labels(text, order)
    with Image.open(image_path) as handle:
        handle.verify()
    with Image.open(image_path) as handle:
        orientation = handle.getexif().get(274, 1)
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size
    if min(width, height) < 1:
        raise ValueError("Empty source image")
    pixel = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
    return {"source_image": image_path.relative_to(source).as_posix(), "source_annotation": label_path.relative_to(source).as_posix(),
            "source_image_sha256": sha(image_path), "source_annotation_sha256": sha(label_path),
            "pixel_sha256": pixel, "dhash": dhash(image), "source_size": [width, height],
            "exif_orientation_corrected": orientation not in (1, None), "targets": targets,
            "native_class_ids": sorted({box[0] for box in boxes}), "native_instances": dict(Counter(str(box[0]) for box in boxes)),
            "suspected_derived_filename": bool(DERIVED_NAME.search(image_path.stem)),
            "native_boxes": boxes}


def inspect_source(source, order):
    paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError("Publisher archive contains no source images")
    tasks = []
    for image in paths:
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise ValueError("Every publisher source image must have its native same-directory YOLO label")
        tasks.append((image, label, source, order))
    # Fail closed on any malformed source row. Do not silently publish a partial
    # audit or train on whichever source annotations happened to parse first.
    result = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, item in enumerate(pool.map(source_row, tasks), 1):
            result.append(item)
            if index % 500 == 0:
                print(f"PECCD publisher photos/labels verified {index}/{len(tasks)}", flush=True)
    return result


def existing_photos(cache_path=None):
    """All original core splits, screened-source holdouts and older industrial data."""
    rows = []
    metadata = []
    path = ROOT / "data/dacl10k-yolo/records.json"
    metadata.append(path)
    for item in read(path):
        rows.append({"image": f"data/dacl10k-yolo/images/{item['split']}/{item['stem']}.jpg", "pixel_sha256": item["pixel_sha256"]})
    for source, splits in (("damsegment", ("train", "test")), ("codebrim", ("train", "val", "test"))):
        for split in splits:
            path = ROOT / f"data/{source}-training/{split}.json"
            metadata.append(path)
            rows.extend({"image": i["image"], "pixel_sha256": i["pixel_sha256"]} for i in read(path)["items"])
    optional = ROOT / "data/s2ds/pixel-audit.json"
    if optional.exists():
        metadata.append(optional)
        rows.extend({"image": i["image"], "pixel_sha256": i["pixel_sha256"]} for i in read(optional)["items"])
    optional = ROOT / "data/ostrava-corrosion-yolo/records.json"
    if optional.exists():
        metadata.append(optional)
        rows.extend({"image": f"data/ostrava-corrosion-yolo/images/{i['split']}/{i['stem']}.jpg", "pixel_sha256": i["pixel_sha256"]} for i in read(optional))
    paths = [ROOT / row["image"] for row in rows]
    signatures = {row["image"]: path.stat().st_size for row, path in zip(rows, paths)}
    if len(signatures) != len(rows):
        raise ValueError("Existing overlap inventory contains duplicate image paths")
    cached = {}
    old_cache = ROOT / "data/s2ds/existing-fingerprints.json"
    if old_cache.exists():
        previous = read(old_cache)
        if len(previous["image_sizes"]) != len(previous["dhash"]):
            raise ValueError("Existing dHash cache order/length changed")
        for (name, size), value in zip(previous["image_sizes"].items(), previous["dhash"]):
            normalized = name.replace("\\", "/")
            if signatures.get(normalized) == size:
                cached[normalized] = value
    own_cache = Path(cache_path) if cache_path else ROOT / "data/peccd-training/existing-fingerprints.json"
    own_cache.parent.mkdir(parents=True, exist_ok=True)
    metadata_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in metadata}
    if own_cache.exists():
        previous = read(own_cache)
        if previous.get("image_sizes") == signatures and previous.get("metadata_sha256") == metadata_hashes:
            return rows, np.asarray(previous["dhash"], dtype=np.uint64), {"photos": len(rows), "cache_sha256": sha(own_cache), "metadata_sha256": metadata_hashes}
    missing = [row["image"] for row in rows if row["image"] not in cached]
    def fingerprint(relative):
        with Image.open(ROOT / relative) as handle:
            return dhash(ImageOps.exif_transpose(handle).convert("RGB"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, value in zip(missing, pool.map(fingerprint, missing)):
            cached[name] = value
    hashes = [cached[row["image"]] for row in rows]
    save(own_cache, {"image_sizes": signatures, "metadata_sha256": metadata_hashes, "dhash": hashes,
                     "reused_s2ds_cache_sha256": sha(old_cache) if old_cache.exists() else None})
    return rows, np.asarray(hashes, dtype=np.uint64), {"photos": len(rows), "cache_sha256": sha(own_cache),
            "metadata_sha256": metadata_hashes,
            "cache_scope": "Cached existing dHashes matched by path and current file byte size; source manifests pinned by SHA256"}


def screen_source(items, previous, previous_hashes, group_prefix="peccd-near:"):
    hashes = np.asarray([row["dhash"] for row in items], dtype=np.uint64)
    old_pixels = {row["pixel_sha256"] for row in previous}
    parent = list(range(len(items)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    seen_pixels = {}
    for index, row in enumerate(items):
        if row["pixel_sha256"] in seen_pixels:
            parent[find(index)] = find(seen_pixels[row["pixel_sha256"]])
        else:
            seen_pixels[row["pixel_sha256"]] = index
        for other in np.flatnonzero(np.bitwise_count(hashes[:index] ^ hashes[index]) <= NEAR_DISTANCE):
            parent[find(index)] = find(int(other))
    groups = defaultdict(list)
    for index in range(len(items)):
        groups[find(index)].append(index)
    kept, review = [], []
    counters = Counter()
    for indices in groups.values():
        reasons = []
        if any(items[i]["pixel_sha256"] in old_pixels for i in indices):
            reasons.append("existing_exact_decoded_pixel_overlap")
        if len(previous_hashes) and any((np.bitwise_count(previous_hashes ^ hashes[i]) <= NEAR_DISTANCE).any() for i in indices):
            reasons.append("existing_near_candidate_review_pending")
        if any(items[i]["suspected_derived_filename"] for i in indices):
            reasons.append("suspected_crop_or_augmentation_filename_review_pending")
        if len({tuple(items[i]["targets"]) for i in indices}) > 1:
            reasons.append("within_near_group_conflicting_photo_labels_review_pending")
        group = group_prefix + min(items[i]["pixel_sha256"] for i in indices)
        if reasons:
            counters["excluded_groups"] += 1
            counters["excluded_photos"] += len(indices)
            for reason in reasons:
                counters[reason] += 1
            review.extend({"source_image": items[i]["source_image"], "group_id": group, "reasons": reasons} for i in indices)
        else:
            index = min(indices, key=lambda i: items[i]["source_image"])
            kept.append({**items[index], "group_id": group, "near_group_members": len(indices)})
            counters["within_source_collapsed_photos"] += len(indices) - 1
    return kept, review, {"within_source_groups": len(groups), "eligible_representatives": len(kept), **dict(counters),
                         "near_rule": "64-bit whole-photo dHash Hamming<=6 transitive groups; decoded-pixel SHA exact check",
                         "review_policy": "All detected existing-near/derived-name/conflicting-label candidates excluded pending review; no model scores or heldout labels used",
                         "limitation": "Whole-photo hash screen does not prove crop, scene, building, photographer or site independence"}


def stratified_select(items, maximum=MAX_SELECTED, seed=SEED):
    if not 1 <= maximum <= MAX_SELECTED:
        raise ValueError("Bounded source selection must be between 1 and 500 photos")
    buckets = defaultdict(list)
    for item in sorted(items, key=lambda row: row["source_image"]):
        buckets[tuple(item["targets"][:2])].append(item)
    rng = random.Random(seed)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    selected = []
    positions = {key: 0 for key in buckets}
    while len(selected) < maximum:
        added = False
        for key in sorted(buckets):
            if positions[key] < len(buckets[key]) and len(selected) < maximum:
                selected.append(buckets[key][positions[key]])
                positions[key] += 1
                added = True
        if not added:
            break
    return selected


def unknown_pixel_target(path):
    np.savez_compressed(path, mask=np.zeros((7, 80, 80), dtype=np.uint8), known=np.zeros(7, dtype=np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/industrial-source-probe/PECCD.rar")
    parser.add_argument("--download-metadata", type=Path, default=ROOT / "data/industrial-source-probe/PECCD.download.json")
    parser.add_argument("--classes-member", help="Explicit publisher classes-file member, if several metadata files need inspection")
    parser.add_argument("--max-selected", type=int, default=MAX_SELECTED)
    args = parser.parse_args()
    if not 1 <= args.max_selected <= MAX_SELECTED:
        raise ValueError("At most 500 independently screened source representatives")
    output = ROOT / "data/peccd-training"
    if (output / "train.json").exists():
        raise ValueError("Preserve the already prepared experiment")
    provenance = validate_download(args.archive.resolve(), args.download_metadata.resolve())
    tar = shutil.which("tar")
    if not tar:
        raise ValueError("Windows bsdtar is required; do not install an alternative silently")
    members = archive_members(args.archive.resolve(), tar)
    source = output / "source"
    extraction = extract_verified(args.archive.resolve(), members, source, tar)
    order, class_evidence = author_classes(source, args.classes_member)
    items = inspect_source(source, order)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "source-audit.json", {"archive": provenance, "extraction": extraction, "class_evidence": class_evidence, "items": items})
    previous, previous_hashes, fingerprint_audit = existing_photos()
    eligible, pending, overlap_audit = screen_source(items, previous, previous_hashes)
    save(output / "overlap-review.json", {"status": "detected_candidates_excluded_pending_review", "items": pending, "audit": overlap_audit})
    selected = stratified_select(eligible, args.max_selected, SEED)
    if not selected:
        raise ValueError("No eligible publisher photos after conservative overlap/diversity screen")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    shared_pixel = output / "masks/photo-unknown.npz"
    unknown_pixel_target(shared_pixel)
    prepared = []
    for index, row in enumerate(selected):
        original = source / row["source_image"]
        if sha(original) != row["source_image_sha256"] or sha(source / row["source_annotation"]) != row["source_annotation_sha256"]:
            raise ValueError("Publisher image/annotation changed after full source audit")
        with Image.open(original) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        destination = output / "images" / f"peccd-{index:04d}.jpg"
        image.save(destination, quality=95)
        # Preserve all native photo labels; no crop/rotation-derived relabelling.
        if parse_labels((source / row["source_annotation"]).read_text(encoding="utf-8-sig"), order)[0] != row["targets"]:
            raise ValueError("Photo labels changed during full-photo conversion")
        prepared.append({"image": destination.relative_to(ROOT).as_posix(), "targets": row["targets"], "domain": "peccd",
                         "source_split": "train_only_unsplit", "author_split": None, "scene_unknown": True,
                         "pixel_target": shared_pixel.relative_to(ROOT).as_posix(), "pixel_target_sha256": sha(shared_pixel),
                         "source_image": (source / row["source_image"]).relative_to(ROOT).as_posix(),
                         "source_annotation": (source / row["source_annotation"]).relative_to(ROOT).as_posix(),
                         "source_image_sha256": row["source_image_sha256"], "source_annotation_sha256": row["source_annotation_sha256"],
                         "pixel_sha256": row["pixel_sha256"], "image_sha256": sha(destination),
                         "group_id": row["group_id"], "native_class_ids": row["native_class_ids"],
                         "source_size": row["source_size"], "converted_size": list(image.size),
                         "exif_orientation_corrected": row["exif_orientation_corrected"]})
    native_counts = {str(k): {"original_photo_positive": sum(k in row["native_class_ids"] for row in items),
                             "selected_photo_positive": sum(k in row["native_class_ids"] for row in selected),
                             "original_instances": sum(row["native_instances"].get(str(k), 0) for row in items)} for k in range(6)}
    public_evidence = {k: value for k, value in class_evidence.items() if k != "archive_member"}
    audit = {"status": "prepared", "recipe": RECIPE, "source": SOURCE_URL, "license": "CC BY 4.0",
             "source_archive_sha256": ARCHIVE_SHA256, "source_archive_bytes": ARCHIVE_BYTES,
             "download_sidecar_sha256": provenance["download_sidecar_sha256"], "extraction": extraction,
             "source_annotation_count": len(items), "source_photos_fully_audited": len(items),
             "native_class_order_evidence": public_evidence, "native_class_counts": native_counts,
             "author_split": None, "source_split": "train_only_unsplit", "max_selected": args.max_selected,
             "selected_photos": len(prepared), "selection_seed": SEED, "selection_policy": "Fixed-seed equal round-robin four crack/spalling presence strata, after source-only diversity and existing-overlap exclusion; no validation scores",
             "source_split_policy": "No author split or scene IDs supplied by inspected archive; all selected photos supplemental TRAIN only. No evaluation split created, existing core/validation/test unchanged.",
             "existing_photos_screened": fingerprint_audit["photos"], "existing_fingerprint_cache_sha256": fingerprint_audit["cache_sha256"],
             "existing_metadata_set_sha256": hashlib.sha256(json.dumps(fingerprint_audit["metadata_sha256"], sort_keys=True).encode()).hexdigest(),
             "overlap_screen": overlap_audit, "suspected_derived_filename_photos": sum(row["suspected_derived_filename"] for row in items),
             "known_labels": NAMES[:2], "unknown_labels": NAMES[2:],
             "native_mapping": {name: ("concrete_crack" if name in CRACK_CLASSES else "concrete_spalling" if name == "spalling" else "unmapped; other source types do not assert any remaining facility label") for name in order},
             "label_policy": "First two photo labels follow native six-class annotation presence/absence. Three crack subtypes merged; only native Spalling maps to spalling. Scaling never maps to spalling; holes never becomes cavity. Remaining five categories unknown, never normal/background.",
             "pixel_policy": "Publisher normalized boxes are not pixel truth; all seven spatial known flags zero and all masks zero, including source-photo negatives.",
             "per_label_train_counts": {label: {"positive": sum(row["targets"][k] == 1 for row in prepared), "negative": sum(row["targets"][k] == 0 for row in prepared), "unknown": sum(row["targets"][k] == -1 for row in prepared)} for k, label in enumerate(NAMES)},
             "per_label_pixel_cells": {label: {"positive": 0, "negative": 0} for label in NAMES},
             "source_environment": "Post-earthquake damaged buildings in Syrian Coast; no verified factory-site or prospective field evaluation",
             "limitations": "All selected source photographs are TRAIN-only. Whole-photo SHA/dHash screen and collapsed representatives do not establish building/scene/crop independence or field performance. Suspected near overlap is excluded, not relabelled. The supplementary photo count is not a measured accuracy gain.",
             "preparation_script_sha256": sha(Path(__file__))}
    save(ROOT / "reports/facility-peccd-data-audit.json", audit)
    audit["audit_file_sha256"] = sha(ROOT / "reports/facility-peccd-data-audit.json")
    save(output / "train.json", {"split": "train", "classes": NAMES, "items": prepared, "audit": audit})
    print(json.dumps({"status": audit["status"], "source_photos_fully_audited": len(items), "selected_photos": len(prepared), "overlap_screen": overlap_audit, "per_label_train_counts": audit["per_label_train_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
