"""Prepare a bounded ConViD V4 TRAIN-only, positive-only full-photo supplement.

Author folder names assert only their own positive class. No other photo label
or segmentation background is inferred. Source/pick/review ledgers stay ignored.
The 100-per-folder picks are frozen before looking up existing photo fingerprints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_damsegment import dhash
from scripts.prepare_facility_data import NAMES
from scripts.prepare_peccd_supplement import DERIVED_NAME, existing_photos, screen_source, unknown_pixel_target
from scripts.train_facility_target import read, save, sha

SOURCE_URL = "https://data.mendeley.com/datasets/fx3rthfjhy/4"
API_BASE = "https://data.mendeley.com/public-api/datasets/fx3rthfjhy"
RECIPE = "convid_v4_folder_positive_photo_train_only_v1"
FOLDERS = ("crack", "Spalling")
PER_FOLDER = 100
SEED = 52
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": SOURCE_URL}
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_bytes(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def positive_targets(folder):
    if folder not in FOLDERS:
        raise ValueError("Only the two exact author folder labels are supported")
    targets = [-1] * 7
    targets[0 if folder == "crack" else 1] = 1
    return targets


def official_index(folders, files):
    """Validate official folder IDs, finished originals, SHA and direct URLs."""
    identities = {}
    for folder in folders:
        if folder.get("name") in FOLDERS:
            if folder["name"] in identities or not UUID.fullmatch(str(folder.get("id", ""))):
                raise ValueError("Author folder identity is ambiguous")
            identities[folder["name"]] = folder["id"]
    if set(identities) != set(FOLDERS):
        raise ValueError("Required exact author folder labels not found")
    records = []
    for folder in FOLDERS:
        if not isinstance(files.get(folder), list) or not files[folder]:
            raise ValueError("Official folder file list is missing")
        for item in files[folder]:
            details = item.get("content_details", {})
            identity = str(item.get("id", ""))
            expected = f"https://data.mendeley.com/public-files/datasets/fx3rthfjhy/files/{identity}/file_downloaded"
            digest = details.get("sha256_hash", "")
            size = details.get("size")
            name = item.get("filename", "")
            if (not UUID.fullmatch(identity) or item.get("folder_id") != identities[folder]
                    or item.get("status") != "COMPLETED" or not SHA256.fullmatch(str(digest))
                    or details.get("download_url") != expected or details.get("content_type") != "image/jpeg"
                    or not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= 25_000_000
                    or item.get("size") != size or not isinstance(name, str)
                    or Path(name).suffix.lower() not in (".jpg", ".jpeg")
                    or any(c in name for c in ("/", "\\", ":", "\n", "\r")) or name in (".", "..")):
                raise ValueError("File is not a verified official original JPEG metadata record")
            records.append({"id": identity, "folder": folder, "folder_id": identities[folder],
                            "sha256": digest, "url": expected, "filename": name, "bytes": size})
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Official source IDs repeat across folders")
    return {"dataset_url": SOURCE_URL, "version": 4, "records": sorted(records, key=lambda row: (row["folder"], row["id"]))}


def fixed_picks(records):
    """No fingerprints, heldout labels or scores are accepted by this function."""
    rng = random.Random(SEED)
    picked = []
    for folder in FOLDERS:
        candidates = sorted((row for row in records if row["folder"] == folder), key=lambda row: row["id"])
        rng.shuffle(candidates)
        picked.extend(candidates[:PER_FOLDER])
    return picked


def get_metadata(metadata_dir, fetch=False):
    metadata_dir.mkdir(parents=True, exist_ok=True)
    folder_path = metadata_dir / "convid-folders.json"
    if fetch:
        response = requests.get(API_BASE + "/folders/4", headers=HEADERS, timeout=60)
        response.raise_for_status()
        save(folder_path, response.json())
    folders = read(folder_path)
    files = {}
    for folder in FOLDERS:
        path = metadata_dir / f"convid-{folder}-files.json"
        if fetch:
            matched = [row for row in folders if row.get("name") == folder]
            if len(matched) != 1 or not UUID.fullmatch(str(matched[0].get("id", ""))):
                raise ValueError("Required exact author folder identity missing")
            values, offset = [], 0
            while True:
                response = requests.get(API_BASE + "/files", params={"folder_id": matched[0]["id"], "version": 4, "$start": offset, "$limit": 1000}, headers=HEADERS, timeout=60)
                response.raise_for_status()
                page = response.json()
                if not isinstance(page, list):
                    raise ValueError("Official file metadata is not a list")
                values.extend(page)
                if len(page) < 1000:
                    break
                offset += 1000
                if offset > 10_000:
                    raise ValueError("Unexpected unbounded metadata pagination")
            save(path, values)
        files[folder] = read(path)
    index = official_index(folders, files)
    evidence = {path.name: sha(path) for path in (folder_path, *(metadata_dir / f"convid-{folder}-files.json" for folder in FOLDERS))}
    return index, evidence


def download_original(task):
    record, output = task
    identity = record["id"]
    if not UUID.fullmatch(identity) or not SHA256.fullmatch(record["sha256"]):
        raise ValueError("Unexpected source ID/hash")
    expected = f"https://data.mendeley.com/public-files/datasets/fx3rthfjhy/files/{identity}/file_downloaded"
    if record["url"] != expected or urlsplit(record["url"]).netloc != "data.mendeley.com":
        raise ValueError("Source download must use its official public-file URL")
    destination = output / (identity + ".jpg")
    if destination.is_file() and destination.stat().st_size == record["bytes"] and sha(destination) == record["sha256"]:
        return destination
    temporary = destination.with_suffix(".jpg.partial")
    for attempt in range(3):
        try:
            digest, count = hashlib.sha256(), 0
            with requests.get(record["url"], headers=HEADERS, timeout=(30, 60), stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        count += len(block)
                        if count > record["bytes"]:
                            raise ValueError("Download exceeds publisher-declared byte size")
                        digest.update(block)
                        handle.write(block)
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("Individual publisher JPEG bytes/SHA256 mismatch; never use partial source")
            temporary.replace(destination)
            return destination
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable download state")


def inspect_photo(record, source):
    path = source / (record["id"] + ".jpg")
    if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise ValueError("Original bytes changed after download")
    with Image.open(path) as handle:
        handle.verify()
    with Image.open(path) as handle:
        orientation = handle.getexif().get(274, 1)
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size
    if min(width, height) < 1:
        raise ValueError("Empty source image")
    return {**record, "source_image": record["folder"] + "/" + record["filename"],
            "pixel_sha256": hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest(),
            "dhash": dhash(image), "source_size": [width, height],
            "exif_orientation_corrected": orientation not in (1, None),
            "targets": positive_targets(record["folder"]),
            "suspected_derived_filename": bool(DERIVED_NAME.search(Path(record["filename"]).stem))}


def original_manifest_hashes():
    paths = [ROOT / "data/facility-spatial-training/train.json", ROOT / "data/dacl10k-yolo/records.json"]
    paths += [ROOT / f"data/{source}-training/{split}.json" for source, splits in (("damsegment", ("train", "test")), ("codebrim", ("train", "val", "test"))) for split in splits]
    optional = ROOT / "data/s2ds/pixel-audit.json"
    if optional.exists():
        paths.append(optional)
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data/industrial-source-probe")
    parser.add_argument("--fetch-metadata", action="store_true", help="Refetch official metadata only; never download full dataset/archive")
    parser.add_argument("--index-only", action="store_true", help="Freeze official index and source-only picks, without downloads or existing-data lookup")
    args = parser.parse_args()
    output = ROOT / "data/convid-training"
    if (output / "train.json").exists():
        raise ValueError("Preserve an already prepared experiment")
    output.mkdir(parents=True, exist_ok=True)
    index, metadata_evidence = get_metadata(args.metadata_dir, args.fetch_metadata)
    index_path = output / "SOURCE-INDEX.json"
    index_bytes = canonical_bytes(index)
    if index_path.exists() and index_path.read_bytes() != index_bytes:
        raise ValueError("Frozen official source index changed")
    index_path.write_bytes(index_bytes)
    picks = fixed_picks(index["records"])
    pick_document = {"seed": SEED, "per_folder_limit": PER_FOLDER, "picks": picks,
                     "policy": "Source-metadata-only selection frozen before existing photograph fingerprints or heldout data are read; no replacement after screening"}
    pick_path = output / "SOURCE-PICKS.json"
    pick_bytes = canonical_bytes(pick_document)
    if pick_path.exists() and pick_path.read_bytes() != pick_bytes:
        raise ValueError("Frozen pre-screen source picks changed")
    pick_path.write_bytes(pick_bytes)
    print(json.dumps({"status": "source_picks_frozen", "official_folder_photos": dict(Counter(row["folder"] for row in index["records"])),
                      "picked": dict(Counter(row["folder"] for row in picks)), "download_bytes": sum(row["bytes"] for row in picks),
                      "source_index_sha256": sha(index_path), "source_pick_sha256": sha(pick_path)}), flush=True)
    if args.index_only:
        return
    original_hashes = original_manifest_hashes()
    source = output / "originals"
    source.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, _ in enumerate(pool.map(download_original, [(row, source) for row in picks]), 1):
            if count % 20 == 0:
                print(f"ConViD individual official JPEG SHA verified {count}/{len(picks)}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(lambda row: inspect_photo(row, source), picks))
    save(output / "source-audit.json", {"metadata_sha256": metadata_evidence, "items": items})
    previous, previous_hashes, fingerprint_audit = existing_photos(output / "existing-fingerprints.json")
    selected, pending, overlap_audit = screen_source(items, previous, previous_hashes, "convid-near:")
    save(output / "overlap-review.json", {"status": "detected_candidates_excluded_pending_review", "items": pending, "audit": overlap_audit})
    if not selected:
        raise ValueError("No eligible source photographs after conservative overlap/diversity screening")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    pixel_path = output / "masks/photo-unknown.npz"
    unknown_pixel_target(pixel_path)
    prepared = []
    for row in selected:
        original = source / (row["id"] + ".jpg")
        if sha(original) != row["sha256"]:
            raise ValueError("Official source JPEG changed before conversion")
        with Image.open(original) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        destination = output / "images" / (row["id"] + ".jpg")
        image.save(destination, quality=95)
        prepared.append({"image": destination.relative_to(ROOT).as_posix(), "targets": row["targets"], "domain": "convid",
                         "source_split": "train_only_unsplit", "author_split": None, "scene_unknown": True,
                         "source_id": row["id"], "source_folder": row["folder"], "source_sha256": row["sha256"],
                         "source_original": original.relative_to(ROOT).as_posix(), "pixel_target": pixel_path.relative_to(ROOT).as_posix(),
                         "pixel_target_sha256": sha(pixel_path), "prepared_image_sha256": sha(destination),
                         "pixel_sha256": row["pixel_sha256"], "group_id": row["group_id"],
                         "source_size": row["source_size"], "converted_size": list(image.size),
                         "exif_orientation_corrected": row["exif_orientation_corrected"]})
    if original_manifest_hashes() != original_hashes:
        raise ValueError("Original TRAIN/validation/test manifests changed during supplemental preparation")
    audit = {"status": "prepared", "recipe": RECIPE, "source": SOURCE_URL, "license": "CC BY 4.0",
             "source_version": 4, "source_index_path": index_path.relative_to(ROOT).as_posix(), "source_index_sha256": sha(index_path),
             "source_pick_sha256": sha(pick_path), "metadata_evidence_sha256": metadata_evidence,
             "official_folder_photo_counts": dict(Counter(row["folder"] for row in index["records"])),
             "max_selected": PER_FOLDER * 2, "selection_seed": SEED,
             "picked_before_existing_lookup": len(picks), "picked_per_folder": dict(Counter(row["folder"] for row in picks)),
             "verified_original_photos": len(items), "verified_original_bytes": sum(row["bytes"] for row in picks),
             "selected_photos": len(prepared), "selected_per_folder": dict(Counter(row["source_folder"] for row in prepared)),
             "selection_policy": "Fixed seed 52 selects up to 100 exact author crack-folder and 100 Spalling-folder photos before existing-data lookup. Screening only excludes/collapses these frozen picks; shortages are not refilled; no heldout labels or model scores used.",
             "source_split": "train_only_unsplit", "author_split": None,
             "original_manifests_unchanged_sha256": original_hashes,
             "existing_photos_screened": fingerprint_audit["photos"], "existing_fingerprint_cache_sha256": fingerprint_audit["cache_sha256"],
             "existing_metadata_set_sha256": hashlib.sha256(canonical_bytes(fingerprint_audit["metadata_sha256"])).hexdigest(),
             "overlap_screen": overlap_audit, "photo_label_policy": "Exactly the author folder class is positive; all six other labels unknown, including the other crack/spalling class. No confirmed absences or normal photos are asserted.",
             "pixel_policy": "No native segmentation annotations were supplied; all seven spatial known flags zero and all masks zero. Folder labels never become pixel masks or background truth.",
             "per_label_train_counts": {label: {"positive": sum(row["targets"][k] == 1 for row in prepared), "negative": sum(row["targets"][k] == 0 for row in prepared), "unknown": sum(row["targets"][k] == -1 for row in prepared)} for k, label in enumerate(NAMES)},
             "per_label_pixel_cells": {label: {"positive": 0, "negative": 0} for label in NAMES},
             "author_scene_ids_available": False, "source_evaluation_created": False,
             "source_environment": "Author describes consumer-phone photographs of outdoor concrete defects in Pune, India; no verified factory-site evaluation.",
             "limitations": "Positive-only folder labels do not establish absence or support source false-positive evaluation. Whole-photo SHA/dHash screening and representative collapse do not prove crop, scene, building, site or photographer independence. Extra TRAIN photographs are not a measured accuracy gain. Public report contains aggregate counts/checksums; raw source photos and detailed IDs/paths remain ignored.",
             "preparation_script_sha256": sha(Path(__file__))}
    public_audit = ROOT / "reports/facility-convid-data-audit.json"
    # Keep local inventory paths private; the trainer needs its index path only
    # inside the ignored manifest. Aggregate public hashes remain reproducible.
    save(public_audit, {key: value for key, value in audit.items() if key != "source_index_path"})
    audit["audit_file_sha256"] = sha(public_audit)
    save(output / "train.json", {"split": "train", "classes": NAMES, "items": prepared, "audit": audit})
    print(json.dumps({"status": "prepared", "selected_photos": len(prepared), "overlap_screen": overlap_audit,
                      "per_label_train_counts": audit["per_label_train_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
