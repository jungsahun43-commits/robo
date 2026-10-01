"""Audit partial labels and keep identical/visually similar patches in one split."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/damsegment/source"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def targets_from_categories(categories, classes):
    if not set(categories).issubset({0, 1}): raise ValueError("Unknown publisher category")
    result = [-1] * len(classes)
    for category, label in enumerate(("concrete_crack", "concrete_spalling")):
        result[classes.index(label)] = int(category in categories)
    return result


def dhash(image):
    tiny = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes(), "big")


def grouped_splits(items):
    """Transitive grouping: extension aliases, exact pixels and near dHash."""
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        i, j = root(i), root(j)
        if i != j: parent[max(i, j)] = min(i, j)
    aliases, hashes = {}, {}
    for i, item in enumerate(items):
        for lookup, key in ((aliases, item["patch_id"]), (hashes, item["pixel_sha256"])):
            if key in lookup: union(i, lookup[key])
            else: lookup[key] = i
        for j in range(i):
            if (int(item["dhash"], 16) ^ int(items[j]["dhash"], 16)).bit_count() <= 6:
                union(i, j)
    members = {}
    for i in range(len(items)): members.setdefault(root(i), []).append(i)
    sizes = []
    for indices in members.values():
        signature = min(items[i]["pixel_sha256"] for i in indices)
        group = hashlib.sha256(("damsegment-seed42:" + signature).encode()).hexdigest()
        split = "test" if int(group[:8], 16) % 10 < 2 else "train"
        sizes.append(len(indices))
        for i in indices: items[i].update(group_id=group, split=split)
    return sizes


def candidate(task):
    path, classes, categories, patch_id = task
    with Image.open(path) as handle: image = handle.convert("RGB")
    width, height = image.size
    digest = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
    return {"source": path.relative_to(ROOT).as_posix(), "patch_id": patch_id,
            "pixel_sha256": digest, "dhash": f"{dhash(image):016x}",
            "source_size": [width, height], "targets": targets_from_categories(categories, classes)}


def main():
    source = read(ROOT / "datasets/damsegment_source.json")
    from scripts.optimize_facilities import sha
    verified = []
    for entry in source["files"]:
        archive = ROOT / "data/facility-archives" / entry["local_archive"]
        if archive.stat().st_size != entry["bytes"] or sha(archive) != entry["sha256"]:
            raise ValueError("Publisher archive not verified")
        verified.append({"name": entry["name"], "sha256": entry["sha256"], "bytes": entry["bytes"]})
    classes = list(yaml.safe_load((ROOT / "data/dacl10k-yolo/data.yaml").read_text())["names"].values())
    tasks, checks = [], 0
    for path in sorted((SOURCE / "Damage Detection/Images").glob("*.jpg")):
        doc = read(SOURCE / "Damage Detection/Labels/Pascal VOC" / f"{path.stem}.json")
        categories = {item["category_id"] for item in doc["annotations"]}
        yolo = SOURCE / "Damage Detection/Labels/Yolo" / f"{path.stem}.txt"
        if categories != {int(line.split()[0]) for line in yolo.read_text().splitlines() if line.strip()}:
            raise ValueError("YOLO and JSON class presence differ")
        level = {"E": "Easy", "M": "Medium", "H": "Hard"}[path.stem[0]]
        folder = SOURCE / "Damage Segmentaion" / level
        maskpath = folder / "Labels/Mask" / f"{path.stem}_mask.png"
        with Image.open(maskpath) as handle: mask = np.asarray(handle.convert("RGB"))
        color_masks = [np.all(mask == color, axis=-1) for color in ((255, 0, 0), (0, 0, 255), (0, 0, 0))]
        if not np.all(color_masks[0] | color_masks[1] | color_masks[2]):
            raise ValueError("Unexpected publisher mask color")
        mask_categories = {i for i in (0, 1) if color_masks[i].any()}
        if mask_categories != categories: raise ValueError(f"Mask and JSON class presence differ: {path.name}, mask={mask_categories}, JSON={categories}")
        segdoc = read(folder / "Labels/Pascal VOC" / f"{path.stem}.json")
        if categories != {item["category_id"] for item in segdoc["annotations"]}:
            raise ValueError("Segmentation and detection annotation disagree")
        with Image.open(path) as image, Image.open(folder / "Images" / path.name) as copy:
            if image.size != (doc["image"]["width"], doc["image"]["height"]) or not np.array_equal(np.asarray(image.convert("RGB")), np.asarray(copy.convert("RGB"))):
                raise ValueError("Detection/segmentation photo mismatch")
        if (mask.shape[1], mask.shape[0]) != (doc["image"]["width"], doc["image"]["height"]):
            raise ValueError("Mask dimension mismatch")
        checks += 1
        tasks.append((path, classes, categories, "detection:" + path.stem))
    normal = sorted(p for p in (SOURCE / "Damage Classification/Non-Crack").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    for path in normal: tasks.append((path, classes, set(), "normal:" + path.stem))
    if checks != 1500 or len(normal) != 1000: raise ValueError("Incomplete publisher photos")
    with ThreadPoolExecutor(max_workers=4) as pool: items = list(pool.map(candidate, tasks))
    original = read(ROOT / "data/dacl10k-yolo/records.json")
    existing = {item["pixel_sha256"] for item in original}
    overlaps = [item for item in items if item["pixel_sha256"] in existing]
    items = [item for item in items if item["pixel_sha256"] not in existing]
    sizes = grouped_splits(items)
    unique, seen = [], {}
    for item in items:
        key = item["pixel_sha256"]
        if key in seen:
            if seen[key]["targets"] != item["targets"]: raise ValueError("Conflicting exact duplicate labels")
            if seen[key]["split"] != item["split"]: raise ValueError("Duplicate split leakage")
        else: seen[key] = item; unique.append(item)
    output = ROOT / "data/damsegment-training"
    (output / "images").mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(unique):
        path = output / "images" / f"dam-{index:05d}.jpg"
        with Image.open(ROOT / item["source"]) as handle:
            image = handle.convert("RGB"); image.save(path, quality=95)
        item["image"] = path.relative_to(ROOT).as_posix()
    split_counts = dict(Counter(item["split"] for item in unique))
    label_counts = {split: {label: {"positive": sum(i["targets"][j] == 1 for i in unique if i["split"] == split),
                                   "negative": sum(i["targets"][j] == 0 for i in unique if i["split"] == split),
                                   "unknown": sum(i["targets"][j] == -1 for i in unique if i["split"] == split)}
                           for j, label in enumerate(classes)} for split in ("train", "test")}
    for split in ("train", "test"):
        for label in ("concrete_crack", "concrete_spalling"):
            if min(label_counts[split][label]["positive"], label_counts[split][label]["negative"]) < 10:
                raise ValueError("Insufficient examples in predeclared split; inspect group policy")
    audit = {"status": "prepared", "source": source["homepage"], "license": source["license"], "verified_files": verified,
             "downloaded_rgb_files": 5000, "ambiguous_classification_files_excluded": 1000,
             "identical_segmentation_copies_not_counted_twice": checks, "usable_before_dedup": len(tasks),
             "duplicate_pixels_removed": len(items) - len(unique), "unique_labeled_patches": len(unique),
             "overlap_with_dacl_removed": len(overlaps), "group_count": len(sizes), "largest_group_files": max(sizes),
             "split_counts": split_counts, "per_split_labels": label_counts,
             "group_policy": "Same patch ID/extension, exact pixels or dHash Hamming <=6; connected components, SHA256(seed42:minPixelHash) bucket 0/1 of 10 reserved test",
             "label_policy": "Downloaded masks: 0 crack/red RGB; 1 spalling/blue RGB (opposite the paper's illustrative Fig.5 palette); intact Non-Crack negatives for these two only; other five targets -1",
             "limitation": "Single dam patches; raw scene/campaign IDs unavailable. Similarity grouping cannot prove scene independence. Reserved patch test is not an independent site test."}
    for split in ("train", "test"):
        selected = [item for item in unique if item["split"] == split]
        (output / f"{split}.json").write_text(json.dumps({"split": split, "classes": classes, "items": selected, "audit": audit}, indent=2), encoding="utf-8")
    (ROOT / "reports/facility-round4-data-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__": main()
