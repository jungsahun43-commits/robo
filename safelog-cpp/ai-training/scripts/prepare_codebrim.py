"""Audit author XML and preserve source-photo groups in official CODEBRIM splits."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_facility_target import read, save, sha

MAPPING = {"Crack": "concrete_crack", "Spallation": "concrete_spalling",
           "CorrosionStain": "rust_stain", "ExposedBars": "exposed_rebar",
           "Efflorescence": "efflorescence"}


def labels(node, classes):
    tags = {child.tag: int(child.text) for child in node}
    if set(tags) != {"Background", *MAPPING} or not set(tags.values()) <= {0, 1}:
        raise ValueError("Unexpected author XML labels")
    if tags["Background"] and any(tags[tag] for tag in MAPPING):
        raise ValueError("Background also labeled defective")
    if not tags["Background"] and not any(tags[tag] for tag in MAPPING):
        raise ValueError("Defect without known defect label")
    targets = [-1] * len(classes)
    for tag, name in MAPPING.items(): targets[classes.index(name)] = tags[tag]
    return targets


def parent_id(name):
    match = re.fullmatch(r"(image_\d+)_crop_\d+\.png", name)
    if not match: raise ValueError(f"Unexpected crop filename {name}")
    return match[1]


def candidate(task):
    path, split, target = task
    with Image.open(path) as handle: image = handle.convert("RGB")
    w, h = image.size
    digest = hashlib.sha256(f"{w}x{h}:".encode() + image.tobytes()).hexdigest()
    return {"source": path.relative_to(ROOT).as_posix(), "split": split,
            "targets": target, "parent_id": parent_id(path.name), "pixel_sha256": digest,
            "source_size": [w, h], "domain": "codebrim"}


def main():
    metadata = read(ROOT / "data/codebrim/metadata.json")
    source = next(f for f in metadata["files"] if f["key"] == "CODEBRIM_classification_dataset.zip")
    archive = ROOT / "data/facility-archives/codebrim.zip"
    with archive.open("rb") as stream: md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if archive.stat().st_size != source["size"] or md5 != source["checksum"].removeprefix("md5:"):
        raise ValueError("Official archive MD5 not verified")
    folder = ROOT / "data/codebrim/source/classification_dataset"
    classes = read(ROOT / "data/damsegment-training/train.json")["classes"]
    annotations = {}
    for category in ("background", "defects"):
        for node in ET.parse(folder / "metadata" / f"{category}.xml").getroot():
            key = (category, node.attrib["name"])
            if key in annotations: raise ValueError("Duplicate XML annotation")
            annotations[key] = labels(node, classes)
    tasks = []
    for split in ("train", "val", "test"):
        for category in ("background", "defects"):
            for path in sorted((folder / split / category).glob("*.png")):
                key = (category, path.name)
                if key not in annotations: raise ValueError("Photo without author annotation")
                tasks.append((path, split, annotations[key]))
    expected = {"train": 6481, "val": 616, "test": 632}
    if dict(Counter(t[1] for t in tasks)) != expected: raise ValueError("Publisher photo counts changed")
    items = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, item in enumerate(pool.map(candidate, tasks), 1):
            items.append(item)
            if i % 1000 == 0: print(f"CODEBRIM photo audit {i}/{len(tasks)}", flush=True)
    groups = defaultdict(set)
    for item in items: groups[item["parent_id"]].add(item["split"])
    if any(len(s) > 1 for s in groups.values()): raise ValueError("Author parent photo crosses splits")
    existing = {r["pixel_sha256"] for r in read(ROOT / "data/dacl10k-yolo/records.json")}
    for split in ("train", "test"):
        existing |= {r["pixel_sha256"] for r in read(ROOT / f"data/damsegment-training/{split}.json")["items"]}
    pixels = defaultdict(list)
    for item in items: pixels[item["pixel_sha256"]].append(item)
    excluded_groups, reasons = set(), Counter()
    for digest, copies in pixels.items():
        if len({tuple(i["targets"]) for i in copies}) > 1: raise ValueError("Conflicting pixel-identical labels")
        if digest in existing or len({i["split"] for i in copies}) > 1:
            excluded_groups |= {i["parent_id"] for i in copies}
            reasons["existing_dataset_overlap" if digest in existing else "cross_split_identical_pixels"] += len(copies)
    kept, seen = [], set()
    for item in items:
        if item["parent_id"] in excluded_groups or item["pixel_sha256"] in seen: continue
        seen.add(item["pixel_sha256"]); kept.append(item)
    output = ROOT / "data/codebrim-training"
    (output / "images").mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(kept):
        dest = output / "images" / f"codebrim-{item['split']}-{i:05d}.jpg"
        with Image.open(ROOT / item["source"]) as handle:
            image = handle.convert("RGB"); image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            image.save(dest, quality=95)
        item["image"] = dest.relative_to(ROOT).as_posix()
        item["group_id"] = "codebrim:" + item["parent_id"]
        if i % 1000 == 0: print(f"CODEBRIM training conversion {i}/{len(kept)}", flush=True)
    audit = {"status": "prepared", "source": "https://zenodo.org/records/2620293", "source_md5": md5,
             "source_sha256": sha(archive), "author_counts": expected, "usable_patches": len(kept),
             "parent_photos": len({i["parent_id"] for i in kept}), "publisher_cross_split_parent_groups": 0,
             "excluded_parent_groups": len(excluded_groups), "exclusion_reasons": dict(reasons),
             "split_counts": dict(Counter(i["split"] for i in kept)), "mapping": MAPPING,
             "per_split_labels": {s: {name: {"positive": sum(i["targets"][j] == 1 for i in kept if i["split"] == s),
                                               "negative": sum(i["targets"][j] == 0 for i in kept if i["split"] == s),
                                               "unknown": sum(i["targets"][j] == -1 for i in kept if i["split"] == s)}
                                      for j, name in enumerate(classes)} for s in expected},
             "label_policy": "Five XML defect labels known; wet_surface and surface_cavity always unknown, including background",
             "license": "CODEBRIM custom noncommercial research/educational terms; no raw or modified data redistribution; data-free models shared under same terms",
             "license_sha256": sha(ROOT / "data/codebrim/license.md"),
             "group_policy": "Official train/val/test, each original image ID belongs to exactly one split; exclude entire parent group if exact-pixel cross-split/existing-data overlap",
             "limitation": "Cropped bridge photos; parent-photo independence does not establish independent bridges/sites or smartphone field performance"}
    for split in expected:
        save(output / f"{split}.json", {"split": split, "classes": classes,
                                      "items": [i for i in kept if i["split"] == split], "audit": audit})
    save(ROOT / "reports/facility-target-codebrim-data-audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__": main()
