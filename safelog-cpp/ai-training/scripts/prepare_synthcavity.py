"""Prepare partial-label supplemental TRAIN photos from verified cavity masks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]


def mask_union(paths, size):
    union = np.zeros((size[1], size[0]), dtype=bool)
    for path in paths:
        with Image.open(path) as handle:
            mask = np.asarray(handle)
            if handle.size != size or mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 255}):
                raise ValueError(f"Expected same-size binary cavity mask: {path}")
            union |= mask > 0
    return union


def prepare(item):
    image_path, output, classes = item
    number = re.fullmatch(r"render(?:_noise)?(\d+)", image_path.stem).group(1)
    masks = [image_path.with_name(f"{prefix}{number}.png") for prefix in ("gt_cavity", "gt_render_cavities")]
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    pixels = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
    union = mask_union(masks, image.size)
    # Keep full-frame labels; random cropping could remove the positive defect.
    positive = int(union.any())
    targets = [-1] * len(classes)
    targets[classes.index("surface_cavity")] = positive
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    image.save(output, format="JPEG", quality=95)
    return {"image": output.relative_to(ROOT).as_posix(), "source": image_path.relative_to(ROOT).as_posix(),
            "scene_id": f"{image_path.parent.relative_to(ROOT).as_posix()}/{number}",
            "masks": [p.relative_to(ROOT).as_posix() for p in masks], "targets": targets,
            "pixel_sha256": pixels, "cavity_pixels": int(union.sum()), "source_size": [width, height]}


def main():
    source_root = ROOT / "data/synthcavity"
    source = json.loads((source_root / "SOURCE.json").read_text(encoding="utf-8"))
    classes = list(yaml.safe_load((ROOT / "data/dacl10k-yolo/data.yaml").read_text())["names"].values())
    paths = sorted(p for p in source_root.rglob("render*.png") if re.fullmatch(r"render(?:_noise)?\d+", p.stem))
    if not paths: raise ValueError("No publisher rendered RGB photos found")
    output = ROOT / "data/synthcavity-training/images"
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(p, output / f"synth-{i:05d}.jpg", classes) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(prepare, tasks))
    real = json.loads((ROOT / "data/dacl10k-yolo/records.json").read_text(encoding="utf-8"))
    real_hashes = {item["pixel_sha256"] for item in real}
    unique, seen, overlaps, duplicates = [], set(), 0, 0
    for item in items:
        if item["pixel_sha256"] in real_hashes:
            overlaps += 1
        elif item["pixel_sha256"] in seen:
            duplicates += 1
        else:
            unique.append(item)
            seen.add(item["pixel_sha256"])
    if overlaps: raise ValueError("Synthetic photos overlap real source photos; investigate before training")
    positive = sum(item["targets"][classes.index("surface_cavity")] for item in unique)
    audit = {"status": "prepared", "split": "train", "source": source["homepage"], "license": source["license"],
             "rendered_rgb_candidates": len(paths), "supplemental_images": len(unique), "positive_cavity_photos": positive,
             "negative_cavity_photos": len(unique) - positive, "duplicate_pixels_removed": duplicates,
             "procedural_scenes": len({item["scene_id"] for item in unique}),
             "overlap_with_dacl_train_val_test": overlaps, "verified_files": source["verified_files"],
             "label_policy": "Only cavity presence from gt_cavity OR gt_render_cavities; six unannotated classes remain -1 (unknown).",
             "scope": "Procedural supplemental TRAIN photos, not real field photos or independent validation."}
    manifest = {"split": "train", "classes": classes, "items": unique, "audit": audit}
    (output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "reports/facility-round3-data-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__": main()
