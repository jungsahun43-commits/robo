"""Derive YOLO region boxes from public facility segmentation annotations.

These are visible defect regions, not structural diagnoses or source benchmark scores.
Original archives, images and annotations are never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DACL_MAPPING = {
    # This historical model key combines Crack + Alligator Crack, not material classes.
    # The API canonicalizes it to surface_crack.
    "Crack": "concrete_crack", "ACrack": "concrete_crack",
    "Spalling": "concrete_spalling", "Rust": "rust_stain",
    "ExposedRebars": "exposed_rebar", "Wetspot": "wet_surface",
    "Efflorescence": "efflorescence", "Cavity": "surface_cavity",
}
NAMES = list(dict.fromkeys(DACL_MAPPING.values()))


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_item(task: tuple[Path, Path | dict, str, str], output: Path, names: list[str], max_side: int) -> dict:
    source, annotation, split, stem = task
    with Image.open(source) as handle:
        orientation = handle.getexif().get(274, 1)
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size
    pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
    boxes = []
    rejected = 0
    ignored = Counter()
    if isinstance(annotation, dict):
        if (annotation["imageWidth"], annotation["imageHeight"]) != (width, height):
            raise ValueError(f"Annotation/image size mismatch: {source}")
        for shape in annotation["shapes"]:
            label = shape["label"]
            if label not in DACL_MAPPING:
                ignored[label] += 1
                continue
            points = shape["points"]
            if shape.get("shape_type", "polygon") != "polygon" or len(points) < 3:
                raise ValueError(f"Unexpected polygon annotation: {source}")
            if not all(len(p) == 2 and all(math.isfinite(float(v)) for v in p) for p in points):
                raise ValueError(f"Invalid polygon coordinates: {source}")
            xs, ys = zip(*points)
            x1, x2 = max(0., min(xs)), min(float(width), max(xs))
            y1, y2 = max(0., min(ys)), min(float(height), max(ys))
            if x2 <= x1 or y2 <= y1:
                rejected += 1
                continue
            boxes.append((names.index(DACL_MAPPING[label]), x1, y1, x2, y2))
    else:
        mask = np.asarray(Image.open(annotation).convert("L"))
        if mask.shape != (height, width):
            raise ValueError(f"Mask/image size mismatch: {source}")
        count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        for x, y, w, h, area in stats[1:count]:
            # Ignore mask specks <64 ORIGINAL pixels; report their count explicitly.
            if area < 64:
                rejected += 1
                continue
            boxes.append((0, float(x), float(y), float(x + w), float(y + h)))
    # Same-label polygons can overlap in a semantic source. Deduplicate identical boxes.
    unique_boxes = list(dict.fromkeys(boxes))
    duplicate_boxes = len(boxes) - len(unique_boxes)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image_dir, label_dir = output / "images" / split, output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_dir / f"{stem}.jpg", quality=95)
    label_dir.joinpath(f"{stem}.txt").write_text("\n".join(
        f"{class_id} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"
        for class_id, x1, y1, x2, y2 in unique_boxes
    ) + ("\n" if unique_boxes else ""), encoding="utf-8")
    return {"source": str(source.relative_to(ROOT / "data")), "split": split,
            "stem": stem, "pixel_sha256": pixel_hash, "file_sha256": digest_file(source),
            "boxes": dict(Counter(names[b[0]] for b in unique_boxes)),
            "rejected_regions": rejected, "identical_boxes_removed": duplicate_boxes,
            "exif_orientation_corrected": orientation not in (1, None),
            "ignored_labels": dict(ignored)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("dacl10k", "ostrava-corrosion"))
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source = ROOT / "data" / args.dataset
    output = ROOT / "data" / f"{args.dataset}-yolo"
    if (output / "PREPARATION.json").exists():
        manifest_path = output / "PREPARATION.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "api_label_aliases" not in manifest:
            manifest["api_label_aliases"] = {"concrete_crack": "surface_crack"} if args.dataset == "dacl10k" else {}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Already prepared: {output}")
        return 0
    if not (source / "SOURCE.json").exists():
        raise SystemExit("Download and checksum verification must finish first.")
    tasks = []
    names = NAMES if args.dataset == "dacl10k" else ["metal_corrosion"]
    if args.dataset == "dacl10k":
        annotation_dirs = list(source.rglob("annotations/train"))
        if len(annotation_dirs) != 1:
            raise ValueError(f"Expected one annotations/train directory, got {annotation_dirs}")
        base = annotation_dirs[0].parent.parent
        for original, count in (("train", 6935), ("validation", 975)):
            paths = sorted((base / "annotations" / original).glob("*.json"))
            if len(paths) != count:
                raise ValueError(f"Unexpected {original} count: {len(paths)} != {count}")
            for path in paths:
                annotation = json.loads(path.read_text(encoding="utf-8"))
                image_name = annotation.get("imageName") or Path(annotation["imagePath"]).name
                image_path = base / "images" / original / image_name
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                # Source validation is held out as our test; source training alone supplies train/val.
                split = "test" if original == "validation" else (
                    "val" if int(hashlib.sha256(image_name.encode()).hexdigest(), 16) % 10 == 0 else "train")
                tasks.append((image_path, annotation, split, f"dacl-{path.stem}"))
    else:
        base = source / "corrosion-in-industrial-complexes-in-ostrava"
        for original, split, expected in (("train", "train", 75), ("validation", "val", 15), ("test", "test", 15)):
            images = sorted((base / original / "images").glob("*.jpg"))
            if len(images) != expected:
                raise ValueError(f"Unexpected {original} count: {len(images)} != {expected}")
            for image in images:
                mask = base / original / "masks" / f"{image.stem}.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                tasks.append((image, mask, split, f"ostrava-{image.stem}"))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(lambda task: write_item(task, output, names, args.max_side), tasks), 1):
            records.append(record)
            if i % 250 == 0 or i == len(tasks):
                print(f"Prepared {i}/{len(tasks)}", flush=True)
    # Exact duplicates across partitions invalidate evaluation; fail before generating training YAML.
    for hash_key in ("pixel_sha256", "file_sha256"):
        seen = {}
        for record in records:
            old = seen.setdefault(record[hash_key], record)
            if old["split"] != record["split"]:
                raise ValueError(f"Cross-split duplicate ({hash_key}): {old['source']} / {record['source']}")
    counts = Counter(r["split"] for r in records)
    report = {"source": args.dataset, "images": len(records), "split_counts": dict(counts),
              "names": names, "max_side": args.max_side,
              "label_type": "derived bounding regions from semantic masks/polygons; not official segmentation benchmark",
              "class_mapping": DACL_MAPPING if args.dataset == "dacl10k" else {"mask>0 connected component": "metal_corrosion"},
              "api_label_aliases": {"concrete_crack": "surface_crack"} if args.dataset == "dacl10k" else {},
              "minimum_mask_area_original_pixels": 64 if args.dataset == "ostrava-corrosion" else None,
              "split_method": "official validation -> held-out test; SHA256 filename modulo 10 internal validation" if args.dataset == "dacl10k" else "official train/validation/test preserved",
              "cross_split_exact_file_and_decoded_pixel_duplicates": 0,
              "limitation": "Source scene/bridge identifiers unavailable; exact-image checks do not establish scene independence.",
              "counts_by_split": {s: dict(sum((Counter(r["boxes"]) for r in records if r["split"] == s), Counter())) for s in counts},
              "rejected_regions": sum(r["rejected_regions"] for r in records),
              "exif_orientation_corrected_images": sum(r["exif_orientation_corrected"] for r in records),
              "identical_boxes_removed": sum(r["identical_boxes_removed"] for r in records)}
    (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "PREPARATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    yaml = f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
