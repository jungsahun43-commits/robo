from __future__ import annotations

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "chvg"
OUTPUT = ROOT / "data" / "chvg-yolo"
CLASSES = ["person", "head", "helmet_white", "helmet_yellow", "helmet_blue", "helmet_red", "vest", "safety_glass"]
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
SOURCE_MAP = {
    "person": "person", "head": "head", "white": "helmet_white", "yellow": "helmet_yellow",
    "blue": "helmet_blue", "red": "helmet_red", "vest": "vest", "glass": "safety_glass",
}


def split_for(filename: str) -> str:
    scene = filename.split("_jpg.rf.", 1)[0]
    bucket = int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit("먼저 download_dataset.py chvg를 실행하세요.")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    counts: Counter[str] = Counter()
    images: Counter[str] = Counter()
    invalid = 0

    for xml_path in sorted(SOURCE.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename", "")
        image_path = SOURCE / filename
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        if not image_path.exists() or width <= 0 or height <= 0:
            invalid += 1
            continue
        split = split_for(filename)
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        labels: list[str] = []
        for item in root.findall("object"):
            mapped = SOURCE_MAP.get(item.findtext("name", "").strip())
            box = item.find("bndbox")
            if mapped is None or box is None:
                invalid += 1
                continue
            xmin = max(0.0, float(box.findtext("xmin", "0")))
            ymin = max(0.0, float(box.findtext("ymin", "0")))
            xmax = min(float(width), float(box.findtext("xmax", "0")))
            ymax = min(float(height), float(box.findtext("ymax", "0")))
            if xmax <= xmin or ymax <= ymin:
                invalid += 1
                continue
            x, y = (xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height)
            w, h = (xmax - xmin) / width, (ymax - ymin) / height
            labels.append(f"{CLASS_ID[mapped]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            counts[mapped] += 1
        link_or_copy(image_path, image_dir / image_path.name)
        (label_dir / f"{image_path.stem}.txt").write_text(
            ("\n".join(labels) + "\n") if labels else "", encoding="utf-8"
        )
        images[split] += 1

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (OUTPUT / "data.yaml").write_text(
        f"path: {OUTPUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print({"images": dict(images), "instances": dict(counts), "invalid": invalid})
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
