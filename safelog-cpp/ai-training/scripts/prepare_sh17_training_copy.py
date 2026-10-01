"""Keep all SH17 samples while reducing repeated high resolution JPEG decoding."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_sh17 import CLASSES, DATASET, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="SH17 전체 이미지의 학습용 사본 생성")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_side < 640:
        raise SystemExit("학습용 사본은 최대 변 길이를 640 이상으로 지정하세요.")
    import cv2

    cv2.setNumThreads(1)
    destination = ROOT / "data" / f"sh17-{args.max_side}"
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    splits = {
        name: [value.strip() for value in (DATASET / f"{name}_files.txt").read_text(encoding="utf-8").splitlines() if value.strip()]
        for name in ("train", "val")
    }
    if set(splits["train"]) & set(splits["val"]):
        raise RuntimeError("SH17 학습·검증 분할 중복")
    filenames = splits["train"] + splits["val"]

    def prepare(filename: str) -> None:
        source = DATASET / "images" / filename
        label = DATASET / "labels" / f"{source.stem}.txt"
        ids = Counter(int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
        xml = ET.parse(DATASET / "voc_labels" / f"{source.stem}.xml")
        aliases = {"tools": "tool", "face-mask-medical": "face-mask"}
        names = Counter(aliases.get(item.findtext("name"), item.findtext("name")) for item in xml.findall("object"))
        if Counter({CLASSES[key]: value for key, value in ids.items()}) != names:
            raise RuntimeError(f"공식 YOLO 번호와 VOC 클래스가 다릅니다: {filename}")
        output = destination / "images" / filename
        if not output.exists():
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"이미지 해독 실패: {filename}")
            height, width = image.shape[:2]
            ratio = min(1.0, args.max_side / max(height, width))
            if ratio < 1.0:
                image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
            options = [cv2.IMWRITE_JPEG_QUALITY, 95] if source.suffix.lower() in {".jpg", ".jpeg"} else []
            if not cv2.imwrite(str(output), image, options):
                raise RuntimeError(f"학습용 이미지 저장 실패: {filename}")
        # YOLO boxes use normalized coordinates, which remain valid after resizing.
        shutil.copyfile(label, destination / "labels" / label.name)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, _ in enumerate(executor.map(prepare, filenames), 1):
            if index % 250 == 0 or index == len(filenames):
                print(f"SH17 사본·클래스 검증 {index}/{len(filenames)}", flush=True)

    for split, names in splits.items():
        (destination / f"{split}_abs.txt").write_text(
            "\n".join((destination / "images" / name).resolve().as_posix() for name in names) + "\n", encoding="utf-8"
        )
    lines = [f"path: {destination.resolve().as_posix()}", "train: train_abs.txt", "val: val_abs.txt", "names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(CLASSES)]
    (destination / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "PREPARATION.json").write_text(json.dumps({
        "source": "SH17 version 1", "max_side": args.max_side, "jpeg_quality": 95,
        "images": len(filenames), "splits": {key: len(value) for key, value in splits.items()},
        "class_id_source": "https://github.com/ahmadmughees/SH17dataset/blob/master/sh17.yaml",
        "voc_class_counts_verified": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {destination / 'data.yaml'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
