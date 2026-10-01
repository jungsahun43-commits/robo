from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


def parse_simple_yaml(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names and line.startswith((" ", "\t")) and ":" in line:
            key, value = line.strip().split(":", 1)
            if key.isdigit():
                names[int(key)] = value.strip().strip("'\"")
                continue
        in_names = False
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            try:
                result[key.strip()] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                result[key.strip()] = value
    if names:
        result["names"] = names
    return result


def image_directory(root: Path, split: object) -> Path:
    value = split[0] if isinstance(split, list) else split
    return (root / str(value)).resolve()


def images_for_split(root: Path, split: object) -> list[Path]:
    location = image_directory(root, split)
    if location.is_file():
        images = []
        for line in location.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                candidate = Path(value)
                images.append(candidate if candidate.is_absolute() else (root / candidate).resolve())
        return images
    return [path for path in location.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 구조와 라벨 검사")
    parser.add_argument("yaml", type=Path)
    args = parser.parse_args()
    config = parse_simple_yaml(args.yaml.resolve())
    root_value = config.get("path", ".")
    root = (args.yaml.parent / str(root_value)).resolve()
    if not root.exists():
        root = args.yaml.parent.resolve()
    names = config.get("names", {})
    class_count = len(names) if isinstance(names, dict) else len(names)

    report: dict[str, object] = {"yaml": str(args.yaml), "classes": names, "splits": {}}
    failed = False
    for split in ("train", "val", "test"):
        if split not in config:
            continue
        images = images_for_split(root, config[split])
        counts: Counter[int] = Counter()
        missing = 0
        invalid = 0
        for image in images:
            label = label_for(image)
            if not label.exists():
                missing += 1
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                    valid = len(fields) == 5 and 0 <= class_id < class_count and all(0 <= value <= 1 for value in coordinates)
                except (ValueError, IndexError):
                    valid = False
                    class_id = -1
                if valid:
                    counts[class_id] += 1
                else:
                    invalid += 1
        report["splits"][split] = {
            "images": len(images), "missing_labels": missing,
            "invalid_annotations": invalid, "instances": dict(sorted(counts.items()))
        }
        failed = failed or not images or missing > 0 or invalid > 0

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
