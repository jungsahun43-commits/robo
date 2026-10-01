from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "datasets" / "sources.json"
DATA = ROOT / "data"


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            parts = Path(item.filename).parts
            if "__MACOSX" in parts or any(part.startswith("._") for part in parts):
                continue
            target = (destination / item.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"압축 파일에 안전하지 않은 경로가 있습니다: {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def find_yaml(directory: Path) -> Path:
    candidates = [path for path in directory.rglob("*.yaml") if path.name in {"data.yaml", "construction-ppe.yaml"}]
    if not candidates:
        raise RuntimeError("데이터셋 YAML을 찾지 못했습니다.")
    return min(candidates, key=lambda value: len(value.parts))


def normalize_layout(extracted: Path, destination: Path) -> Path:
    yaml_path = find_yaml(extracted)
    source_root = yaml_path.parent
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(source_root), str(destination))
    output_yaml = find_yaml(destination)
    if output_yaml.name != "data.yaml":
        output_yaml.rename(destination / "data.yaml")
    output_yaml = destination / "data.yaml"
    lines = output_yaml.read_text(encoding="utf-8").splitlines()
    normalized: list[str] = []
    path_replaced = False
    for line in lines:
        if line.lstrip().startswith("path:") and not path_replaced:
            normalized.append(f"path: {destination.resolve().as_posix()}")
            path_replaced = True
        elif line.lstrip().startswith("download:"):
            continue
        else:
            normalized.append(line)
    if not path_replaced:
        normalized.insert(0, f"path: {destination.resolve().as_posix()}")
    output_yaml.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    return output_yaml


def preserve_layout(extracted: Path, destination: Path) -> Path | None:
    children = list(extracted.iterdir())
    source_root = children[0] if len(children) == 1 and children[0].is_dir() else extracted
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(source_root), str(destination))
    try:
        return find_yaml(destination)
    except RuntimeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="공식 SafeLog 학습 데이터셋 다운로드")
    available = list(json.loads(SOURCES.read_text(encoding="utf-8")))
    parser.add_argument("dataset", choices=available)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    source = sources[args.dataset]
    destination = DATA / args.dataset
    yaml_path = destination / "data.yaml"
    marker = destination / "SOURCE.json"
    if (yaml_path.exists() or marker.exists()) and not args.force:
        print(f"이미 준비됨: {yaml_path if yaml_path.exists() else destination}")
        return 0

    DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DATA) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "dataset.zip"

        last_percent = -1

        def progress(block: int, size: int, total: int) -> None:
            nonlocal last_percent
            if total > 0:
                percent = min(100, block * size * 100 // total)
                if percent != last_percent:
                    print(f"\r다운로드 {percent:3d}%", end="", flush=True)
                    last_percent = percent

        print(f"출처: {source['homepage']}")
        urllib.request.urlretrieve(source["url"], archive, progress)
        print("\n압축 해제 중...")
        extracted = temporary_path / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        yaml_path = (normalize_layout(extracted, destination)
                     if source.get("layout") == "yolo" else preserve_layout(extracted, destination))

    (destination / "SOURCE.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: {yaml_path or destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
