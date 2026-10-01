from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "sh17"
CLASSES = (
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot",
    "tool", "glasses", "gloves", "helmet", "hands", "head", "medical-suit",
    "shoes", "safety-suit", "safety-vest",
)
# Class IDs must match the author's sh17.yaml, not the README's display order.
# https://github.com/ahmadmughees/SH17dataset/blob/master/sh17.yaml


def prepare_list(source: Path, output: Path) -> int:
    paths: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        image = (DATASET / "images" / name).resolve()
        label = DATASET / "labels" / f"{Path(name).stem}.txt"
        if not image.exists() or not label.exists():
            raise RuntimeError(f"이미지 또는 라벨 누락: {name}")
        paths.append(image.as_posix())
    output.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return len(paths)


def main() -> int:
    train_names = set((DATASET / "train_files.txt").read_text(encoding="utf-8").splitlines())
    val_names = set((DATASET / "val_files.txt").read_text(encoding="utf-8").splitlines())
    overlap = (train_names & val_names) - {""}
    if overlap:
        raise RuntimeError(f"학습·검증 분할 중복: {len(overlap)}개")
    train_count = prepare_list(DATASET / "train_files.txt", DATASET / "train_abs.txt")
    val_count = prepare_list(DATASET / "val_files.txt", DATASET / "val_abs.txt")
    lines = [
        f"path: {DATASET.resolve().as_posix()}",
        f"train: {(DATASET / 'train_abs.txt').resolve().as_posix()}",
        f"val: {(DATASET / 'val_abs.txt').resolve().as_posix()}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SH17 준비 완료: train={train_count}, val={val_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
