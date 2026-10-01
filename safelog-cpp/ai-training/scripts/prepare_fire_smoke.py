from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "indoor-fire-smoke"


def main() -> int:
    yaml_path = DATASET / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit("먼저 download_dataset.py indoor-fire-smoke를 실행하세요.")
    yaml_path.write_text(
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: fire\n  1: smoke\n",
        encoding="utf-8",
    )
    print(f"준비 완료: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
