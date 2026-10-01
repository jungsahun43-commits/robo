from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".config" / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(root / ".config" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".config" / "matplotlib"))
    for variable in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR"):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="SafeLog YOLO 모델 ONNX 변환")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"모델 가중치를 찾을 수 없습니다: {args.weights}")

    from ultralytics import YOLO

    result = YOLO(str(args.weights.resolve())).export(
        format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=True
    )
    print(f"ONNX 변환 완료: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
