from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".config" / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(root / ".config" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".config" / "matplotlib"))
    for variable in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR"):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="SafeLog PPE YOLO 파인튜닝")
    parser.add_argument("--data", type=Path, default=Path("data/construction-ppe/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="ppe-baseline")
    parser.add_argument("--device", default=None, help="예: 0 또는 cpu; 생략하면 자동 선택")
    parser.add_argument("--cache", choices=("none", "disk", "ram"), default="none")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="같은 이름의 last.pt에서 이어 학습")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"데이터 설정을 찾을 수 없습니다: {args.data}")
    from ultralytics import YOLO

    checkpoint = root / "runs" / args.name / "weights" / "last.pt"
    if args.resume and not checkpoint.exists():
        raise SystemExit(f"이어 학습 체크포인트가 없습니다: {checkpoint}")
    if not args.resume and checkpoint.exists():
        raise SystemExit("이미 존재하는 학습명입니다. --resume 또는 새로운 --name을 사용하세요.")
    if args.resume:
        import torch
        if torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch") == -1:
            print(f"이미 완료된 학습입니다: {args.name}")
            return 0
    model = YOLO(str(checkpoint) if args.resume else args.model)
    options = dict(
        data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        project=str((root / "runs").resolve()), name=args.name,
        exist_ok=True, seed=42, deterministic=True, patience=args.patience, workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        plots=True, save=True, verbose=True,
    )
    if args.device is not None:
        options["device"] = args.device
    model.train(resume=True, **options) if args.resume else model.train(**options)
    (root / "runs" / args.name / "TRAINING.json").write_text(
        json.dumps({"status": "complete", "model": args.model, "data": str(args.data.resolve()),
                    "requested_epochs": args.epochs, "seed": 42, "imgsz": args.imgsz}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
