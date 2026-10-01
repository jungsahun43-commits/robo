from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="학습 CSV에서 현재 진행 상황 확인")
    parser.add_argument("--name", default="sh17-ppe")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "runs" / args.name / "results.csv"
    if not path.exists():
        print(json.dumps({"run": args.name, "status": "preparing"}))
        return 0
    with path.open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("metrics/mAP50(B)") and row.get("time")]
    if not rows:
        print(json.dumps({"run": args.name, "status": "first_epoch"}))
        return 0
    last = rows[-1]
    print(json.dumps({
        "run": args.name, "epoch": int(last["epoch"]), "elapsed_minutes": round(float(last["time"]) / 60, 1),
        "val_map50": round(float(last["metrics/mAP50(B)"]), 4),
        "best_val_map50": round(max(float(row["metrics/mAP50(B)"]) for row in rows), 4),
        "val_map50_95": round(float(last["metrics/mAP50-95(B)"]), 4),
        "train_box_loss": round(float(last["train/box_loss"]), 4),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
