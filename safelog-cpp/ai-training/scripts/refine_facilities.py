"""A reproducible first refinement round. No test images are used for selection."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (("facility-corrosion", "ostrava-corrosion", 40, 12),
               ("facility-dacl", "dacl10k", 24, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    for baseline, dataset, epochs, patience in EXPERIMENTS:
        refined = baseline + "-refined"
        data = f"data/{dataset}-yolo/data.yaml"
        if not args.evaluate_only:
            subprocess.run([sys.executable, "scripts/train.py", "--model", f"runs/{baseline}/weights/best.pt",
                            "--name", refined, "--data", data, "--epochs", str(epochs),
                            "--patience", str(patience), "--batch", "8", "--imgsz", "960",
                            "--cache", "disk", "--workers", "4", "--device", args.device,
                            "--facility-refine"], cwd=ROOT, check=True)
        for name in (baseline, refined):
            for size in (960, 1280):
                suffix = "" if size == 960 else f"-{size}"
                subprocess.run([sys.executable, "scripts/evaluate.py", f"runs/{name}/weights/best.pt", data,
                                "--split", "val", "--imgsz", str(size), "--device", args.device,
                                "--output", f"runs/evaluation-{name}-val{suffix}.json"], cwd=ROOT, check=True)
    print("Refinement and validation complete. Select/calibrate on validation before test evaluation.", flush=True)


if __name__ == "__main__":
    main()
