"""Finish a completed refinement round in validation-before-test order."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    for name in ("facility-dacl-refined", "facility-corrosion-refined"):
        path = ROOT / f"runs/{name}/TRAINING.json"
        if not path.exists() or json.loads(path.read_text())["status"] != "complete":
            raise SystemExit(f"Finish refinement first: {name}")
    presence_training = ROOT / "runs/facility-presence/TRAINING.json"
    if not presence_training.exists():
        run("scripts/train_facility_presence.py")
    run("scripts/optimize_facilities.py", "select")
    run("scripts/optimize_facility_presence.py", "select")
    run("scripts/optimize_facilities.py", "test")
    run("scripts/optimize_facility_presence.py", "test")
    profile = json.loads((ROOT / "reports/facility-inference-profile.json").read_text(encoding="utf-8"))
    for name, entry in profile["models"].items():
        weights = f"runs/{name}/weights/best.pt"
        run("scripts/evaluate.py", weights, f"data/{entry['dataset']}-yolo/data.yaml", "--split", "test",
            "--imgsz", str(entry["imgsz"]), "--device", "0")
        run("scripts/export_onnx.py", weights, "--imgsz", str(entry["imgsz"]))
    run("scripts/optimize_facility_presence.py", "export")
    run("scripts/build_facility_optimization_report.py")
    run("scripts/package_models.py", "--require-facilities", "--require-optimization")


if __name__ == "__main__":
    main()
