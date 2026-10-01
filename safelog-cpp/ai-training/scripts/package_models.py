from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ("ppe-baseline", "fire-smoke", "chvg-ppe", "sh17-ppe")


def main() -> int:
    registry = []
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for run in RUNS:
        metrics = json.loads((ROOT / "runs" / f"evaluation-{run}.json").read_text(encoding="utf-8"))
        files = []
        for suffix in ("pt", "onnx"):
            source = ROOT / "runs" / run / "weights" / f"best.{suffix}"
            target = models / f"{run}.{suffix}"
            shutil.copyfile(source, target)
            files.append({"path": f"models/{target.name}", "bytes": target.stat().st_size,
                          "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        registry.append({"name": run, "files": files, "evaluation_split": metrics.get("split", "test"),
                         "metrics": {key: metrics[key] for key in ("precision", "recall", "map50", "map50_95")}})
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    manifest = report_dir / "model-registry.json"
    manifest.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    bundle = artifacts / "safelog-trained-models.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in registry:
            for item in entry["files"]:
                archive.write(ROOT / item["path"], item["path"])
        archive.write(manifest, "reports/model-registry.json")
        archive.write(ROOT / "datasets" / "sources.json", "datasets/sources.json")
    print(f"팀 공유 모델 ZIP: {bundle} ({bundle.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
