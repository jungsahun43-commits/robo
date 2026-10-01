from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ("ppe-baseline", "fire-smoke", "chvg-ppe", "sh17-ppe", "ppe-sh17-transfer")


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
        archive.write(ROOT / "reports" / "selected-models.json", "reports/selected-models.json")
        archive.write(ROOT / "reports" / "TRAINING_RESULTS_KO.md", "reports/TRAINING_RESULTS_KO.md")
        archive.writestr("MODELS_README_KO.md", """# SafeLog 학습 모델 사용법

1. GitHub의 feature/ai-engine 브랜치를 내려받는다.
2. 이 ZIP의 내용을 safelog-cpp/ai-training 안에 압축 해제한다.
   ai-training/models/*.pt와 ai-training/reports/selected-models.json이 있어야 한다.
3. PowerShell에서 ai-training 폴더로 이동해 ./setup_windows.ps1을 실행한다.
   NVIDIA GPU가 없는 PC는 ./setup_windows.ps1 -CpuOnly를 사용한다.
4. ./start_ai_server.ps1 -AllModels를 실행한다.
   앱의 AI 서버 주소는 http://서버PC의LAN주소:8080이다. 같은 PC에서 테스트하면 http://127.0.0.1:8080이다.
5. 모델 선택 및 성능은 reports/TRAINING_RESULTS_KO.md를 확인한다.

pt는 Python AI 서버에서 사용하고, onnx는 향후 C++ 런타임 연결용이다.
이 ZIP에는 데이터셋 원본과 개발용 가상환경이 포함되어 있지 않다.
datasets/sources.json의 출처와 라이선스 정보를 확인한다.
AI 출력은 안전관리자가 검토해야 하는 제안이다.
""")
    print(f"팀 공유 모델 ZIP: {bundle} ({bundle.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
