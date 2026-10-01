"""Choose the compliance detector using validation only; never use test to select."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK_CLASSES = ("no_helmet", "no_goggle", "no_gloves", "no_boots")


def main() -> int:
    scores = {}
    for run in ("ppe-baseline", "ppe-sh17-transfer"):
        path = ROOT / "runs" / "selection" / f"{run}-val.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["split"] != "val":
            raise RuntimeError("모델 선택에는 val 분할만 사용합니다.")
        scores[run] = sum(report["per_class"][name]["map50"] for name in RISK_CLASSES) / len(RISK_CLASSES)
    selected = max(scores, key=scores.get)
    output = ROOT / "reports" / "selected-models.json"
    output.parent.mkdir(exist_ok=True)
    result = {"primary_ppe": selected, "fire": "fire-smoke", "auxiliary": ["chvg-ppe", "sh17-ppe"],
              "selection_split": "val", "criterion": "mean AP50 of four explicit missing-PPE classes",
              "scores": scores}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
