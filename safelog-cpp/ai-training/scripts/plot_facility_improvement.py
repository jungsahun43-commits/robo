from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_facility_report import LABELS_KO


def main():
    original = json.loads((ROOT / "reports/facility-optimization-test.json").read_text(encoding="utf-8"))
    dacl = next(e for e in original["experiments"] if e["model"] == "facility-dacl-optimized")
    final = json.loads((ROOT / "reports/facility-presence-test.json").read_text())["per_class"]
    labels = list(final)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = np.arange(len(labels))
    for axis, key, title in zip(axes, ("miss_fraction", "false_positive_rate"),
                               ("항목별 사진 미탐률 — 낮을수록 좋음", "항목별 사진 오탐률 — 낮을수록 좋음")):
        old = [dacl["baseline"][label]["photo"][key] * 100 for label in labels]
        new = [final[label]["combined"][key] * 100 for label in labels]
        bars1 = axis.bar(x - .18, old, .36, label="최초 모델", color="#8395a7")
        bars2 = axis.bar(x + .18, new, .36, label="최종 보강", color="#2878b5")
        axis.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
        axis.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
        axis.set_xticks(x, [LABELS_KO[label] for label in labels])
        axis.set_ylabel("비율 (%)")
        axis.set_title(title, loc="left", pad=14)
        axis.set_ylim(0, min(100, max(old + new) * 1.25 + 3))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", frameon=False)
    fig.suptitle("시설 모델 보강 비교 | 기존 시험 사진 975장", fontsize=17, y=.99)
    fig.text(.02, .01, "사진의 해당 항목 존재 여부만 평가합니다. 결함 위치·현장 안전 판정 정확도는 아닙니다.\n검증 사진으로 모델·탐지 기준을 선택한 뒤 기존 시험 분할에서 비교했습니다. 외부 시설 현장 검증은 필요합니다.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .955))
    output = ROOT / "artifacts/facility-performance-comparison.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
