"""Post-training native19 plot; no frozen132 sources or data are changed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "reports/facility-dense-auxiliary-study-comparison.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    if result["technical_verification"]["status"] != "passed" or result["actual_new_training_epochs"] != 6:
        raise ValueError("Only verified real completed training may be plotted")
    entries = result["experiments"]
    labels = ["Original initializer", "Reused baseline", "Native19 spatial task"]
    errors = [e["worst_error"] * 100 for e in entries]
    crack = [e["small_dacl_polygon_area_below_one_percent"]["concrete_crack"]["false_negatives"] for e in entries]
    spalling = [e["small_dacl_polygon_area_below_one_percent"]["concrete_spalling"]["false_negatives"] for e in entries]
    colors = ["#697586", "#3474A6", "#A25522"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    x = np.arange(3)
    axes[0].bar(x, errors, color=colors, width=.62)
    axes[0].axhline(5, color="#C43F3F", linestyle="--", linewidth=1.2, label="Strict target <5% for all12 rates")
    axes[0].set_ylim(0, max(errors) + 5)
    axes[0].set_ylabel("Worst source-VAL FNR/FPR (%)")
    axes[0].set_title("Original full-photo validation")
    axes[0].legend(loc="lower right", fontsize=8)
    for index, value in enumerate(errors): axes[0].text(index, value + .6, f"{value:.2f}%", ha="center", weight="bold")
    axes[1].bar(x, crack, width=.62, color="#3474A6", label="Crack (93 positive cases)")
    axes[1].bar(x, spalling, bottom=crack, width=.62, color="#DB9560", label="Spalling (105 positive cases)")
    axes[1].set_ylabel("False-negative class-photo cases")
    axes[1].set_ylim(0, max(np.asarray(crack) + np.asarray(spalling)) + 18)
    axes[1].set_title("DACL annotated area <1%")
    axes[1].legend(loc="upper left", fontsize=8)
    for index, (a, b) in enumerate(zip(crack, spalling)):
        axes[1].text(index, a / 2, str(a), ha="center", color="white", weight="bold")
        axes[1].text(index, a + b / 2, str(b), ha="center", weight="bold")
        axes[1].text(index, a + b + 1.2, f"Total {a+b}", ha="center", weight="bold")
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    fig.suptitle("Native19 spatial supervision package: actual6 new epochs", fontsize=14)
    fig.text(.05, .025, "Same public VAL repeatedly used for epoch/threshold selection; exploratory, not independent factory accuracy.\n"
             "6,225 existing TRAIN photos with independent19 masks. Small subset:198 class-photo cases, not198 unique photos.", fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.2, left=.08, right=.98, wspace=.27)
    destination = ROOT / "reports/facility-dense-auxiliary-study-comparison.png"
    if destination.exists(): raise ValueError("Preserve completed plot")
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__": main()
