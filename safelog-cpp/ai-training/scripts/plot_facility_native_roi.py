"""Plot measured native-ROI-study aggregates; never open images or model files.

The PNG contains source-validation maxima and small-positive photo miss counts.
Its sidecar records the aggregate and figure hashes, without individual cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

RUNS = (
    "facility-presence-target-roi-control",
    "facility-presence-target-native-roi-control",
    "facility-presence-target-native-roi-native",
)
TITLES = ("Initial ROI\n640 px", "Pre-downsampled ROI\n640 px", "Direct native ROI\n640 px")
LEGEND_TITLES = ("Initial ROI (640)", "Pre-downsampled ROI (640)", "Direct native ROI (640)")
COLORS = ("#737373", "#0072B2", "#D55E00")
DOMAINS = ("dacl", "damsegment", "codebrim")
DOMAIN_COUNTS = (710, 424, 611)
TASKS = ("concrete_crack", "concrete_spalling")
SMALL_COUNTS = (93, 105)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, name):
    require(type(value) is int and value >= 0, f"Invalid {name} count")
    return value


def rate(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"Invalid {name} rate")
    return float(value)


def measurements(result):
    """Check aggregate denominators and reconstruct the plotted twelve-rate max."""
    require(isinstance(result, dict)
            and result.get("schema") == "facility_native_roi_study_comparison_v1",
            "Use the native-ROI-study aggregate report")
    require(result.get("deployed") is False
            and result.get("source_test_inference_executed") is False
            and result.get("label_changes") == 0
            and result.get("additional_expert_confirmed_labels") == 0,
            "Plot scope requires an unchanged-label, undeployed source-VAL study")
    entries = result.get("experiments")
    require(isinstance(entries, list) and len(entries) == 3
            and all(isinstance(entry, dict) for entry in entries), "Three measured models are required")
    require(len({entry.get("run") for entry in entries}) == 3
            and {entry.get("run") for entry in entries} == set(RUNS), "Unexpected native ROI models")
    by_run = {entry["run"]: entry for entry in entries}
    rows = []
    for index, name in enumerate(RUNS):
        entry = by_run[name]
        require(entry.get("imgsz") == 640, "Input resolution changed")
        epochs = count(entry.get("actual_epochs"), "actual epoch")
        require(epochs > 0 and (index == 0 or epochs == 6), "Measured six-epoch native ROI pair required")
        per_class = entry.get("per_class", {})
        require(set(per_class) == set(TASKS), "Both evaluated target classes are required")
        observed = []
        for task in TASKS:
            domains = per_class[task].get("domains", {})
            require(set(domains) == set(DOMAINS), "All three validation domains are required")
            for domain, total in zip(DOMAINS, DOMAIN_COUNTS):
                point = domains[domain]
                tp, fn, fp, tn = (count(point.get(key), key) for key in ("tp", "fn", "fp", "tn"))
                require(tp + fn > 0 and fp + tn > 0 and tp + fn + fp + tn == total,
                        "Source validation support changed")
                require(point.get("positive_photos") == tp + fn
                        and point.get("negative_photos") == fp + tn, "Aggregate support differs from confusion counts")
                for key, actual in (("fnr", fn / (tp + fn)), ("fpr", fp / (fp + tn))):
                    require(math.isclose(rate(point.get(key), key), actual, rel_tol=0, abs_tol=1e-12),
                            "Recorded rate differs from confusion counts")
                    observed.append(actual)
        worst = max(observed)
        require(math.isclose(rate(entry.get("worst_error"), "maximum error"), worst,
                             rel_tol=0, abs_tol=1e-12), "Headline maximum differs from its twelve rates")
        require(entry.get("target_passed") is all(value < .05 for value in observed),
                "Strict five-percent flag differs from measured rates")
        small = entry.get("small_dacl_polygon_area_below_one_percent", {})
        require(set(small) == set(TASKS), "Both small-positive subsets are required")
        misses = []
        for task, expected in zip(TASKS, SMALL_COUNTS):
            point = small[task]
            require(count(point.get("positive_photos"), "small positive") == expected,
                    "Small-positive denominators must remain 93 and 105")
            fn = count(point.get("false_negatives"), "small false negative")
            require(fn <= expected, "Small-positive misses exceed their support")
            if "fnr" in point:
                require(math.isclose(rate(point["fnr"], "small false negative"), fn / expected,
                                     rel_tol=0, abs_tol=1e-12), "Small-positive rate differs from counts")
            misses.append(fn)
        rows.append({"run": name, "imgsz": entry["imgsz"], "actual_epochs": epochs,
                     "maximum_fnr_fpr": worst, "small_false_negatives": misses,
                     "small_positive_cases": list(SMALL_COUNTS)})
    return rows


def render_plot(rows, output, *, title="Native ROI pixel comparison on source validation"):
    """Render already checked aggregate measurements; no intervals are invented."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.25))
    left, right = axes
    maxima = [100 * row["maximum_fnr_fpr"] for row in rows]
    bars = left.bar(range(3), maxima, color=COLORS, width=.58)
    left.bar_label(bars, labels=[f"{value:.2f}%" for value in maxima], padding=4, fontsize=10)
    left.axhline(5, color="#228833", linestyle="--", linewidth=1.3)
    left.text(.98, 5, "Strict target: <5%", transform=left.get_yaxis_transform(),
              ha="right", va="bottom", color="#228833", fontsize=8)
    left.set_xticks(range(3), TITLES)
    left.set_ylim(0, max(10, max(maxima) * 1.24))
    left.set_ylabel("Largest crack/spalling FNR or FPR (%)")
    left.set_title("A  Maximum across 12 source-VAL rates", loc="left", fontsize=10, pad=12)
    width = .23
    for index, row in enumerate(rows):
        positions = [task + (index - 1) * width for task in range(2)]
        bars = right.bar(positions, row["small_false_negatives"], width=width,
                         color=COLORS[index], label=LEGEND_TITLES[index])
        right.bar_label(bars, labels=[f"{fn}/{n}" for fn, n in
                                     zip(row["small_false_negatives"], SMALL_COUNTS)],
                        padding=3, fontsize=8)
    right.set_xticks(range(2), ("Crack (n=93)", "Spalling (n=105)"))
    highest = max(fn for row in rows for fn in row["small_false_negatives"])
    right.set_ylim(0, max(10, highest * 1.32))
    right.set_ylabel("False-negative class-photo cases")
    right.set_title("B  Small DACL positives (<1% polygon area)", loc="left", fontsize=10, pad=12)
    right.legend(frameon=False, fontsize=8, loc="upper left")
    for axis in axes:
        axis.grid(axis="y", alpha=.22)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13, y=.98)
    fig.text(.5, .025,
             "Repeated source-VAL selection; no held-out TEST or factory accuracy. Small cases are class-photo pairs; one photo may count twice.\n"
             "Lower is better. Matched640PNG TRAIN detail pixels; existing targets and full-photo VAL; photo presence only, without location or structural safety decisions.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .93), w_pad=3)
    fig.savefig(output, dpi=220, facecolor="white", metadata={"Title": title})
    plt.close(fig)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/facility-native-roi-study-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/facility-native-roi-study-comparison.png")
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output.resolve()
    sidecar = output.with_suffix(".plot.json")
    require(output.suffix.lower() == ".png" and source not in (output, sidecar), "Use a separate PNG and provenance output")
    require(not output.exists() and not sidecar.exists(), "Preserve existing plot evidence")
    result = json.loads(source.read_text(encoding="utf-8"))
    rows = measurements(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_plot(rows, output)
    provenance = {"schema": "facility_native_roi_aggregate_plot_v1", "source_aggregate_sha256": sha(source),
                  "plot_sha256": sha(output), "script_sha256": sha(Path(__file__)),
                  "metric": "Maximum of 12 source-VAL crack/spalling FNR/FPR rates; small-positive class-photo FN counts",
                  "source_test_inference_executed": False, "industrial_field_performance_measured": False,
                  "deployed": False, "experiments": rows}
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
