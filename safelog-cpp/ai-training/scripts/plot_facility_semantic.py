"""Plot checked semantic-feature source-VAL aggregates; no image/model inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scripts.report_facility_small_region import TARGETS
from scripts.report_facility_retention import RETENTION_GATE, retention_measurements
from scripts.report_facility_native_roi import validate_measurements

RUNS = ("facility-presence-target-roi-control", "facility-presence-target-head-lr-low",
        "facility-presence-target-semantic-features")
TITLES = ("Initial ROI\n640 px", "Reused low-head-LR\n640 px", "Frozen ConvNeXt features\n640 px")
COLORS = ("#737373", "#0072B2", "#D55E00")


def require(condition, message):
    if not condition: raise ValueError(message)


def measurements(result):
    require(result.get("schema") == "facility_semantic_study_comparison_v1", "Semantic aggregate report is required")
    require(result.get("deployed") is False and result.get("app_model_promoted") is False
            and result.get("source_test_inference_executed") is False, "Only undeployed source-VAL measurements may be plotted")
    require(all(type(result.get(k)) is int and result[k] == 0 for k in
        ("label_changes", "additional_expert_confirmed_labels", "new_photo_targets", "new_pixel_targets", "new_control_training_epochs"))
        and type(result.get("actual_new_completed_training_epochs")) is int and result["actual_new_completed_training_epochs"] == 6
        and result.get("previous_training_epochs_counted_again") is False
        and result.get("repeated_source_val_adaptation") is True
        and result.get("equal_compute_or_parameter_budget_asserted") is False,
        "New truth, independent selection or equal compute cannot be asserted")
    entries = result.get("experiments")
    require(isinstance(entries, list) and [e.get("run") for e in entries] == list(RUNS), "Exactly the fixed three models are required")
    from scripts.report_facility_detail import GATE
    actual = validate_measurements(entries, {"reference": RUNS[0], "control": RUNS[1], "treatment": RUNS[2], "research_candidate_gate": GATE})
    retention = retention_measurements(entries, RETENTION_GATE)
    require(result.get("retention_gate") == retention and result.get("research_gate") == actual["research_gate"],
            "Recorded nomination gates differ from actual aggregate measurements")
    rows = []
    for index, entry in enumerate(entries):
        require(type(entry.get("imgsz")) is int and entry["imgsz"] == 640
                and type(entry.get("actual_epochs")) is int and entry["actual_epochs"] > 0
                and (index == 0 or entry["actual_epochs"] == 6), "Actual input/epoch budget changed")
        rows.append({"run": entry["run"], "maximum_fnr_fpr": entry["worst_error"],
            "small_false_negatives": [entry["small_dacl_polygon_area_below_one_percent"][task]["false_negatives"] for task in TARGETS],
            "small_positive_cases": [93, 105],
            "dacl_exposed_rebar_ap": retention["dacl_exposed_rebar"][("reference_ap", "control_ap", "treatment_ap")[index]],
            "maximum_other_ap_drop_vs_reference": retention["worst_other_class_ap_decline_by_model"][index]["maximum_other_ap_drop_vs_reference"]})
    return rows


def render_plot(rows, output, *, outcomes=None):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))
    maximum, iron, other, small = axes.flat
    rates = [row["maximum_fnr_fpr"] * 100 for row in rows]
    bars = maximum.bar(range(3), rates, color=COLORS, width=.58)
    maximum.bar_label(bars, labels=[f"{v:.2f}%" for v in rates], padding=4, fontsize=10)
    maximum.axhline(5, color="#228833", linestyle="--", linewidth=1.2)
    maximum.set_ylim(0, max(10, max(rates) * 1.24))
    maximum.set_ylabel("Largest crack/spalling FNR or FPR (%)")
    maximum.set_title("A  Maximum of 12 source-VAL error rates; strict target <5%", loc="left", fontsize=10)
    ap = [row["dacl_exposed_rebar_ap"] for row in rows]
    bars = iron.bar(range(3), ap, color=COLORS, width=.58)
    iron.bar_label(bars, labels=[f"{v:.4f}" for v in ap], padding=4, fontsize=10)
    iron.axhline(ap[1] + .02, color="#228833", linestyle="--", linewidth=1.2)
    iron.set_ylim(0, 1); iron.set_ylabel("Ranking AP on known DACL labels")
    iron.set_title("B  Exposed rebar; required recovery +0.02 vs reused control", loc="left", fontsize=10)
    drops = [row["maximum_other_ap_drop_vs_reference"] for row in rows]
    bars = other.bar(range(3), drops, color=COLORS, width=.58)
    other.bar_label(bars, labels=[f"{v:.4f}" for v in drops], padding=4, fontsize=10)
    other.axhline(.02, color="#228833", linestyle="--", linewidth=1.2)
    other.set_ylim(0, max(.04, max(drops) * 1.25)); other.set_ylabel("Largest other-class AP decrease vs initial")
    other.set_title("C  Retention among 8 known source-class pairs; guard <=0.02", loc="left", fontsize=10)
    for axis in (maximum, iron, other): axis.set_xticks(range(3), TITLES, fontsize=8)
    for index, row in enumerate(rows):
        positions = [j + (index - 1) * .23 for j in range(2)]
        bars = small.bar(positions, row["small_false_negatives"], width=.23, color=COLORS[index], label=TITLES[index].replace("\n", " "))
        small.bar_label(bars, labels=[f"{fn}/{n}" for fn, n in zip(row["small_false_negatives"], (93, 105))], padding=3, fontsize=8)
    small.set_xticks(range(2), ("Crack (n=93)", "Spalling (n=105)"))
    small.set_ylim(0, max(10, max(fn for row in rows for fn in row["small_false_negatives"]) * 1.34))
    small.set_ylabel("False-negative class-photo cases")
    small.set_title("D  Small DACL positives (<1% original polygon area)", loc="left", fontsize=10)
    small.legend(frameon=False, fontsize=7, loc="upper left")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.22); axis.set_axisbelow(True); axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen ImageNet ConvNeXt features: one six-epoch follow-up", fontsize=13, y=.98)
    line = "Retention/research/strict5% gates reported independently."
    if outcomes is not None:
        line = "  ".join(f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in outcomes.items())
    fig.text(.5, .027, line + "\nRepeated source-VAL selection; no held-out TEST or factory accuracy. Only the candidate adds6 new epochs.\n"
        "Frozen extra encoder and three learned zero-initialized residual heads add parameters/compute; epoch/exposure matched, not equal FLOPs.",
        ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .94), w_pad=3, h_pad=3)
    fig.savefig(output, dpi=220, facecolor="white", metadata={"Title": "Frozen semantic features source-VAL comparison"})
    plt.close(fig)


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/facility-semantic-study-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/facility-semantic-study-comparison.png")
    args = parser.parse_args(argv); source, output = args.input.resolve(), args.output.resolve(); sidecar = output.with_suffix(".plot.json")
    require(output.suffix.lower() == ".png" and source not in (output, sidecar)
            and not output.exists() and not sidecar.exists(), "Use new separate PNG/provenance outputs")
    from scripts.verify_facility_resolution import read
    result = read(source); rows = measurements(result)
    outcomes = {"Retention": result["retention_gate"]["retention_candidate_nominated"],
                "Research": result["research_gate"]["research_candidate_nominated"], "Strict<5%": result["experiments"][2]["target_passed"]}
    output.parent.mkdir(parents=True, exist_ok=True); render_plot(rows, output, outcomes=outcomes)
    value = {"schema": "facility_semantic_aggregate_plot_v1", "source_aggregate_sha256": sha(source), "plot_sha256": sha(output),
             "script_sha256": sha(Path(__file__)), "experiments": rows, "outcomes": outcomes,
             "source_test_inference_executed": False, "industrial_field_performance_measured": False, "deployed": False,
             "equal_compute_or_parameter_budget_asserted": False, "accuracy_estimated_by_plot": False}
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    return value


if __name__ == "__main__": main()
