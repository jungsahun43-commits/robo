"""Plot the four-model source-VAL follow-up aggregates without new inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scripts.report_facility_retention_strength import RUNS, RETENTION_GATE, measurements, require
from scripts.report_facility_detail import GATE

TITLES = ("Initial ROI", "Reused weight0", "Previous weight1", "Follow-up weight4")
COLORS = ("#737373", "#0072B2", "#CCBB44", "#D55E00")
SMALL_COUNTS = (93, 105)


def plotted_measurements(result):
    require(result.get("schema") == "facility_retention_strength_study_comparison_v1",
            "Use the actual strength follow-up comparison")
    require(result.get("actual_new_completed_training_epochs") == 6 and result.get("new_control_training_epochs") == 0
            and result.get("previous_pair_completed_epochs_counted_again") is False
            and result.get("weight_chosen_after_previous_source_val_results") is True
            and result.get("repeated_source_val_adaptation") is True
            and result.get("source_test_inference_executed") is False and result.get("app_model_promoted") is False
            and result.get("deployed") is False and result.get("teacher_predictions_are_new_truth") is False,
            "Plot requires the declared reused-control, adaptive source-VAL scope")
    entries = result.get("experiments", [])
    protocol = {"reference": RUNS[0], "control": RUNS[1], "treatment": RUNS[3],
        "research_candidate_gate": GATE, "retention_candidate_gate": RETENTION_GATE}
    measured = measurements(entries, protocol)
    for key in ("retention_gate", "previous_weight1_retention_gate", "worst_other_class_ap_decline_by_model",
                "four_model_other_class_ap", "increment_vs_weight1"):
        require(result.get(key) == measured[key], "Stored follow-up AP/gate differs from original measurements")
    require(result.get("verified_known_class_ap_measurements") == 56
            and result.get("verified_known_other_class_ap_measurements") == 32, "Four-model unique AP count differs")
    iron = next(row for row in measured["four_model_other_class_ap"]
                if row["domain"] == "dacl" and row["class"] == "exposed_rebar")
    rows = []
    for index, entry in enumerate(entries):
        require(type(entry.get("imgsz")) is int and entry["imgsz"] == 640
                and type(entry.get("actual_epochs")) is int and entry["actual_epochs"] == 6,
                "All four actual six-epoch640 models are required")
        small = entry["small_dacl_polygon_area_below_one_percent"]
        rows.append({"run": entry["run"], "maximum_fnr_fpr": entry["worst_error"],
            "dacl_exposed_rebar_ap": iron[("reference_ap", "control_ap", "weight1_ap", "weight4_ap")[index]],
            "maximum_other_ap_drop_vs_reference": measured["worst_other_class_ap_decline_by_model"][index]["maximum_other_ap_drop_vs_reference"],
            "small_false_negatives": [small[task]["false_negatives"] for task in ("concrete_crack", "concrete_spalling")],
            "small_positive_cases": list(SMALL_COUNTS), "newly_trained_this_followup": index == 3})
    return rows


def render_plot(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.3))
    maximum, iron, decline, small = axes.flat
    values = [100*row["maximum_fnr_fpr"] for row in rows]
    bars = maximum.bar(range(4), values, color=COLORS, width=.6)
    maximum.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4, fontsize=9)
    maximum.axhline(5, color="#228833", linestyle="--", linewidth=1.3)
    maximum.text(.98, 5, "Strict target: <5%", transform=maximum.get_yaxis_transform(),
                 ha="right", va="bottom", color="#228833", fontsize=8)
    maximum.set_ylim(0, max(10, max(values)*1.24))
    maximum.set_ylabel("Largest crack/spalling FNR or FPR (%)")
    maximum.set_title("A  Maximum among12 source-VAL rates", loc="left", fontsize=10, pad=12)
    ap = [row["dacl_exposed_rebar_ap"] for row in rows]
    bars = iron.bar(range(4), ap, color=COLORS, width=.6)
    iron.bar_label(bars, labels=[f"{v:.4f}" for v in ap], padding=4, fontsize=9)
    iron.set_ylim(0, 1)
    iron.set_ylabel("AP on known DACL exposed-rebar labels")
    iron.set_title("B  DACL exposed rebar: ranking AP", loc="left", fontsize=10, pad=12)
    drops = [row["maximum_other_ap_drop_vs_reference"] for row in rows]
    bars = decline.bar(range(4), drops, color=COLORS, width=.6)
    decline.bar_label(bars, labels=[f"{v:.4f}" for v in drops], padding=4, fontsize=9)
    decline.axhline(.02, color="#228833", linestyle="--", linewidth=1.3)
    decline.text(.98, .02, "Retention guard: at most0.02", transform=decline.get_yaxis_transform(),
                 ha="right", va="bottom", color="#228833", fontsize=8)
    decline.set_ylim(0, max(.04, max(drops)*1.25))
    decline.set_ylabel("Largest other-class AP decrease vs initial")
    decline.set_title("C  Maximum decrease among8 known source-class pairs", loc="left", fontsize=10, pad=12)
    for axis in (maximum, iron, decline):
        axis.set_xticks(range(4), ("Initial\nROI", "Reused\nweight0", "Previous\nweight1", "Follow-up\nweight4"))
    width = .18
    for index, row in enumerate(rows):
        positions = [task+(index-1.5)*width for task in range(2)]
        bars = small.bar(positions, row["small_false_negatives"], width=width, color=COLORS[index], label=TITLES[index])
        small.bar_label(bars, labels=[f"{fn}/{n}" for fn, n in zip(row["small_false_negatives"], SMALL_COUNTS)], padding=3, fontsize=7)
    highest = max(fn for row in rows for fn in row["small_false_negatives"])
    small.set_ylim(0, max(10, highest*1.35))
    small.set_xticks(range(2), ("Crack (n=93)", "Spalling (n=105)"))
    small.set_ylabel("False-negative class-photo cases")
    small.set_title("D  Small DACL positives (<1% polygon area)", loc="left", fontsize=10, pad=12)
    small.legend(frameon=False, fontsize=7, loc="upper left")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.22); axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Known-class retention: weight4 follow-up with a reused control", fontsize=13, y=.98)
    fig.text(.5, .024,
        "Weight4 was chosen after previous source-VAL results; repeated validation adaptation, without held-out TEST or factory accuracy.\n"
        "Only weight4 was newly trained (6epochs). A/C/D: lower is better; B: higher. Teacher signals are a regularizer, not new ground truth.",
        ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .94), w_pad=3, h_pad=3)
    fig.savefig(output, dpi=220, facecolor="white", metadata={"Title": "Known-class retention strength follow-up"})
    plt.close(fig)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"reports/facility-retention-strength-study-comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/facility-retention-strength-study-comparison.png")
    args = parser.parse_args(argv); source, output = args.input.resolve(), args.output.resolve()
    sidecar = output.with_suffix(".plot.json")
    require(output.suffix.lower() == ".png" and source not in (output, sidecar), "Use separate aggregate/PNG/provenance paths")
    require(not output.exists() and not sidecar.exists(), "Preserve the existing measured plot")
    result = json.loads(source.read_text(encoding="utf-8")); rows = plotted_measurements(result)
    output.parent.mkdir(parents=True, exist_ok=True); render_plot(rows, output)
    record = {"schema": "facility_retention_strength_aggregate_plot_v1", "source_aggregate_sha256": sha(source),
        "plot_sha256": sha(output), "script_sha256": sha(Path(__file__)), "experiments": rows,
        "actual_new_completed_training_epochs": 6, "new_control_training_epochs": 0,
        "repeated_source_val_adaptation": True, "source_test_inference_executed": False,
        "industrial_field_performance_measured": False, "deployed": False}
    sidecar.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
