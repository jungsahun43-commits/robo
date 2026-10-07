"""Run one fixed semantic-feature candidate and verified completion pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_semantic_study import PROTOCOL, REFERENCE, RUN_NAME, VARIANT, validate_protocol, require, write
from scripts.train_facility_target import read, sha

DIRECTORY = "runs/facility-semantic-background-candidate"
PREFLIGHT = "runs/facility-semantic-preflight.json"
SOURCE_RECORD = "runs/facility-semantic-source-before-training.json"
TEST_RECORD = "runs/facility-semantic-test-results.json"
PLOT_OUTPUT = "reports/facility-semantic-study-comparison.png"


def training_command(python=sys.executable):
    return [python, "-u", "scripts/train_facility_semantic.py", "--name", RUN_NAME,
        "--variant", VARIANT, "--initial", f"runs/{REFERENCE}/best.pt",
        "--auxiliary-manifest", "data/facility-auxiliary-training/train.json", "--study-protocol", PROTOCOL,
        "--epochs", "6", "--backbone-lr", "0.00004", "--head-lr", "0.0001"]


def completion_commands(python=sys.executable):
    return [
        ("source-val", [python, "-u", "scripts/evaluate_facility_semantic.py", "select", "--name", RUN_NAME, "--grids", "1"]),
        ("small-region-audit", [python, "-u", "scripts/analyze_facility_target.py", "--name", RUN_NAME, "--aggregate-only"]),
        ("technical-verification", [python, "-u", "scripts/verify_facility_semantic.py", "--test-results", TEST_RECORD]),
        ("result-report", [python, "-u", "scripts/report_facility_semantic.py"]),
        ("result-plot", [python, "-u", "-m", "scripts.plot_facility_semantic", "--output", PLOT_OUTPUT]),
    ]


def main():
    protocol = validate_protocol(read(ROOT / PROTOCOL), ROOT)
    protocol_sha = sha(ROOT / PROTOCOL)
    preflight = read(ROOT / PREFLIGHT); source = read(ROOT / SOURCE_RECORD)
    require(preflight.get("status") == "passed" and preflight.get("protocol_sha256") == protocol_sha
            and source.get("protocol_sha256") == protocol_sha
            and source.get("preflight_sha256") == sha(ROOT / PREFLIGHT)
            and source.get("source_sha256") == protocol["source_sha256"],
            "Before-training technical/source proof must match the fixed head-LR protocol")
    require(not (ROOT / "runs" / RUN_NAME).exists(), "Preserve existing candidate training evidence")
    directory = ROOT / DIRECTORY
    directory.mkdir(exist_ok=False)
    state = {"schema": "facility_semantic_background_candidate_v1", "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "protocol_sha256": protocol_sha, "source_sha256": protocol["source_sha256"], "stages": []}

    def save():
        temporary = directory / "status.tmp"
        write(temporary, state); temporary.replace(directory / "status.json")

    def execute(title, command):
        stage = {"stage": title, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
        state["stages"].append(stage); save()
        try:
            with (directory / f"{title}.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                stage["child_pid"] = child.pid; save(); code = child.wait()
        except BaseException as error:
            stage.update(status="failed", error_type=type(error).__name__, finished_utc=datetime.now(timezone.utc).isoformat())
            state["status"] = "failed"; save(); raise
        stage.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(), status="complete" if code == 0 else "failed")
        if code:
            state["status"] = "failed"; save(); raise SystemExit(code)
        save()
        return stage

    save()
    stage = execute("semantic", training_command())
    training = read(ROOT / "runs" / RUN_NAME / "TRAINING.json")
    if (training.get("status") != "complete" or training.get("actual_epochs") != 6
            or training.get("head_lr") != .0001 or training.get("study_protocol_sha256") != protocol_sha):
        stage["status"] = "failed"; state["status"] = "failed"; save()
        raise ValueError("Candidate did not complete the fixed six-epoch semantic-feature training")
    stage.update(actual_epochs=6, weights_sha256=training["weights_sha256"]); save()
    for title, command in completion_commands():
        execute(title, command)
    state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat()); save()


if __name__ == "__main__":
    main()
