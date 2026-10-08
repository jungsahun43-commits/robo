"""Post-training adapter restoring the declared third VAL source, not training.

The frozen trainer evaluated all3 sources every epoch but omitted the legacy
additional_validation metadata consumed by the old standalone evaluator. Keep
TRAINING.json and the incomplete selection intact; supply the original pinned
CODEBRIM manifest in memory for one full3-source grid1 evaluation.
"""
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_rc_positive import NAME, CONTROL, PROTOCOL, read, sha, validate_protocol
from scripts.fetch_rc2119 import write_new
from scripts import evaluate_facility_target as frozen


def augmented_metadata(training, source, validation):
    if (training.get("status") != "complete" or source.get("path") != "data/codebrim-training/val.json"
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
            or validation.get("split") != "val" or validation.get("classes") != training.get("classes")
            or len(validation.get("items", [])) != 611 or training.get("additional_validation") is not None):
        raise ValueError("Only the original declared611-photo CODEBRIM VAL metadata may be supplied")
    result = deepcopy(training)
    result["additional_validation"] = deepcopy(source)
    return result


def main():
    protocol = validate_protocol(read(ROOT / PROTOCOL))
    run = (ROOT / "runs" / NAME).resolve()
    evidence = run / "complete-val-adapter.json"
    if evidence.exists(): raise ValueError("Preserve completed standalone evaluation adapter")
    training_path = run / "TRAINING.json"; training = read(training_path)
    source = read(ROOT / f"runs/{CONTROL}/TRAINING.json")["additional_validation"]
    if sha(ROOT / source["path"]) != source["sha256"]: raise ValueError("Original CODEBRIM VAL bytes changed")
    enriched = augmented_metadata(training, source, read(ROOT / source["path"]))
    selection = read(run / "TARGET-SELECTION.json")
    if selection["validation_counts"] != {"dacl": 710, "damsegment": 424}:
        raise ValueError("Only the known incomplete two-domain selection may be preserved")
    preserved = run / "original-incomplete-two-domain-post-eval"
    if preserved.exists(): raise ValueError("Never overwrite incomplete original evaluation")
    preserved.mkdir()
    paths = [(run / "TARGET-SELECTION.json", "TARGET-SELECTION.json"),
             (run / "ERROR-SIZE-AUDIT.json", "ERROR-SIZE-AUDIT.json"),
             (ROOT / "reports" / f"{NAME}-target-validation.json", "public-target-validation.json"),
             (ROOT / "reports" / f"{NAME}-error-size-audit.json", "public-error-size-audit.json")]
    old = {}
    for path, name in paths:
        target = (preserved / name).resolve()
        if not path.resolve().is_relative_to(ROOT) or not target.is_relative_to(run):
            raise ValueError("Preservation move leaves intended workspace/run")
        old[name] = sha(path); path.rename(target)
    training_sha = sha(training_path); weights_sha = sha(run / "best.pt")
    original_read = frozen.read
    def read_with_explicit_val(path):
        if Path(path).resolve() == training_path: return deepcopy(enriched)
        return original_read(path)
    try:
        frozen.read = read_with_explicit_val
        frozen.select(NAME, "cuda", grids=(1,))
    finally:
        frozen.read = original_read
    if read(run / "TARGET-SELECTION.json")["validation_counts"] != {"dacl": 710, "damsegment": 424, "codebrim": 611}:
        raise ValueError("Full original three-source evaluation missing")
    subprocess.run([sys.executable, "scripts/analyze_facility_target.py", "--name", NAME, "--aggregate-only"], cwd=ROOT, check=True)
    if sha(training_path) != training_sha or sha(run / "best.pt") != weights_sha:
        raise ValueError("Reporting adapter modified actual training/checkpoint")
    write_new(evidence, {"schema": "rc_positive_complete_val_adapter_v1", "status": "passed",
        "adapter_source_sha256": sha(Path(__file__)), "frozen_evaluator_source_sha256": sha(ROOT / "scripts/evaluate_facility_target.py"),
        "actual_training_metadata_sha256": training_sha, "weights_sha256": weights_sha,
        "original_training_metadata_unchanged": True, "original_incomplete_outputs_preserved_sha256": old,
        "additional_validation": source, "final_validation_counts": {"dacl": 710, "damsegment": 424, "codebrim": 611},
        "grid": 1, "new_training_epochs": 0, "test_inference_executed": False,
        "scope": "Post-training only; restores an omitted evaluator metadata link, not labels, scores, thresholds or training settings"})
    print("Complete3-source VAL and small-region audit restored; training and original incomplete outputs preserved")


if __name__ == "__main__": main()
