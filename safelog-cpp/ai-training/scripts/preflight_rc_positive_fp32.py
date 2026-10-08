"""Maintenance-only strict-FP32 preflight; declared training bytes stay frozen.

Installed cuDNN TF32 caused the disposable CPU/GPU comparison to fail. Preserve
that diagnostic, temporarily disable TF32 for this one disposable check, then
restore installed settings. The actual six-epoch trainer runs separately using
the same installed settings as the historical comparator.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.fetch_rc2119 import sha, write_new
from scripts import train_facility_rc_positive as frozen


def main():
    runs = (ROOT / "runs").resolve()
    original = (runs / "facility-rc-positive-preflight").resolve()
    preserved = (runs / "facility-rc-positive-preflight-installed-tf32-failure").resolve()
    # Checked absolute paths precede the directory move, which preserves every
    # failed diagnostic byte. No deletion or replacement is permitted.
    if not all(p.is_relative_to(runs) and p != runs for p in (original, preserved)):
        raise ValueError("Diagnostic move leaves the intended runs directory")
    if not original.is_dir() or preserved.exists() or {p.name for p in original.iterdir()} != {"disposable.pt"}:
        raise ValueError("Only the known incomplete first preflight may be preserved")
    failed_digest = sha(original / "disposable.pt")
    original.rename(preserved)
    settings = {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    previous_argv = sys.argv[:]
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        sys.argv = ["train_facility_rc_positive.py", "--preflight"]
        frozen.main()
    finally:
        torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]
        sys.argv = previous_argv
    write_new(original / "strict-fp32-maintenance.json", {
        "status": "passed", "scope": "Disposable preflight numerical verification only; outside the frozen123 training sources",
        "maintenance_script_sha256": sha(Path(__file__)), "frozen_trainer_sha256": sha(ROOT / "scripts/train_facility_rc_positive.py"),
        "installed_settings_restored": settings, "strict_fp32_used_only_for_disposable_preflight": True,
        "actual_training_epochs": 0, "failed_first_preflight_checkpoint_sha256": failed_digest,
        "failed_first_preflight_preserved": True, "first_disposable_update_count": 1, "second_disposable_update_count": 1,
        "verified_preflight_sha256": sha(original / "preflight.json"),
        "diagnosis_sha256": sha(runs / "facility-rc-positive-cpu-gpu-probe.json")})


if __name__ == "__main__": main()
