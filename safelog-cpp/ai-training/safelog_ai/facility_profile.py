"""Shared by calibration and HTTP inference to keep postprocessing identical."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_profile(path: Path) -> dict:
    if not path.exists():
        return {"models": {}}
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("selection_split") != "val":
        raise ValueError("Facility profiles must be selected on validation, not test.")
    entries = list(profile["models"].values())
    if profile.get("photo_classifier"):
        entries.append(profile["photo_classifier"])
    for entry in entries:
        if not 32 <= entry["imgsz"] <= 2048 or entry["imgsz"] % 32:
            raise ValueError("Invalid facility inference size")
        if not entry["thresholds"] or not all(.01 <= float(v) <= 1 for v in entry["thresholds"].values()):
            raise ValueError("Invalid facility thresholds")
    return profile


def verify_weights(path: Path, entry: dict) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry["weights_sha256"]:
        raise ValueError(f"Profile and weights do not match: {path.name}")


def prediction_floor(entry: dict, default: float = .25) -> float:
    return min(map(float, entry["thresholds"].values())) if entry else default


def accepted(label: str, confidence: float, entry: dict, default: float = .25) -> bool:
    return confidence >= float(entry.get("thresholds", {}).get(label, default))
