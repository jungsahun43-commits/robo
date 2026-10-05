"""Pure conditional sampler coupling for unchanged original TRAIN labels.

The one planned intervention increases weighted exposure of full DACL photos
whose original Spalling target is absent and at least one of four native tags
is present. Replacement stays within the original Crack/Spalling joint state.
No prediction, review outcome, polygon, or image feature selects a row.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from numbers import Real
from collections.abc import Mapping

import numpy as np

from safelog_ai.auxiliary_classifier import AUX_CLASSES

FACTOR = 1.5
RELATED_TAGS = ("Rockpocket", "WConccor", "Hollowareas", "Cavity")
DOMAINS = ("dacl", "damsegment", "codebrim")
ELIGIBLE_STRATA = ("dacl/full/00", "dacl/full/10")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(items, full_count):
    _require(isinstance(items, list) and items and type(full_count) is int
             and 0 < full_count <= len(items), "Invalid original TRAIN row boundary")
    images = set()
    for index, item in enumerate(items):
        _require(isinstance(item, Mapping) and item.get("domain") in DOMAINS
                 and isinstance(item.get("image"), str) and item["image"]
                 and item["image"] not in images, "Invalid or duplicate original TRAIN row")
        images.add(item["image"])
        targets = item.get("targets")
        _require(isinstance(targets, list) and len(targets) == 7
                 and all(type(v) is int and v in (-1, 0, 1) for v in targets),
                 "Original targets must remain seven known/unknown integer labels")
        _require(item.get("split", "train") == "train"
                 and item.get("target_split", "train") == "train", "Held-out row is forbidden")
        if index < full_count:
            _require("parent_image" not in item, "Original full row cannot be a derived crop")
        else:
            _require(isinstance(item.get("parent_image"), str)
                     and item.get("parent_split") == "train", "Derived row requires its TRAIN parent")
    parents = {item["image"]: item for item in items[:full_count]}
    for item in items[full_count:]:
        parent = parents.get(item["parent_image"])
        _require(parent is not None and parent["domain"] == item["domain"],
                 "Derived row must keep its original TRAIN source")


def coupling_strata(items, full_count) -> np.ndarray:
    """Identify source, original full/crop, and exact two-target joint state."""
    _rows(items, full_count)
    result = []
    for index, item in enumerate(items):
        crack, spalling = item["targets"][:2]
        state = f"{crack}{spalling}" if crack >= 0 and spalling >= 0 else f"unknown:{crack},{spalling}"
        result.append(f"{item['domain']}/{'full' if index < full_count else 'crop'}/{state}")
    return np.asarray(result)


def eligible_rows(items, full_count, auxiliary: Mapping[str, list[int]]) -> np.ndarray:
    """Select original native tags with no use of individual model outcomes."""
    _rows(items, full_count)
    _require(isinstance(auxiliary, Mapping), "Native tag membership must be a mapping")
    dacl = {item["image"]: item for item in items[:full_count] if item["domain"] == "dacl"}
    _require(set(auxiliary) == set(dacl), "Native tags must cover exactly original full DACL TRAIN")
    tag_indices = [AUX_CLASSES.index(tag) for tag in RELATED_TAGS]
    spalling_index = AUX_CLASSES.index("Spalling")
    result = np.zeros(len(items), dtype=bool)
    for index, item in enumerate(items[:full_count]):
        if item["domain"] != "dacl":
            continue
        tags = auxiliary[item["image"]]
        _require(isinstance(tags, list) and len(tags) == len(AUX_CLASSES)
                 and all(type(v) is int and v in (0, 1) for v in tags),
                 "Native tags must remain original nineteen known integer labels")
        _require(item["targets"][1] in (0, 1)
                 and tags[spalling_index] == item["targets"][1],
                 "Native Spalling must match the original photo label")
        result[index] = (item["targets"][1] == 0 and any(tags[k] for k in tag_indices))
        if result[index]:
            _require(item["targets"][0] in (0, 1), "Eligible Crack/Spalling state must be known")
    return result


def coupling_probabilities(p: float, factor: float = FACTOR) -> dict[str, float]:
    """Return the exact upward reweighting and maximal-coupling probability.

    Boundary cases p=0, p=1 and factor=1 are exact no-ops. Production planning
    fixes factor at 1.5; the factor argument permits mathematical boundary tests.
    """
    _require(not isinstance(p, (bool, np.bool_)) and isinstance(p, Real)
             and math.isfinite(float(p)) and 0 <= p <= 1, "Invalid eligible weighted probability")
    _require(not isinstance(factor, (bool, np.bool_)) and isinstance(factor, Real)
             and math.isfinite(float(factor)) and factor >= 1, "Only finite upward coupling is defined")
    p = float(p)
    factor = float(factor)
    denominator = 1 + (factor - 1) * p
    # The algebraic form avoids cancellation when p is one representable
    # float below one. Boundary no-ops still do not advance the RNG stream.
    replacement = 0.0 if p in (0.0, 1.0) or factor == 1 else (factor - 1) * p / denominator
    changed = (1 - p) * replacement
    treatment = p + changed
    return {"original_eligible_probability": p, "treatment_eligible_probability": treatment,
            "noneligible_replacement_probability": replacement,
            "minimum_changed_probability": changed}


def _draws(value, rows):
    _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in "iu"
             and np.all(value >= 0) and np.all(value < rows), "Draw indices must be valid integer rows")
    return value.astype(np.int64, copy=False)


def paired_epoch(base_indices: np.ndarray, weights: np.ndarray, eligible: np.ndarray,
                 strata: np.ndarray, rng: np.random.Generator,
                 factor: float = FACTOR) -> np.ndarray:
    """Keep eligible draws and couple only noneligible draws in identical strata.

    Eligible replacement rows follow the original weights conditional on their
    stratum. This preserves each original stratum's mass and every drawn joint
    target at the same position. Other five photo labels and auxiliary tag
    exposure can change; the labels themselves remain untouched.
    """
    _require(isinstance(weights, np.ndarray) and weights.ndim == 1
             and weights.dtype.kind in "iuf" and np.isfinite(weights).all()
             and (weights >= 0).all() and weights.sum() > 0, "Invalid original sampler weights")
    rows = weights.size
    _require(isinstance(eligible, np.ndarray) and eligible.shape == (rows,)
             and eligible.dtype == np.bool_, "Eligibility must be a boolean array")
    _require(isinstance(strata, np.ndarray) and strata.shape == (rows,)
             and strata.dtype.kind in "US", "Strata must be a string array")
    _require(isinstance(rng, np.random.Generator), "A separate NumPy generator is required")
    _require(not np.any(eligible & ~np.isin(strata, ELIGIBLE_STRATA)),
             "Only full known-negative DACL joint states are eligible")
    coupling_probabilities(0, factor)  # Validate the mathematical factor before any no-op.
    base = _draws(base_indices, rows)
    treatment = base.copy()
    for key in ELIGIBLE_STRATA:
        members = strata == key
        candidates = np.flatnonzero(members & eligible)
        total = float(weights[members].sum())
        if total == 0:
            _require(not np.any(strata[base] == key), "A zero-mass stratum cannot appear in base draws")
            continue
        mass = float(weights[candidates].sum())
        probabilities = coupling_probabilities(mass / total, factor)
        replacement = probabilities["noneligible_replacement_probability"]
        if replacement == 0:
            continue
        positions = np.flatnonzero((strata[base] == key) & ~eligible[base])
        changed = positions[rng.random(positions.size) < replacement]
        if changed.size:
            treatment[changed] = rng.choice(candidates, size=changed.size, replace=True,
                                             p=weights[candidates] / mass)
    _require(np.array_equal(strata[base], strata[treatment]), "Paired stratum preservation failed")
    _require(np.array_equal(base[eligible[base]], treatment[eligible[base]]),
             "Originally eligible draws must remain unchanged")
    return treatment


def draw_sha256(indices: np.ndarray) -> str:
    """Match the actual native trainer's concatenated little-endian int64 hash."""
    _require(isinstance(indices, np.ndarray) and indices.ndim == 1
             and indices.dtype.kind in "iu" and np.all(indices >= 0), "Invalid row-order indices")
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def draw_counts(indices: np.ndarray, strata: np.ndarray) -> dict[str, int]:
    """Aggregate private indices without publishing individual photo identities."""
    _require(isinstance(strata, np.ndarray) and strata.ndim == 1 and strata.dtype.kind in "US",
             "Strata must be a string array")
    indices = _draws(indices, strata.size)
    return dict(sorted(Counter(strata[indices].tolist()).items()))
