"""Photo error operating points: validation selection, never test calibration."""
import numpy as np


def wilson_interval(errors, total):
    """Two-sided 95% binomial interval; independent-photo assumption is required."""
    if not total: return None
    z = 1.959963984540054
    p = errors / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [float(max(0., center-radius)), float(min(1., center+radius))]


def rates(target, prediction):
    target, prediction = np.asarray(target, bool), np.asarray(prediction, bool)
    tp, fp, fn, tn = (int(x.sum()) for x in (target & prediction, ~target & prediction, target & ~prediction, ~target & ~prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fnr": fn / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "fnr_ci95": wilson_interval(fn, tp+fn), "fpr_ci95": wilson_interval(fp, fp+tn),
            "photo_error_fraction": (fp + fn) / len(target) if len(target) else None}


def under_target(items, limit=.05):
    return bool(items) and all(item[key] is not None and item[key] < limit for item in items for key in ("fnr", "fpr"))


def operating_point(domains):
    """One common threshold chosen on >=2 labelled VALIDATION domains only."""
    if not domains: raise ValueError("No validation domains")
    for target, score in domains.values():
        if not np.isfinite(score).all() or len(target) != len(score) or len(set(np.asarray(target).tolist())) != 2:
            raise ValueError("Validation domain must have both positive and negative gold labels")
    combined = np.unique(np.concatenate([np.asarray(score) for _, score in domains.values()]))
    thresholds = np.unique(np.r_[0., (combined[:-1] + combined[1:]) / 2, np.nextafter(combined[-1], 2.)])
    choices = []
    for cutoff in thresholds:
        metric = {name: rates(target, np.asarray(score) >= cutoff) for name, (target, score) in domains.items()}
        fractions = [item[key] for item in metric.values() for key in ("fnr", "fpr")]
        choices.append(((max(fractions), sum(fractions), sum(item["fp"] for item in metric.values()), -cutoff), float(cutoff), metric))
    _, threshold, metrics = min(choices, key=lambda row: row[0])
    return {"threshold": threshold, "domains": metrics,
            "worst_error": max(item[key] for item in metrics.values() for key in ("fnr", "fpr")),
            "target_passed": under_target(list(metrics.values()))}
