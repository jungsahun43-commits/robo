"""RGB diagnostics only: no duplicate acceptance threshold or defect predictor."""
import numpy as np


def orientation_rgb_probe(left, right):
    left = np.asarray(left); right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 3 or left.shape[0] != left.shape[1] or left.shape[2] != 3:
        raise ValueError('Equal square RGB arrays required')
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError('Raw diagnostic RGB uint8 required')
    reference = left.astype(np.float64) / 255.
    trials = []
    for flip in (False, True):
        value = np.fliplr(right) if flip else right
        for turns in range(4):
            delta = np.abs(reference - np.rot90(value, turns).astype(np.float64) / 255.)
            trials.append({'mirror': flip, 'quarter_turns': turns, 'mean_absolute_RGB_difference': float(delta.mean()),
                           'RGB_RMSE': float(np.sqrt(np.square(delta).mean())),
                           'absolute_RGB_difference_p95': float(np.quantile(delta, .95))})
    return min(trials, key=lambda row: row['mean_absolute_RGB_difference'])
