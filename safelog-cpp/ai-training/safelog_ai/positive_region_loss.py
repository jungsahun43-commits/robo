"""Supervise asserted foreground cells without inventing background truth."""
import torch
from torch.nn import functional as F


def positive_region_loss(logits, asserted, sample_mask, weight=.25):
    """Positive focal loss on explicit crack/spalling cells, scaled by exposure.

    A zero in asserted means unknown, not a negative pixel. Other five classes
    are never supervised by this loss. The original loss remains unchanged for
    core data; this is an additional, deliberately limited research intervention.
    """
    if (logits.ndim != 4 or logits.shape != asserted.shape or logits.shape[1] != 7
            or sample_mask.shape != (len(logits),) or sample_mask.dtype != torch.bool or not 0 <= weight <= 1
            or not torch.isfinite(asserted).all() or not ((asserted == 0) | (asserted == 1)).all()):
        raise ValueError("Expected seven registered binary foreground maps and a sample mask")
    if asserted[:, 2:].any():
        raise ValueError("This study asserts only crack/spalling foreground")
    active = asserted[:, :2].bool() & sample_mask[:, None, None, None].bool()
    scores = logits[:, :2].float()
    point_loss = -F.logsigmoid(scores) * torch.sigmoid(-scores)
    counts = active.flatten(1).sum(1)
    per_photo = (point_loss * active).flatten(1).sum(1) / counts.clamp_min(1)
    return weight * per_photo.mean()
