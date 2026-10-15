"""Per-photo positive/negative pixel means for small procedural edit support."""
import torch
from torch.nn import functional as F
from .ksdd_supervised import ksdd_photo_score


def balanced_pixel_bce(logits, mask, valid):
    if logits.shape != mask.shape or mask.shape != valid.shape or logits.ndim != 4:
        raise ValueError('Aligned BCHW logits/mask/valid required')
    target, known = mask.float(), valid.float()
    positive = target * known
    negative = (1 - target) * known
    pc, nc = positive.flatten(1).sum(1), negative.flatten(1).sum(1)
    if bool((nc == 0).any()):
        raise ValueError('A valid background region per image is required')
    p = (F.softplus(-logits.float()) * positive).flatten(1).sum(1) / pc.clamp_min(1)
    n = (F.softplus(logits.float()) * negative).flatten(1).sum(1) / nc
    return torch.where(pc > 0, .5 * (p + n), n).mean()


def balanced_target_loss(logits, mask, valid, _unused_legacy_positive_weight=None):
    # Runner retains the historical argument for diagnostics only. It is not a loss weight.
    pixel = balanced_pixel_bce(logits, mask, valid)
    known, target = valid.float(), mask.float() * valid.float()
    prob = logits.sigmoid()
    intersection = (prob * target * known).flatten(1).sum(1)
    denominator = ((prob + target) * known).flatten(1).sum(1)
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    score = ksdd_photo_score(prob[:, 0], valid[:, 0].bool())
    labels = (target.flatten(1).sum(1) > 0).float()
    return pixel + dice + .5 * F.binary_cross_entropy(score.clamp(1e-6, 1 - 1e-6), labels)
