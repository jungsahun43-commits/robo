"""Within-source ranking supervision for asserted original TRAIN full photos.

The caller supplies its existing minibatch: this helper does not resample,
read data, infer labels, or calibrate the seven public probabilities.
"""
from __future__ import annotations

from numbers import Integral
import torch
from torch.nn import functional as F

DOMAIN_IDS = (0, 1, 2)
TARGET_INDICES = (0, 1)


def target_ranking_loss(logits, labels, known, domains, full_mask,
                        target_indices=TARGET_INDICES):
    """Return ``(float32 loss, aggregate pair/group counts)``.

    ``full_mask`` is true for original full photos (the trainer's crop flag
    must therefore be inverted). Labels with ``known == 0`` are ignored.
    Every contributing domain/target group averages all its positive-negative
    pairs; contributing group means then receive equal weight. If there is no
    valid pair, the returned zero remains connected to logits for backward().

    Metadata has no filenames or per-photo scores. Same-source common logit
    offsets cancel in pair differences; calibrated probabilities are not
    guaranteed by this additional objective.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError('Ranking logits must be a floating batch-by-class tensor')
    if (not isinstance(target_indices, (tuple, list)) or not target_indices
            or any(not isinstance(k, Integral) or isinstance(k, bool)
                   or not 0 <= k < logits.shape[1] for k in target_indices)
            or len(set(target_indices)) != len(target_indices)):
        raise ValueError('Ranking target indices must be distinct existing class indices')
    device = logits.device
    labels, known, domains, full_mask = [torch.as_tensor(value, device=device)
                                        for value in (labels, known, domains, full_mask)]
    if labels.shape != logits.shape or known.shape != logits.shape:
        raise ValueError('Ranking labels/known mask must match the public logit shape')
    if domains.shape != (logits.shape[0],) or full_mask.shape != (logits.shape[0],):
        raise ValueError('Ranking source and full-photo masks must have one value per row')
    if (not torch.isfinite(labels).all() or not ((labels == -1) | (labels == 0) | (labels == 1)).all()
            or not torch.isfinite(known).all() or not ((known == 0) | (known == 1)).all()
            or ((known == 1) & (labels == -1)).any()):
        raise ValueError('Ranking supervision must be binary when asserted; unknown never becomes a negative')
    if (domains.dtype == torch.bool or not torch.isfinite(domains).all()
            or not torch.isin(domains, torch.tensor(DOMAIN_IDS, device=device)).all()):
        raise ValueError('Ranking sources must be recorded original domain IDs 0, 1 or 2')
    if not torch.isfinite(full_mask).all() or not ((full_mask == 0) | (full_mask == 1)).all():
        raise ValueError('Ranking full-photo mask must be binary')
    # Cast before subtraction/softplus, including under autocast. The finite
    # range guard also prevents subtraction overflow of unusually large logits.
    scores = logits.float()
    if not torch.isfinite(scores).all() or (scores.abs() > torch.finfo(torch.float32).max / 4).any():
        raise ValueError('Ranking logits must fit the finite float32 difference range')
    means, groups = [], []
    selected_full = full_mask.bool()
    for domain in DOMAIN_IDS:
        source_full = (domains == domain) & selected_full
        for k in target_indices:
            asserted = source_full & (known[:, k] == 1)
            positive = scores[asserted & (labels[:, k] == 1), k]
            negative = scores[asserted & (labels[:, k] == 0), k]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            # softplus stays finite even for AMP-originated +/-60,000 logits.
            means.append(F.softplus(negative[:, None] - positive[None, :]).mean())
            groups.append({'domain': domain, 'target_index': int(k),
                           'positive_rows': positive.numel(), 'negative_rows': negative.numel(),
                           'pair_count': positive.numel() * negative.numel()})
    loss = torch.stack(means).mean() if means else scores.reshape(-1)[:1].sum() * 0.
    return loss, {'pair_count': sum(group['pair_count'] for group in groups),
                  'contributing_groups': len(groups), 'groups': groups}
