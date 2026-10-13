"""ROI absence supervision is classification, not invented dense pixel truth."""
import torch
from torch.nn import functional as F


def normal_roi_loss(logits):
    if logits.ndim!=4 or logits.shape[1]!=1 or logits.shape[0]<1:
        raise ValueError('Nonempty single-class spatial logits required')
    values=logits.flatten(1)
    top=values.topk(min(64,values.shape[1]),dim=1).values.mean(1)
    return F.softplus(top).mean()
