"""Bounded training residual sampling; unknown labels never contribute."""
import math
import torch


def hard_multipliers(targets, probabilities, target_indices, strength=2.):
    targets=torch.as_tensor(targets,dtype=torch.float64)
    probabilities=torch.as_tensor(probabilities,dtype=torch.float64)
    if (targets.ndim!=2 or targets.shape!=probabilities.shape or not target_indices
            or not math.isfinite(strength) or not 0<=strength<=4
            or not torch.isfinite(probabilities).all()
            or not ((probabilities>=0)&(probabilities<=1)).all()
            or not ((targets==-1)|(targets==0)|(targets==1)).all()
            or any(k<0 or k>=targets.shape[1] for k in target_indices)):
        raise ValueError('Invalid hard-sampling labels, scores, indices or strength')
    selected=targets[:,target_indices]
    residual=(probabilities[:,target_indices]-selected.clamp_min(0)).abs()
    residual=torch.where(selected>=0,residual,0.)
    return 1+strength*residual.amax(1).square()
