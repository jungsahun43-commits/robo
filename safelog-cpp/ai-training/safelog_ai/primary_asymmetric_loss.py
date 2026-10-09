"""ASL-style primary photo loss; other-five focal and unknown semantics retained.

Reference: Ridnik et al., ICCV2021, Asymmetric Loss for Multi-Label
Classification, and the authors' Alibaba-MIIL/ASL train.py example.
Only concrete_crack/spalling receive gamma_pos0/gamma_neg4/clip0.05.
This independent implementation keeps existing source positive weights,
primary emphasis2, per-photo known-label normalization and batch mean.
AI visual judgements never become targets. No global autograd mode is changed.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

GAMMA_POS=0.0
GAMMA_NEG=4.0
NEGATIVE_CLIP=0.05

def primary_asymmetric_photo_loss(logits,labels,known,positive_weights):
    if not (logits.ndim==2 and logits.shape[0]>0 and logits.shape[1]==7
        and logits.is_floating_point() and torch.isfinite(logits).all()
        and labels.shape==known.shape==logits.shape and labels.device==known.device==logits.device
        and torch.isfinite(labels).all() and torch.isfinite(known).all()
        and ((known==0)|(known==1)).all()
        and (((labels==0)|(labels==1))|(known==0)).all()):
        raise ValueError("Finite Bx7 logits, known binary targets and explicit unknown mask required")
    if not (positive_weights.shape in ((7,),logits.shape) and positive_weights.device==logits.device
        and torch.isfinite(positive_weights).all() and (positive_weights>0).all()):
        raise ValueError("Original finite positive source weights required")
    # Only computational placeholders are sanitized. Unknown entries retain
    # zero loss/gradient and do not enter the known-label denominator.
    targets=torch.where(known>0,labels,0.)
    weights=positive_weights[None,:]if positive_weights.ndim==1 else positive_weights
    balance=torch.where(targets>0,weights,1.)
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
    probability=logits.sigmoid()
    correct=torch.where(targets>0,probability,1-probability)
    original=bce*(1-correct).pow(1.)*balance*known
    # Stable FP32 log-domain positive term retains a gradient for very wrong
    # positive logits; clipping never fills a missing label. Negative focal
    # weights are detached, matching the authors' training example.
    x=logits[:,:2].float();y=targets[:,:2].float();p=x.sigmoid()
    negative_probability=(1-p+NEGATIVE_CLIP).clamp(max=1.)
    log_positive=F.logsigmoid(x)
    log_negative=negative_probability.log()
    ce=-(y*log_positive+(1-y)*log_negative)
    focus=torch.where(y>0,torch.ones_like(p),(1-negative_probability).pow(GAMMA_NEG)).detach()
    primary=ce*focus*balance[:,:2]*known[:,:2]*2.
    combined=torch.cat((primary,original[:,2:]),dim=1)
    return (combined.sum(1)/known.sum(1).clamp_min(1)).mean()
