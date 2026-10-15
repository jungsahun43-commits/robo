"""One FIT-derived global class-budget correction, preserving original state GT."""
import numpy as np
import torch
from torch.nn import functional as F


def coefficient_weights(present,probability=None):
    present=np.asarray(present,dtype=np.float64)
    if present.ndim!=2 or present.shape[1]!=4 or np.any((present!=0)&(present!=1)) or np.any(present.sum(1)==0):
        raise ValueError('Four-class nonempty presence rows required')
    p=np.ones(len(present))/len(present) if probability is None else np.asarray(probability,dtype=np.float64)
    if p.shape!=(len(present),) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError('Sampling probabilities required')
    budget=((present/present.sum(1,keepdims=True))*p[:,None]).sum(0)
    if np.any(budget<=0):raise ValueError('Every original state must have FIT support')
    return 1/(4*budget)


def weighted_state_loss(logits,target,weights):
    if logits.ndim!=4 or logits.shape[1]!=4 or tuple(logits.shape[0:1]+logits.shape[2:])!=tuple(target.shape):raise ValueError('Matching four-class maps required')
    weight=torch.as_tensor(weights,dtype=torch.float32,device=logits.device)
    if weight.shape!=(4,) or not torch.isfinite(weight).all() or torch.any(weight<=0):raise ValueError('Four positive fixed class weights required')
    valid=target!=255
    if torch.any(valid&((target<0)|(target>3))):raise ValueError('Undocumented state class')
    safe=target.masked_fill(~valid,0).long();truth=F.one_hot(safe,4).permute(0,3,1,2).float()*valid[:,None]
    counts=truth.sum((2,3));present=counts>0;number=present.sum(1).clamp_min(1)
    logp=logits.float().log_softmax(1);ce=-(logp*truth).sum((2,3))/counts.clamp_min(1)
    prob=logp.exp()*valid[:,None];intersection=(prob*truth).sum((2,3))
    dice=1-(2*intersection+1)/(prob.sum((2,3))+counts+1)
    # Do not renormalize weights within each photo: expected FIT class budgets are equal.
    return ((ce+dice)*present*weight[None]).sum(1)/number
