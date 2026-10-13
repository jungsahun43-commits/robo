"""Protect the TRAIN positive-pixel loss against a harmful actual optimizer step.

Inspired by gradient projection, but this is a one-sided parameter-displacement
constraint, not the original symmetric PCGrad algorithm. It is a first-order
TRAIN constraint, NOT a guarantee of validation recall or industrial safety.
"""
import torch
from torch.nn import functional as F


def positive_recall_loss(logits,mask,valid):
    if logits.shape!=mask.shape or mask.shape!=valid.shape:
        raise ValueError('Matching belt logits and spatial publisher truth required')
    if not (((mask==0)|(mask==1)).all() and ((valid==0)|(valid==1)).all()):
        raise ValueError('Binary publisher mask and spatial validity required')
    known=mask*valid
    if known.sum()<=0: raise ValueError('Known positive publisher pixels required')
    return -(F.logsigmoid(logits)*known).sum()/known.sum()


@torch.no_grad()
def project_step(parameters,before,recall_gradients):
    """Project the actual Adam displacement, including momentum and decay.

    Optimizer moments remain the raw moments. No claim of standard PCGrad or
    Adam convergence is made. Projection is checked after FP32 parameter writes.
    """
    if not parameters or len(parameters)!=len(before) or len(before)!=len(recall_gradients):
        raise ValueError('Matching ordered parameters, snapshots and gradients required')
    active=[]
    for p,b,g in zip(parameters,before,recall_gradients):
        if p.shape!=b.shape or p.device!=b.device: raise ValueError('Parameter snapshot mismatch')
        if g is None: continue
        if g.shape!=p.shape or g.device!=p.device: raise ValueError('Gradient mismatch')
        if not (torch.isfinite(p).all() and torch.isfinite(b).all() and torch.isfinite(g).all()):
            raise ValueError('Finite values required for displacement projection')
        active.append((p,b,g.double()))
    if not active: return {'projected':False,'dot_before':0.,'dot_after':0.,'tolerance':1e-9,'gradient_norm2':0.}
    norm=sum(g.square().sum() for p,b,g in active)
    def dot(): return sum((g*(p.double()-b.double())).sum() for p,b,g in active)
    initial=dot(); tolerance=1e-9; projected=False
    if norm>1e-20 and initial>0:
        projected=True
        # Up to two tiny numerical repairs for the FP32 write-rounding error.
        for _ in range(3):
            value=dot()
            if value<=0: break
            coefficient=(value+1e-9)/norm
            for p,b,g in active: p.copy_((p.double()-coefficient*g).to(p.dtype))
    after=dot()
    if norm>1e-20 and after>tolerance:
        raise RuntimeError('Recall step projection could not meet numerical tolerance')
    return {'projected':projected,'dot_before':float(initial),'dot_after':float(after),
        'tolerance':tolerance,'gradient_norm2':float(norm)}
