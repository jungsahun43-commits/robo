"""FP32-write-aware version of the one-sided TRAIN recall displacement guard."""
import torch
from .belt_recall_step import positive_recall_loss


@torch.no_grad()
def project_step(parameters,before,recall_gradients):
    if not parameters or len(parameters)!=len(before) or len(before)!=len(recall_gradients):
        raise ValueError('Matching ordered parameters, snapshots and gradients required')
    active=[]
    for p,b,g in zip(parameters,before,recall_gradients):
        if p.shape!=b.shape or p.device!=b.device: raise ValueError('Parameter snapshot mismatch')
        if not (torch.isfinite(p).all() and torch.isfinite(b).all()): raise ValueError('Finite parameters required')
        if g is None: continue
        if g.shape!=p.shape or g.device!=p.device or not torch.isfinite(g).all(): raise ValueError('Finite matching gradient required')
        active.append((p,b,g.double()))
    if not active:
        return {'projected':False,'reverted':False,'dot_before':0.,'dot_after':0.,'tolerance':1e-9,'gradient_norm2':0.,'write_margin':0.}
    norm=sum(g.square().sum() for p,b,g in active)
    def dot(): return sum((g*(p.double()-b.double())).sum() for p,b,g in active)
    initial=dot(); projected=False; reverted=False; margin=0.; tolerance=1e-9
    if norm>1e-20 and initial>0:
        projected=True
        for _ in range(3):
            value=dot()
            if value<=0: break
            # Bound the aggregate dot error from writes to actual parameter dtype.
            bound=sum((g.abs()*torch.maximum(p.double().abs(),b.double().abs())*torch.finfo(p.dtype).eps*4).sum() for p,b,g in active)
            margin=max(float(bound),1e-8)
            coefficient=(value+margin)/norm
            for p,b,g in active: p.copy_((p.double()-coefficient*g).to(p.dtype))
    after=dot()
    if after>tolerance:
        # Preserve the exact old weights if FP32 cannot represent a safe step.
        for p,b in zip(parameters,before): p.copy_(b)
        reverted=True; after=dot()
    assert after<=tolerance
    return {'projected':projected,'reverted':reverted,'dot_before':float(initial),'dot_after':float(after),
        'tolerance':tolerance,'gradient_norm2':float(norm),'write_margin':margin}
