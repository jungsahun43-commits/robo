"""Soft primary prediction regularization on known original TRAIN rows only."""
from numbers import Real
import math
import torch
from torch.nn import functional as F

def primary_preservation_loss(student,teacher,known,original_rows,temperature=2.):
    """T² mean Bernoulli KL for crack/spalling; teacher is never a new label."""
    if isinstance(temperature,bool)or not isinstance(temperature,Real)or not math.isfinite(float(temperature))or temperature<=0:
        raise ValueError('Finite positive temperature required')
    for value in(student,teacher):
        if not isinstance(value,torch.Tensor)or value.ndim!=2 or value.shape[1]!=7 or value.shape[0]<1 or not value.is_floating_point()or not torch.isfinite(value).all():
            raise ValueError('Finite Bx7 logits required')
    if student.shape!=teacher.shape or student.device!=teacher.device:raise ValueError('Teacher/student shape/device differs')
    if not isinstance(known,torch.Tensor)or known.shape!=student.shape or known.device!=student.device or not torch.isfinite(known).all()or not((known==0)|(known==1)).all():
        raise ValueError('Matching binary known mask required')
    if not isinstance(original_rows,torch.Tensor)or original_rows.dtype!=torch.bool or original_rows.shape!=(student.shape[0],)or original_rows.device!=student.device:
        raise ValueError('Matching original-row boolean mask required')
    t=float(temperature);a=student[:,:2].float()/t;b=teacher.detach()[:,:2].float()/t
    mask=known[:,:2].detach().float()*original_rows[:,None].float();count=mask.sum()
    if not count.item():return a.sum()*0.
    probability=b.sigmoid()
    kl=(probability*(F.logsigmoid(b)-F.logsigmoid(a))+(1-probability)*(F.logsigmoid(-b)-F.logsigmoid(-a))).clamp_min(0.)
    loss=t*t*(kl*mask).sum()/count
    if not torch.isfinite(loss):raise ValueError('Nonfinite primary retention loss')
    return loss
