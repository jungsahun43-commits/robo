"""Whole-image model response as an image feature, never a new ground-truth label."""
import numpy as np
import torch
from torch.nn import functional as F
from .native_corrosion_grid import grid
from .corrosion_partial_mask import native_probability


def whole_geometry(size):
    w,h=size
    if min(w,h)<=0:raise ValueError('Positive native image dimensions required')
    scale=min(512/w,512/h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    return {'native_size':[w,h],'resized_size':[rw,rh],'left':(512-rw)//2,'top':(512-rh)//2}


def native_context(probability512,size):
    p=torch.as_tensor(probability512,dtype=torch.float32)
    if p.shape!=(512,512) or not torch.isfinite(p).all() or torch.any((p<0)|(p>1)):raise ValueError('Whole512 finite probability required')
    probability=native_probability(p,whole_geometry(size))
    return (torch.logit(probability.clamp(1e-6,1-1e-6))/8).clamp(-1,1).numpy()


def context_tile512(field,box):
    if field.ndim!=2 or not np.isfinite(field).all() or np.max(np.abs(field))>1:raise ValueError('Finite bounded context field required')
    x,y,w,h=(box[k] for k in ('left','top','width','height'))
    if min(x,y)<0 or not 0<w<=512 or not 0<h<=512 or x+w>field.shape[1] or y+h>field.shape[0]:raise ValueError('Context tile outside native image')
    out=np.zeros((512,512),np.float32);out[:h,:w]=field[y:y+h,x:x+w]
    return out


def context_tile128(field,box):
    t=torch.from_numpy(context_tile512(field,box))[None,None]
    return F.interpolate(t,size=(128,128),mode='bilinear',align_corners=False)[0].numpy()


def fixed_probe_tiles(size):
    boxes=grid(size);indices=sorted({0,len(boxes)//2,len(boxes)-1})
    return [boxes[i] for i in indices]
