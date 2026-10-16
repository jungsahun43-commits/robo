"""Derived binary supervision; unannotated/minimal VT corrosion stays unknown."""
import numpy as np
from PIL import Image

SIDE=512


def partial_target(source,raw):
    raw=np.asarray(raw)
    if raw.ndim!=2:raise ValueError('A native two dimensional annotation is required')
    if source=='vt':
        if not set(np.unique(raw))<={0,1,2,3,255}:raise ValueError('Unknown VT state')
        return np.where((raw>=1)&(raw<=3),1,255).astype(np.uint8)
    if source=='ostrava':
        if not set(np.unique(raw))<={0,169}:raise ValueError('Unexpected publisher L mask')
        return (raw>0).astype(np.uint8)
    raise ValueError('Unknown annotation contract')


def mask_view(target):
    if target.ndim!=2 or not set(np.unique(target))<={0,1,255}:raise ValueError('Binary or unknown native target required')
    h,w=target.shape;scale=min(SIDE/w,SIDE/h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    left,top=(SIDE-rw)//2,(SIDE-rh)//2
    result=np.full((SIDE,SIDE),255,dtype=np.uint8);padding=np.ones((SIDE,SIDE),dtype=bool)
    result[top:top+rh,left:left+rw]=np.asarray(Image.fromarray(target).resize((rw,rh),Image.Resampling.NEAREST))
    padding[top:top+rh,left:left+rw]=False
    return result,padding,{'native_size':[w,h],'resized_size':[rw,rh],'left':left,'top':top}


def pixel_counts(target,padding=None):
    padding=np.zeros_like(target,dtype=bool) if padding is None else padding
    if padding.shape!=target.shape or np.any(target[padding]!=255):raise ValueError('Padding must remain ignored')
    return {'positive':int((target==1).sum()),'known_negative':int((target==0).sum()),
            'unknown':int(((target==255)&~padding).sum()),'padding':int(padding.sum())}
