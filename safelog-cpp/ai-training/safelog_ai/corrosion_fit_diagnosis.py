"""Threshold-free diagnostics of known sparse/full FIT labels."""
import numpy as np


def summary(probability,truth):
    if probability.shape!=truth.shape or not np.isfinite(probability).all():raise ValueError('Aligned finite probability required')
    if not set(np.unique(truth))<={0,1,255}:raise ValueError('Undocumented label')
    p=np.clip(probability.astype(np.float64),1e-6,1-1e-6);losses=[];classes={}
    for k in (0,1):
        keep=truth==k
        if keep.any():
            value=p[keep];loss=float((-np.log(value) if k else -np.log1p(-value)).mean())
            classes[str(k)]={'pixels':int(keep.sum()),'mean_probability':float(value.mean()),'BCE':loss};losses.append(loss)
    if not losses:raise ValueError('No known FIT supervision')
    return {'balanced_known_BCE':float(np.mean(losses)),'classes':classes,'unknown_pixels':int((truth==255).sum())}


def balanced_difference(a,b,truth):
    if a.shape!=b.shape or a.shape!=truth.shape:raise ValueError('Aligned native views required')
    values=[float(np.abs(a[truth==k]-b[truth==k]).mean()) for k in (0,1) if np.any(truth==k)]
    if not values:raise ValueError('Known pixels required')
    return float(np.mean(values))
