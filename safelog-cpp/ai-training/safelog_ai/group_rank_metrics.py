"""Threshold-free validation ranking under explicit label/group weights."""
from collections import Counter
import numpy as np


def label_group_weights(labels,groups):
    if len(labels)!=len(groups) or set(labels)!={0,1}:raise ValueError('Aligned binary labels/groups required')
    counts=Counter(zip(labels,groups));ng={y:len({g for label,g in counts if label==y}) for y in (0,1)}
    return np.array([1/(ng[y]*counts[(y,g)]) for y,g in zip(labels,groups)],dtype=np.float64)


def weighted_auc(labels,scores,weights):
    y,s,w=np.asarray(labels),np.asarray(scores,dtype=float),np.asarray(weights,dtype=float)
    if y.ndim!=1 or y.shape!=s.shape or y.shape!=w.shape or set(y.tolist())!={0,1} or not np.isfinite(s).all() or not np.isfinite(w).all() or not (w>0).all():raise ValueError('Finite positive aligned ranking weights required')
    order=np.argsort(s,kind='stable');y,s,w=y[order],s[order],w[order];negative=0.;numerator=0.;start=0
    while start<len(y):
        end=start+1
        while end<len(y) and s[end]==s[start]:end+=1
        pos=w[start:end][y[start:end]==1].sum();neg=w[start:end][y[start:end]==0].sum()
        numerator+=pos*(negative+.5*neg);negative+=neg;start=end
    return float(numerator/(w[y==1].sum()*w[y==0].sum()))
