"""An optional research corroboration gate; existing rust decisions remain intact."""
import math
import torch


def mask_photo_score(probability):
    if probability.ndim!=3 or tuple(probability.shape[1:])!=(512,512):
        raise ValueError('Full512 probability maps required')
    if not torch.isfinite(probability).all():raise ValueError('Finite map required')
    return probability.flatten(1).topk(math.ceil(.02*512*512),dim=1).values.mean(1)


def corroborated(parent_score,parent_cut,mask_score,mask_cut):
    return (parent_score>parent_cut)&(mask_score>mask_cut)


def passes_guard(baseline,candidate,groups,selected_epoch):
    return bool(selected_epoch>0 and min(candidate['positive_photos'],candidate['negative_photos'])>=20 and min(groups.values())>=3
        and candidate['fpr']<=.2 and candidate['fnr']<=.2
        and baseline['fpr']-candidate['fpr']>=.05-1e-12 and candidate['fnr']-baseline['fnr']<=.05+1e-12)
