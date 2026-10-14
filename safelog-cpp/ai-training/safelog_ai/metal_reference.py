"""Normal-reference patch comparison on the SAME accepted common backbone.

Inspired by PatchCore's nominal patch memory, not a paper reproduction. This
fits a non-gradient reference buffer; it does not train a second neural model.
"""
import math
import torch
from torch.nn import functional as F
from .equipment_expansion import ExpandedMain, features_for_equipment

ARCH = 'safelog-equipment-metal-reference-v1'
CATEGORY = 'mpdd_metal_plate'


def patch_features(features):
    merged = features_for_equipment(features)
    return F.normalize(F.avg_pool2d(merged, 3, stride=1, padding=1, count_include_pad=False), dim=1, eps=1e-6)


def nearest_cosine(query, reference, query_chunk=512, reference_chunk=2048):
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1] or min(len(query),len(reference))<1:
        raise ValueError('Nonempty matching query and normal reference features required')
    if min(query_chunk, reference_chunk)<1 or not (torch.isfinite(query).all() and torch.isfinite(reference).all()):
        raise ValueError('Finite features and positive chunk budgets required')
    if (query.norm(dim=1)<1e-8).any() or (reference.norm(dim=1)<1e-8).any():
        raise ValueError('Zero vectors cannot represent normal appearance')
    query=F.normalize(query.float(),dim=1);reference=F.normalize(reference.float(),dim=1)
    values=[]
    for start in range(0,len(query),query_chunk):
        chunk=query[start:start+query_chunk];best=chunk.new_full((len(chunk),),-1.)
        for offset in range(0,len(reference),reference_chunk):
            best=torch.maximum(best,(chunk@reference[offset:offset+reference_chunk].T).amax(1))
        values.append((1-best).clamp(0,2))
    return torch.cat(values)


def normal_budget_threshold(scores, alpha=.20):
    if len(scores)<20 or not all(math.isfinite(x) for x in scores) or not 0<alpha<1:
        raise ValueError('At least20 finite normalCAL scores and explicit error budget required')
    rank=min(len(scores),math.ceil((len(scores)+1)*(1-alpha)))
    return sorted(scores)[rank-1],rank


class MetalReferenceMain(ExpandedMain):
    def __init__(self, reference_count):
        super().__init__()
        if reference_count<1:raise ValueError('Positive reference bank size required')
        self.register_buffer('metal_reference',torch.zeros(reference_count,1000))

    def forward_equipment(self, image, category):
        if category!=CATEGORY:return super().forward_equipment(image,category)
        target=patch_features(self.backbone(image));b,c,h,w=target.shape
        query=target.permute(0,2,3,1).reshape(-1,c)
        distances=nearest_cosine(query,self.metal_reference).reshape(b,1,h,w)
        return F.interpolate(distances,size=image.shape[-2:],mode='bilinear',align_corners=False)[:,0]
