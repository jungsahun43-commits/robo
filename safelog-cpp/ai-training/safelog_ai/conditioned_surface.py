"""One new decoder conditioned on the existing frozen reconstruction residual."""
import torch
from torch import nn
from torch.nn import functional as F
from .ksdd_supervised import SurfaceMaskHead
from .real_mask_server import RuntimeRealMaskMain
from .equipment_expansion import features_for_equipment,CATEGORIES as ALL_CATEGORIES

VERSION='safelog-target-conditioned-v19'
CATEGORIES=('mpdd_bracket_black','mpdd_tubes')


class ConditionedMaskHead(SurfaceMaskHead):
    def __init__(self):
        super().__init__();self.fusion[0]=nn.Conv2d(73,64,3,padding=1)

    def forward(self,low,high,residual):
        low=low.half().float();high=high.half().float();residual=residual.half().float()
        if residual.ndim!=4 or residual.shape[:2]!=(len(low),1):raise ValueError('One frozen residual channel per image')
        high=F.interpolate(self.high_projection(high),size=low.shape[-2:],mode='bilinear',align_corners=False)
        residual=F.interpolate(residual,size=low.shape[-2:],mode='bilinear',align_corners=False)
        return self.fusion(torch.cat((low,high,residual),1))


def warm_start(head,source):
    state={n:t.clone() for n,t in source.items()}
    weight=state['fusion.0.weight'];state['fusion.0.weight']=torch.cat((weight,weight.new_zeros((64,1,3,3))),1)
    head.load_state_dict(state,strict=True)


def frozen_residual(model,low,high,categories):
    if len(categories)!=len(low) or any(c not in CATEGORIES for c in categories):raise ValueError('Explicit supported target category per photo required')
    target=features_for_equipment({'low':low.half().float(),'high':high.half().float()})
    ids=torch.tensor([ALL_CATEGORIES.index(c) for c in categories],device=low.device,dtype=torch.long)
    predicted=model.equipment_head(target,ids)
    return (1-F.cosine_similarity(predicted.float(),target.float(),dim=1)).clamp(0,2)[:,None].half()


class ConditionedSurfaceMain(RuntimeRealMaskMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.target_conditioned_head=ConditionedMaskHead()

    def forward_conditioned_features(self,low,high,residual,size=(512,512)):
        return F.interpolate(self.target_conditioned_head(low,high,residual),size=size,mode='bilinear',align_corners=False)[:,0]

    def forward_conditioned_mask(self,native,category):
        values=self.backbone(native);low,high=values['low'].half(),values['high'].half()
        residual=frozen_residual(self,low,high,[category]*len(native))
        return self.forward_conditioned_features(low,high,residual,native.shape[-2:])
