"""Normal-only neighboring feature prediction on the accepted shared backbone.

Removing the center feature is not a blind spot in RGB: backbone receptive
fields still overlap. No foreground or defect mask is inferred as ground truth.
"""
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .equipment_expansion import features_for_equipment
from .loco_context import ContextMain

VERSION = 'safelog-neighbor-surfaces-v15'
CATEGORIES = ('mpdd_bracket_black', 'mpdd_tubes')


def neighbor_view(rgb):
    if min(rgb.size) < 64: raise ValueError('Small photo requires manual inspection')
    scale = min(512/rgb.width, 512/rgb.height)
    size = (max(1,round(rgb.width*scale)),max(1,round(rgb.height*scale)))
    offset = ((512-size[0])//2,(512-size[1])//2)
    result = Image.new('RGB',(512,512),(127,127,127))
    result.paste(rgb.resize(size,Image.Resampling.BICUBIC),offset)
    return result, {'size':[512,512],'source_size':list(rgb.size),'scale':scale,'offset':list(offset)}


def neighbor_targets(features):
    # Cache and live inference use the same quantization boundary.
    return features_for_equipment(features).half().float()


class NeighborHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('FIT_mean',torch.zeros(1,1000,1,1))
        self.register_buffer('FIT_scale',torch.ones(1,1000,1,1))
        kernel=torch.ones(1000,1,3,3)/8
        kernel[:,:,1,1]=0
        self.register_buffer('neighbor_kernel',kernel)
        self.predictor=nn.Sequential(nn.Conv2d(1000,128,1),nn.ReLU(),
            nn.Conv2d(128,64,1),nn.ReLU(),nn.Conv2d(64,1000,1))

    def standardize(self,target):
        if target.ndim!=4 or target.shape[1]!=1000 or min(target.shape[-2:])<3:
            raise ValueError('Shared-backbone feature grid required')
        return ((target-self.FIT_mean)/self.FIT_scale).clamp(-8,8)

    def surrounding(self,z):
        return F.conv2d(F.pad(z,(1,1,1,1),mode='reflect'),self.neighbor_kernel,groups=1000)

    def forward(self,target):
        return self.predictor(self.surrounding(self.standardize(target)))

    def evidence(self,target):
        errors=(self(target)-self.standardize(target)).square().mean(1)
        k=max(1,round(errors.shape[-2]*errors.shape[-1]*.01))
        return errors,errors.flatten(1).topk(k,dim=1).values.mean(1)


class NeighborMain(ContextMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs,neighbor_categories=CATEGORIES):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        if not set(neighbor_categories)<=set(CATEGORIES):raise ValueError('Unsupported neighbor category')
        self.neighbor_heads=nn.ModuleDict({c:NeighborHead() for c in neighbor_categories})

    def forward_neighbor(self,native,category):
        if category not in self.neighbor_heads:raise ValueError('This category has no accepted neighbor head')
        return self.neighbor_heads[category].evidence(neighbor_targets(self.backbone(native)))
