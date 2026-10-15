"""Actual patch acceptance labels with spatial MIL, never pixel defect truth."""
import math
import torch
from torch import nn
from .fine_feature_probe import forward_with_fine
from .real_mask_server import RuntimeRealMaskMain

VERSION = 'safelog-gear-spatial-mil-v29'


def mil_logit(logit_map):
    if logit_map.ndim != 4 or logit_map.shape[1] != 1:
        raise ValueError('One scalar activation map per patch required')
    values = logit_map.float().flatten(1)
    if not torch.isfinite(values).all(): raise ValueError('Finite activations required')
    count = max(1, math.ceil(.10 * values.shape[1]))
    return values.topk(count, dim=1).values.mean(1)


class GearSpatialHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(24, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


class GearSpatialMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.gear_spatial_head = GearSpatialHead()
        self.register_buffer('gear_spatial_mean', torch.zeros(1,24,1,1))
        self.register_buffer('gear_spatial_std', torch.ones(1,24,1,1))

    def forward_gear_spatial_features(self, fine):
        if fine.ndim != 4 or tuple(fine.shape[1:]) != (24,64,64):
            raise ValueError('Preserved fine24 at64x64 from RGB256 patch required')
        value = ((fine.half().float() - self.gear_spatial_mean) / self.gear_spatial_std.clamp_min(1e-3)).clamp(-8,8)
        return mil_logit(self.gear_spatial_head(value))

    def forward_gear_spatial_patch(self, image):
        if tuple(image.shape[1:]) != (3,256,256): raise ValueError('RGB256 preprocessed source patch required')
        return self.forward_gear_spatial_features(forward_with_fine(self.backbone,image)['fine'])
