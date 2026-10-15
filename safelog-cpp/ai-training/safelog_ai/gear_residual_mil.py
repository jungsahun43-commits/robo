"""One shared encoder: frozen global descriptor classifier plus learned bounded MIL."""
import torch
from .gearbox_linear import LinearGearboxMain
from .gearbox_supervised import gearbox_descriptor
from .gear_spatial_mil import GearSpatialHead,mil_logit
from .fine_feature_probe import forward_with_fine

VERSION='safelog-gear-residual-mil-v30'


def bounded_correction(logits):return .5*torch.tanh(logits.float())


class GearResidualMain(LinearGearboxMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.gear_spatial_head=GearSpatialHead()
        self.register_buffer('gear_spatial_mean',torch.zeros(1,24,1,1))
        self.register_buffer('gear_spatial_std',torch.ones(1,24,1,1))

    def forward_gear_residual_features(self,fine,descriptor):
        if fine.ndim!=4 or tuple(fine.shape[1:])!=(24,64,64) or len(fine)!=len(descriptor):
            raise ValueError('Aligned fine24@64x64 and original descriptor required')
        values=((fine.half().float()-self.gear_spatial_mean)/self.gear_spatial_std.clamp_min(1e-3)).clamp(-8,8)
        correction=bounded_correction(mil_logit(self.gear_spatial_head(values)))
        return self.forward_gearbox_descriptor(descriptor)+correction

    def forward_gear_residual_patch(self,image):
        if tuple(image.shape[1:])!=(3,256,256):raise ValueError('RGB256 preprocessed source patch required')
        features=forward_with_fine(self.backbone,image)
        return self.forward_gear_residual_features(features['fine'],gearbox_descriptor(features))
