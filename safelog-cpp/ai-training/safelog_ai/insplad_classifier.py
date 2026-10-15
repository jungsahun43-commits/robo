"""Visible corrosion of a cropped upper shackle; one unchanged shared encoder."""
import torch
from torch import nn
from .real_mask_server import RuntimeRealMaskMain
from .gearbox_supervised import gearbox_descriptor
from .fine_feature_probe import forward_with_fine


class CorrosionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(3072))
        self.register_buffer('std',torch.ones(3072))
        self.head=nn.Sequential(nn.Linear(3072,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,descriptor):
        if descriptor.ndim!=2 or descriptor.shape[1]!=3072:
            raise ValueError('Shared fine/low/high pooled3072 descriptor required')
        if not torch.isfinite(descriptor).all():raise ValueError('Finite descriptor required')
        value=((descriptor.float()-self.mean)/self.std.clamp_min(1e-3)).clamp(-8,8)
        return self.head(value).flatten()


class InspladMain(RuntimeRealMaskMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.insplad_classifier=CorrosionClassifier()

    def forward_insplad_descriptor(self,descriptor):
        return self.insplad_classifier(descriptor)

    def forward_insplad_upper_shackle(self,image):
        return self.forward_insplad_descriptor(gearbox_descriptor(forward_with_fine(self.backbone,image)))
