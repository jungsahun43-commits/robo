"""Visible-wear acceptance of publisher surface patches, one preserved encoder."""
import torch
from torch import nn
from .fine_feature_probe import forward_with_fine
from .real_mask_server import RuntimeRealMaskMain

VERSION = 'safelog-gearbox-patch-v26'
DIM = 3072


def gearbox_descriptor(features):
    result = []
    for name, channels in (('fine', 24), ('low', 40), ('high', 960)):
        value = features[name]
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError('Preserved fine24/low40/high960 encoder required')
        value = value.half().float().flatten(2)
        result.extend((value.mean(2), value.std(2, correction=0), value.amax(2)))
    return torch.cat(result, 1).half().float()


class GearboxMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.gearbox_head = nn.Sequential(nn.Linear(DIM, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))
        self.register_buffer('gearbox_mean', torch.zeros(DIM))
        self.register_buffer('gearbox_std', torch.ones(DIM))

    def forward_gearbox_descriptor(self, descriptor):
        if descriptor.ndim != 2 or descriptor.shape[1] != DIM:
            raise ValueError('One3072-channel descriptor per publisher patch required')
        value = ((descriptor.float() - self.gearbox_mean) / self.gearbox_std.clamp_min(1e-3)).clamp(-8, 8)
        return self.gearbox_head(value).flatten()

    def forward_gearbox_patch(self, image):
        return self.forward_gearbox_descriptor(gearbox_descriptor(forward_with_fine(self.backbone, image)))
