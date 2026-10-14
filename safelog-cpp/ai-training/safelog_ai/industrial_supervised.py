"""Real-labelled ball-screw wear head on the accepted single shared backbone."""
import torch
from torch import nn
from .native_component_views import NativeComponentMain

ARCH = 'safelog-industrial-supervised-wear-v8'
WEAR_CATEGORY = 'ball_screw_pitting'
DESCRIPTOR_DIM = 3000
VARIANTS = ('linear', 'mlp')


def wear_descriptor(features):
    """Preserve low-level texture and high-level appearance, one encoder pass."""
    if features['low'].shape[1] != 40 or features['high'].shape[1] != 960:
        raise ValueError('Accepted low40/high960 backbone required')
    values = []
    for name in ('low', 'high'):
        value = features[name].float().flatten(2)
        values.extend((value.mean(2), value.std(2, correction=0), value.amax(2)))
    return torch.cat(values, 1)


def wear_head(variant):
    if variant == 'linear': return nn.Linear(DESCRIPTOR_DIM, 1)
    if variant == 'mlp':
        return nn.Sequential(nn.Linear(DESCRIPTOR_DIM, 256), nn.ReLU(), nn.Dropout(.2),
                             nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))
    raise ValueError('Explicit registered wear head required')


class IndustrialSupervisedMain(NativeComponentMain):
    def __init__(self, reference_counts, native_specs, wear_variant):
        super().__init__(reference_counts, native_specs)
        self.wear_variant = wear_variant
        self.wear_head = wear_head(wear_variant)
        self.register_buffer('wear_mean', torch.zeros(DESCRIPTOR_DIM))
        self.register_buffer('wear_std', torch.ones(DESCRIPTOR_DIM))

    def standardized_wear(self, descriptor):
        if descriptor.ndim != 2 or descriptor.shape[1] != DESCRIPTOR_DIM:
            raise ValueError('One3000-channel wear descriptor per input required')
        return ((descriptor.float()-self.wear_mean)/self.wear_std.clamp_min(1e-4)).clamp(-8, 8)

    def forward_wear_descriptor(self, descriptor):
        return self.wear_head(self.standardized_wear(descriptor)).flatten()

    def forward_wear(self, image):
        return self.forward_wear_descriptor(wear_descriptor(self.backbone(image)))
