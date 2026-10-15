"""One finer-grid target decoder on the unchanged shared industrial encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .conditioned_surface import ConditionedMaskHead, frozen_residual
from .fine_feature_probe import forward_with_fine
from .real_mask_server import RuntimeRealMaskMain

VERSION = 'safelog-fine-target-v20'
CATEGORIES = ('mpdd_bracket_black', 'mpdd_tubes')


class FineTargetHead(ConditionedMaskHead):
    def __init__(self):
        super().__init__()
        self.fine_projection = nn.Sequential(nn.Conv2d(24, 16, 1), nn.ReLU())
        self.fusion[0] = nn.Conv2d(89, 64, 3, padding=1)

    def forward(self, low, high, residual, fine):
        if fine.ndim != 4 or fine.shape[:2] != (len(low), 24):
            raise ValueError('Aligned 24-channel fine feature per image required')
        if fine.shape[-2:] != tuple(d * 2 for d in low.shape[-2:]):
            raise ValueError('Fine grid must be twice the existing low grid')
        low, high, residual, fine = (v.half().float() for v in (low, high, residual, fine))
        size = fine.shape[-2:]
        up = lambda value: F.interpolate(value, size=size, mode='bilinear', align_corners=False)
        values = (up(low), up(self.high_projection(high)), up(residual), self.fine_projection(fine))
        return self.fusion(torch.cat(values, 1))


def initialize_fine_head(head, source):
    """Copy v19's 73 channels; fine projection is new and its fusion inputs start at zero."""
    state = {name: value.clone() for name, value in source.items()}
    weight = state['fusion.0.weight']
    assert weight.shape == (64, 73, 3, 3)
    state['fusion.0.weight'] = torch.cat((weight, weight.new_zeros((64, 16, 3, 3))), 1)
    missing, extra = head.load_state_dict(state, strict=False)
    assert not extra and set(missing) == {'fine_projection.0.weight', 'fine_projection.0.bias'}


def trainable_fine_head(head):
    head.requires_grad_(True)
    head.high_projection.requires_grad_(False)
    return [parameter for parameter in head.parameters() if parameter.requires_grad]


class FineTargetMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.fine_target_head = FineTargetHead()

    def forward_fine_target_features(self, low, high, residual, fine, size=(512, 512)):
        return F.interpolate(self.fine_target_head(low, high, residual, fine), size=size,
                             mode='bilinear', align_corners=False)[:, 0]

    def forward_fine_target_mask(self, native, category):
        values = forward_with_fine(self.backbone, native)
        low, high = values['low'].half(), values['high'].half()
        residual = frozen_residual(self, low, high, [category] * len(native))
        return self.forward_fine_target_features(low, high, residual, values['fine'], native.shape[-2:])
