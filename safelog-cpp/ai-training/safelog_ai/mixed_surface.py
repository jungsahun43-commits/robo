"""One candidate trained on two real-mask sources; accepted branches are immutable."""
import torch
from torch.nn import functional as F
from .ksdd_supervised import SurfaceMaskHead
from .real_mask_server import RuntimeRealMaskMain

VERSION = 'safelog-mixed-real-mask-v17'


def mixed_logits(head, low, high, output_size):
    """Keep each source's own spatial coordinates, including its letterbox."""
    return F.interpolate(head(low.half().float(), high.half().float()),
                         size=tuple(output_size), mode='bilinear', align_corners=False)


def freeze_projection(head):
    head.requires_grad_(True)
    head.high_projection.requires_grad_(False)
    return [p for p in head.parameters() if p.requires_grad]


class MixedSurfaceMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.mixed_surface_head = SurfaceMaskHead()

    def forward_mixed_features(self, low, high, output_size=(512, 512)):
        return mixed_logits(self.mixed_surface_head, low, high, output_size)[:, 0]

    def forward_mixed_mask(self, native):
        features = self.backbone(native)
        return self.forward_mixed_features(features['low'], features['high'])
