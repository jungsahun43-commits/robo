"""Normal-only global-context reconstruction; one accepted shared backbone.

It is a hypothesis for long-range configuration evidence, not a supervised
clamp counter, wire tracker, or reproduction of the LOCO paper's method.
"""
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .equipment_expansion import features_for_equipment
from .ksdd_photo_only import KSDDPhotoMain

VERSION = 'safelog-loco-context-v13'
CATEGORY = 'loco_splicing_connectors'


def context_view(rgb):
    if min(rgb.size) < 64: raise ValueError('Context view needs at least64 pixels; manual review required')
    scale = min(512/rgb.width, 256/rgb.height)
    size = (max(1, round(rgb.width*scale)), max(1, round(rgb.height*scale)))
    offset = ((512-size[0])//2, (256-size[1])//2)
    view = Image.new('RGB', (512, 256), (127, 127, 127))
    view.paste(rgb.resize(size, Image.Resampling.BICUBIC), offset)
    return view, {'size': [512, 256], 'source_size': list(rgb.size), 'offset': list(offset), 'scale': scale}


def context_targets(features):
    # The same FP16 roundtrip is used by cached training and live inference.
    return F.adaptive_avg_pool2d(features_for_equipment(features), (4, 8)).half().float()


class ContextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('FIT_mean', torch.zeros(1, 1000, 1, 1))
        self.register_buffer('FIT_scale', torch.ones(1, 1000, 1, 1))
        self.encoder = nn.Sequential(nn.Conv2d(1000, 64, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 128))
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 2048), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 8)), nn.Conv2d(64, 1000, 1))

    def standardize(self, target):
        if target.shape[1:] != (1000, 4, 8): raise ValueError('Fixed common-backbone spatial target required')
        return ((target-self.FIT_mean)/self.FIT_scale).clamp(-8, 8)

    def forward(self, target):
        return self.decoder(self.encoder(self.standardize(target)))

    def evidence(self, target):
        error = (self(target)-self.standardize(target)).square().mean(1)
        return error, error.flatten(1).topk(8, dim=1).values.mean(1)


class ContextMain(KSDDPhotoMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.context_head = ContextHead()

    def forward_context(self, native):
        return self.context_head.evidence(context_targets(self.backbone(native)))

    def forward_context_and_location(self, original, native):
        location = super().forward_equipment(original, CATEGORY)
        evidence, score = self.forward_context(native)
        return location, evidence, score
