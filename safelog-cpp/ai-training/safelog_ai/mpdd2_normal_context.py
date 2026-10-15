"""Normal-only spatial reconstruction for the MPDD2 white-bracket image domain."""
import torch
from torch import nn
from torch.nn import functional as F
from .equipment_expansion import features_for_equipment
from .real_mask_server import RuntimeRealMaskMain

VERSION = 'safelog-mpdd2-normal-context-v40'


def white_targets(features):
    return F.adaptive_avg_pool2d(features_for_equipment(features), (4, 4)).half().float()


class WhiteContextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('FIT_mean', torch.zeros(1, 1000, 1, 1))
        self.register_buffer('FIT_scale', torch.ones(1, 1000, 1, 1))
        self.encoder = nn.Sequential(nn.Conv2d(1000, 64, 1), nn.GELU(), nn.Flatten(),
            nn.Linear(1024, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 64))
        self.decoder = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 1024), nn.GELU(),
            nn.Unflatten(1, (64, 4, 4)), nn.Conv2d(64, 1000, 1))

    def standardize(self, target):
        if target.ndim != 4 or target.shape[1:] != (1000, 4, 4):
            raise ValueError('Fixed shared-encoder white-bracket target required')
        return ((target - self.FIT_mean) / self.FIT_scale).clamp(-8, 8)

    def forward(self, target):
        return self.decoder(self.encoder(self.standardize(target)))

    def evidence(self, target):
        # Spatial reconstruction evidence has no verified pixel annotations.
        error = (self(target) - self.standardize(target)).square().mean(1)
        return error, error.flatten(1).topk(4, dim=1).values.mean(1)


class MPDD2NormalMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.white_context_head = WhiteContextHead()

    def forward_mpdd2_white(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, 256, 256):
            raise ValueError('Only MPDD2 aligned 256x256 white-bracket view supported')
        return self.white_context_head.evidence(white_targets(self.backbone(image)))
