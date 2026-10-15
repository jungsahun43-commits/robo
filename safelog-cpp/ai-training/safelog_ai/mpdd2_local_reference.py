"""Normal patch registration on the same encoder; no gradient optimization."""
import torch
from torch.nn import functional as F
from .fine_feature_probe import forward_with_fine
from .metal_reference import nearest_cosine
from .real_mask_server import RuntimeRealMaskMain

VERSION = 'safelog-mpdd2-local-reference-v41'


def white_local_features(values):
    fine, low = values['fine'].float(), values['low'].float()
    if fine.shape[1:] != (24, 64, 64) or low.shape[1:] != (40, 32, 32):
        raise ValueError('Fixed shared fine24/low40 geometry required')
    up = F.interpolate(low, size=(64, 64), mode='bilinear', align_corners=False)
    joined = torch.cat((F.normalize(fine, dim=1, eps=1e-6), F.normalize(up, dim=1, eps=1e-6)), dim=1)
    joined = F.avg_pool2d(joined, 3, stride=1, padding=1, count_include_pad=False)
    # A constant background token preserves a nonzero representation in blank areas.
    joined = torch.cat((joined, torch.full_like(joined[:, :1], .01)), dim=1)
    return F.normalize(joined, dim=1, eps=1e-6).half().float()


def reference_evidence(features, bank):
    if features.ndim != 4 or features.shape[1:] != (65, 64, 64):
        raise ValueError('Aligned 65-channel reference feature grid required')
    batch = len(features)
    query = features.permute(0, 2, 3, 1).reshape(-1, 65)
    distances = nearest_cosine(query, bank, query_chunk=1024, reference_chunk=8192).reshape(batch, 64, 64)
    return distances, distances.flatten(1).topk(41, dim=1).values.mean(1)


class MPDD2ReferenceMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs, white_reference_count=28928):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.register_buffer('mpdd2_white_reference', torch.zeros(white_reference_count, 65, dtype=torch.float16))

    def forward_mpdd2_white_reference(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, 256, 256):
            raise ValueError('Only MPDD2 white aligned256x256 source domain supported')
        features = white_local_features(forward_with_fine(self.backbone, image))
        return reference_evidence(features, self.mpdd2_white_reference)
