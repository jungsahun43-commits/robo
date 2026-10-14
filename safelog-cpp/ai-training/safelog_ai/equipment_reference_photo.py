"""Normal FIT patch memory for photo presence; preserve reconstruction locations.

One backbone pass supplies both outputs. The reference distance map is separate
evidence and must not be described as the original localization prediction.
"""
import torch
from torch.nn import functional as F
from .equipment_expansion import ExpandedMain, CATEGORIES, features_for_equipment, anomaly_map
from .metal_reference import patch_features, nearest_cosine

ARCH = 'safelog-equipment-reference-photo-v1'
REFERENCE_CATEGORIES = ('ad_cable', 'ad_metal_nut')


class ReferencePhotoMain(ExpandedMain):
    def __init__(self, reference_counts):
        super().__init__()
        if any(c not in REFERENCE_CATEGORIES or not isinstance(n, int) or n < 1 for c, n in reference_counts.items()):
            raise ValueError('Positive patch counts for supported explicit categories required')
        self.reference_counts = dict(reference_counts)
        for category, count in self.reference_counts.items():
            # Storage precision is fixed before CAL/TEST. Search normalizes in FP32.
            self.register_buffer('normal_reference_' + category, torch.zeros(count, 1000, dtype=torch.float16))

    def forward_equipment_evidence(self, image, category):
        if category not in CATEGORIES:
            raise ValueError('Explicit supported equipment category required')
        features = self.backbone(image)
        target = features_for_equipment(features)
        ids = torch.full((len(image),), CATEGORIES.index(category), device=image.device, dtype=torch.long)
        localization = anomaly_map(self.equipment_head(target, ids), target, image.shape[-2:])
        if category not in self.reference_counts:
            return localization, localization
        patches = patch_features(features)
        b, channels, height, width = patches.shape
        reference = getattr(self, 'normal_reference_' + category)
        distance = nearest_cosine(patches.permute(0, 2, 3, 1).reshape(-1, channels), reference)
        evidence = F.interpolate(distance.reshape(b, 1, height, width), size=image.shape[-2:],
                                 mode='bilinear', align_corners=False)[:, 0]
        return localization, evidence
