"""Train a shared component discriminator on normal vs synthetic feature noise.

Inspired by SimpleNet, not its architecture or benchmark reproduction. All
accepted neural tensors and normal reference buffers remain frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .equipment_reference_photo import ReferencePhotoMain
from .equipment_expansion import CATEGORIES, features_for_equipment, anomaly_map
from .metal_reference import patch_features
from .component_reference import NEW_CATEGORIES

ARCH = 'safelog-equipment-component-discriminator-v5'


def synthetic_features(normal, noise, standard_deviation=.005):
    if normal.shape != noise.shape or normal.ndim != 2 or not 0 < standard_deviation < 1:
        raise ValueError('Matching normal feature noise with explicit positive scale required')
    return F.normalize(F.normalize(normal.float(), dim=1)+noise.float()*standard_deviation, dim=1, eps=1e-6)


class ComponentDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(NEW_CATEGORIES), 16)
        self.layers = nn.Sequential(nn.Linear(1016, 256), nn.LeakyReLU(.2), nn.Linear(256, 128), nn.LeakyReLU(.2), nn.Linear(128, 1))

    def forward(self, features, category_ids):
        if features.ndim != 2 or features.shape[1] != 1000 or category_ids.shape != (len(features),):
            raise ValueError('One supported explicit category per1000-channel feature required')
        if ((category_ids < 0) | (category_ids >= len(NEW_CATEGORIES))).any(): raise ValueError('Unknown component category')
        condition = self.embedding(category_ids)*.05
        return self.layers(torch.cat((F.normalize(features.float(), dim=1), condition), dim=1)).flatten()


class DiscriminatorMain(ReferencePhotoMain):
    def __init__(self, reference_counts, active_components):
        super().__init__(reference_counts)
        if len(set(active_components)) != len(active_components) or any(c not in NEW_CATEGORIES for c in active_components):
            raise ValueError('Unique explicit trained discriminator components required')
        self.active_components = tuple(active_components)
        self.component_discriminator = ComponentDiscriminator()

    def forward_equipment_evidence(self, image, category):
        if category not in self.active_components: return super().forward_equipment_evidence(image, category)
        features = self.backbone(image); target = features_for_equipment(features)
        ids = torch.full((len(image),), CATEGORIES.index(category), device=image.device, dtype=torch.long)
        location = anomaly_map(self.equipment_head(target, ids), target, image.shape[-2:])
        patches = patch_features(features); b, channels, height, width = patches.shape
        flat = patches.permute(0, 2, 3, 1).reshape(-1, channels)
        category_ids = torch.full((len(flat),), NEW_CATEGORIES.index(category), device=image.device, dtype=torch.long)
        logits = self.component_discriminator(flat, category_ids).reshape(b, 1, height, width)
        # Bounded synthetic-class score, NOT real defect probability.
        evidence = F.interpolate(logits.sigmoid(), size=image.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        return location, evidence
