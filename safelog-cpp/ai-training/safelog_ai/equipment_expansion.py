"""New equipment outputs attached to the existing ONE shared image backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from .unified_main import UnifiedMain, TASKS as OLD_TASKS

ARCH = 'safelog-unified-equipment-expansion-v1'
CATEGORIES = ('ad_cable', 'ad_screw', 'ad_metal_nut', 'loco_splicing_connectors',
              'mpdd_bracket_black', 'mpdd_bracket_brown', 'mpdd_bracket_white',
              'mpdd_connector', 'mpdd_metal_plate', 'mpdd_tubes', 'ksdd2_surface')
SIZE = 256


def features_for_equipment(features):
    """Use the existing low/high features; no second pretrained encoder."""
    low, high = features['low'], features['high']
    if low.shape[1] != 40 or high.shape[1] != 960:
        raise ValueError('Expected the accepted MobileNet shared backbone')
    merged = torch.cat([F.normalize(F.adaptive_avg_pool2d(low.float(), high.shape[-2:]), dim=1, eps=1e-6),
                        F.normalize(high.float(), dim=1, eps=1e-6)], 1)
    return F.normalize(merged.float(), dim=1, eps=1e-6)


class EquipmentHead(nn.Module):
    """A single small category-conditioned feature reconstructor for11 types."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(CATEGORIES), 16)
        self.encoder = nn.Sequential(nn.Conv2d(1000, 128, 1), nn.ReLU(), nn.Conv2d(128, 32, 1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(48, 128, 1), nn.ReLU(), nn.Conv2d(128, 1000, 1))

    def forward(self, features, category_ids):
        if features.ndim != 4 or features.shape[1] != 1000 or category_ids.shape != (len(features),):
            raise ValueError('One explicit category per feature image required')
        if ((category_ids < 0) | (category_ids >= len(CATEGORIES))).any():
            raise ValueError('Unknown equipment category cannot be inferred as normal')
        latent = self.encoder(features.detach())
        condition = self.embedding(category_ids)[:, :, None, None].expand(-1, -1, *latent.shape[-2:])
        return self.decoder(torch.cat([latent, condition], 1))


class ExpandedMain(UnifiedMain):
    def __init__(self):
        super().__init__(); self.equipment_head = EquipmentHead()

    def initialize_accepted(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        if extra or any(not n.startswith('equipment_head.') for n in missing):
            raise ValueError('All accepted model tensors must match; only new head may be absent')
        return self

    def forward_equipment(self, image, category):
        if category not in CATEGORIES:
            raise ValueError('Explicit supported equipment category required')
        target = features_for_equipment(self.backbone(image))
        ids = torch.full((len(image),), CATEGORIES.index(category), device=image.device, dtype=torch.long)
        reconstructed = self.equipment_head(target, ids)
        return anomaly_map(reconstructed, target, image.shape[-2:])

    def forward_task(self, image, task):
        if task in OLD_TASKS: return super().forward_task(image, task)
        if task == 'equipment_surface':
            raise ValueError('equipment_surface also requires an explicit equipment category')
        raise ValueError('Unsupported inspection task')


def reconstruction_loss(predicted, target):
    if predicted.shape != target.shape: raise ValueError('Matching fixed feature shapes required')
    return (1-F.cosine_similarity(predicted.float(), target.detach().float(), dim=1)).mean()


def anomaly_map(predicted, target, size=(SIZE, SIZE)):
    distance = (1-F.cosine_similarity(predicted.float(), target.float(), dim=1)).clamp_min(0)
    # Coarse feature evidence, not confirmed pixel segmentation.
    return F.interpolate(distance[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]


def photo_score(maps):
    # Fixed before evaluation. Avoid one isolated activation deciding the photo.
    count = max(1, round(maps.shape[-2]*maps.shape[-1]*.01))
    return maps.flatten(1).topk(count, dim=1).values.mean(1)


def stage_gate(fpr, fnr, normal_count, defect_count):
    if min(normal_count, defect_count) < 20 or fpr is None or fnr is None:
        return {'state': 'insufficient_evaluation', 'ready_for_next_stage': False}
    if not (0 <= fpr <= 1 and 0 <= fnr <= 1): raise ValueError('Rates must be between0 and1')
    return {'state': 'target15_met' if max(fpr, fnr) <= .15 else 'target20_met' if max(fpr, fnr) <= .20 else 'needs_improvement',
            'ready_for_next_stage': max(fpr, fnr) <= .20}
