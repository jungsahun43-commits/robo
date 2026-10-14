"""Fine screw appearance and coherent whole-template connector comparison."""
import torch
from torch.nn import functional as F
from .component_reference import ComponentReferenceMain
from .equipment_expansion import CATEGORIES, features_for_equipment, anomaly_map, photo_score
from .metal_reference import patch_features, nearest_cosine

ARCH = 'safelog-equipment-component-detail-v4'
METHODS = {'ad_screw': 'fine_low41', 'loco_splicing_connectors': 'whole_template_top10'}


def fine_features(features):
    low = features['low'].float()
    if low.shape[1] != 40: raise ValueError('Accepted low40 features required')
    low = F.normalize(F.avg_pool2d(low, 3, stride=1, padding=1, count_include_pad=False), dim=1, eps=1e-6)
    # A fixed background token makes zero-activation patches valid descriptors.
    return F.normalize(torch.cat((low, torch.full_like(low[:, :1], .01)), 1), dim=1, eps=1e-6)


def whole_template_distance(features, bank, chunk=32):
    b, channels, height, width = features.shape; patches = height*width
    if (bank.ndim != 2 or bank.shape[1] != channels or len(bank) == 0 or len(bank) % patches or chunk < 1 or
            not torch.isfinite(bank).all() or not torch.isfinite(features).all() or
            (bank.norm(dim=1) < 1e-8).any() or (features.norm(dim=1) < 1e-8).any()):
        raise ValueError('Finite nonzero, ordered whole normal templates with matching patch grids required')
    query = F.normalize(features.float(), dim=1).permute(0, 2, 3, 1).reshape(b, patches, channels)
    templates = F.normalize(bank.float(), dim=1).reshape(-1, patches, channels)
    count = max(1, round(patches*.10)); best = query.new_full((b,), float('inf')); chosen = query.new_zeros((b, patches))
    for start in range(0, len(templates), chunk):
        distance = (1-torch.einsum('bpc,npc->bnp', query, templates[start:start+chunk])).clamp(0, 2)
        score = distance.topk(count, dim=2).values.mean(2); value, index = score.min(1)
        take = value < best; selected_map = distance[torch.arange(b, device=query.device), index]
        chosen = torch.where(take[:, None], selected_map, chosen); best = torch.minimum(best, value)
    return chosen.reshape(b, height, width), best


class DetailLayoutMain(ComponentReferenceMain):
    def __init__(self, reference_counts, reference_methods):
        if set(reference_methods) != set(reference_counts): raise ValueError('One explicit reference method per stored category required')
        if any(m != 'patch' and METHODS.get(c) != m for c, m in reference_methods.items()): raise ValueError('Unsupported component detail method')
        standard = {c: n for c, n in reference_counts.items() if reference_methods[c] != 'fine_low41'}
        super().__init__(standard)
        for category, count in reference_counts.items():
            if reference_methods[category] == 'fine_low41':
                if type(count) is not int or count < 1: raise ValueError('Positive fine reference count required')
                self.register_buffer('normal_reference_'+category, torch.zeros(count, 41, dtype=torch.float16))
        self.reference_counts = dict(reference_counts); self.reference_methods = dict(reference_methods)

    def forward_component_outputs(self, image, category):
        method = self.reference_methods.get(category, 'patch')
        if method == 'patch':
            location, reference = super().forward_equipment_evidence(image, category)
            return location, reference, photo_score(reference)
        if category not in CATEGORIES: raise ValueError('Explicit supported category required')
        features = self.backbone(image); target = features_for_equipment(features)
        ids = torch.full((len(image),), CATEGORIES.index(category), device=image.device, dtype=torch.long)
        location = anomaly_map(self.equipment_head(target, ids), target, image.shape[-2:])
        bank = getattr(self, 'normal_reference_'+category)
        if method == 'fine_low41':
            fine = fine_features(features); b, channels, height, width = fine.shape
            query = fine.permute(0, 2, 3, 1).reshape(-1, channels)
            reference = F.interpolate(nearest_cosine(query, bank).reshape(b, 1, height, width), size=image.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            return location, reference, photo_score(reference)
        native_map, score = whole_template_distance(patch_features(features), bank)
        reference = F.interpolate(native_map[:, None], size=image.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        return location, reference, score

    def forward_equipment_evidence(self, image, category):
        location, reference, _ = self.forward_component_outputs(image, category)
        return location, reference
