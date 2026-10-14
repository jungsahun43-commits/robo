"""Additional metal component evidence without replacing accepted neural paths."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .industrial_supervised import IndustrialSupervisedMain
from .equipment_expansion import CATEGORIES, features_for_equipment, anomaly_map, photo_score
from .metal_reference import patch_features, nearest_cosine

ARCH = 'safelog-industrial-surface-reference-v9'
SURFACES = ('mpdd_bracket_black', 'mpdd_bracket_brown', 'mpdd_bracket_white', 'mpdd_connector', 'mpdd_tubes')
METHODS = ('pooled256', 'aspect512')


def surface_view(rgb, method):
    if method not in METHODS: raise ValueError('Registered metal component view required')
    if method == 'pooled256': return rgb.resize((256, 256), Image.Resampling.BICUBIC), {'size': [256, 256], 'method': method}
    scale = min(512/rgb.width, 512/rgb.height)
    size = (max(1, round(rgb.width*scale)), max(1, round(rgb.height*scale)))
    offset = ((512-size[0])//2, (512-size[1])//2)
    value = Image.new('RGB', (512, 512), (127, 127, 127)); value.paste(rgb.resize(size, Image.Resampling.BICUBIC), offset)
    return value, {'method': method, 'size': [512, 512], 'scale': scale, 'offset': list(offset), 'source_size': list(rgb.size)}


def surface_evidence(features, bank, size):
    values = patch_features(features); b, channels, height, width = values.shape
    distances = nearest_cosine(values.permute(0, 2, 3, 1).reshape(-1, channels), bank, query_chunk=2048, reference_chunk=8192)
    heat = F.interpolate(distances.reshape(b, 1, height, width), size=size, mode='bilinear', align_corners=False)[:, 0]
    return heat, photo_score(heat)


class IndustrialSurfaceMain(IndustrialSupervisedMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant); self.surface_specs = dict(surface_specs)
        for category, spec in surface_specs.items():
            if category not in SURFACES or spec['method'] not in METHODS or type(spec['patches']) is not int or spec['patches'] < 1:
                raise ValueError('Explicit supported surface category, method and bank count required')
            self.register_buffer('surface_reference_'+category, torch.zeros(spec['patches'], 1000, dtype=torch.float16))

    def forward_surface(self, original, view, category):
        if category not in self.surface_specs: raise ValueError('Surface reference not selected')
        if self.surface_specs[category]['method'] == 'pooled256':
            features = self.backbone(original); target = features_for_equipment(features)
            ids = torch.full((len(original),), CATEGORIES.index(category), device=original.device, dtype=torch.long)
            location = anomaly_map(self.equipment_head(target, ids), target, original.shape[-2:])
        else:
            location = super().forward_equipment(original, category); features = self.backbone(view)
        evidence, score = surface_evidence(features, getattr(self, 'surface_reference_'+category), view.shape[-2:])
        return location, evidence, score
