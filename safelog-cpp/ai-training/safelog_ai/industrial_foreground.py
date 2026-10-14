"""Input-only foreground evidence on the accepted single shared backbone.

These colour masks are part candidates, not defect annotations or measured
segmentation truth. A failed mask requests review; it never means healthy.
"""
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .industrial_surface_reference import surface_view
from .ksdd_photo_only import KSDDPhotoMain
from .metal_reference import patch_features, nearest_cosine

CATEGORIES = ('mpdd_bracket_black', 'mpdd_tubes')
VERSION = 'safelog-industrial-foreground-v12'


def foreground_view(rgb):
    if min(rgb.size) < 64:
        raise ValueError('Foreground evidence needs at least64 pixels; manual review required')
    view, geometry = surface_view(rgb, 'aspect512')
    x, y = geometry['offset']; scale = geometry['scale']
    w, h = max(1, round(rgb.width*scale)), max(1, round(rgb.height*scale))
    local = np.asarray(view)[y:y+h, x:x+w]
    lab = cv2.cvtColor(local.astype(np.float32)/255., cv2.COLOR_RGB2Lab)
    border = np.zeros((h, w), dtype=bool)
    band = max(1, round(min(h, w)*.05))
    border[:band] = True; border[-band:] = True
    border[:, :band] = True; border[:, -band:] = True
    samples = lab[border][::max(1, math.ceil(int(border.sum())/4096))]
    # Deterministic farthest-point initialisation and eight Lloyd iterations.
    centres = [samples[np.argmin(np.linalg.norm(samples-np.median(samples, axis=0), axis=1))]]
    for _ in range(2):
        distance = ((samples[:, None]-np.asarray(centres)[None])**2).sum(2).min(1)
        centres.append(samples[np.argmax(distance)])
    centres = np.asarray(centres)
    for _ in range(8):
        assignment = ((samples[:, None]-centres[None])**2).sum(2).argmin(1)
        centres = np.stack([samples[assignment == i].mean(0) if (assignment == i).any() else centres[i] for i in range(3)])
    distances = np.sqrt(((lab[:, :, None]-centres[None, None])**2).sum(3).min(2))
    threshold = max(12., float(np.quantile(distances[border], .99)))
    mask = (distances > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 32:
            kept[labels == i] = 1
    coverage = float(kept.mean())
    if not .01 <= coverage <= .75:
        raise ValueError('Foreground area outside1to75percent; manual review required')
    result = np.zeros((512, 512), dtype=bool); result[y:y+h, x:x+w] = kept > 0
    geometry.update(foreground_fraction=coverage, input_border_Lab_threshold=threshold,
                    mask_source='input_colour_only_not_defect_annotation', mask_quality_verified=False)
    return view, torch.from_numpy(result), geometry


def token_mask(mask, shape):
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError('A boolean input foreground mask per photo is required')
    return F.adaptive_avg_pool2d(mask[:, None].float(), shape)[:, 0] >= .25


def foreground_score(heat, mask):
    if heat.shape != mask.shape or not torch.isfinite(heat).all():
        raise ValueError('Finite evidence and matching foreground required')
    values = []
    for value, selected in zip(heat, mask):
        candidate = value[selected]
        if len(candidate) < 1:
            raise ValueError('Empty foreground requests review')
        values.append(candidate.topk(max(1, round(len(candidate)*.01))).values.mean())
    return torch.stack(values)


def foreground_evidence(features, bank, mask):
    patches = patch_features(features); b, c, h, w = patches.shape
    tokens = token_mask(mask, (h, w)); maps = []
    for value, selected in zip(patches, tokens):
        if int(selected.sum()) < 4:
            raise ValueError('Fewer than4 foreground feature cells; manual review required')
        # Distances outside foreground cells never contribute to photo score.
        query = value.permute(1, 2, 0)[selected]
        distance = nearest_cosine(query, bank, query_chunk=2048, reference_chunk=8192)
        coarse = value.new_zeros((h, w)); coarse[selected] = distance
        maps.append(coarse)
    heat = F.interpolate(torch.stack(maps)[:, None], size=mask.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    return heat, foreground_score(heat, mask)


class ForegroundMain(KSDDPhotoMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs, foreground_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.foreground_specs = dict(foreground_specs)
        for category, spec in self.foreground_specs.items():
            if category not in CATEGORIES or type(spec['patches']) is not int or spec['patches'] < 4:
                raise ValueError('Explicit fitted foreground category and nonempty bank required')
            self.register_buffer('foreground_reference_'+category, torch.zeros(spec['patches'], 1000, dtype=torch.float16))

    def forward_foreground(self, original, view, mask, category):
        if category not in self.foreground_specs:
            raise ValueError('Foreground reference not fitted')
        location = super().forward_equipment(original, category)
        evidence, score = foreground_evidence(self.backbone(view), getattr(self, 'foreground_reference_'+category), mask)
        return location, evidence, score
