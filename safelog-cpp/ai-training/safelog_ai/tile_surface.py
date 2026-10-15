"""Fixed overlapping RGB detail views, one preserved encoder, no category shortcut."""
import math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .fine_target_surface import FineTargetHead, trainable_fine_head
from .fine_feature_probe import forward_with_fine
from .real_mask_server import RuntimeRealMaskMain
from .magnetic_surface import surface_view
from .ksdd_supervised import ksdd_photo_score

VERSION = 'safelog-neu-detail-v22'


def tile_boxes(size):
    """Nine half-width/half-height views at start, middle, end of each axis."""
    width, height = size
    if min(size) < 128:
        raise ValueError('Detail view requires at least 128 pixels on each source axis')
    w, h = math.ceil(width / 2), math.ceil(height / 2)
    xs, ys = (0, (width - w) // 2, width - w), (0, (height - h) // 2, height - h)
    return [(x, y, x + w, y + h) for y in ys for x in xs]


def tile_view(rgb, box, truth=None):
    if tuple(box) not in tile_boxes(rgb.size):
        raise ValueError('Only the declared source-independent nine windows are supported')
    crop = rgb.crop(box)
    if truth is not None and truth.size != rgb.size:
        raise ValueError('Original mask and RGB coordinates differ')
    label = None if truth is None else truth.crop(box)
    return surface_view(crop, label)


def per_tile_loss(logits, mask, valid):
    """Vector of per-view balanced BCE + Dice + .5 photo BCE, without area dilution."""
    if logits.shape != mask.shape or mask.shape != valid.shape or logits.ndim != 4:
        raise ValueError('Aligned BCHW tensors required')
    target, known = mask.float(), valid.float()
    positive, negative = target * known, (1 - target) * known
    pc, nc = positive.flatten(1).sum(1), negative.flatten(1).sum(1)
    if bool((nc == 0).any()):
        raise ValueError('Each crop requires some valid annotation-negative background')
    p = (F.softplus(-logits.float()) * positive).flatten(1).sum(1) / pc.clamp_min(1)
    n = (F.softplus(logits.float()) * negative).flatten(1).sum(1) / nc
    pixel = torch.where(pc > 0, .5 * (p + n), n)
    prob = logits.sigmoid()
    intersection = (prob * positive).flatten(1).sum(1)
    denom = ((prob + target) * known).flatten(1).sum(1)
    dice = 1 - (2 * intersection + 1) / (denom + 1)
    score = ksdd_photo_score(prob[:, 0], valid[:, 0].bool())
    photo = F.binary_cross_entropy(score.clamp(1e-6, 1 - 1e-6), (pc > 0).float(), reduction='none')
    return pixel + dice + .5 * photo


def group_photo_scores(patch_scores, photo_ids):
    if len(patch_scores) != len(photo_ids):
        raise ValueError('One original-photo id per crop score required')
    scores = {}
    for key, value in zip(photo_ids, patch_scores):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Finite crop scores required')
        scores[key] = max(scores.get(key, -math.inf), value)
    return scores


class DetailSurfaceMain(RuntimeRealMaskMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.detail_surface_head = FineTargetHead()

    def forward_detail_features(self, low, high, fine, size=(512, 512)):
        # Exactly the same zero channel for every source, category and inference view.
        zero = low.new_zeros((len(low), 1, *high.shape[-2:]))
        logits = self.detail_surface_head(low, high, zero, fine)
        return F.interpolate(logits, size=size, mode='bilinear', align_corners=False)[:, 0]

    def forward_detail_view(self, image):
        values = forward_with_fine(self.backbone, image)
        return self.forward_detail_features(values['low'], values['high'], values['fine'], image.shape[-2:])
