"""Publisher-known hard background mining; positive reference is never a new label."""
import math
import torch
from torch.nn import functional as F
from .belt_segmentation import loss as base_loss

VARIANTS = ('anchor_control', 'hard_background')
HARD_WEIGHT = .05
ANCHOR_WEIGHT = 2.


def checked_grids(logits, mask, valid):
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != mask.shape or mask.shape != valid.shape:
        raise ValueError('Matching N1HW logits, publisher mask and valid grids required')
    if not torch.isfinite(logits).all() or not ((mask == 0) | (mask == 1)).all() or not ((valid == 0) | (valid == 1)).all():
        raise ValueError('Finite logits and explicit binary publisher supervision required')


def background_loss(logits, mask, valid):
    checked_grids(logits, mask, valid)
    # Polygon rasterization is approximate: omit a three-pixel boundary band.
    far_background = (valid > 0) & (F.max_pool2d(mask.float(), 7, stride=1, padding=3) == 0)
    pointwise = F.softplus(logits.float())
    losses = []
    for i in range(len(logits)):
        values = pointwise[i][far_background[i]]
        if values.numel():
            k = max(1, math.ceil(.01*values.numel()))
            losses.append(values.topk(k).values.mean())
        else:
            losses.append(logits[i].sum()*0.)
    return torch.stack(losses).mean()


def positive_anchor(logits, reference_logits, mask, valid):
    checked_grids(logits, mask, valid)
    if reference_logits.shape != logits.shape or not torch.isfinite(reference_logits).all():
        raise ValueError('Matching finite frozen-reference logits required')
    reference = reference_logits.detach().float().sigmoid()
    selected = (mask > 0) & (valid > 0) & (reference >= .5)
    deficit = (reference-logits.float().sigmoid()).clamp_min(0).square()
    per_photo = (deficit*selected).flatten(1).sum(1)/selected.flatten(1).sum(1).clamp_min(1)
    return per_photo.mean()


def training_loss(variant, logits, reference_logits, truth):
    if variant not in VARIANTS:
        raise ValueError('Explicit predeclared background study variant required')
    mask, valid = truth['mask'], truth['valid']
    value = base_loss(logits, mask, valid, 20)+ANCHOR_WEIGHT*positive_anchor(logits, reference_logits, mask, valid)
    if variant == 'hard_background': value = value+HARD_WEIGHT*background_loss(logits, mask, valid)
    return value


def select_background(results, baseline):
    gates = {}
    for name, result in results.items():
        gates[name] = {
            'background_pixel_FPR_relative_reduction_at_least_10pct': result['pixel_fpr'] <= baseline['pixel_fpr']*.9,
            'pixel_IoU_not_worse': result['pixel_iou'] >= baseline['pixel_iou'],
            'photo_FNR_not_worse': result['photo_rates']['fnr'] <= baseline['photo_rates']['fnr'],
            'pixel_FNR_not_worse': result['pixel_fnr'] <= baseline['pixel_fnr'],
            'zero_GT_overlap_photos_not_worse': result['positive_photos_without_any_gt_overlap'] <= baseline['positive_photos_without_any_gt_overlap'],
        }
    accepted = [name for name in results if all(gates[name].values())]
    selected = min(accepted, key=lambda n: (results[n]['pixel_fpr'], -results[n]['pixel_iou'], n != 'anchor_control')) if accepted else 'prior_native_mix'
    return {'selected': selected, 'guards': gates, 'normal_belt_photo_FPR_available': False,
            'known_VAL_reused': True, 'reserved_TEST_used': False, 'independent_factory_verified': False, 'app_default_promoted': False}
