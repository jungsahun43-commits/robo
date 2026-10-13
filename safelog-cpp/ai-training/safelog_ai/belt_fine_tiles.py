"""Prespecified finer, geometry-only views of the existing ONE neural model."""
import hashlib
import json
import math
import torch
from torch.nn import functional as F
from .belt_tiles import uniform_boxes, stitch, policy as coarse_policy
from .unified_main_server import tensor

BASELINE = 'global_tile_mean'
MODES = (BASELINE, 'global_fine_mean', 'global_multiscale_mean')


def fine_boxes(width, height):
    if min(width, height) < 1:
        raise ValueError('Positive dimensions required')
    long, short = max(width, height), min(width, height)
    if long <= 640 or long > 3*short:
        return []
    side = min(short, max(640, math.ceil(.25*long), math.ceil(.4*short)))
    def positions(length):
        count = math.ceil((length-side)/(.8*side))
        return [0] if count == 0 else sorted(set(round(i*(length-side)/count) for i in range(count+1)))
    xs, ys = positions(width), positions(height)
    boxes = [(x, y, x+side, y+side) for y in ys for x in xs]
    assert len(boxes) <= 15
    return boxes


def policy(mode):
    if mode not in MODES:
        raise ValueError('Explicit supported mode required')
    if mode == BASELINE:
        return coarse_policy(mode)
    value = {'version': 'safelog-belt-fine-tiles-v1', 'beltMode': mode,
        'model_count': 1, 'annotations_at_inference': False,
        'fine_geometry': 'side=min(short,max640,ceil.25long,ceil.4short); uniform end-inclusive positions with gap<=.8side; max15 squares',
        'coarse_geometry': coarse_policy(BASELINE)['geometry'],
        'tiny_or_panorama_fallback': 'global only if maxedge<=640 or aspect>3',
        'stitch': coarse_policy(BASELINE)['stitch'],
        'fusion': '.5global+.5fine' if mode == 'global_fine_mean' else '.5global+.25coarse+.25fine',
        'threshold': .5, 'new_neural_updates': 0}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**value, 'sha256': digest}


def project(maps, boxes, size, shape, box):
    native = stitch(maps, boxes, *size)
    left, top, width, height = box
    grid = maps.new_zeros(shape)
    grid[:, :, top:top+height, left:left+width] = F.interpolate(native, size=(height, width), mode='bilinear', align_corners=False)
    return grid


def fuse(global_prob, coarse, fine, box, mode):
    if mode not in MODES:
        raise ValueError('Explicit supported mode required')
    if fine is None:
        if coarse is not None:
            raise ValueError('Inconsistent fallback geometry')
        return global_prob
    if coarse is None or global_prob.shape != fine.shape or coarse.shape != fine.shape:
        raise ValueError('Matching full-grid probabilities required')
    left, top, width, height = box
    region = (slice(None), slice(None), slice(top, top+height), slice(left, left+width))
    result = global_prob.clone()
    if mode == BASELINE:
        result[region] = .5*global_prob[region]+.5*coarse[region]
    elif mode == 'global_fine_mean':
        result[region] = .5*global_prob[region]+.5*fine[region]
    else:
        result[region] = .5*global_prob[region]+.25*coarse[region]+.25*fine[region]
    return result


@torch.inference_mode()
def components(model, rgb, device='cpu', include_coarse=True, include_fine=True):
    image, box = tensor(rgb, 'belt_damage')
    global_prob = model.forward_task(image.to(device), 'belt_damage').sigmoid()
    geometries = {'coarse': uniform_boxes(*rgb.size), 'fine': fine_boxes(*rgb.size)}
    grids, counts = {}, {}
    for name, enabled in [('coarse', include_coarse), ('fine', include_fine)]:
        boxes = geometries[name] if enabled else []
        counts[name] = len(boxes)
        if not boxes:
            grids[name] = None
            continue
        maps = []
        for offset in range(0, len(boxes), 2):
            batch = torch.cat([tensor(rgb.crop(b), 'belt_damage')[0] for b in boxes[offset:offset+2]]).to(device)
            maps.append(model.forward_task(batch, 'belt_damage').sigmoid())
        grids[name] = project(torch.cat(maps), boxes, rgb.size, global_prob.shape, box)
    return global_prob, grids['coarse'], grids['fine'], box, counts


@torch.inference_mode()
def predict(model, rgb, mode, device='cpu'):
    if mode not in MODES:
        raise ValueError('Explicit supported mode required')
    # The accepted baseline remains the exact historical implementation.
    if mode == BASELINE:
        from .belt_tiles import predict as old_predict
        return old_predict(model, rgb, mode, device)
    global_prob, coarse, fine, box, counts = components(model, rgb, device,
        include_coarse=mode == 'global_multiscale_mean')
    if mode == 'global_fine_mean' and fine is not None:
        coarse = fine  # Ignored by this mode; retain matching-grid validation.
    return fuse(global_prob, coarse, fine, box, mode), box, sum(counts.values())


def guards(result, baseline, slices, baseline_slices):
    return {
        'IoU_gain_at_least_half_percentage_point': result['pixel_iou'] >= baseline['pixel_iou']+.005,
        'background_pixel_FPR_not_worse': result['pixel_fpr'] <= baseline['pixel_fpr'],
        'pixel_FNR_not_worse': result['pixel_fnr'] <= baseline['pixel_fnr'],
        'photo_FNR_not_worse': result['photo_rates']['fnr'] <= baseline['photo_rates']['fnr'],
        'zero_GT_overlap_not_worse': result['positive_photos_without_any_gt_overlap'] <= baseline['positive_photos_without_any_gt_overlap'],
        'small_damage_IoU_not_worse': slices['GT-area:<=0.25%']['pixel_iou'] >= baseline_slices['GT-area:<=0.25%']['pixel_iou'],
        'small_damage_pixel_FNR_not_worse': slices['GT-area:<=0.25%']['pixel_fnr'] <= baseline_slices['GT-area:<=0.25%']['pixel_fnr'],
        'each_known_sequence_pixel_FNR_within_one_percentage_point': all(
            slices[k]['pixel_fnr'] <= baseline_slices[k]['pixel_fnr']+.01 for k in baseline_slices if k.startswith('sequence:'))}
