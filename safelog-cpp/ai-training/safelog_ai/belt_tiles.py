"""Geometry-only native views of ONE belt model; annotations never enter inference."""
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from .unified_main_server import tensor

MODES = ('global_only', 'global_tile_mean')
CONFIGS = ('accepted_tiles', 'trained_global', 'trained_tiles')


def uniform_boxes(width, height):
    """At most three long-axis x two short-axis squares, with complete coverage."""
    if min(width, height) < 1:
        raise ValueError('Positive native dimensions required')
    long, short = max(width, height), min(width, height)
    if long <= 640 or long > 3*short: return []
    side = min(short, max(640, math.ceil(.4*long), math.ceil(.6*short)))
    lp = sorted(set([0, (long-side)//2, long-side])); sp = sorted(set([0, short-side]))
    if width >= height: return [(x, y, x+side, y+side) for y in sp for x in lp]
    return [(x, y, x+side, y+side) for y in lp for x in sp]


def raster_view(image, annotation, box):
    """TRAIN-only label transform. Empty view truth is local mask, not photo health."""
    if image.size != (annotation['imageWidth'], annotation['imageHeight']): raise ValueError('Native geometry mismatch')
    left, top, right, bottom = box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height and right-left == bottom-top):
        raise ValueError('Native in-bounds square view required')
    scale = 640/(right-left); mask = Image.new('L', (640, 640)); draw = ImageDraw.Draw(mask)
    for shape in annotation['shapes']:
        if shape['label'] != 'crack': raise ValueError('Unknown publisher label')
        if not all(math.isfinite(float(v)) for p in shape['points'] for v in p): raise ValueError('Nonfinite publisher vertices')
        points = [((x-left)*scale, (y-top)*scale) for x, y in shape['points']]
        if shape['shape_type'] == 'polygon' and len(points) >= 3: draw.polygon(points, fill=255)
        elif shape['shape_type'] == 'rectangle' and len(points) == 2:
            xs, ys = zip(*points); draw.rectangle((min(xs), min(ys), max(xs), max(ys)), fill=255)
        else: raise ValueError('Publisher polygon/rectangle required')
    return image.crop(box).resize((640, 640), Image.Resampling.BILINEAR), mask, Image.new('L', (640, 640), 255)


def stitch(maps, boxes, width, height):
    if len(maps) != len(boxes) or not boxes: raise ValueError('Matching probability maps and covered boxes required')
    canvas = maps.new_zeros((1, 1, height, width)); weight = canvas.clone()
    for i, (left, top, right, bottom) in enumerate(boxes):
        side = right-left; p = F.interpolate(maps[i:i+1], size=(side, side), mode='bilinear', align_corners=False)
        window = torch.hann_window(side, periodic=False, device=maps.device, dtype=maps.dtype).clamp_min(.05)
        window = window[:, None]*window[None, :]
        canvas[:, :, top:bottom, left:right] += p*window
        weight[:, :, top:bottom, left:right] += window
    if not (weight > 0).all(): raise ValueError('Tile coverage hole; cannot silently fill missing pixels')
    return canvas/weight


def policy(mode):
    if mode not in MODES: raise ValueError('Explicit supported inference mode required')
    value = {'version': 'safelog-belt-uniform-tiles-v1', 'beltMode': mode, 'model_count': 1,
        'geometry': 'max6 squares; side=min(short,max640,ceil.4long,ceil.6short); uniform3long x2short',
        'tiny_or_panorama_fallback': 'global only if maxedge<=640 or aspect>3',
        'stitch': 'positive-floor Hann weighted mean at native coordinates then bilinear to original640 valid grid',
        'fusion': '0.5global probability +0.5tile probability, threshold unchanged', 'annotations_at_inference': False}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {**value, 'sha256': digest}


@torch.inference_mode()
def predict(model, rgb, mode, device='cpu'):
    if mode not in MODES: raise ValueError('Explicit inference mode required')
    image, box = tensor(rgb, 'belt_damage'); global_prob = model.forward_task(image.to(device), 'belt_damage').sigmoid()
    boxes = uniform_boxes(*rgb.size) if mode == 'global_tile_mean' else []
    if not boxes: return global_prob, box, 0
    maps = []
    # Batch2 bounds CPU memory and gives the same fixed geometry on CUDA.
    for offset in range(0, len(boxes), 2):
        images = torch.cat([tensor(rgb.crop(b), 'belt_damage')[0] for b in boxes[offset:offset+2]]).to(device)
        maps.append(model.forward_task(images, 'belt_damage').sigmoid())
    native = stitch(torch.cat(maps), boxes, *rgb.size)
    left, top, width, height = box
    tiles_grid = global_prob.new_zeros(global_prob.shape)
    tiles_grid[:, :, top:top+height, left:left+width] = F.interpolate(native, size=(height, width), mode='bilinear', align_corners=False)
    fused = global_prob.clone()
    fused[:, :, top:top+height, left:left+width] = .5*global_prob[:, :, top:top+height, left:left+width]+.5*tiles_grid[:, :, top:top+height, left:left+width]
    return fused, box, len(boxes)


def select_tiles(results, baseline):
    gates = {}
    for name, r in results.items():
        gates[name] = {'IoU_gain_at_least_half_percentage_point': r['pixel_iou'] >= baseline['pixel_iou']+.005,
            'background_pixel_FPR_not_worse': r['pixel_fpr'] <= baseline['pixel_fpr'],
            'photo_FNR_not_worse': r['photo_rates']['fnr'] <= baseline['photo_rates']['fnr'],
            'pixel_FNR_not_worse': r['pixel_fnr'] <= baseline['pixel_fnr'],
            'zero_GT_overlap_photos_not_worse': r['positive_photos_without_any_gt_overlap'] <= baseline['positive_photos_without_any_gt_overlap']}
    accepted = [n for n in results if all(gates[n].values())]
    selected = min(accepted, key=lambda n: (-results[n]['pixel_iou'], results[n]['pixel_fpr'], n != 'trained_global')) if accepted else 'prior_native_mix'
    return {'selected': selected, 'guards': gates, 'normal_belt_photo_FPR_available': False, 'known_VAL_reused': True,
        'reserved_TEST_used': False, 'independent_factory_verified': False, 'app_default_promoted': False}
