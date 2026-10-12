"""TRAIN-only native belt crops, preserving publisher geometry and padding truth."""
import math
import numpy as np
from PIL import Image, ImageDraw
import torch


def crop_box(annotation):
    """A fixed contextual square around the first publisher shape; no model scores."""
    width, height = annotation['imageWidth'], annotation['imageHeight']
    shapes = annotation['shapes']
    if not shapes or min(width, height) < 1:
        raise ValueError('A positive publisher annotation with native geometry is required')
    points = shapes[0]['points']
    xs, ys = zip(*points)
    if not all(math.isfinite(float(x)) for pair in points for x in pair):
        raise ValueError('Finite publisher vertices required')
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    side = min(width, height, max(720, math.ceil(1.5*span)))
    left = max(0, min(width-side, round((min(xs)+max(xs)-side)/2)))
    top = max(0, min(height-side, round((min(ys)+max(ys)-side)/2)))
    return left, top, left+side, top+side


def native_crop(image, annotation):
    if image.size != (annotation['imageWidth'], annotation['imageHeight']):
        raise ValueError('Native photo and annotation geometry differ')
    box = crop_box(annotation)
    left, top, right, bottom = box
    scale = 640/(right-left)
    mask = Image.new('L', (640, 640))
    draw = ImageDraw.Draw(mask)
    for shape in annotation['shapes']:
        if shape['label'] != 'crack':
            raise ValueError('Unknown publisher label cannot become a crack')
        points = [((float(x)-left)*scale, (float(y)-top)*scale) for x, y in shape['points']]
        if shape['shape_type'] == 'polygon' and len(points) >= 3:
            draw.polygon(points, fill=255)
        elif shape['shape_type'] == 'rectangle' and len(points) == 2:
            xs, ys = zip(*points)
            draw.rectangle((min(xs), min(ys), max(xs), max(ys)), fill=255)
        else:
            raise ValueError('Supported publisher polygon or rectangle required')
    if not np.any(np.asarray(mask)):
        raise ValueError('Positive native crop lost its publisher target')
    return image.crop(box).resize((640, 640), Image.Resampling.BILINEAR), mask, Image.new('L', (640, 640), 255), box


def paired_flip(images, mask, valid, flips):
    """Clone before mirroring so persistent caches and other truth stay intact."""
    if images.ndim != 4 or mask.shape != valid.shape or mask.ndim != 4:
        raise ValueError('Batched images and matching spatial mask/valid grids required')
    if images.shape[0] != mask.shape[0] or images.shape[-2:] != mask.shape[-2:]:
        raise ValueError('Image and supervision coordinates differ')
    images, mask, valid = images.clone(), mask.clone(), valid.clone()
    images[flips] = images[flips].flip(-1)
    mask[flips.to(mask.device)] = mask[flips.to(mask.device)].flip(-1)
    valid[flips.to(valid.device)] = valid[flips.to(valid.device)].flip(-1)
    return images, mask, valid


def configure_belt_only(model):
    from .unified_main import freeze_bn_statistics
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    model.belt_head.requires_grad_(True)
    model.eval()
    model.belt_head.train()
    freeze_bn_statistics(model)
    return list(model.belt_head.parameters())


def non_belt_state(model):
    return {name: value for name, value in model.state_dict().items() if not name.startswith('belt_head.')}


def select_detail(rows, baseline):
    gates = {}
    for name, result in rows.items():
        gates[name] = {
            'IoU_gain_at_least_half_percentage_point': result['pixel_iou'] >= baseline['pixel_iou']+.005,
            'photo_FNR_not_worse': result['photo_rates']['fnr'] <= baseline['photo_rates']['fnr'],
            'pixel_FNR_not_worse': result['pixel_fnr'] <= baseline['pixel_fnr'],
            'zero_GT_overlap_photos_not_worse': result['positive_photos_without_any_gt_overlap'] <= baseline['positive_photos_without_any_gt_overlap'],
        }
    accepted = [name for name in rows if all(gates[name].values())]
    selected = min(accepted, key=lambda name: (-rows[name]['pixel_iou'], rows[name]['pixel_fnr'], name != 'full_control')) if accepted else 'prior_v1'
    return {'selected': selected, 'guards': gates, 'normal_belt_photo_FPR_available': False,
            'independent_factory_verified': False, 'app_default_promoted': False}
