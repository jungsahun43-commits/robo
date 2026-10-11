"""Whole-photo foreign-object presence; no belt-health or localization verdict."""
import math

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

ARCH = 'resnet18-foreign-presence-v1'
CLASS = 'conveyor_foreign_object'
SIZE = 384
THRESHOLD = 0.5


def create_model(pretrained=False):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def letterbox(image):
    rgb = image.convert('RGB')
    scale = SIZE/max(rgb.size)
    w, h = (max(1, round(v*scale)) for v in rgb.size)
    output = Image.new('RGB', (SIZE, SIZE), (123, 116, 103))
    output.paste(rgb.resize((w, h), Image.Resampling.BILINEAR), ((SIZE-w)//2, (SIZE-h)//2))
    return torch.from_numpy(np.array(output)).permute(2, 0, 1)


def normalize(images, device='cpu'):
    values = images.to(device).float()/255
    mean = values.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = values.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (values-mean)/std


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p, denominator = successes/total, 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0, center-radius), min(1, center+radius)]


def summarize(observations, threshold=THRESHOLD):
    if not 0 < threshold < 1:
        raise ValueError('Threshold must lie between zero and one')
    counts = dict(tp=0, fn=0, fp=0, tn=0)
    for row in observations:
        label, score = row['label'], row['score']
        if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Finite binary labels and sigmoid scores required')
        predicted = score > threshold
        key = ('tp' if predicted else 'fn') if label else ('fp' if predicted else 'tn')
        counts[key] += 1
    positives, negatives = counts['tp']+counts['fn'], counts['fp']+counts['tn']
    fnr, fpr = counts['fn']/positives if positives else None, counts['fp']/negatives if negatives else None
    return {**counts, 'positive_photos': positives, 'negative_photos': negatives,
            'fnr': fnr, 'fpr': fpr, 'fnr_frame_binomial_wilson95': wilson(counts['fn'], positives),
            'fpr_frame_binomial_wilson95': wilson(counts['fp'], negatives),
            'intervals_assume_independent_frames_not_proven_here': True,
            'photo_fpr_fnr_below_5_percent': fpr is not None and fnr is not None and fpr < .05 and fnr < .05}
