"""Independent equipment appearance research model, not a mechanical diagnosis."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import gaussian_blur

CATEGORIES = ('pcb1', 'pcb2', 'pcb3', 'pcb4')
SCHEMA = 'safelog.equipment.visual-anomaly.v1'
ARCHITECTURE = 'resnet18-pyramid-student-teacher-v1'


class EquipmentPyramid(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3

    def forward(self, image):
        a = self.layer1(self.stem(image))
        b = self.layer2(a)
        c = self.layer3(b)
        return a, b, c


def matching_loss(student, teacher):
    if len(student) != 3 or len(teacher) != 3:
        raise ValueError('Expected three feature scales')
    return sum((F.normalize(s.float(), dim=1) - F.normalize(t.detach().float(), dim=1))
               .square().sum(dim=1).mean() / 2 for s, t in zip(student, teacher))


def appearance_maps(student, teacher, size=256):
    maps = [F.interpolate((F.normalize(s.float(), dim=1) -
                           F.normalize(t.float(), dim=1)).square().sum(dim=1, keepdim=True) / 2,
                          size=(size, size), mode='bilinear', align_corners=False)
            for s, t in zip(student, teacher)]
    # Multi-scale product, followed by fixed smoothing; no test-dependent scaling.
    merged = maps[0] * maps[1] * maps[2]
    return gaussian_blur(merged, [25, 25], [4.0, 4.0]).squeeze(1)


def normal_calibration_threshold(scores, alpha=0.05):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all():
        raise ValueError('At least 20 finite held-out normal scores required')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie between zero and one')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else float('inf')


def classification_rates(labels, scores, threshold):
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite equally sized binary labels and scores required')
    pred = score > threshold
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, tn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'positive_photos': tp + fn, 'negative_photos': fp + tn,
            'fnr': fn / (tp + fn) if tp + fn else None,
            'fpr': fp / (fp + tn) if fp + tn else None}


def rank_auroc(labels, scores):
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if score.shape != y.shape or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid AUROC inputs')
    positive, negative = int(y.sum()), int((y == 0).sum())
    if not positive or not negative:
        return None
    order = np.argsort(score, kind='stable')
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def inspection_result(category, score, threshold, checkpoint_sha256):
    if category not in CATEGORIES:
        raise ValueError('Unsupported equipment category: no general machinery or belt inference')
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError('Nonfinite appearance score')
    return {'schema': SCHEMA, 'equipment_category': category,
            'assessment': 'appearance_anomaly_suspected' if score > threshold else 'no_visual_anomaly_detected',
            'anomaly_score': score, 'threshold': threshold, 'score_is_probability': False,
            'mechanical_failure_state': 'not_assessed', 'human_confirmation_required': True,
            'model_sha256': checkpoint_sha256,
            'scope': 'Aligned close-up of the specified PCB type; external benchmark research candidate'}
