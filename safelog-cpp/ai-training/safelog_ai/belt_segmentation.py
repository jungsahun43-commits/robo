"""A fresh one-class belt segmentation model; no building checkpoint transfer."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

ARCH = 'safelog-belt-lraspp-polygon-v1'


def create_model(pretrained=False):
    net = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
                                  weights_backbone=None)
    head = net.classifier
    head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, 1, 1)
    head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, 1, 1)
    return net


def loss(logits, mask, valid, positive_weight):
    if logits.shape != mask.shape or mask.shape != valid.shape:
        raise ValueError('Belt mask shapes differ')
    weights = torch.as_tensor(positive_weight, device=logits.device, dtype=torch.float32)
    pointwise = F.binary_cross_entropy_with_logits(logits.float(), mask.float(), pos_weight=weights, reduction='none')
    bce = (pointwise * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.float().sigmoid() * valid
    truth = mask.float() * valid
    intersection = (probability * truth).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + truth.flatten(1).sum(1)
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice


def pixel_counts(probability, mask, valid, threshold=0.5):
    predicted, truth, known = probability > threshold, mask > 0, valid > 0
    return {name: int(value.sum()) for name, value in {
        'tp': predicted & truth & known, 'fn': ~predicted & truth & known,
        'fp': predicted & ~truth & known, 'tn': ~predicted & ~truth & known}.items()}
