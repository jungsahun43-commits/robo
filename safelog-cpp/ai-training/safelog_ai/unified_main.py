"""Extend the existing facility main weights with two heads and ONE shared backbone."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .auxiliary_classifier import AuxiliaryClassifier
from .belt_segmentation import create_model as belt_model, loss as belt_loss
from .spatial_classifier import spatial_loss

ARCH = 'safelog-unified-facility-belt-foreign-v1'
TASKS = ('facility_surface', 'belt_damage', 'conveyor_foreign')
FACILITY_CLASSES = ('concrete_crack', 'concrete_spalling', 'rust_stain', 'exposed_rebar',
                    'wet_surface', 'efflorescence', 'surface_cavity')
SIZE = 640


class UnifiedMain(AuxiliaryClassifier):
    def __init__(self):
        super().__init__(7, False)
        self.belt_head = deepcopy(belt_model(False).classifier)
        self.foreign_head = nn.Sequential(nn.Dropout(.2), nn.Linear(960, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward_task(self, image, task):
        if task not in TASKS:
            raise ValueError('Explicit supported inspection task required')
        features = self.backbone(image)
        pooled = features['high'].mean((-2, -1))
        if task == 'belt_damage':
            return F.interpolate(self.belt_head(features), size=image.shape[-2:], mode='bilinear', align_corners=False)
        if task == 'conveyor_foreign':
            return self.foreign_head(pooled).flatten()
        maps = self.segmentation_head(features)
        global_logits = self.photo_head(pooled); spatial = maps.flatten(2).topk(32, dim=-1).values.mean(-1)
        mix = self.mix.sigmoid()[None, :]
        return global_logits*(1-mix)+spatial*mix, maps, self.auxiliary_head(pooled)

    def forward(self, image):
        return self.forward_task(image, 'facility_surface')[0]


def initialize_from_main(main_state, belt_state):
    model = UnifiedMain()
    missing, unexpected = model.load_state_dict(main_state, strict=False)
    if unexpected or any(not k.startswith(('belt_head.', 'foreign_head.')) for k in missing):
        raise ValueError('Existing main weights must match completely; only new heads may be missing')
    belt = {k.removeprefix('classifier.'): v for k, v in belt_state.items() if k.startswith('classifier.')}
    model.belt_head.load_state_dict(belt, strict=True)
    return model


def freeze_bn_statistics(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def masked_photo(logits, labels, known, weights):
    if logits.shape != labels.shape or labels.shape != known.shape:
        raise ValueError('Matching photo targets and explicit known mask required')
    # Unknowns are computational placeholders only, with exactly zero gradient.
    target = torch.where(known > 0, labels, 0.)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    probability = logits.sigmoid(); correct = torch.where(target > 0, probability, 1-probability)
    balanced = torch.where(target > 0, weights[None, :], 1.)
    return ((bce*(1-correct)*balanced*known).sum(1)/known.sum(1).clamp_min(1)).mean()


def preserve_known(logits, teacher_logits, known):
    x, t = logits.float()/2, teacher_logits.detach().float()/2
    p = t.sigmoid()
    kl = p*(F.logsigmoid(t)-F.logsigmoid(x))+(1-p)*(F.logsigmoid(-t)-F.logsigmoid(-x))
    return 4*(kl.clamp_min(0)*known).sum()/known.sum().clamp_min(1)


def loss_for_task(task, output, truth, weights=None, teacher_logits=None):
    if task == 'belt_damage':
        return belt_loss(output, truth['mask'], truth['valid'], 20)
    if task == 'conveyor_foreign':
        return F.binary_cross_entropy_with_logits(output, truth['label'])
    if task != 'facility_surface':
        raise ValueError('Unknown task cannot train a head')
    photo, maps, auxiliary = output
    return (masked_photo(photo, truth['labels'], truth['known'], weights['photo'])
        + .5*spatial_loss(maps, truth['mask'], truth['pixel_known'], weights['pixel'])
        + .25*masked_photo(auxiliary, truth['aux'], truth['aux_known'], weights['aux'])
        + 4*preserve_known(photo, teacher_logits, truth['known']))
