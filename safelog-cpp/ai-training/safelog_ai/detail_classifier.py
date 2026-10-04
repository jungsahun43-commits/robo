"""Stride-4 feature residual; existing photo, 80-cell-map and tag meanings stay.

The 160x160 feature grid at a 640 input is pooled to the original 80x80 map
before its existing supervision and photo pooling. It is not new fine-resolution
ground truth. Only the final new projection starts at zero.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F

from .auxiliary_classifier import AuxiliaryClassifier

ARCH = 'lraspp_mobilenet_facility_detail_s4_v1'
DETAIL_RECIPE = {
    'recipe': 'zero_initialized_stride4_feature_residual_v1',
    'backbone_feature_block': '3',
    'feature_channels': 24,
    'feature_stride': 4,
    'hidden_channels': 48,
    'group_norm_groups': 8,
    'depthwise_kernel': 3,
    'residual_resize': 'adaptive_average_pool_to_original_map_shape',
    'new_projection_initialization': 'zero_weight_and_bias',
    'new_parameter_count_for_seven_classes': 2119,
    'supervision': 'Existing original 80x80 known source cells; no 160x160 target invented',
    'public_photo_pooling': 'Original top32 spatial-map logits, photo head and learned mixture unchanged',
}


class DetailClassifier(AuxiliaryClassifier):
    def __init__(self, classes=7, pretrained=False):
        super().__init__(classes, pretrained)
        if ('3' not in self.backbone or self.backbone['3'].out_channels != 24
                or self.backbone.return_layers != {'4': 'low', '16': 'high'}):
            raise ValueError('Installed MobileNet backbone differs from the declared stride-4 recipe')
        self.backbone.return_layers = {**self.backbone.return_layers, '3': 'detail'}
        # Preserve the shared constructor's CPU RNG stream. The additional
        # module must not change subsequent paired DataLoader worker seeds.
        with torch.random.fork_rng(devices=[]):
            self.detail_head = nn.Sequential(
                nn.Conv2d(24, 48, 1, bias=False),
                nn.GroupNorm(8, 48),
                nn.SiLU(),
                nn.Conv2d(48, 48, 3, padding=1, groups=48, bias=False),
                nn.GroupNorm(8, 48),
                nn.SiLU(),
                nn.Conv2d(48, classes, 1),
            )
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)

    def forward_training(self, image):
        features = self.backbone(image)
        original_maps = self.segmentation_head(features)
        fine_residual = self.detail_head(features['detail'])
        maps = original_maps + F.adaptive_avg_pool2d(fine_residual, original_maps.shape[-2:])
        pooled = features['high'].mean((-2, -1))
        global_logits = self.photo_head(pooled)
        spatial_logits = maps.flatten(2).topk(32, dim=-1).values.mean(-1)
        mixture = self.mix.sigmoid()[None, :]
        return (global_logits * (1 - mixture) + spatial_logits * mixture,
                maps, self.auxiliary_head(pooled))


def load_auxiliary_initializer(model, state_dict):
    """Load exactly the legacy auxiliary states, allowing only the new branch.

    This accepts an original AuxiliaryClassifier or a fresh DetailClassifier.
    Existing tensors must have identical names, shapes and dtypes. No legacy
    tensor may be omitted and no unknown/new checkpoint tensor is accepted.
    Only aggregate evidence is returned; this function reads no checkpoint file.
    """
    if type(model) not in (AuxiliaryClassifier, DetailClassifier) or not isinstance(state_dict, Mapping):
        raise ValueError('Initializer requires a declared auxiliary/detail model and state mapping')
    expected = model.state_dict()
    new_keys = {key for key in expected if key.startswith('detail_head.')}
    shared_keys = set(expected) - new_keys
    if set(state_dict) != shared_keys:
        raise ValueError('Initializer must contain every legacy tensor and no undeclared tensor')
    for key in shared_keys:
        value = state_dict[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Legacy initializer tensor shape/dtype/value differs: {key}')
    zero_projection = None
    if new_keys:
        projection = model.detail_head[-1]
        zero_projection = bool(torch.count_nonzero(projection.weight) == 0
                               and torch.count_nonzero(projection.bias) == 0)
        if not zero_projection:
            raise ValueError('Fresh detail residual must have an exactly zero output projection')
    mismatch = model.load_state_dict(state_dict, strict=not new_keys)
    if set(mismatch.missing_keys) != new_keys or mismatch.unexpected_keys:
        raise ValueError('Only the declared detail-head tensors may be absent during transfer')
    loaded = model.state_dict()
    if not all(torch.equal(loaded[key].detach().cpu(), state_dict[key].detach().cpu()) for key in shared_keys):
        raise ValueError('Transferred legacy tensors do not equal the initializer')
    return {'shared_state_tensors_equal': True,
            'shared_state_tensor_count': len(shared_keys),
            'new_state_tensor_count': len(new_keys),
            'new_output_projection_zero': zero_projection}
