"""Learn a narrow/broad map-logit contrast; no new spatial or photo truth.

Top256 is chosen before learning, not searched on validation. These order
statistics do not preserve adjacency or imply scene/industrial understanding.
Signed gates may favour either narrow peaks or broader evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
import torch

from .auxiliary_classifier import AuxiliaryClassifier

ARCH = 'lraspp_mobilenet_facility_pool_context_v1'
CONTEXT_RECIPE = {
    'recipe': 'zero_initialized_narrow_broad_pool_contrast_v1',
    'narrow_topk': 32,
    'broad_topk': 256,
    'gate_transform': 'tanh_signed_per_class',
    'pool_formula': 'narrow + tanh(context_gate) * (broad - narrow)',
    'initial_gate': 0.,
    'additional_parameter_count_for_seven_classes': 7,
    'maps_and_auxiliary_unchanged': True,
    'backbone_passes': 1,
    'new_photo_labels': 0,
    'spatial_adjacency_or_scene_semantics_asserted': False,
}


class ContextClassifier(AuxiliaryClassifier):
    def __init__(self, classes=7, pretrained=False):
        super().__init__(classes, pretrained)
        self.context_gate = torch.nn.Parameter(torch.zeros(classes))

    def forward_training(self, image):
        features = self.backbone(image)
        maps = self.segmentation_head(features)
        pooled = features['high'].mean((-2, -1))
        global_logits = self.photo_head(pooled)
        ordered = maps.flatten(2)
        narrow = ordered.topk(32, dim=-1).values.mean(-1)
        broad = ordered.topk(min(256, ordered.shape[-1]), dim=-1).values.mean(-1)
        spatial_logits = narrow + self.context_gate.tanh()[None, :] * (broad - narrow)
        mixture = self.mix.sigmoid()[None, :]
        return (global_logits * (1 - mixture) + spatial_logits * mixture,
                maps, self.auxiliary_head(pooled))


def load_auxiliary_initializer(model, state_dict):
    """Transfer every legacy state exactly; allow only a fresh zero gate."""
    if type(model) not in (AuxiliaryClassifier, ContextClassifier) or not isinstance(state_dict, Mapping):
        raise ValueError('Initializer requires a declared auxiliary/context model and state mapping')
    expected = model.state_dict()
    new_keys = {'context_gate'} if type(model) is ContextClassifier else set()
    shared_keys = set(expected) - new_keys
    if set(state_dict) != shared_keys:
        raise ValueError('Initializer must contain every legacy tensor and no undeclared tensor')
    for key in shared_keys:
        value = state_dict[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Legacy initializer tensor shape/dtype/value differs: {key}')
    zero = None
    if new_keys:
        zero = bool(torch.count_nonzero(model.context_gate) == 0)
        if not zero:
            raise ValueError('Fresh context gate must be exactly zero')
    mismatch = model.load_state_dict(state_dict, strict=not new_keys)
    if set(mismatch.missing_keys) != new_keys or mismatch.unexpected_keys:
        raise ValueError('Only the declared context gate may be absent during transfer')
    loaded = model.state_dict()
    if not all(torch.equal(loaded[k].detach().cpu(), state_dict[k].detach().cpu()) for k in shared_keys):
        raise ValueError('Transferred legacy tensors do not equal the initializer')
    return {'shared_state_tensors_equal': True,
            'shared_state_tensor_count': len(shared_keys),
            'new_state_tensor_count': len(new_keys),
            'new_output_projection_zero': zero}
