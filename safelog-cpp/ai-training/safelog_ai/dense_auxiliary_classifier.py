"""Training-only dense DACL supervision alongside the unchanged public heads.

The added branch reads the same MobileNet feature dictionary as the original
heads. It does not replace a public head or run the backbone a second time.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import torch
from torch import nn
from torch.nn import functional as F

from safelog_ai.auxiliary_classifier import (
    ARCH as ORIGINAL_ARCH,
    AUX_CLASSES,
    AuxiliaryClassifier,
)

ARCH = "lraspp_mobilenet_facility_dense_auxiliary_v1"
NEW_PREFIXES = ("dense_auxiliary_head.",)
DENSE_INITIALIZATION_SEED = 56
HIGH_ADAPTER_CHANNELS = 64
ORIGINAL_STATE_TENSORS = 324


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DenseAuxiliaryHead(nn.Module):
    """Zero output projection on a normally initialized 64-channel adapter.

At initialization, the dense output is exactly zero. The output projection
receives gradients on the first update; gradients from this branch reach its
adapter and the shared backbone once that projection has learned nonzero
weights. No BatchNorm buffers or stochastic layers are added.
    """

    def __init__(self):
        super().__init__()
        self.high_adapter = nn.Sequential(
            nn.Conv2d(960, HIGH_ADAPTER_CHANNELS, kernel_size=1),
            nn.ReLU(inplace=False),
        )
        self.projection = nn.Conv2d(
            40 + HIGH_ADAPTER_CHANNELS, len(AUX_CLASSES), kernel_size=1,
        )
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        require(isinstance(features, Mapping) and "low" in features and "high" in features,
                "Dense auxiliary head needs the original low/high features")
        low, high = features["low"], features["high"]
        require(isinstance(low, torch.Tensor) and isinstance(high, torch.Tensor)
                and low.ndim == high.ndim == 4 and low.shape[0] == high.shape[0]
                and low.shape[1] == 40 and high.shape[1] == 960,
                "Original MobileNet low/high channel geometry differs")
        adapted = self.high_adapter(high)
        adapted = F.interpolate(adapted, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.projection(torch.cat((low, adapted), dim=1))


class DenseAuxiliaryClassifier(AuxiliaryClassifier):
    """Keep public seven classes and photo auxiliary nineteen; add dense19."""

    def __init__(self, classes=7, pretrained=False, *, dense_seed=DENSE_INITIALIZATION_SEED):
        require(type(classes) is int and classes == 7 and pretrained is False,
                "Dense auxiliary research model needs seven classes and offline construction")
        require(type(dense_seed) is int and 0 <= dense_seed < 2 ** 63,
                "Dense auxiliary initialization seed must be a nonnegative integer")
        super().__init__(classes, pretrained=False)
        # The adapter seed is explicit and does not advance the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(dense_seed)
            self.dense_auxiliary_head = DenseAuxiliaryHead()
        self.dense_initialization_seed = dense_seed

    def forward_dense_training(self, image):
        """One backbone call; return photo7, maps7, photo_aux19, dense_aux19.

The arithmetic of the original three outputs is kept in its original order.
Inherited forward_training / forward_details / forward do not invoke the new
branch and retain their original return contract.
        """
        features = self.backbone(image)
        maps = self.segmentation_head(features)
        pooled = features["high"].mean((-2, -1))
        global_logits = self.photo_head(pooled)
        spatial_logits = maps.flatten(2).topk(32, dim=-1).values.mean(-1)
        mixture = self.mix.sigmoid()[None, :]
        photo = global_logits * (1 - mixture) + spatial_logits * mixture
        auxiliary = self.auxiliary_head(pooled)
        dense = self.dense_auxiliary_head(features)
        require(dense.shape[0] == maps.shape[0] and dense.shape[1] == len(AUX_CLASSES)
                and dense.shape[-2:] == maps.shape[-2:],
                "Dense auxiliary maps must retain the original low grid")
        return photo, maps, auxiliary, dense


def original_state(model):
    require(isinstance(model, DenseAuxiliaryClassifier), "Expected dense auxiliary classifier")
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith(NEW_PREFIXES)}


def zero_dense_projection(model):
    require(isinstance(model, DenseAuxiliaryClassifier), "Expected dense auxiliary classifier")
    return all(bool(torch.count_nonzero(value) == 0)
               for value in model.dense_auxiliary_head.projection.state_dict().values())


def load_original_state(model, checkpoint, classes, split_sha):
    """Validate and transfer every original tensor without altering new state."""
    require(isinstance(model, DenseAuxiliaryClassifier) and isinstance(checkpoint, Mapping),
            "Expected dense auxiliary model and original checkpoint")
    require(isinstance(classes, (list, tuple)) and len(classes) == 7
            and all(isinstance(name, str) and name for name in classes)
            and len(set(classes)) == 7 and isinstance(split_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", split_sha) is not None,
            "Original public classes or split SHA256 are invalid")
    require(checkpoint.get("architecture") == ORIGINAL_ARCH
            and checkpoint.get("classes") == list(classes)
            and checkpoint.get("auxiliary_classes") == list(AUX_CLASSES)
            and type(checkpoint.get("imgsz")) is int and checkpoint["imgsz"] == 640
            and checkpoint.get("split_sha256") == split_sha,
            "Original facility initializer metadata differs")
    source, expected = checkpoint.get("state_dict"), original_state(model)
    require(isinstance(source, Mapping) and set(source) == set(expected)
            and len(expected) == ORIGINAL_STATE_TENSORS,
            "Every original 324 state tensor must be present with its original name")
    require(zero_dense_projection(model), "New dense output projection must start exactly zero")
    for name, value in source.items():
        require(isinstance(value, torch.Tensor) and value.shape == expected[name].shape
                and value.dtype == expected[name].dtype and bool(torch.isfinite(value).all()),
                f"Original state shape/dtype/value differs: {name}")
    combined = model.state_dict()
    combined.update(source)
    model.load_state_dict(combined, strict=True)
    require(all(torch.equal(value.detach().cpu(), source[name].detach().cpu())
                for name, value in original_state(model).items()),
            "Original transferred tensors differ from the fixed initializer")
    return {
        "shared_state_tensors_equal": True,
        "shared_state_tensor_count": ORIGINAL_STATE_TENSORS,
        "original_state_tensor_count": ORIGINAL_STATE_TENSORS,
        "strict_state_load": True,
        "new_state_tensor_count": len(model.state_dict()) - ORIGINAL_STATE_TENSORS,
        "new_dense_output_projection_zero": zero_dense_projection(model),
        "additional_trainable_parameters": sum(p.numel() for p in model.dense_auxiliary_head.parameters()),
        "full_original_model_state_preserved": True,
    }


def model_inventory(model):
    require(isinstance(model, DenseAuxiliaryClassifier), "Expected dense auxiliary classifier")
    dense = model.dense_auxiliary_head
    return {
        "architecture": ARCH,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_tensor_count": len(model.state_dict()),
        "original_state_tensor_count": len(original_state(model)),
        "new_dense_head_state_tensor_count": len(dense.state_dict()),
        "new_dense_head_parameter_count": sum(p.numel() for p in dense.parameters()),
        "dense_output_projection_state_tensor_count": len(dense.projection.state_dict()),
        "dense_output_projection_parameter_count": sum(p.numel() for p in dense.projection.parameters()),
        "dense_output_projection_zero": zero_dense_projection(model),
        "dense_head_trainable": all(p.requires_grad for p in dense.parameters()),
        "dense_initialization_seed": model.dense_initialization_seed,
        "new_batchnorm_modules": sum(isinstance(module, nn.modules.batchnorm._BatchNorm)
                                     for module in dense.modules()),
        "low_channels": 40,
        "high_channels": 960,
        "high_adapter_channels": HIGH_ADAPTER_CHANNELS,
        "dense_classes": list(AUX_CLASSES),
        "public_class_count": 7,
        "dense_map_grid_for_640": [80, 80],
        "public_output_contract_changed": False,
        "dense_branch_directly_changes_public_outputs": False,
        "additional_backbone_passes": 0,
    }
