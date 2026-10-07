"""Add zero-initialized facility residuals from a fixed ImageNet feature encoder."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import convnext_tiny

from safelog_ai.auxiliary_classifier import AuxiliaryClassifier, AUX_CLASSES, ARCH as ORIGINAL_ARCH
from safelog_ai.retention_distillation import state_sha256

ARCH = "lraspp_mobilenet_facility_semantic_residual_v1"
OFFICIAL_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
WEIGHT_ENUM = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
WEIGHT_SHA_PREFIX = "983f1562"
HEAD_PREFIXES = ("semantic_map_head.", "semantic_photo_head.", "semantic_auxiliary_head.")
NEW_PREFIXES = ("semantic_encoder.",) + HEAD_PREFIXES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class SemanticFeatureEncoder(nn.Module):
    """Offline construction; explicit local loading supplies official weights."""
    def __init__(self):
        super().__init__()
        network = convnext_tiny(weights=None)
        self.features = network.features
        self.pool_norm = network.classifier[0]
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        require(type(mode) is bool, "Encoder mode must be a boolean")
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image):
        require(image.ndim == 4 and image.shape[1] == 3 and image.shape[-2] % 32 == 0
                and image.shape[-1] % 32 == 0, "Semantic encoder needs a three-channel stride32-compatible image")
        value = image; low = None
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index == 3:
                low = value
        pooled = self.pool_norm(value.mean((-2, -1), keepdim=True)).flatten(1)
        require(low is not None and low.shape[1] == 192 and pooled.shape[1] == 768,
                "Official ConvNeXt Tiny feature layout changed")
        return low, pooled


class SemanticResidualClassifier(AuxiliaryClassifier):
    def __init__(self, classes=7, pretrained=False):
        require(type(classes) is int and classes == 7 and pretrained is False,
                "Research semantic model requires seven classes and offline construction")
        super().__init__(classes, pretrained=False)
        self.semantic_encoder = SemanticFeatureEncoder()
        self.semantic_map_head = nn.Conv2d(192, classes, 1)
        self.semantic_photo_head = nn.Linear(768, classes)
        self.semantic_auxiliary_head = nn.Linear(768, len(AUX_CLASSES))
        for module in (self.semantic_map_head, self.semantic_photo_head, self.semantic_auxiliary_head):
            nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "semantic_encoder"):
            self.semantic_encoder.eval()
        return self

    def forward_training(self, image):
        features = self.backbone(image)
        maps = self.segmentation_head(features)
        low, semantic_pooled = self.semantic_encoder(image)
        residual = self.semantic_map_head(low)
        require(residual.shape == maps.shape, "Semantic map must preserve the original low-grid geometry")
        maps = maps + residual
        pooled = features["high"].mean((-2, -1))
        global_logits = self.photo_head(pooled) + self.semantic_photo_head(semantic_pooled)
        spatial_logits = maps.flatten(2).topk(32, dim=-1).values.mean(-1)
        mixture = self.mix.sigmoid()[None, :]
        aux = self.auxiliary_head(pooled) + self.semantic_auxiliary_head(semantic_pooled)
        return global_logits * (1 - mixture) + spatial_logits * mixture, maps, aux


def original_state(model):
    return {name: value for name, value in model.state_dict().items() if not name.startswith(NEW_PREFIXES)}


def zero_semantic_heads(model):
    return all(bool(torch.count_nonzero(value) == 0) for name, value in model.state_dict().items()
               if name.startswith(HEAD_PREFIXES))


def load_original_state(model, checkpoint, classes, split_sha):
    require(isinstance(model, SemanticResidualClassifier) and isinstance(checkpoint, Mapping), "Expected semantic model and original checkpoint")
    require(checkpoint.get("architecture") == ORIGINAL_ARCH and checkpoint.get("classes") == list(classes)
            and checkpoint.get("auxiliary_classes") == list(AUX_CLASSES) and checkpoint.get("imgsz") == 640
            and checkpoint.get("split_sha256") == split_sha, "Original facility initializer metadata differs")
    source = checkpoint.get("state_dict"); expected = original_state(model)
    require(isinstance(source, Mapping) and set(source) == set(expected) and len(expected) == 324,
            "Every original 324 state tensor must be present with its original name")
    require(zero_semantic_heads(model), "All six new residual head tensors must start exactly zero")
    for name, value in source.items():
        require(isinstance(value, torch.Tensor) and value.shape == expected[name].shape
                and value.dtype == expected[name].dtype and torch.isfinite(value).all(), f"Original state shape/dtype/value differs: {name}")
    combined = model.state_dict(); combined.update(source)
    model.load_state_dict(combined, strict=True)
    require(all(torch.equal(value.detach().cpu(), source[name].detach().cpu()) for name, value in original_state(model).items()),
            "Original transferred tensors differ from the fixed initializer")
    return {"shared_state_tensors_equal": True, "shared_state_tensor_count": 324, "original_state_tensor_count": 324,
        "strict_state_load": True, "new_state_tensor_count": len(model.state_dict()) - 324,
        "new_semantic_head_state_tensor_count": 6, "new_semantic_heads_zero": zero_semantic_heads(model),
        "additional_trainable_parameters": 21345, "full_original_model_state_preserved": True}


def semantic_pretrained_subset(model, source):
    """Map only official features and classifier[0] pooling LayerNorm."""
    require(isinstance(model, SemanticResidualClassifier) and isinstance(source, Mapping), "Expected official named ConvNeXt tensors")
    expected = model.semantic_encoder.state_dict(); mapped = {}
    for name, value in expected.items():
        official = "classifier.0." + name[len("pool_norm."):] if name.startswith("pool_norm.") else name
        tensor = source.get(official)
        require(isinstance(tensor, torch.Tensor) and tensor.shape == value.shape and tensor.dtype == value.dtype
                and torch.isfinite(tensor).all(), f"Official semantic state shape/dtype/value differs: {official}")
        mapped[name] = tensor
    expected_official = {"classifier.0." + name[len("pool_norm."):] if name.startswith("pool_norm.") else name for name in expected}
    discarded = {"classifier.2.weight": (1000, 768), "classifier.2.bias": (1000,)}
    require(set(source) == expected_official | set(discarded), "Official Tiny checkpoint inventory differs")
    for name, shape in discarded.items():
        value = source[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == torch.float32
                and torch.isfinite(value).all(), "Discarded ImageNet classifier inventory differs")
    return mapped


def load_semantic_pretrained(model, path):
    path = Path(path)
    digest = file_sha256(path)
    require(digest.startswith(WEIGHT_SHA_PREFIX), "Official ConvNeXt Tiny filename hash prefix does not match")
    source = torch.load(path, map_location="cpu", weights_only=True)
    mapped = semantic_pretrained_subset(model, source)
    model.semantic_encoder.load_state_dict(mapped, strict=True)
    model.semantic_encoder.requires_grad_(False).eval()
    require(all(torch.equal(value.detach().cpu(), mapped[name].detach().cpu()) for name, value in model.semantic_encoder.state_dict().items()),
            "Loaded official semantic tensors differ")
    return {"weights_sha256": digest, "encoder_state_sha256": semantic_encoder_state_sha256(model),
        "encoder_state_tensor_count": len(mapped), "encoder_parameter_count": sum(p.numel() for p in model.semantic_encoder.parameters()),
        "strict_encoder_load": True, "official_pooling_norm_transferred": True,
        "discarded_image_net_classifier_tensors": ["classifier.2.weight", "classifier.2.bias"],
        "all_encoder_parameters_frozen": all(not p.requires_grad for p in model.semantic_encoder.parameters()),
        "encoder_eval": not model.semantic_encoder.training, "new_semantic_heads_zero": zero_semantic_heads(model)}


def semantic_encoder_state_sha256(model):
    require(isinstance(model, SemanticResidualClassifier), "Expected the semantic residual model")
    return state_sha256(model.semantic_encoder.state_dict())


def model_inventory(model):
    require(isinstance(model, SemanticResidualClassifier), "Expected the semantic residual model")
    heads = [p for name, p in model.named_parameters() if name.startswith(HEAD_PREFIXES)]
    return {"parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_encoder_parameter_count": sum(p.numel() for p in model.semantic_encoder.parameters()),
        "new_semantic_head_parameter_count": sum(p.numel() for p in heads),
        "state_tensor_count": len(model.state_dict()), "original_state_tensor_count": len(original_state(model)),
        "encoder_state_tensor_count": len(model.semantic_encoder.state_dict()), "new_head_state_tensor_count": 6,
        "all_encoder_parameters_frozen": all(not p.requires_grad for p in model.semantic_encoder.parameters()),
        "encoder_eval": not model.semantic_encoder.training,
        "all_encoder_modules_eval": all(not module.training for module in model.semantic_encoder.modules()),
        "new_semantic_heads_trainable": all(p.requires_grad for p in heads), "new_semantic_heads_zero": zero_semantic_heads(model),
        "low_channels": 192, "pooled_channels": 768, "low_feature_grid_for_640": [80, 80]}
