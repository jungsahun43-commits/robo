"""Use existing student BatchNorm running statistics while learning all weights.

This changes training forward normalization as well as stopping running-buffer
updates. It does not freeze affine weights, add tensors, or change eps/momentum.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn

from safelog_ai.retention_distillation import state_sha256

BUFFER_NAMES = ("running_mean", "running_var", "num_batches_tracked")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def batchnorm_modules(model):
    """Return the validated named tracked, affine BatchNorm modules."""
    _require(isinstance(model, nn.Module), "Expected a student module")
    result = tuple((name, module) for name, module in model.named_modules()
                   if isinstance(module, nn.modules.batchnorm._BatchNorm))
    _require(result, "The declared student must contain BatchNorm modules")
    for name, module in result:
        _require(module.track_running_stats is True and module.affine is True,
                 "Frozen BatchNorm requires tracked running statistics and affine weights")
        _require(type(module.num_features) is int and module.num_features > 0,
                 "Invalid BatchNorm channel count")
        for key in ("running_mean", "running_var", "weight", "bias"):
            value = getattr(module, key)
            _require(isinstance(value, torch.Tensor) and value.shape == (module.num_features,)
                     and value.is_floating_point() and torch.isfinite(value).all(),
                     f"Invalid finite BatchNorm vector: {name}.{key}")
        _require((module.running_var >= 0).all(), "BatchNorm running variance must be nonnegative")
        counter = module.num_batches_tracked
        _require(isinstance(counter, torch.Tensor) and counter.shape == torch.Size([])
                 and counter.dtype == torch.int64 and counter.item() >= 0,
                 "BatchNorm must preserve its scalar int64 batch counter")
        _require(not isinstance(module.eps, bool) and isinstance(module.eps, Real)
                 and math.isfinite(float(module.eps)) and module.eps > 0,
                 "BatchNorm eps must be a finite positive real")
        _require(module.momentum is None or (not isinstance(module.momentum, bool)
                 and isinstance(module.momentum, Real) and math.isfinite(float(module.momentum))
                 and 0 <= module.momentum <= 1), "Invalid existing BatchNorm momentum")
    return result


def batchnorm_configuration(model):
    """Immutable forward hyperparameters and channel layout, excluding modes."""
    return {name: {"channels": module.num_features, "eps": float(module.eps),
                   "momentum": None if module.momentum is None else float(module.momentum)}
            for name, module in batchnorm_modules(model)}


def batchnorm_inventory(model):
    modules = batchnorm_modules(model)
    layers = len(modules); channels = sum(module.num_features for _, module in modules)
    affine = sum(2 for _, module in modules)
    buffers = sum(len(BUFFER_NAMES) for _, module in modules)
    return {"module_count": layers, "layer_count": layers, "channels": channels, "channel_count": channels,
        "affine_parameter_tensor_count": affine, "running_buffer_tensor_count": buffers,
        "buffer_tensor_count": buffers, "all_batchnorm_eval": all(not module.training for _, module in modules),
        "affine_parameters_trainable": all(module.weight.requires_grad and module.bias.requires_grad for _, module in modules),
        "configuration": batchnorm_configuration(model),
        "modules": [{"name": name, "channels": module.num_features, "eps": float(module.eps),
            "momentum": None if module.momentum is None else float(module.momentum),
            "training": module.training, "affine_trainable": module.weight.requires_grad and module.bias.requires_grad}
            for name, module in modules]}


def batchnorm_state_sha256(model):
    """Bind mean/variance/counter bytes and tensor identity for every BN layer."""
    state = {}
    for name, module in batchnorm_modules(model):
        prefix = name + "." if name else ""
        state.update({prefix + key: getattr(module, key) for key in BUFFER_NAMES})
    return state_sha256(state)


def apply_frozen_batchnorm(model):
    """Apply after every model.train(); preserve affine and all other weights.

    Trainable affine is a precondition; this function never enables gradients
    on an accidentally supplied frozen teacher or freezes student parameters.
    """
    modules = batchnorm_modules(model)
    _require(all(module.weight.requires_grad and module.bias.requires_grad for _, module in modules),
             "Student BatchNorm affine weights must remain trainable")
    before = batchnorm_configuration(model)
    for _, module in modules:
        module.eval()
    _require(batchnorm_configuration(model) == before, "BatchNorm eps/momentum/channel layout changed")
    return batchnorm_inventory(model)
