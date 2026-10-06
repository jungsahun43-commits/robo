"""Preserve known non-primary photo predictions with a frozen TRAIN teacher.

Teacher probabilities are a regularizer, not asserted labels. The two primary
classes and every publisher-unknown photo label contribute no retention loss.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

TEMPERATURE = 2.0
EXCLUDED_INDICES = (0, 1)
RETAINED_INDICES = (2, 3, 4, 5, 6)
RETAINED_CLASSES = ("rust_stain", "exposed_rebar", "wet_surface", "efflorescence", "surface_cavity")
STATE_HASH_RECIPE = "sha256 sorted state names; uint64le length + canonical name/dtype/shape JSON; uint64le length + contiguous CPU raw tensor bytes"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def masked_bernoulli_kl(student_logits, teacher_logits, known, temperature=TEMPERATURE):
    """Return T² mean KL(teacher || student) over known original other-five.

    Inputs are Bx7 photo logits and a Bx7 known-label mask. The teacher is
    detached regardless of the caller's autograd context. Computation is FP32
    even when model logits arrive from CUDA autocast in FP16/BF16.
    """
    _require(not isinstance(temperature, bool) and isinstance(temperature, Real)
             and math.isfinite(float(temperature)) and float(temperature) > 0,
             "Temperature must be a finite positive real")
    for name, logits in (("student", student_logits), ("teacher", teacher_logits)):
        _require(isinstance(logits, torch.Tensor) and logits.ndim == 2
                 and logits.shape[0] > 0 and logits.shape[1] == 7
                 and logits.is_floating_point() and torch.isfinite(logits).all(),
                 f"Expected finite floating Bx7 {name} photo logits")
    _require(teacher_logits.shape == student_logits.shape
             and teacher_logits.device == student_logits.device,
             "Teacher/student photo shape and device must match")
    _require(isinstance(known, torch.Tensor) and known.shape == student_logits.shape
             and known.device == student_logits.device
             and (known.is_floating_point() or known.dtype == torch.bool)
             and torch.isfinite(known).all() and ((known == 0) | (known == 1)).all(),
             "Known mask must be matching finite binary photo entries")
    temperature = float(temperature)
    student = student_logits[:, RETAINED_INDICES].float() / temperature
    teacher = teacher_logits.detach()[:, RETAINED_INDICES].float() / temperature
    mask = known[:, RETAINED_INDICES].detach().float()
    count = mask.sum()
    if count.item() == 0:
        # Keep a student autograd path so all-unknown batches remain valid.
        return student.sum() * 0.0
    probability = teacher.sigmoid()
    positive = probability * (F.logsigmoid(teacher) - F.logsigmoid(student))
    negative = (1 - probability) * (F.logsigmoid(-teacher) - F.logsigmoid(-student))
    # KL is nonnegative; roundoff near identical probabilities can be a few
    # FP32 ulps negative. This bound does not assert or fill unknown labels.
    pointwise = (positive + negative).clamp_min(0.0)
    result = temperature * temperature * (pointwise * mask).sum() / count
    _require(torch.isfinite(result), "Retention KL must remain finite")
    return result


teacher_bernoulli_kl = masked_bernoulli_kl


def state_sha256(state_or_model):
    """Bind every teacher parameter/buffer, including scalar integer buffers."""
    state = state_or_model.state_dict() if isinstance(state_or_model, nn.Module) else state_or_model
    _require(isinstance(state, Mapping) and state and all(isinstance(name, str) and name for name in state),
             "Expected a nonempty named tensor state")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                 and (not tensor.is_floating_point() or torch.isfinite(tensor).all()),
                 "Teacher state must contain finite dense tensors")
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for value in (header, payload):
            digest.update(len(value).to_bytes(8, "little")); digest.update(value)
    return digest.hexdigest()
