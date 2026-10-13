"""Separate first-order TRAIN recall constraints for every positive photo.

No constraint on validation recall is implied. Raw Adam moments are retained.
Only the actual parameter displacement is corrected; an infeasible or rounded
unsafe correction restores the exact preceding parameters.
"""
import math
import torch
from torch.nn import functional as F


def positive_photo_losses(logits, mask, valid):
    if logits.ndim != 4 or logits.shape != mask.shape or mask.shape != valid.shape:
        raise ValueError('Matching BCHW logits, publisher masks and validity required')
    if not (torch.isfinite(logits).all() and
            (((mask == 0) | (mask == 1)).all()) and
            (((valid == 0) | (valid == 1)).all())):
        raise ValueError('Finite logits and binary publisher truth required')
    known = mask * valid
    counts = known.flatten(1).sum(1)
    if (counts <= 0).any():
        raise ValueError('Every protected photo must contain known positive pixels')
    return -(F.logsigmoid(logits) * known).flatten(1).sum(1) / counts


@torch.no_grad()
def project_photo_step(parameters, before, photo_gradients, max_cycles=64):
    """Project an Adam step onto all photo halfspaces using dual coordinates.

    The tiny Gram system is solved in float64. All dot products are checked
    again after writes to the actual parameter dtype. Numerical write margins
    may make opposing constraints infeasible, in which case exact restore is
    safe. This is not a symmetric gradient-surgery or convergence guarantee.
    """
    if not parameters or len(parameters) != len(before) or not photo_gradients:
        raise ValueError('Parameters, snapshots and at least one photo required')
    if max_cycles < 1:
        raise ValueError('Positive cycle budget required')
    for p, b in zip(parameters, before):
        if p.shape != b.shape or p.device != b.device or p.dtype != b.dtype:
            raise ValueError('Matching parameter snapshot required')
        if not p.is_floating_point() or not (torch.isfinite(p).all() and torch.isfinite(b).all()):
            raise ValueError('Finite floating parameters required')
    matrix = []
    for gradients in photo_gradients:
        if len(gradients) != len(parameters):
            raise ValueError('One ordered gradient per parameter required')
        pieces = []
        for p, g in zip(parameters, gradients):
            if g is None:
                pieces.append(torch.zeros_like(p).flatten().double())
            else:
                if g.shape != p.shape or g.device != p.device or not torch.isfinite(g).all():
                    raise ValueError('Finite matching gradient required')
                pieces.append(g.flatten().double())
        matrix.append(torch.cat(pieces))
    gradients = torch.stack(matrix)
    previous = torch.cat([p.flatten().double() for p in before])
    proposed = torch.cat([p.flatten().double() for p in parameters])
    delta = proposed - previous
    initial = gradients @ delta
    norms = gradients.square().sum(1)
    tolerance = 1e-9
    projected = bool((initial > 0).any())
    reverted = False
    cycles = 0
    margins = torch.zeros_like(initial)
    active = norms > 1e-20
    if projected and active.any():
        eps = torch.cat([torch.full_like(p, torch.finfo(p.dtype).eps).flatten().double() for p in parameters])
        margins = (4 * gradients.abs() * torch.maximum(previous.abs(), proposed.abs()) * eps).sum(1).clamp_min(1e-10)
        unit = gradients[active] / norms[active].sqrt().unsqueeze(1)
        gram = (unit @ unit.T).cpu().tolist()
        q = (unit @ delta + margins[active] / norms[active].sqrt()).cpu().tolist()
        original_q = list(q)
        dual = [0.] * len(q)
        # Coordinate minimization of the nonnegative dual of Euclidean
        # halfspace projection. Budget fixed before training, no metric search.
        for cycle in range(max_cycles):
            for i in range(len(q)):
                updated = max(0., dual[i] + q[i] / gram[i][i])
                change = updated - dual[i]
                dual[i] = updated
                for j in range(len(q)):
                    q[j] -= change * gram[j][i]
            cycles = cycle + 1
            if max(q) <= 1e-12:
                break
        if not all(math.isfinite(v) for v in dual + q + original_q):
            raise RuntimeError('Nonfinite displacement projection')
        corrected = previous + delta - unit.T @ unit.new_tensor(dual)
        cursor = 0
        for p in parameters:
            size = p.numel()
            p.copy_(corrected[cursor:cursor+size].reshape(p.shape).to(p.dtype))
            cursor += size
    effective = torch.cat([p.flatten().double() for p in parameters]) - previous
    after = gradients @ effective
    if not torch.isfinite(after).all() or (after > tolerance).any():
        for p, b in zip(parameters, before):
            p.copy_(b)
        reverted = True
        after = gradients @ (torch.cat([p.flatten().double() for p in parameters]) - previous)
    assert torch.isfinite(after).all() and (after <= tolerance).all()
    return {'projected': projected, 'reverted': reverted, 'photo_constraints': len(photo_gradients),
            'dot_before': initial.tolist(), 'dot_after': after.tolist(), 'tolerance': tolerance,
            'gradient_norm2': norms.tolist(), 'write_margins': margins.tolist(), 'cycles': cycles}
