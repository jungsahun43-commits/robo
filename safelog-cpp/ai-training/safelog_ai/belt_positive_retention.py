"""Retain TRAIN teacher scores while publisher-positive views train the belt head."""
import torch
from torch.nn import functional as F


def pixel_retention(student, teacher, valid):
    if student.shape != teacher.shape or valid.shape != student.shape:
        raise ValueError('Matching belt logits and spatial validity required')
    if not ((valid==0)|(valid==1)).all():
        raise ValueError('Binary spatial validity required')
    s,t=student.float(),teacher.detach().float(); p=t.sigmoid()
    value=(p*(F.logsigmoid(t)-F.logsigmoid(s))+(1-p)*(F.logsigmoid(-t)-F.logsigmoid(-s))).clamp_min(0)
    return (value*valid).sum()/valid.sum().clamp_min(1)
