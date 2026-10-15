"""Actual-label cross-group contrastive loss prototype; group ids are not inputs."""
import torch
from torch.nn import functional as F


def cross_group_loss(embedding,labels,groups,temperature=.2):
    if embedding.ndim!=2 or len(embedding)!=len(labels) or len(groups)!=len(labels) or temperature<=0:raise ValueError('Aligned embedding/labels/groups and positive temperature required')
    if not torch.isfinite(embedding).all() or set(labels.detach().cpu().tolist())!={0,1}:raise ValueError('Finite embeddings and both actual classes required')
    y=labels.to(embedding.device);different=torch.tensor([[a!=b for b in groups] for a in groups],dtype=torch.bool,device=embedding.device)
    positive=different & (y[:,None]==y[None,:])
    if not bool(positive.any(1).all()):raise ValueError('Each anchor needs a same-label positive in another group')
    z=F.normalize(embedding.float(),dim=1,eps=1e-12);logits=(z@z.T)/temperature
    numerator=torch.logsumexp(logits.masked_fill(~positive,-torch.inf),dim=1)
    denominator=torch.logsumexp(logits.masked_fill(~different,-torch.inf),dim=1)
    return (denominator-numerator).mean()
