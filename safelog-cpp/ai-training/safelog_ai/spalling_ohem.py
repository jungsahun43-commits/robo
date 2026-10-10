"""Mine only asserted spalling-background pixels; retain every other term.

This is a pixel-level adaptation of hard-example mining, not a reproduction
of the original region-detector OHEM. Selection is based on current negative
focal loss. Each photo retains ceil(0.1 * known background cells); its selected
sum is multiplied by background_count / selected_count, preserving the old
global denominator. This deliberately focuses the negative objective on the
hard tail. It is not an unbiased estimator of the original background loss.
"""
import math
import torch
from torch.nn import functional as F

HARD_FRACTION=.1
SPALLING_INDEX=1
def require(condition,message):
    if not condition:raise ValueError(message)

def spalling_ohem_spatial_loss(logits,target,known,positive_weights,hard_fraction=HARD_FRACTION,return_diagnostics=False):
    require(type(hard_fraction)in(float,int)and not isinstance(hard_fraction,bool)
        and math.isfinite(hard_fraction)and 0<hard_fraction<=1,'Invalid hard fraction')
    require(logits.ndim==4 and logits.shape[1]==7 and target.shape==logits.shape
        and known.shape==logits.shape[:2]and positive_weights.shape==(7,),'Spatial contract differs')
    require(all(torch.isfinite(t).all()for t in(logits,target,known,positive_weights))
        and ((known==0)|(known==1)).all()and (target>=0).all()and (target<=1).all()
        and (positive_weights>0).all(),'Nonfinite or invalid supervision')
    expanded=known[:,:,None,None];weights=positive_weights[None,:,None,None]
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    probability=logits.sigmoid();correct=torch.where(target>0,probability,1-probability)
    balanced=torch.where(target>0,weights,1.)
    cells=bce*(1-correct)*balanced*expanded
    y=target[:,SPALLING_INDEX]
    # Exactly the original focal negative term. Soft/nonzero foreground stays
    # in the unmodified base; unavailable channels never enter selection.
    negative=((y==0)&known[:,SPALLING_INDEX,None,None].bool()).flatten(1)
    values=cells[:,SPALLING_INDEX].flatten(1)
    count=negative.sum(1);selected=(count.to(torch.float64)*hard_fraction).ceil().long()
    capacity=max(1,min(values.shape[1],math.ceil(values.shape[1]*hard_fraction)))
    ranked=values.detach().masked_fill(~negative,-torch.inf).topk(capacity,dim=1)
    valid=torch.arange(capacity,device=logits.device)[None,:]<selected[:,None]
    selected_mask=torch.zeros_like(negative).scatter(1,ranked.indices,valid)
    multiplier=torch.where(negative,selected_mask.to(values.dtype)*(count.to(values.dtype)/selected.clamp_min(1))[:,None],1.)
    mined_spalling=(values*multiplier).reshape_as(y)
    modified=torch.cat((cells[:,:1],mined_spalling[:,None],cells[:,2:]),dim=1)
    denominator=(known.sum()*logits.shape[-1]*logits.shape[-2]).clamp_min(1)
    intersection=(probability*target).sum((-2,-1))
    dice=1-(2*intersection+1)/(probability.sum((-2,-1))+target.sum((-2,-1))+1)
    positives=(target.sum((-2,-1))>0)*known
    unchanged_dice=.5*(dice*positives).sum()/positives.sum().clamp_min(1)
    base=cells.sum()/denominator+unchanged_dice
    result=modified.sum()/denominator+unchanged_dice
    if not return_diagnostics:return result
    return result,{'known_background_cells':count.detach(),'selected_background_cells':selected.detach(),
        'unmodified_spatial_loss':base.detach(),'negative_objective_correction':(result-base).detach(),
        'spalling_foreground_cells_retained':((y>0)&known[:,SPALLING_INDEX,None,None].bool()).sum().detach()}
