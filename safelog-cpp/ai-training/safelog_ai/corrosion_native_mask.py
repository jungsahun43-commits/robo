"""Native input, sparse labels and overlap-corrected supervision on one encoder."""
import numpy as np
import torch
from torch.nn import functional as F
from .native_corrosion_grid import grid, annotation_tile
from .ostrava_rust_mask import OstravaRustMain, RustMaskHead
from .fine_feature_probe import forward_with_fine


def coverage_map(size):
    w,h=size
    coverage=np.zeros((h,w),dtype=np.uint16)
    for b in grid(size):
        coverage[b['top']:b['top']+b['height'],b['left']:b['left']+b['width']]+=1
    if not np.all(coverage>0):raise ValueError('Uncovered native input pixel')
    return coverage


def rgb_tile(rgb,box):
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Native uint8 RGB required')
    x,y,w,h=(box[k] for k in ('left','top','width','height'))
    if min(x,y)<0 or not 0<w<=512 or not 0<h<=512 or x+w>rgb.shape[1] or y+h>rgb.shape[0]:raise ValueError('Invalid native RGB bounds')
    result=np.empty((512,512,3),dtype=np.uint8);result[:]=(124,116,104)
    result[:h,:w]=rgb[y:y+h,x:x+w]
    return result


def supervised_weights(truth,coverage,box,counts):
    """Half the photo budget per known class, divided by its unique native count."""
    if truth.shape!=coverage.shape or min(counts)<=0:raise ValueError('Both native classes and aligned coverage required')
    target=annotation_tile(truth,box)
    x,y,w,h=(box[k] for k in ('left','top','width','height'))
    inv=np.zeros((512,512),dtype=np.float32)
    inv[:h,:w]=1/coverage[y:y+h,x:x+w].astype(np.float32)
    weight=inv*((target==0)*(.5/counts[0])+(target==1)*(.5/counts[1]))
    return target,weight.astype(np.float32)


def native_loss(logits,target,weights):
    if logits.shape!=target.shape or logits.shape!=weights.shape:raise ValueError('Aligned tile logits, GT and weights required')
    if torch.any((target!=0)&(target!=1)&(target!=255)):raise ValueError('Undocumented sparse label')
    if torch.any(weights<0) or not torch.isfinite(weights).all() or torch.any(weights[target==255]!=0):raise ValueError('Unknown/padding must have exactly zero loss weight')
    return (F.binary_cross_entropy_with_logits(logits.float(),(target==1).float(),reduction='none')*weights).sum()


def add_probability(total,probability,box):
    if probability.shape!=(512,512) or not np.isfinite(probability).all():raise ValueError('Finite tile probability512 required')
    x,y,w,h=(box[k] for k in ('left','top','width','height'))
    total[y:y+h,x:x+w]+=probability[:h,:w]


class CorrosionNativeMain(OstravaRustMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.corrosion_native_head=RustMaskHead()

    def forward_corrosion_native_features(self,fine,low,high):
        return self.corrosion_native_head(fine,low,high)

    def forward_corrosion_native_tiles(self,image):
        values=forward_with_fine(self.backbone,image)
        return self.forward_corrosion_native_features(values['fine'],values['low'],values['high'])
