"""Cross-view teacher regularization; teacher probabilities are not new annotations."""
import numpy as np
import torch
from torch.nn import functional as F
from .native_corrosion_grid import annotation_tile
from .ostrava_rust_mask import OstravaRustMain,RustMaskHead
from .fine_feature_probe import forward_with_fine


def retained_weights(truth,coverage,box):
    target=annotation_tile(truth,box);counts=[int((truth==k).sum()) for k in (0,1)]
    present=sum(v>0 for v in counts)
    if not present or coverage.shape!=truth.shape:raise ValueError('Known native class support required')
    x,y,w,h=(box[k] for k in ('left','top','width','height'))
    weight=np.zeros((512,512),np.float32)
    for k,count in enumerate(counts):
        if count:weight[:h,:w]+=(target[:h,:w]==k)/(present*count*coverage[y:y+h,x:x+w].astype(np.float64))
    return target,weight


def teacher_tile(probability,box):
    if probability.ndim!=2 or not np.isfinite(probability).all():raise ValueError('Finite native teacher probability required')
    x,y,w,h=(box[k] for k in ('left','top','width','height'))
    result=np.full((512,512),.5,np.float32);result[:h,:w]=probability[y:y+h,x:x+w]
    return result


def teacher_kl(logits,teacher,weights):
    if logits.shape!=teacher.shape or logits.shape!=weights.shape:raise ValueError('Aligned teacher/student/weights required')
    if not torch.isfinite(teacher).all() or torch.any((teacher<0)|(teacher>1)) or torch.any(weights<0):raise ValueError('Invalid teacher probability/weight')
    p=teacher.detach().float().clamp(1e-6,1-1e-6)
    entropy=-(p*p.log()+(1-p)*torch.log1p(-p))
    divergence=(F.binary_cross_entropy_with_logits(logits.float(),p,reduction='none')-entropy).clamp_min(0)
    return (divergence*weights).sum()


class CorrosionRetainedMain(OstravaRustMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.corrosion_retained_head=RustMaskHead()

    @property
    def corrosion_native_head(self):
        # Compatibility with local v81 grid helpers, no duplicate registered state.
        return self.corrosion_retained_head

    def forward_corrosion_retained_features(self,fine,low,high):return self.corrosion_retained_head(fine,low,high)
    def forward_corrosion_native_features(self,fine,low,high):return self.forward_corrosion_retained_features(fine,low,high)

    def forward_corrosion_retained_tiles(self,image):
        values=forward_with_fine(self.backbone,image)
        return self.forward_corrosion_retained_features(values['fine'],values['low'],values['high'])
