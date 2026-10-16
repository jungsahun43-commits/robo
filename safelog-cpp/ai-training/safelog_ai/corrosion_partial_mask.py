"""A separately named binary branch with explicit partial-label loss budgets."""
import torch
from torch.nn import functional as F
from .ostrava_rust_mask import OstravaRustMain,RustMaskHead
from .fine_feature_probe import forward_with_fine


def loss_components(logits,target):
    if logits.ndim!=3 or logits.shape!=target.shape:raise ValueError('Aligned batch maps required')
    if torch.any((target!=0)&(target!=1)&(target!=255)):raise ValueError('Undocumented binary label')
    pos=target==1;neg=target==0;valid=pos|neg
    bce=F.binary_cross_entropy_with_logits(logits.float(),pos.float(),reduction='none')
    pc=pos.sum((1,2));nc=neg.sum((1,2))
    positive=(bce*pos).sum((1,2))/pc.clamp_min(1)
    negative=(bce*neg).sum((1,2))/nc.clamp_min(1)
    prob=logits.sigmoid()*valid
    dice=1-(2*(prob*pos).sum((1,2))+1)/(prob.sum((1,2))+pc+1)
    return positive,negative,dice,pc>0,nc>0


def partial_loss(logits,target,fullmask):
    fullmask=torch.as_tensor(fullmask,dtype=torch.bool,device=logits.device)
    if fullmask.shape!=(len(logits),) or not fullmask.any() or fullmask.all():raise ValueError('Both sources required')
    if torch.any((target==0)&~fullmask[:,None,None]):raise ValueError('VT unknown cannot become known background')
    p,n,d,hasp,hasn=loss_components(logits,target)
    mean=lambda values,which:(values*which).sum()/which.sum().clamp_min(1)
    # Fixed .5 positive / .5 known-negative BCE budget. Positive half is split
    # equally between sources; only the full-mask source contributes Dice.
    return .25*mean(p,hasp&fullmask)+.25*mean(p,hasp&~fullmask)+.5*mean(n,hasn&fullmask)+mean(d,fullmask)


def native_probability(probability,geometry):
    if tuple(probability.shape)!=(512,512):raise ValueError('Full512 probability required')
    w,h=geometry['native_size'];rw,rh=geometry['resized_size'];left,top=geometry['left'],geometry['top']
    return F.interpolate(probability[None,None,top:top+rh,left:left+rw].float(),size=(h,w),mode='bilinear',align_corners=False)[0,0]


def known_confusion(probability,truth,cut):
    if probability.shape!=truth.shape:raise ValueError('Aligned native masks required')
    if torch.any((truth!=0)&(truth!=1)&(truth!=255)):raise ValueError('Undocumented native label')
    pred=probability>float(cut);p=truth==1;n=truth==0
    return torch.stack(((pred&p).sum(),(~pred&p).sum(),(pred&n).sum(),(~pred&n).sum()))


class CorrosionPartialMain(OstravaRustMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.corrosion_partial_head=RustMaskHead()

    def forward_corrosion_partial_features(self,fine,low,high):return self.corrosion_partial_head(fine,low,high)

    def forward_corrosion_partial(self,image):
        values=forward_with_fine(self.backbone,image)
        return self.forward_corrosion_partial_features(values['fine'],values['low'],values['high'])
