"""Native fine-mask candidate on the existing one shared industrial encoder."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .ksdd_supervised import SurfaceMaskHead,ksdd_photo_score,restore_map
from .mixed_surface import mixed_logits,freeze_projection
from .real_mask_server import RuntimeRealMaskMain

VERSION='safelog-commutator-native-v18'
CATEGORY='ksdd1_commutator_surface'
WIDTH,HEIGHT=512,1408


def commutator_view(rgb,truth=None):
    if min(rgb.size)<64:raise ValueError('Surface input too small for this research view')
    scale=min(WIDTH/rgb.width,HEIGHT/rgb.height)
    w,h=max(1,round(rgb.width*scale)),max(1,round(rgb.height*scale));x,y=(WIDTH-w)//2,(HEIGHT-h)//2
    image=Image.new('RGB',(WIDTH,HEIGHT),(127,127,127));image.paste(rgb.resize((w,h),Image.Resampling.BICUBIC),(x,y))
    valid=torch.zeros(HEIGHT,WIDTH,dtype=torch.bool);valid[y:y+h,x:x+w]=True
    mask=torch.zeros(HEIGHT,WIDTH,dtype=torch.uint8)
    if truth is not None:
        if truth.size!=rgb.size:raise ValueError('Original image/mask coordinate mismatch')
        mask[y:y+h,x:x+w]=torch.from_numpy(np.array(truth.convert('L').resize((w,h),Image.Resampling.NEAREST))>0)
    return image,mask,valid,{'source_size':list(rgb.size),'view_size':[WIDTH,HEIGHT],'resized_size':[w,h],'offset':[x,y]}


def fine_mask_loss(logits,mask,valid,positive_weight):
    if logits.shape!=mask.shape or mask.shape!=valid.shape:raise ValueError('Aligned mask/valid required')
    known=valid.float();target=mask.float()*known
    bce=F.binary_cross_entropy_with_logits(logits.float(),target,pos_weight=logits.new_tensor(float(positive_weight)),reduction='none')
    bce=(bce*known).sum()/known.sum().clamp_min(1)
    prob=logits.sigmoid();intersection=(prob*target*known).flatten(1).sum(1)
    denominator=((prob+target)*known).flatten(1).sum(1)
    dice=1-((2*intersection+1)/(denominator+1)).mean()
    score=ksdd_photo_score(prob[:,0],valid[:,0].bool())
    labels=(target.flatten(1).sum(1)>0).float()
    return bce+dice+.5*F.binary_cross_entropy(score.clamp(1e-6,1-1e-6),labels)


class CommutatorSurfaceMain(RuntimeRealMaskMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.commutator_surface_head=SurfaceMaskHead()

    def forward_commutator_features(self,low,high,output_size=(HEIGHT,WIDTH)):
        return mixed_logits(self.commutator_surface_head,low,high,output_size)[:,0]

    def forward_commutator_mask(self,native):
        features=self.backbone(native)
        return self.forward_commutator_features(features['low'],features['high'],native.shape[-2:])
