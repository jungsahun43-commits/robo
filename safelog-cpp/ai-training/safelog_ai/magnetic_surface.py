"""Shared real-mask surface branch. Original accepted branches stay frozen."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .loco_context import ContextMain
from .ksdd_supervised import SurfaceMaskHead,mask_loss,ksdd_photo_score

VERSION='safelog-real-mask-transfer-v16'
CATEGORY='magnetic_tile_surface'
TARGETS=(CATEGORY,'mpdd_bracket_black','mpdd_tubes')
SIZE=512


def surface_view(rgb,truth=None):
    if min(rgb.size)<64:raise ValueError('Input too small; manual inspection required')
    scale=min(SIZE/rgb.width,SIZE/rgb.height)
    w,h=max(1,round(rgb.width*scale)),max(1,round(rgb.height*scale))
    x,y=(SIZE-w)//2,(SIZE-h)//2
    view=Image.new('RGB',(SIZE,SIZE),(127,127,127));view.paste(rgb.resize((w,h),Image.Resampling.BICUBIC),(x,y))
    valid=torch.zeros(SIZE,SIZE,dtype=torch.bool);valid[y:y+h,x:x+w]=True
    mask=torch.zeros(SIZE,SIZE,dtype=torch.uint8)
    if truth is not None:
        if truth.size!=rgb.size:raise ValueError('Matching source mask size required')
        binary=Image.fromarray((np.asarray(truth.convert('L'))>=128).astype('uint8')*255)
        mask[y:y+h,x:x+w]=torch.from_numpy(np.array(binary.resize((w,h),Image.Resampling.NEAREST))>0)
    return view,mask,valid,{'source_size':list(rgb.size),'view_size':[SIZE,SIZE],'resized_size':[w,h],'offset':[x,y]}


def source_probability(probability,geometry):
    x,y=geometry['offset'];w,h=geometry['resized_size'];sw,sh=geometry['source_size']
    return F.interpolate(probability[...,y:y+h,x:x+w].reshape(-1,1,h,w),size=(sh,sw),mode='bilinear',align_corners=False).reshape(*probability.shape[:-2],sh,sw)


class RealMaskMain(ContextMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.transfer_surface_head=SurfaceMaskHead()

    def forward_surface_features(self,low,high):
        return F.interpolate(self.transfer_surface_head(low.half().float(),high.half().float()),size=(SIZE,SIZE),mode='bilinear',align_corners=False)[:,0]

    def forward_surface(self,native):
        features=self.backbone(native)
        return self.forward_surface_features(features['low'],features['high'])


def photo_score(probability,valid):
    return ksdd_photo_score(probability,valid)
