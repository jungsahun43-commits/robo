"""Actual corrosion pixel supervision on one unchanged shared encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from .insplad_suspension import InspladSuspensionMain
from .fine_feature_probe import forward_with_fine

SIDE=512


def source_view(rgb,mask):
    if rgb.size!=mask.size or rgb.width!=rgb.height:
        raise ValueError('Aligned square original image and actual mask required')
    view=rgb.resize((SIDE,SIDE),Image.Resampling.BICUBIC)
    target=torch.from_numpy(np.array(mask.convert('L').resize((SIDE,SIDE),Image.Resampling.NEAREST))>0)
    return view,target


def native_probability(probability,size=(2048,2048)):
    if probability.ndim!=2 or tuple(probability.shape)!=(SIDE,SIDE):
        raise ValueError('Full square512 probability map required')
    return F.interpolate(probability[None,None].float(),size=size,mode='bilinear',align_corners=False)[0,0]


def pixel_loss(logits,target):
    if logits.shape!=target.shape or logits.ndim!=3:
        raise ValueError('Matching batch, height and width required')
    truth=target.float();bce=F.binary_cross_entropy_with_logits(logits.float(),truth,reduction='none')
    positive=truth.flatten(1).sum(1);negative=(1-truth).flatten(1).sum(1)
    pos=(bce*truth).flatten(1).sum(1)/positive.clamp_min(1)
    neg=(bce*(1-truth)).flatten(1).sum(1)/negative.clamp_min(1)
    balanced=(pos*(positive>0)+neg*(negative>0))/((positive>0).float()+(negative>0).float())
    probability=logits.sigmoid();intersection=(probability*truth).flatten(1).sum(1)
    dice=1-(2*intersection+1)/(probability.flatten(1).sum(1)+positive+1)
    return balanced+dice


def confusion_at(probability,truth,thresholds):
    if probability.shape!=truth.shape:raise ValueError('Matching original-coordinate maps required')
    truth=truth.bool();result=[]
    for cut in thresholds:
        pred=probability>float(cut)
        result.append(torch.stack(((pred&truth).sum(),(~pred&truth).sum(),(pred&~truth).sum(),(~pred&~truth).sum())))
    return torch.stack(result)


def pixel_rates(counts):
    tp,fn,fp,tn=map(int,counts)
    return {'tp':tp,'fn':fn,'fp':fp,'tn':tn,'fnr':fn/(tp+fn) if tp+fn else None,
            'fpr':fp/(fp+tn) if fp+tn else None,'IoU':tp/(tp+fn+fp) if tp+fn+fp else None,
            'Dice':2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else None}


class RustMaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        for name,c in (('fine',24),('low',40),('high',960)):
            self.register_buffer(name+'_mean',torch.zeros(1,c,1,1))
            self.register_buffer(name+'_std',torch.ones(1,c,1,1))
        self.fine_projection=nn.Sequential(nn.Conv2d(24,16,1),nn.GELU())
        self.high_projection=nn.Sequential(nn.Conv2d(960,32,1),nn.GELU())
        self.fusion=nn.Sequential(nn.Conv2d(88,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Dropout2d(.1),
            nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)

    @torch.no_grad()
    def fit_normalization(self,features):
        for name in ('fine','low','high'):
            value=features[name].detach().cpu().double()
            mean=value.mean((0,2,3),keepdim=True)
            std=value.std((0,2,3),correction=0,keepdim=True).clamp_min(.001)
            getattr(self,name+'_mean').copy_(mean.to(getattr(self,name+'_mean')))
            getattr(self,name+'_std').copy_(std.to(getattr(self,name+'_std')))

    def forward(self,fine,low,high):
        values={}
        for name,value,c,size in (('fine',fine,24,128),('low',low,40,64),('high',high,960,32)):
            if tuple(value.shape[1:])!=(c,size,size):raise ValueError('Declared frozen RGB512 encoder grid required')
            values[name]=((value.half().float()-getattr(self,name+'_mean'))/getattr(self,name+'_std')).clamp(-8,8)
        up=lambda x:F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
        x=torch.cat((self.fine_projection(values['fine']),up(values['low']),up(self.high_projection(values['high']))),1)
        return F.interpolate(self.fusion(x),size=(SIDE,SIDE),mode='bilinear',align_corners=False)[:,0]


class OstravaRustMain(InspladSuspensionMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.ostrava_rust_head=RustMaskHead()

    def forward_ostrava_features(self,fine,low,high):
        return self.ostrava_rust_head(fine,low,high)

    def forward_ostrava_mask(self,image):
        values=forward_with_fine(self.backbone,image)
        return self.forward_ostrava_features(values['fine'],values['low'],values['high'])
