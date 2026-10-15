"""Four publisher appearance states on the unchanged shared industrial encoder."""
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from .ostrava_rust_mask import OstravaRustMain,RustMaskHead
from .fine_feature_probe import forward_with_fine

SIDE=512


def state_view(rgb,target):
    if rgb.size!=(target.shape[1],target.shape[0]) or target.ndim!=2:
        raise ValueError('Aligned embedded RGB/polygon raster required')
    if not set(np.unique(target))<={0,1,2,3,255}:raise ValueError('Undocumented state label')
    w,h=rgb.size;scale=min(SIDE/w,SIDE/h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    left,top=(SIDE-rw)//2,(SIDE-rh)//2
    view=Image.new('RGB',(SIDE,SIDE),(124,116,104));view.paste(rgb.resize((rw,rh),Image.Resampling.BICUBIC),(left,top))
    mask=np.full((SIDE,SIDE),255,dtype=np.uint8)
    mask[top:top+rh,left:left+rw]=np.asarray(Image.fromarray(target.astype(np.uint8)).resize((rw,rh),Image.Resampling.NEAREST))
    return np.array(view),mask,{'native_size':[w,h],'resized_size':[rw,rh],'left':left,'top':top}


def native_probabilities(probability,geometry):
    if tuple(probability.shape)!=(4,SIDE,SIDE):raise ValueError('Four full512 state probabilities required')
    w,h=geometry['native_size'];rw,rh=geometry['resized_size'];left,top=geometry['left'],geometry['top']
    return F.interpolate(probability[None,:,top:top+rh,left:left+rw].float(),size=(h,w),mode='bilinear',align_corners=False)[0]


def state_loss(logits,target):
    if logits.ndim!=4 or logits.shape[1]!=4 or tuple(logits.shape[0:1]+logits.shape[2:])!=tuple(target.shape):
        raise ValueError('Matching batch and state map required')
    valid=target!=255
    if torch.any(valid&((target<0)|(target>3))):raise ValueError('Undocumented state class')
    safe=target.masked_fill(~valid,0).long();truth=F.one_hot(safe,4).permute(0,3,1,2).float()*valid[:,None]
    counts=truth.sum((2,3));present=counts>0;number=present.sum(1).clamp_min(1)
    logp=logits.float().log_softmax(1);ce=-(logp*truth).sum((2,3))/counts.clamp_min(1)
    prob=logp.exp()*valid[:,None];intersection=(prob*truth).sum((2,3))
    dice=1-(2*intersection+1)/(prob.sum((2,3))+counts+1)
    return ((ce+dice)*present).sum(1)/number


def confusion(probability,truth):
    if probability.shape[0]!=4 or tuple(probability.shape[1:])!=tuple(truth.shape):raise ValueError('Matching native state maps required')
    valid=truth!=255
    if torch.any(valid&((truth<0)|(truth>3))):raise ValueError('Undocumented native class')
    pred=probability.argmax(0)
    return torch.bincount((4*truth[valid].long()+pred[valid]).flatten(),minlength=16).reshape(4,4)


def rates(matrix):
    c=np.asarray(matrix,dtype=np.float64);tp=np.diag(c);actual=c.sum(1);pred=c.sum(0)
    iou=np.divide(tp,actual+pred-tp,out=np.zeros(4),where=(actual+pred-tp)>0)
    dice=np.divide(2*tp,actual+pred,out=np.zeros(4),where=(actual+pred)>0)
    recall=np.divide(tp,actual,out=np.zeros(4),where=actual>0)
    # This is >=Fair versus the author's Good/minimal class, not all rust versus zero rust.
    fn=c[1:,0].sum();fp=c[0,1:].sum()
    return {'matrix_true_rows_pred_columns':c.astype(np.int64).tolist(),'per_class_IoU':iou.tolist(),'per_class_Dice':dice.tolist(),
            'per_class_recall':recall.tolist(),'macroIoU':float(iou.mean()),'macroDice':float(dice.mean()),
            'pixel_accuracy':float(tp.sum()/c.sum()) if c.sum() else None,
            'at_least_Fair_pixel_FPR':float(fp/actual[0]) if actual[0] else None,
            'at_least_Fair_pixel_FNR':float(fn/actual[1:].sum()) if actual[1:].sum() else None}


class CorrosionStateHead(RustMaskHead):
    def __init__(self):
        super().__init__();self.fusion[-1]=nn.Conv2d(32,4,1)
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)

    def initialize_from_rust(self,rust_head):
        values={n:v for n,v in rust_head.state_dict().items() if not n.startswith('fusion.7.')}
        missing,extra=self.load_state_dict(values,strict=False)
        if set(missing)!={'fusion.7.weight','fusion.7.bias'} or extra:raise ValueError('Declared v62 initializer changed')

    def forward(self,fine,low,high):
        values={}
        for name,value,c,size in (('fine',fine,24,128),('low',low,40,64),('high',high,960,32)):
            if tuple(value.shape[1:])!=(c,size,size):raise ValueError('Declared shared RGB512 grids required')
            values[name]=((value.half().float()-getattr(self,name+'_mean'))/getattr(self,name+'_std')).clamp(-8,8)
        up=lambda x:F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
        merged=torch.cat((self.fine_projection(values['fine']),up(values['low']),up(self.high_projection(values['high']))),1)
        return F.interpolate(self.fusion(merged),size=(SIDE,SIDE),mode='bilinear',align_corners=False)


class CorrosionStateMain(OstravaRustMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs);self.corrosion_state_head=CorrosionStateHead()

    def forward_corrosion_state_features(self,fine,low,high):return self.corrosion_state_head(fine,low,high)

    def forward_corrosion_state(self,image):
        x=forward_with_fine(self.backbone,image)
        return self.forward_corrosion_state_features(x['fine'],x['low'],x['high'])
