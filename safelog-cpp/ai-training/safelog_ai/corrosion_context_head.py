"""One native decoder conditioned on an existing frozen whole-image response."""
import torch
from torch import nn
from torch.nn import functional as F
from .ostrava_rust_mask import RustMaskHead,OstravaRustMain
from .fine_feature_probe import forward_with_fine


class ContextRustHead(RustMaskHead):
    def __init__(self):
        super().__init__()
        # Preserve the original constructor RNG stream for the controlled v83 comparison.
        with torch.random.fork_rng(devices=[]):
            self.context_projection=nn.Conv2d(1,64,3,padding=1,bias=False)
            nn.init.zeros_(self.context_projection.weight)

    def forward(self,fine,low,high,context):
        if tuple(context.shape)!=(fine.shape[0],1,128,128) or not torch.isfinite(context).all() or torch.any(context.abs()>1):raise ValueError('Bounded aligned whole-view context128 required')
        values={}
        for name,value,c,size in (('fine',fine,24,128),('low',low,40,64),('high',high,960,32)):
            if tuple(value.shape[1:])!=(c,size,size):raise ValueError('Declared frozen RGB512 encoder grid required')
            values[name]=((value.half().float()-getattr(self,name+'_mean'))/getattr(self,name+'_std')).clamp(-8,8)
        up=lambda x:F.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
        x=torch.cat((self.fine_projection(values['fine']),up(values['low']),up(self.high_projection(values['high']))),1)
        # Keep the old convolution kernel/reduction exactly; zero context adds no initial change.
        x=self.fusion[0](x)+self.context_projection(context.detach().float())
        for layer in self.fusion[1:]:x=layer(x)
        return F.interpolate(x,size=(512,512),mode='bilinear',align_corners=False)[:,0]


class CorrosionContextMain(OstravaRustMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.corrosion_context_head=ContextRustHead()

    @property
    def corrosion_retained_head(self):
        return self.corrosion_context_head

    def forward_corrosion_context_features(self,fine,low,high,context):return self.corrosion_context_head(fine,low,high,context)

    def forward_corrosion_context_tiles(self,image,context):
        f=forward_with_fine(self.backbone,image)
        return self.forward_corrosion_context_features(f['fine'],f['low'],f['high'],context)
