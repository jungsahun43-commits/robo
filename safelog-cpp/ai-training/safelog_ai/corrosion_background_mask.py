"""Same-domain sparse positive and negative ROI supervision on one shared encoder."""
import torch
from .ostrava_rust_mask import OstravaRustMain,RustMaskHead
from .corrosion_partial_mask import loss_components
from .fine_feature_probe import forward_with_fine


def background_loss(logits,target):
    p,n,_,hp,hn=loss_components(logits,target)
    if not torch.all(hp&hn):raise ValueError('Both provided classes must survive in every training view')
    return .5*p+.5*n


class CorrosionBackgroundMain(OstravaRustMain):
    def __init__(self,reference_counts,native_specs,wear_variant,surface_specs):
        super().__init__(reference_counts,native_specs,wear_variant,surface_specs)
        self.corrosion_background_head=RustMaskHead()

    def forward_corrosion_background_features(self,fine,low,high):return self.corrosion_background_head(fine,low,high)

    def forward_corrosion_background(self,image):
        values=forward_with_fine(self.backbone,image)
        return self.forward_corrosion_background_features(values['fine'],values['low'],values['high'])
