"""Higher input resolution with the original 80-cell supervision/pooling grid."""
import torch
from .auxiliary_classifier import AuxiliaryClassifier

ARCH = "lraspp_mobilenet_facility_resolution_grid_v1"
RECIPE = {"input_sizes": [640, 960], "photo_and_loss_grid": [80, 80],
          "projection": "identity_at_80_else_area_downsample", "photo_topk": 32,
          "additional_parameters": 0, "new_pixel_targets": 0}


def project_logits(logits, shape=(80, 80)):
    """Downsample predictions, never interpolate labels or create fine truth."""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or not logits.is_floating_point()
            or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)):
        raise ValueError("Expected floating BCHW logits and a positive grid")
    if any(source < target for source, target in zip(logits.shape[-2:], shape)):
        raise ValueError("The declared study forbids upsampling prediction grids")
    if tuple(logits.shape[-2:]) == tuple(shape):
        return logits
    return torch.nn.functional.interpolate(logits, size=shape, mode="area")


class ResolutionClassifier(AuxiliaryClassifier):
    """Same state tensors as AuxiliaryClassifier; projection adds no weights."""
    def forward_training(self, image):
        features = self.backbone(image)
        maps = project_logits(self.segmentation_head(features))
        pooled = features["high"].mean((-2, -1))
        global_logits = self.photo_head(pooled)
        spatial_logits = maps.flatten(2).topk(32, dim=-1).values.mean(-1)
        mixture = self.mix.sigmoid()[None, :]
        return (global_logits * (1-mixture) + spatial_logits * mixture,
                maps, self.auxiliary_head(pooled))
