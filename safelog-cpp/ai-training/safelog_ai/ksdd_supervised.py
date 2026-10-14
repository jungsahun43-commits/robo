"""Real-mask surface head on the preserved industrial shared encoder."""
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from .industrial_surface_reference import IndustrialSurfaceMain

ARCH = 'safelog-industrial-ksdd-supervised-v10'
CATEGORY = 'ksdd2_surface'
WIDTH, HEIGHT = 256, 704


def ksdd_view(rgb, truth=None):
    scale = min(WIDTH/rgb.width, HEIGHT/rgb.height)
    width, height = max(1, round(rgb.width*scale)), max(1, round(rgb.height*scale))
    offset = ((WIDTH-width)//2, (HEIGHT-height)//2)
    image = Image.new('RGB', (WIDTH, HEIGHT), (127, 127, 127))
    image.paste(rgb.resize((width, height), Image.Resampling.BICUBIC), offset)
    valid = torch.zeros(HEIGHT, WIDTH, dtype=torch.bool)
    x, y = offset; valid[y:y+height, x:x+width] = True
    mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.uint8)
    if truth is not None:
        if truth.size != rgb.size: raise ValueError('Source image and publisher mask sizes differ')
        values = torch.from_numpy(np.array(truth.convert('L').resize((width, height), Image.Resampling.NEAREST)) > 0)
        mask[y:y+height, x:x+width] = values
    geometry = {'source_size': list(rgb.size), 'view_size': [WIDTH, HEIGHT], 'resized_size': [width, height], 'offset': list(offset), 'scale': scale}
    return image, mask, valid, geometry


def restore_map(probability, geometry):
    x, y = geometry['offset']; width, height = geometry['resized_size']; source_w, source_h = geometry['source_size']
    cropped = probability[..., y:y+height, x:x+width]
    return F.interpolate(cropped.reshape(-1, 1, height, width), size=(source_h, source_w), mode='bilinear', align_corners=False).reshape(*probability.shape[:-2], source_h, source_w)


def ksdd_photo_score(probability, valid):
    if probability.shape != valid.shape: raise ValueError('Matching probability and assessed-region mask required')
    return probability.masked_fill(~valid.bool(), 0).flatten(1).topk(32, dim=1).values.mean(1)


class SurfaceMaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_projection = nn.Sequential(nn.Conv2d(960, 32, 1), nn.ReLU())
        self.fusion = nn.Sequential(nn.Conv2d(72, 64, 3, padding=1), nn.ReLU(), nn.Dropout2d(.1),
                                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, low, high):
        projected = F.interpolate(self.high_projection(high.float()), size=low.shape[-2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat((low.float(), projected), 1))


def mask_loss(logits, mask, valid):
    target = mask.float(); known = valid.float()
    if logits.shape != target.shape or target.shape != known.shape: raise ValueError('Matching loss tensors required')
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=logits.new_tensor(10.), reduction='none')
    bce = (bce*known).sum()/known.sum().clamp_min(1)
    probability = logits.sigmoid(); intersection = (probability*target*known).flatten(1).sum(1)
    denominator = ((probability+target)*known).flatten(1).sum(1)
    dice = 1-((2*intersection+1)/(denominator+1)).mean()
    photo = ksdd_photo_score(probability[:, 0], valid[:, 0].bool())
    labels = (target.flatten(1).sum(1) > 0).float()
    classification = F.binary_cross_entropy(photo.clamp(1e-6, 1-1e-6), labels)
    return bce+dice+.5*classification


class KSDDSupervisedMain(IndustrialSurfaceMain):
    def __init__(self, reference_counts, native_specs, wear_variant, surface_specs):
        super().__init__(reference_counts, native_specs, wear_variant, surface_specs)
        self.ksdd_head = SurfaceMaskHead()

    def forward_ksdd_features(self, low, high):
        # The FP16 roundtrip matches the fixed frozen-encoder training cache.
        return F.interpolate(self.ksdd_head(low.half().float(), high.half().float()), size=(HEIGHT, WIDTH), mode='bilinear', align_corners=False)[:, 0]

    def forward_ksdd(self, image):
        features = self.backbone(image)
        return self.forward_ksdd_features(features['low'], features['high'])
