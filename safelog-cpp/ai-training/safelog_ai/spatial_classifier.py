"""Photo presence with training-only spatial supervision, no structural diagnosis."""
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

ARCH = "lraspp_mobilenet_facility_multitask_v1"


class SpatialClassifier(torch.nn.Module):
    def __init__(self, classes=7, pretrained=False):
        super().__init__()
        network = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None,
                                           weights_backbone=None)
        self.backbone = network.backbone
        self.segmentation_head = network.classifier
        self.segmentation_head.low_classifier = torch.nn.Conv2d(40, classes, 1)
        self.segmentation_head.high_classifier = torch.nn.Conv2d(128, classes, 1)
        self.photo_head = torch.nn.Linear(960, classes)
        self.mix = torch.nn.Parameter(torch.zeros(classes))

    def forward_details(self, image):
        features = self.backbone(image)
        maps = self.segmentation_head(features)
        global_logits = self.photo_head(features["high"].mean((-2,-1)))
        spatial_logits = maps.flatten(2).topk(32, dim=-1).values.mean(-1)
        mixture = self.mix.sigmoid()[None, :]
        return global_logits * (1-mixture) + spatial_logits * mixture, maps

    def forward(self, image): return self.forward_details(image)[0]


def spatial_loss(logits, target, known, positive_weights):
    # Unknown pixel regions never become background. CODEBRIM positives supply
    # photo labels only; its asserted absent classes can supply pixel negatives.
    known = known[:, :, None, None]
    weights = positive_weights[None, :, None, None]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = logits.sigmoid()
    correct = torch.where(target > 0, probability, 1-probability)
    balanced = torch.where(target > 0, weights, 1.)
    focal = (bce * (1-correct) * balanced * known).sum() / (known.sum() * logits.shape[-1] * logits.shape[-2]).clamp_min(1)
    intersection = (probability * target).sum((-2,-1))
    dice = 1-(2*intersection+1)/(probability.sum((-2,-1))+target.sum((-2,-1))+1)
    positives = (target.sum((-2,-1)) > 0) * known[:, :, 0, 0]
    return focal + .5*(dice*positives).sum()/positives.sum().clamp_min(1)
