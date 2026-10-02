"""Multi-label PHOTO evidence. Outputs no defect boxes or structural diagnosis."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def build_model(classes: int, pretrained: bool = False, architecture: str = "efficientnet_b0_multilabel_v1"):
    if architecture == "efficientnet_b0_multilabel_v1":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    elif architecture == "efficientnet_v2_s_multilabel_v1":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("Unsupported facility photo classifier")
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, classes)
    return model


def image_transform(size: int, training: bool = False):
    # Keep the entire photo; a random/center crop could remove the positive defect.
    ops = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(.1, .1, .1, .01)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class PresenceClassifier:
    def __init__(self, weights: Path, device: str | None = None):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.classes = checkpoint["classes"]
        self.imgsz = checkpoint["imgsz"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(len(self.classes), architecture=checkpoint["architecture"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.transform = image_transform(self.imgsz)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[float]:
        data = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return self.model(data).sigmoid()[0].cpu().tolist()
