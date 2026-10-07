"""Offline research model factory without changing the frozen app factory."""
from __future__ import annotations

from pathlib import Path

import torch

from safelog_ai.presence_classifier import PresenceClassifier, build_model, image_transform
from safelog_ai.semantic_residual_classifier import SemanticResidualClassifier, ARCH, require
from safelog_ai.auxiliary_classifier import AUX_CLASSES


def build_research_model(classes, architecture):
    if architecture == ARCH:
        return SemanticResidualClassifier(classes, pretrained=False)
    return build_model(classes, pretrained=False, architecture=architecture)


class ResearchPresenceClassifier(PresenceClassifier):
    def __init__(self, weights: Path, device: str | None = None):
        checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
        self.classes = checkpoint["classes"]; self.imgsz = checkpoint["imgsz"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_research_model(len(self.classes), checkpoint["architecture"])
        if checkpoint["architecture"] == ARCH:
            require(self.imgsz == 640 and checkpoint.get("auxiliary_classes") == list(AUX_CLASSES),
                    "Semantic research checkpoint must retain640 and original19 auxiliary classes")
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = image_transform(self.imgsz)
