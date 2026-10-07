"""Fetch one official pretrained file and bind its verified offline encoder."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.semantic_residual_classifier import (SemanticResidualClassifier, OFFICIAL_URL, WEIGHT_ENUM,
    WEIGHT_SHA_PREFIX, load_semantic_pretrained, file_sha256, model_inventory, require)

WEIGHTS_PATH = "data/pretrained/convnext-tiny-imagenet1k-v1.pth"
METADATA_PATH = "reports/facility-semantic-pretrained-weights.json"


def verify_existing(path):
    model = SemanticResidualClassifier(7, pretrained=False)
    loaded = load_semantic_pretrained(model, path)
    inventory = model_inventory(model)
    source = Path(torchvision.__file__).parent / "models/convnext.py"
    return {"schema": "facility_semantic_pretrained_weights_v1", "status": "verified",
        "created_utc": datetime.now(timezone.utc).isoformat(), "weights_path": WEIGHTS_PATH,
        "official_url": OFFICIAL_URL, "weight_enum": WEIGHT_ENUM, "expected_filename_sha256_prefix": WEIGHT_SHA_PREFIX,
        "weights_sha256": loaded["weights_sha256"], "file_size_bytes": Path(path).stat().st_size,
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "torchvision_convnext_source_sha256": file_sha256(source),
        "encoder_state_sha256": loaded["encoder_state_sha256"],
        "encoder_state_tensor_count": loaded["encoder_state_tensor_count"],
        "encoder_parameter_count": loaded["encoder_parameter_count"],
        "strict_encoder_load": loaded["strict_encoder_load"], "official_pooling_norm_transferred": True,
        "discarded_image_net_classifier_tensors": loaded["discarded_image_net_classifier_tensors"],
        "all_encoder_parameters_frozen": True, "encoder_eval": True,
        "feature_shapes_for_640": {"low": [192, 80, 80], "high": [768, 20, 20], "pooled": [768]},
        "encoder_inventory": inventory, "image_preprocessing": "Existing full640 RGB/MEAN/STD; no224 centre crop",
        "new_training_epochs": 0, "source_val_or_test_inference": False, "app_model_promoted": False}


def main():
    torch.set_num_threads(4)
    target = ROOT / WEIGHTS_PATH; metadata = ROOT / METADATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        require(target.exists(), "Preserved verified metadata has no corresponding weights")
        old = json.loads(metadata.read_text(encoding="utf-8"))
        actual = verify_existing(target)
        for key in ("schema", "status", "weights_path", "official_url", "weight_enum", "weights_sha256",
                    "file_size_bytes", "encoder_state_sha256", "encoder_state_tensor_count", "encoder_parameter_count"):
            require(old.get(key) == actual[key], "Preserved official metadata/weights binding changed")
        print(json.dumps({"status": "existing_verified", "weights_sha256": actual["weights_sha256"],
            "encoder_state_sha256": actual["encoder_state_sha256"]})); return
    if not target.exists():
        temporary = target.with_suffix(".download.part")
        require(not temporary.exists(), "Preserve the incomplete earlier download for inspection")
        with urlopen(OFFICIAL_URL, timeout=60) as response, temporary.open("xb") as stream:
            final = urlparse(response.geturl())
            require(final.scheme == "https" and final.hostname == "download.pytorch.org", "Only the official HTTPS model host is permitted")
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        require(file_sha256(temporary).startswith(WEIGHT_SHA_PREFIX), "Downloaded official hash prefix does not match")
        verify_existing(temporary)
        temporary.replace(target)
    record = verify_existing(target)
    with metadata.open("xb") as stream:
        stream.write((json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print(json.dumps({"status": "verified", "weights_sha256": record["weights_sha256"],
        "encoder_state_sha256": record["encoder_state_sha256"], "encoder_parameter_count": record["encoder_parameter_count"]}))


if __name__ == "__main__":
    main()
