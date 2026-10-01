"""Check unadopted round-3 ONNX parity and prove the default profile is preserved."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.presence_classifier import PresenceClassifier


def sha(path):
    with path.open("rb") as handle: return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    run = ROOT / "runs/facility-presence-synthetic"
    classifier = PresenceClassifier(run / "best.pt", "cpu")
    torch.set_num_threads(4)
    onnx.checker.check_model(onnx.load(run / "best.onnx"))
    fixture = sorted((ROOT / "data/synthcavity-training/images").glob("*.jpg"))[0]
    with Image.open(fixture) as image: inputs = classifier.transform(image).unsqueeze(0)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    runtime = ort.InferenceSession(str(run / "best.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode(): reference = classifier.model(inputs).sigmoid().numpy()
    logits = runtime.run(None, {"images": inputs.numpy()})[0]
    predicted = 1 / (1 + np.exp(-logits))
    difference = float(np.abs(reference - predicted).max())
    assert difference < 1e-4
    shape = runtime.run(None, {"images": inputs.repeat(2, 1, 1, 1).numpy()})[0].shape
    assert shape == (2, 7)
    profile = ROOT / "reports/facility-inference-profile.json"
    old = ROOT / "reports/facility-inference-profile-round1.json"
    assert profile.read_bytes() == old.read_bytes()
    candidate = ROOT / "reports/facility-inference-profile-round3-candidate.json"
    assert sha(candidate) == json.loads((ROOT / "reports/facility-round3-test.json").read_text())["profile_sha256"]
    result = {"status": "passed", "scope": "unadopted round3 classifier ONNX/PyTorch CPU parity; not accuracy or Android verification",
              "classifier": "facility-presence-synthetic", "max_probability_difference": difference,
              "dynamic_batch_shape": list(shape), "weights_sha256": sha(run / "best.pt"), "onnx_sha256": sha(run / "best.onnx"),
              "test_candidate_checksum_matches": True, "default_profile_unchanged": True}
    (ROOT / "reports/facility-round3-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
