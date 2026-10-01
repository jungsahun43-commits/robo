"""Check the actual delivery archive and classifier ONNX/PyTorch parity on CPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.presence_classifier import PresenceClassifier


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    bundle = ROOT / "artifacts/safelog-trained-models.zip"
    expected = bundle.with_suffix(".zip.sha256").read_text().split()[0]
    assert sha(bundle) == expected, "ZIP checksum mismatch"
    registry = json.loads((ROOT / "reports/model-registry.json").read_text(encoding="utf-8"))
    verified = 0
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None, "Corrupted ZIP member"
        for entry in registry:
            for item in entry["files"]:
                assert sha(ROOT / item["path"]) == item["sha256"]
                with archive.open(item["path"]) as handle:
                    assert hashlib.file_digest(handle, "sha256").hexdigest() == item["sha256"]
                verified += 1
    profile_path = ROOT / "reports/facility-inference-profile.json"
    profile = json.loads(profile_path.read_text())
    for report in ("facility-optimization-test.json", "facility-presence-test.json"):
        assert json.loads((ROOT / "reports" / report).read_text())["profile_sha256"] == sha(profile_path)
    for name in (*profile["models"], "facility-presence"):
        onnx.checker.check_model(onnx.load(ROOT / f"models/{name}.onnx"))
    torch.set_num_threads(4)
    classifier = PresenceClassifier(ROOT / "models/facility-presence.pt", "cpu")
    fixture = sorted((ROOT / "data/dacl10k-yolo/images/test").glob("*.jpg"))[0]
    with Image.open(fixture) as image:
        inputs = classifier.transform(image.convert("RGB")).unsqueeze(0)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    runtime = ort.InferenceSession(str(ROOT / "models/facility-presence.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        reference = classifier.model(inputs).sigmoid().numpy()
    logits = runtime.run(None, {"images": inputs.numpy()})[0]
    probability = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(reference, probability, atol=1e-5, rtol=1e-4)
    two = runtime.run(None, {"images": inputs.repeat(2, 1, 1, 1).numpy()})[0]
    assert two.shape == (2, len(classifier.classes)), "Dynamic batch export failed"
    result = {"status": "passed", "bundle_sha256": expected, "bundle_bytes": bundle.stat().st_size,
              "models": len(registry), "model_files_verified": verified, "new_onnx_structural_checks": 3,
              "classifier_cpu_onnx_max_probability_difference": float(abs(reference - probability).max()),
              "classifier_dynamic_batch_shape": list(two.shape), "test_profile_checksums_match": True}
    (ROOT / "reports/facility-optimization-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
