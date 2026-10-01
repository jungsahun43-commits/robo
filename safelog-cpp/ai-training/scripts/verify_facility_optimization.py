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
from safelog_ai.facility_profile import photo_entries


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
    test_reports = ("facility-feedback-test.json",) if "photo_classifiers" in profile else ("facility-optimization-test.json", "facility-presence-test.json")
    for report in test_reports:
        assert json.loads((ROOT / "reports" / report).read_text())["profile_sha256"] == sha(profile_path)
    classifier_names = [entry["model"] for entry in photo_entries(profile)]
    for name in (*profile["models"], *classifier_names):
        onnx.checker.check_model(onnx.load(ROOT / f"models/{name}.onnx"))
    torch.set_num_threads(4)
    fixture = sorted((ROOT / "data/dacl10k-yolo/images/test").glob("*.jpg"))[0]
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    parity = {}
    for name in classifier_names:
        classifier = PresenceClassifier(ROOT / "models" / f"{name}.pt", "cpu")
        with Image.open(fixture) as image:
            inputs = classifier.transform(image.convert("RGB")).unsqueeze(0)
        runtime = ort.InferenceSession(str(ROOT / "models" / f"{name}.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            reference = classifier.model(inputs).sigmoid().numpy()
        logits = runtime.run(None, {"images": inputs.numpy()})[0]
        probability = 1 / (1 + np.exp(-logits))
        np.testing.assert_allclose(reference, probability, atol=1e-5, rtol=1e-4)
        two = runtime.run(None, {"images": inputs.repeat(2, 1, 1, 1).numpy()})[0]
        assert two.shape == (2, len(classifier.classes)), "Dynamic batch export failed"
        parity[name] = {"max_probability_difference": float(abs(reference-probability).max()), "dynamic_batch_shape": list(two.shape)}
    result = {"status": "passed", "bundle_sha256": expected, "bundle_bytes": bundle.stat().st_size,
              "models": len(registry), "model_files_verified": verified, "new_onnx_structural_checks": len(profile["models"]) + len(classifier_names),
              "classifier_cpu_onnx_parity": parity, "test_profile_checksums_match": True}
    (ROOT / "reports/facility-optimization-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
