"""Verify disjoint patches and actual CPU ONNX inference for the trained candidate."""
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_facilities import sha, save
from safelog_ai.presence_classifier import PresenceClassifier


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    train = read(ROOT / "data/damsegment-training/train.json")
    test = read(ROOT / "data/damsegment-training/test.json")
    for key in ("group_id", "pixel_sha256", "image"):
        if {item[key] for item in train["items"]} & {item[key] for item in test["items"]}:
            raise ValueError(f"Supplemental test leakage: {key}")
    if any(item["split"] != "train" for item in train["items"]): raise ValueError("Wrong train split")
    if any(item["split"] != "test" for item in test["items"]): raise ValueError("Wrong test split")
    for item in train["items"] + test["items"]:
        if item["targets"][2:] != [-1] * 5: raise ValueError("Unannotated label became negative")
    run = ROOT / "runs/facility-presence-damsegment"
    training = read(run / "TRAINING.json")
    if training["status"] != "complete" or training["supplemental_images"] != len(train["items"]):
        raise ValueError("Incomplete supplemental training")
    torch.set_num_threads(4)
    model = PresenceClassifier(run / "best.pt", "cpu")
    with Image.open(ROOT / test["items"][0]["image"]) as handle:
        inputs = model.transform(handle.convert("RGB")).unsqueeze(0)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    runtime = ort.InferenceSession(str(run / "best.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode(): reference = model.model(inputs).sigmoid().numpy()
    logits = runtime.run(None, {"images": inputs.numpy()})[0]
    probability = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(reference, probability, atol=1e-5, rtol=1e-4)
    two = runtime.run(None, {"images": inputs.repeat(2, 1, 1, 1).numpy()})[0]
    if two.shape != (2, 7): raise ValueError("Dynamic batch export failed")
    old = torch.load(ROOT / "runs/facility-presence/best.pt", map_location="cpu", weights_only=True)
    new = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    changed = sum(not torch.equal(old["state_dict"][key], tensor) for key, tensor in new["state_dict"].items())
    if not changed: raise ValueError("Candidate was not trained")
    result = {"status": "passed", "supplemental_train_images": len(train["items"]), "reserved_patch_test_images": len(test["items"]),
              "split_group_and_pixel_overlap": 0, "unknown_labels_preserved": True,
              "model_sha256": sha(run / "best.pt"), "onnx_sha256": sha(run / "best.onnx"), "changed_weight_tensors": changed,
              "classifier_cpu_onnx_parity": {"max_probability_difference": float(abs(reference - probability).max()), "dynamic_batch_shape": list(two.shape)}}
    save(ROOT / "reports/facility-round4-verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
