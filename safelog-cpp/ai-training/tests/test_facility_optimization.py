from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from safelog_ai import server
from safelog_ai.facility_profile import load_profile, verify_weights
from scripts.optimize_facilities import choose_threshold, measure
from scripts.train_facility_presence import average_precision
from scripts.optimize_facility_presence import photo_rates
from safelog_ai.policy import Detection, analyze_detections
import numpy as np


class OptimizationTests(unittest.TestCase):
    def test_classification_ap_handles_tied_scores(self):
        self.assertAlmostEqual(average_precision(np.array([1, 0]), np.array([.5, .5])), .5)
        self.assertAlmostEqual(average_precision(np.array([1, 0]), np.array([.9, .1])), 1.)
        self.assertIsNone(average_precision(np.array([0, 0]), np.array([.9, .1])))

    def test_classification_opinion_does_not_claim_position(self):
        opinion = Detection("surface_crack", .9, "facility-presence", None, "photo_presence")
        result = analyze_detections([opinion])
        self.assertIn("위치를 확정하지", result["description"])
        self.assertIsNone(opinion.box)

    def test_combined_photo_metrics_have_true_negatives(self):
        result = photo_rates(np.array([1, 1, 0, 0], dtype=bool), np.array([1, 0, 1, 0], dtype=bool))
        self.assertEqual(result["false_positive_rate"], .5)
        self.assertEqual(result["miss_fraction"], .5)

    def test_runtime_adds_unlocalized_opinion_and_deduplicates_existing_label(self):
        path = Path("models/facility-dacl-optimized.pt")
        detector = Mock()
        detector.predict.return_value = [SimpleNamespace(names={0: "concrete_crack"}, boxes=SimpleNamespace(
            cls=torch.tensor([0]), conf=torch.tensor([.5]), xyxy=torch.tensor([[0., 0., 2., 2.]])))]
        classifier = Mock()
        classifier.classes = ["concrete_crack", "wet_surface"]
        classifier.predict.return_value = [.95, .9]
        entry = {"model": "facility-presence", "thresholds": {"concrete_crack": .8, "wet_surface": .8}}
        output = io.BytesIO(); Image.new("RGB", (4, 4)).save(output, format="PNG")
        data = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        with patch.object(server, "MODEL_PATHS", [path]), patch.object(server, "models", return_value=[detector]), \
             patch.object(server, "FACILITY_PROFILE", {"models": {}, "photo_classifier": entry}), \
             patch.object(server, "PRESENCE_PATH", Path("models/facility-presence.pt")), \
             patch.object(server, "_presence_models", {Path("models/facility-presence.pt"): classifier}), \
             patch.object(Path, "is_file", return_value=True):
            detections = server.detect(data)
        self.assertEqual([d.label for d in detections], ["surface_crack", "wet_surface"])
        self.assertIsNone(detections[1].box)
        self.assertEqual(detections[1].evidence_scope, "photo_presence")

    def test_profile_cannot_be_selected_on_test(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({"selection_split": "test", "models": {}}))
            with self.assertRaises(ValueError):
                load_profile(path)

    def test_wrong_checkpoint_cannot_use_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"different model")
            with self.assertRaises(ValueError):
                verify_weights(path, {"weights_sha256": "0" * 64})

    def test_presence_and_localization_are_different(self):
        cache = {"images": [{"target_classes": [0], "target_boxes": [[0, 0, 10, 10]],
                  "classes": [0], "confidence": [.9], "boxes": [[20, 20, 30, 30]]}]}
        result = measure(cache, 0, .25)
        self.assertEqual(result["photo"]["tp"], 1)
        self.assertEqual(result["region"]["tp"], 0)
        self.assertIsNone(result["photo"]["false_positive_rate"])
        chosen, audit = choose_threshold(cache, 0, result)
        self.assertEqual(chosen["threshold"], .25)
        self.assertIn("no negative", audit["reason"])

    def test_threshold_search_respects_negative_photos(self):
        def image(positive, confidence):
            return {"target_classes": [0] if positive else [],
                    "target_boxes": [[0, 0, 10, 10]] if positive else [],
                    "classes": [0], "confidence": [confidence], "boxes": [[0, 0, 10, 10]]}
        cache = {"images": [image(True, .3), image(True, .15)] + [image(False, .1)] * 5}
        baseline = measure(cache, 0, .25)
        chosen, _ = choose_threshold(cache, 0, baseline)
        self.assertEqual(chosen["photo"]["recall"], 1.)
        self.assertEqual(chosen["photo"]["false_positive_rate"], 0.)
        self.assertGreater(chosen["threshold"], .1)

    def test_runtime_filters_raw_class_threshold_before_canonicalization(self):
        path = Path("models/facility-dacl-optimized.pt")
        entry = {"imgsz": 1280, "thresholds": {"concrete_crack": .2, "wet_surface": .6}}
        result = SimpleNamespace(names={0: "concrete_crack", 1: "wet_surface"}, boxes=SimpleNamespace(
            cls=torch.tensor([0, 0, 1]), conf=torch.tensor([.19, .3, .5]),
            xyxy=torch.tensor([[0., 0., 2., 2.]] * 3)))
        detector = Mock()
        detector.predict.return_value = [result]
        output = io.BytesIO()
        Image.new("RGB", (4, 4)).save(output, format="PNG")
        data = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        with patch.object(server, "MODEL_PATHS", [path]), patch.object(server, "models", return_value=[detector]), \
             patch.object(server, "FACILITY_PROFILE", {"models": {path.stem: entry}}):
            detections = server.detect(data)
        self.assertEqual([d.label for d in detections], ["surface_crack"])
        self.assertEqual(detector.predict.call_args.kwargs["conf"], .2)
        self.assertEqual(detector.predict.call_args.kwargs["imgsz"], 1280)
        self.assertEqual(detector.predict.call_args.kwargs["max_det"], 300)


if __name__ == "__main__":
    unittest.main()
