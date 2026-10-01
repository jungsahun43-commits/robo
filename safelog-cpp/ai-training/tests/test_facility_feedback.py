from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from safelog_ai import server
from safelog_ai.facility_profile import load_profile
from scripts.refine_facility_feedback import pick_head, NEW_RUN
from scripts import refine_facility_feedback as feedback


class FeedbackTests(unittest.TestCase):
    def test_release_refuses_a_profile_unrelated_to_the_test_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            active = reports / "facility-inference-profile.json"
            active.write_text('{"version": "facility-validation-v2"}')
            before = active.read_bytes()
            (reports / "facility-feedback-test.json").write_text(json.dumps({"profile_sha256": "unrelated"}))
            with patch.object(feedback, "ROOT", root), patch.object(feedback, "PROFILE", active):
                with self.assertRaises(ValueError): feedback.release()
            self.assertEqual(active.read_bytes(), before)

    def test_release_keeps_baseline_when_false_alarms_increase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            active = reports / "facility-inference-profile.json"
            baseline = reports / "facility-inference-profile-round1.json"
            baseline.write_text('{"version": "facility-validation-v2"}')
            active.write_text('{"version": "facility-validation-v3"}')
            (reports / "facility-feedback-test.json").write_text(json.dumps({
                "profile_sha256": feedback.sha(active), "per_class": {"crack": {
                    "round1": {"fp": 2, "fn": 5}, "round2": {"fp": 3, "fn": 4}}}}))
            with patch.object(feedback, "ROOT", root), patch.object(feedback, "PROFILE", active), \
                 patch.object(feedback, "ROUND1", baseline):
                feedback.release()
            self.assertEqual(active.read_bytes(), baseline.read_bytes())
            self.assertEqual(json.loads((reports / "facility-feedback-release.json").read_text())["status"], "kept_round1")

    def test_new_head_requires_no_validation_false_alarm_regression(self):
        target = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        base = np.zeros(6, dtype=bool)
        old = np.array([.9, .2, .85, .1, .1, .1])
        choice = pick_head(target, base, old, .88, np.array([.9, .89, .1, .1, .1, .1]))
        self.assertEqual(choice["selected"]["source"], NEW_RUN)
        self.assertEqual(choice["selected"]["metrics"]["miss_fraction"], 0)
        bad = pick_head(target, base, old, .88, np.array([.9, .2, .91, .1, .1, .1]))
        self.assertEqual(bad["selected"]["source"], "facility-presence")

    def test_ties_keep_the_existing_head(self):
        target = np.array([1, 1, 0, 0], dtype=bool)
        score = np.array([.9, .2, .85, .1])
        result = pick_head(target, np.zeros(4, dtype=bool), score, .88, score)
        self.assertEqual(result["selected"]["source"], "facility-presence")

    def test_profile_rejects_two_active_heads_for_the_same_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile = {"selection_split": "val", "models": {}, "photo_classifiers": [
                {"imgsz": 384, "thresholds": {"wet_surface": .8}},
                {"imgsz": 512, "thresholds": {"wet_surface": .9}}]}
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError): load_profile(path)

    def test_runtime_uses_selected_heads_without_inventing_boxes(self):
        yolo = Mock()
        yolo.predict.return_value = [SimpleNamespace(names={}, boxes=SimpleNamespace(
            cls=torch.empty(0), conf=torch.empty(0), xyxy=torch.empty((0, 4))))]
        old_path, new_path = Path("models/facility-presence.pt"), Path("models/facility-presence-refined.pt")
        old, new = Mock(), Mock()
        old.classes = new.classes = ["concrete_crack", "wet_surface"]
        old.predict.return_value = new.predict.return_value = [.95, .95]
        profile = {"models": {}, "photo_classifiers": [
            {"model": "facility-presence", "thresholds": {"concrete_crack": .8, "wet_surface": 1}},
            {"model": "facility-presence-refined", "thresholds": {"concrete_crack": 1, "wet_surface": .8}}]}
        output = io.BytesIO(); Image.new("RGB", (4, 4)).save(output, format="PNG")
        data = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        with patch.object(server, "MODEL_PATHS", [Path("models/facility-dacl.pt")]), \
             patch.object(server, "models", return_value=[yolo]), patch.object(server, "PRESENCE_PATH", None), \
             patch.object(server, "PRESENCE_PATHS", [old_path, new_path]), patch.object(server, "FACILITY_PROFILE", profile), \
             patch.object(server, "_presence_models", {old_path: old, new_path: new}), patch.object(Path, "is_file", return_value=True):
            detections = server.detect(data)
        self.assertEqual([(d.label, d.model) for d in detections], [("surface_crack", "facility-presence"), ("wet_surface", "facility-presence-refined")])
        self.assertTrue(all(d.box is None and d.evidence_scope == "photo_presence" for d in detections))


if __name__ == "__main__": unittest.main()
