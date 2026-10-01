from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from safelog_ai.policy import Detection, analyze_detections, canonical_label
from safelog_ai import server
from scripts.evaluate_fixed_threshold import match_counts
from scripts.build_facility_report import class_presence


class FacilityTests(unittest.TestCase):
    def test_combined_crack_class_does_not_claim_concrete_material(self):
        label = canonical_label("concrete_crack")
        self.assertEqual(label, "surface_crack")
        self.assertEqual(analyze_detections([Detection(label, .8)])["category"], "표면 균열 의심")

    def test_presence_does_not_require_box_match_and_fpr_requires_negatives(self):
        fixed = {"per_class": {"metal_corrosion": {}}, "per_image": [
            {"counts": {"metal_corrosion": {"tp": 0, "fp": 1, "fn": 2}}},
            {"counts": {"metal_corrosion": {"tp": 0, "fp": 0, "fn": 1}}},
        ]}
        result = class_presence(fixed)["metal_corrosion"]
        self.assertEqual((result["tp"], result["fn"]), (1, 1))
        self.assertIsNone(result["false_positive_rate"])
        self.assertEqual(result["miss_fraction"], .5)
    def test_facility_findings_do_not_claim_pipe_leak_or_structural_failure(self):
        result = analyze_detections([Detection("wet_surface", .8), Detection("metal_corrosion", .9)])
        self.assertIn("의심", result["category"])
        self.assertIn("배관 누수를 확정하지", result["action"])
        self.assertIn("두께·단면", result["action"])
        self.assertNotIn("붕괴", result["description"])

    def test_fire_takes_priority_over_facility_defects(self):
        result = analyze_detections([Detection("fire", .8), Detection("exposed_rebar", .99)])
        self.assertEqual(result["risk_level"], 5)
        self.assertIn("철근 노출 의심", result["hazards"])

    def test_identical_defect_does_not_resolve(self):
        client = TestClient(server.app)
        with patch.object(server, "detect", return_value=[Detection("concrete_crack", .8, "facility-dacl")]):
            result = client.post("/v1/compare-action", json={"beforeImage": "test", "afterImage": "test", "promptVersion": "facility-test-v1"})
        self.assertFalse(result.json()["likelyResolved"])
        self.assertTrue(result.json()["requiresHumanReview"])

    def test_matching_cannot_count_two_predictions_for_one_region(self):
        predictions = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]])
        targets = np.array([[0, 0, 10, 10], [40, 40, 50, 50]])
        self.assertEqual(match_counts(predictions, targets), (1, 2, 1))
        self.assertEqual(match_counts(np.empty((0, 4)), targets), (0, 0, 2))

    def test_inference_matches_facility_training_resolution(self):
        self.assertEqual(server.inference_size(Path("models/facility-dacl.pt")), 960)
        self.assertEqual(server.inference_size(Path("runs/facility-corrosion/weights/best.pt")), 960)
        self.assertEqual(server.inference_size(Path("models/fire-smoke.pt")), 640)


if __name__ == "__main__":
    unittest.main()
