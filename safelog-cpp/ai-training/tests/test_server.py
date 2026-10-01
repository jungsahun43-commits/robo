from __future__ import annotations

import base64
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from safelog_ai import server
from safelog_ai.policy import Detection


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        output = io.BytesIO()
        Image.new("RGB", (4, 4), "white").save(output, format="PNG")
        self.image = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()

    def test_corrupt_image_returns_400(self) -> None:
        image = "data:image/jpeg;base64," + base64.b64encode(b"not a photo").decode()
        response = self.client.post("/v1/analyze-hazard", json={"image": image, "promptVersion": "test-v1"})
        self.assertEqual(response.status_code, 400)

    def test_missing_model_is_service_unavailable(self) -> None:
        with patch.object(server, "MODEL_PATHS", [Path("missing-model-for-test.pt")]), patch.object(server, "_models", None):
            self.assertEqual(self.client.get("/health").json()["status"], "unavailable")
            response = self.client.post("/v1/analyze-hazard", json={"image": self.image, "promptVersion": "test-v1"})
        self.assertEqual(response.status_code, 503)

    def test_hazard_response_keeps_evidence_and_contract(self) -> None:
        evidence = [Detection("helmet", 0.8, "sh17-ppe", (1, 2, 3, 4)), Detection("fire", 0.9, "fire-smoke", (0, 0, 4, 4))]
        with patch.object(server, "detect", return_value=evidence):
            response = self.client.post("/v1/analyze-hazard", json={"image": self.image, "promptVersion": "test-v1"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["riskLevel"], 5)
        self.assertTrue(result["requiresHumanReview"])
        self.assertEqual(result["promptVersion"], "test-v1")
        self.assertEqual(result["detections"][0]["model"], "sh17-ppe")
        self.assertEqual(result["detections"][0]["box"], [1, 2, 3, 4])

    def test_normal_equipment_does_not_imply_missing_equipment(self) -> None:
        with patch.object(server, "detect", return_value=[Detection("head", 0.95, "sh17-ppe")]):
            response = self.client.post("/v1/analyze-hazard", json={"image": self.image, "promptVersion": "test-v1"})
        self.assertEqual(response.json()["riskLevel"], 1)
        self.assertNotIn("안전모 미착용", response.json()["detectedHazards"])

    def test_summary_identifies_template(self) -> None:
        response = self.client.post("/v1/summarize", json={"inspectionContext": "입구 | 화재 | 대피 | Open\n", "promptVersion": "test-v1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["modelName"], "safelog-summary-template-v1")

    def test_missing_prompt_version_is_rejected(self) -> None:
        response = self.client.post("/v1/analyze-hazard", json={"image": self.image})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
