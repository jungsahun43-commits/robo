from __future__ import annotations

import unittest

from safelog_ai.policy import Detection, analyze_detections, likely_resolved


class PolicyTests(unittest.TestCase):
    def test_highest_risk_wins(self) -> None:
        result = analyze_detections([
            Detection("no_gloves", 0.95), Detection("no_helmet", 0.80)
        ])
        self.assertEqual(result["category"], "안전모 미착용")
        self.assertEqual(result["risk_level"], 4)
        self.assertAlmostEqual(result["confidence"], 0.80)

    def test_safe_fallback_requires_human_review(self) -> None:
        result = analyze_detections([Detection("helmet", 0.91)])
        self.assertEqual(result["risk_level"], 1)
        self.assertIn("점검자", result["description"])

    def test_comparison_requires_before_hazard_and_clean_after(self) -> None:
        resolved, remaining, _ = likely_resolved(
            [Detection("no_boots", 0.88)], [Detection("boots", 0.90)]
        )
        self.assertTrue(resolved)
        self.assertEqual(remaining, [])

    def test_remaining_risk_is_reported(self) -> None:
        resolved, remaining, _ = likely_resolved(
            [Detection("no_helmet", 0.88)], [Detection("no_helmet", 0.70)]
        )
        self.assertFalse(resolved)
        self.assertEqual(remaining, ["안전모 미착용"])


if __name__ == "__main__":
    unittest.main()
