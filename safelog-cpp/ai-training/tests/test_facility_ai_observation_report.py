from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.report_facility_ai_observations import (
    ROOT, audit_dimensions, prior_photo_context, render_report,
)


class ObservationReportTests(unittest.TestCase):
    def test_prior_comparison_uses_photo_identity_and_excludes_free_text(self):
        prior = {"schema": "facility_train_AI_visual_observations_v1", "split": "train",
                 "cases": [{"image_sha256": "a"*64, "case_id": "old-name", "note": "private"},
                           {"image_sha256": "b"*64}]}
        observations = {"cases": [{"case_id": "new-name", "image_sha256": "a"*64},
                                  {"case_id": "other", "image_sha256": "c"*64}]}
        before = deepcopy(prior)
        result = prior_photo_context(prior, observations, "d"*64)
        self.assertEqual((result["same_photo_sha256_cases"], result["not_in_prior_observation_cases"]), (1, 1))
        self.assertNotIn("old-name", json.dumps(result)); self.assertNotIn("private", json.dumps(result))
        self.assertEqual(prior, before)

    def test_malformed_or_nontrain_prior_record_rejects(self):
        prior = {"schema": "facility_train_AI_visual_observations_v1", "split": "train",
                 "cases": [{"image_sha256": "a"*64}]}
        malformed = [None, [], {**prior, "split": "test"}, {**prior, "cases": []},
                     {**prior, "cases": [None]}, {**prior, "cases": [{"image_sha256": "bad"}]},
                     {**prior, "cases": prior["cases"]*2}]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                prior_photo_context(value, {"cases": []}, "b"*64)

    def test_image_header_audit_preserves_bytes_and_rejects_changed_asset(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as folder:
            directory = Path(folder); path = directory / "synthetic.png"
            Image.new("RGB", (33, 300)).save(path)
            before = path.read_bytes()
            package = {"cases": [{"case_id": "synthetic", "domain": "codebrim",
                       "image_url": path.name, "image_sha256": hashlib.sha256(before).hexdigest()}]}
            summary, details = audit_dimensions(package, directory)
            self.assertEqual(summary["codebrim"]["min_side_below_64"], 1)
            self.assertEqual(summary["codebrim"]["aspect_above_8"], 1)
            self.assertEqual((details[0]["width"], details[0]["height"]), (33, 300))
            self.assertEqual(path.read_bytes(), before)
            path.write_bytes(before + b"changed")
            with self.assertRaises(ValueError): audit_dimensions(package, directory)

    def test_report_uses_subset_counts_without_fixed_study_observations(self):
        result = {"ai_observations": {"observed_cases": 1, "observed_task_opinions": 2,
                  "by_source": {"dacl": 1}, "by_reason_task_opinions": {"small_damage": 2},
                  "source_package_sha256": "a"*64, "observation_content_sha256": "b"*64},
                  "input_dimensions": {"dacl": {"cases": 4, "min_input_side": 33,
                      "median_min_input_side": 44, "min_side_below_64": 2, "aspect_above_8": 0}},
                  "metadata_only_case_count": 3}
        report = render_report(result)
        self.assertIn("검수 묶음 4건 중 선택한 1건", report)
        self.assertIn("나머지 3건", report)
        self.assertNotIn("130", report); self.assertNotIn("107", report)
        self.assertNotIn("1,060", report); self.assertNotIn("22.28%", report)
        self.assertIn("전문가 판정, 원본 정답 수정, 새 학습과 성능 개선은 이 작업에서 0건", report)


if __name__ == "__main__": unittest.main()
