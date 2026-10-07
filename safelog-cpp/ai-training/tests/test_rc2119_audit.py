"""Native label identity and conservative exclusions precede eligibility."""
import unittest

import numpy as np

from scripts.audit_rc2119 import LABELS, derive_codes, overlap, positive_targets, validate_annotation
from scripts.screen_rc2119_crops import geometric_candidate


class AnnotationTests(unittest.TestCase):
    def rows(self):
        return [{"native_instances": {name: 1}, "native_mask_pixels": {"0": 20, str(i + 1): 5}}
                for i, name in enumerate(LABELS)]

    def test_absent_damage_stays_unknown(self):
        self.assertEqual(positive_targets(["Crack"]), [1, -1, -1, -1, -1, -1, -1])
        self.assertEqual(positive_targets(["Concrete spalling", "Rebar exposure"]), [-1, 1, -1, 1, -1, -1, -1])

    def test_corrosion_and_crushing_not_relabelled(self):
        self.assertEqual(positive_targets(["Rebar corrosion", "Concrete crushing"]), [-1] * 7)
        with self.assertRaises(ValueError): positive_targets(["unknown"])

    def test_numeric_codes_need_paired_evidence(self):
        codes, proof = derive_codes(self.rows())
        self.assertEqual(codes["Concrete spalling"], 2)
        self.assertEqual(proof["Concrete spalling"], {"2": 1})

    def test_conflicting_numeric_evidence_rejected(self):
        rows = self.rows(); rows.append({"native_instances": {"Crack": 1}, "native_mask_pixels": {"0": 10, "2": 5}})
        with self.assertRaises(ValueError): derive_codes(rows)
        with self.assertRaises(ValueError): derive_codes(self.rows()[:-1])

    def test_geometry_dimensions_and_nonfinite_are_rejected(self):
        doc = {"imageWidth": 20, "imageHeight": 10,
               "shapes": [{"label": "Crack", "shape_type": "polygon", "points": [[0, 0], [20, 0], [10, 10]]}]}
        self.assertEqual(len(validate_annotation(doc, (20, 10))), 1)
        with self.assertRaises(ValueError): validate_annotation(doc, (10, 20))
        doc["shapes"][0]["points"][0][0] = float("nan")
        with self.assertRaises(ValueError): validate_annotation(doc, (20, 10))

    def test_outside_geometry_is_not_silently_clipped(self):
        doc = {"imageWidth": 20, "imageHeight": 10,
               "shapes": [{"label": "Crack", "shape_type": "polygon", "points": [[-1, 0], [20, 0], [10, 10]]}]}
        with self.assertRaises(ValueError): validate_annotation(doc, (20, 10))

    def test_exact_overlap_excludes_transitive_whole_group(self):
        rows = [{"image": str(i), "dhash": h, "pixel_sha256": str(i), "orientation": 1}
                for i, h in enumerate((0, 1, 3))]
        result = overlap(rows, {"2"}, np.asarray([2 ** 64 - 1], np.uint64), distance=1)
        self.assertEqual(len({r["group"] for r in result}), 1)
        self.assertTrue(all("existing_exact_native_pixels" in r["excluded_reasons"] for r in result))

    def test_geometry_review_propagates_to_group(self):
        rows = [{"image": str(i), "dhash": h, "pixel_sha256": str(i), "orientation": 1,
                 "annotation_geometry_issue": "bad" if i == 0 else None} for i, h in enumerate((0, 1))]
        result = overlap(rows, set(), np.asarray([2 ** 64 - 1], np.uint64), distance=1)
        self.assertTrue(all("annotation_geometry_review" in r["excluded_reasons"] for r in result))


class CropGeometryTests(unittest.TestCase):
    def test_homography_recovers_known_cropped_view(self):
        rng = np.random.default_rng(61)
        points = rng.uniform(0, 200, (16, 2)).astype(np.float32)
        shifted = points * .7 + np.asarray([20, 30], np.float32)
        proof = geometric_candidate(points, shifted, [(i, i) for i in range(16)])
        self.assertIsNotNone(proof)
        self.assertEqual(proof["inliers"], 16)

    def test_sparse_matches_not_approved(self):
        points = np.arange(20, dtype=np.float32).reshape(10, 2)
        self.assertIsNone(geometric_candidate(points, points, [(i, i) for i in range(7)]))
        # Repeated coordinates cannot establish a crop transform.
        repeated = np.zeros((10, 2), np.float32)
        self.assertIsNone(geometric_candidate(repeated, repeated, [(i, i) for i in range(10)]))


if __name__ == "__main__":
    unittest.main()
