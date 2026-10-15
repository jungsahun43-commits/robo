import unittest
import numpy as np
from safelog_ai.image_pair_probe import orientation_rgb_probe


class PairProbeTests(unittest.TestCase):
    def test_known_rotated_mirrored_duplicate_is_exact_without_assuming_defect_label(self):
        left = np.random.default_rng(36).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        result = orientation_rgb_probe(left, np.rot90(np.fliplr(left)))
        self.assertEqual(result['mean_absolute_RGB_difference'], 0.)
        self.assertEqual(result['RGB_RMSE'], 0.)

    def test_reports_pixel_disagreement_and_rejects_unmatched_geometry(self):
        left = np.zeros((32, 32, 3), dtype=np.uint8); right = np.full_like(left, 255)
        self.assertEqual(orientation_rgb_probe(left, right)['mean_absolute_RGB_difference'], 1.)
        for other in (right[:16], right.astype(np.float32)):
            with self.assertRaises(ValueError): orientation_rgb_probe(left, other)


if __name__ == '__main__': unittest.main()
