import copy
import hashlib
import unittest

import numpy as np

from safelog_ai.auxiliary_classifier import AUX_CLASSES
from safelog_ai.spalling_sampler import (ELIGIBLE_STRATA, coupling_probabilities,
    coupling_strata, eligible_rows, paired_epoch, draw_counts, draw_sha256)


class SpallingSamplerTests(unittest.TestCase):
    def fixture(self):
        items = []
        for image, domain, crack, spalling in (
            ("d0", "dacl", 0, 0), ("d1", "dacl", 0, 0),
            ("d2", "dacl", 1, 0), ("d3", "dacl", 1, 0),
            ("d4", "dacl", 0, 1), ("m0", "damsegment", 0, 0),
            ("c0", "codebrim", 0, 0)):
            items.append({"image": image, "domain": domain,
                          "targets": [crack, spalling, 0, 0, 0, 0, 0]})
        items.append({"image": "crop", "domain": "dacl", "parent_image": "d0",
                      "parent_split": "train", "targets": [0, 0, 0, 0, 0, 0, 0]})
        auxiliary = {}
        for item in items[:5]:
            tags = [0] * len(AUX_CLASSES)
            tags[AUX_CLASSES.index("Spalling")] = item["targets"][1]
            tags[AUX_CLASSES.index("Crack")] = item["targets"][0]
            if item["image"] in ("d0", "d4"):
                tags[AUX_CLASSES.index("Rockpocket")] = 1
            if item["image"] == "d2":
                tags[AUX_CLASSES.index("Cavity")] = 1
            auxiliary[item["image"]] = tags
        weights = np.array([1., 3., 2., 4., 1., 1., 1., .5])
        return items, 7, auxiliary, weights

    def test_probability_boundaries_are_defined_and_invalid_values_rejected(self):
        for p, factor in ((0, 1.5), (1, 1.5), (.4, 1)):
            result = coupling_probabilities(p, factor)
            self.assertEqual(result["original_eligible_probability"], p)
            self.assertEqual(result["treatment_eligible_probability"], p)
            self.assertEqual(result["noneligible_replacement_probability"], 0)
            self.assertEqual(result["minimum_changed_probability"], 0)
        near_one = float(np.nextafter(1.0, 0.0))
        stable = coupling_probabilities(near_one)
        self.assertAlmostEqual(stable["noneligible_replacement_probability"], 1 / 3)
        self.assertGreater(stable["minimum_changed_probability"], 0)
        self.assertGreaterEqual(stable["treatment_eligible_probability"], near_one)
        self.assertLessEqual(stable["treatment_eligible_probability"], 1.0)
        for p, factor in ((True, 1.5), (float("nan"), 1.5), (-.1, 1.5),
                          (1.1, 1.5), (.4, True), (.4, .5), (.4, float("inf"))):
            with self.subTest(p=p, factor=factor), self.assertRaises(ValueError):
                coupling_probabilities(p, factor)

    def test_exact_weighted_coupling_marginals_and_maximal_retention(self):
        # Two eligible rows have combined weighted mass .4; their item fraction
        # is 2/3. This distinguishes weighted probability from a photo census.
        original = np.array([.1, .3, .6])
        eligible = np.array([True, True, False])
        probability = coupling_probabilities(float(original[eligible].sum()))
        self.assertAlmostEqual(probability["treatment_eligible_probability"], .5)
        self.assertAlmostEqual(probability["noneligible_replacement_probability"], 1 / 6)
        conditional = original[:2] / original[:2].sum()
        transition = np.eye(3)
        replacement = probability["noneligible_replacement_probability"]
        transition[2, :2] = replacement * conditional
        transition[2, 2] = 1 - replacement
        joint = original[:, None] * transition
        target = original * np.where(eligible, 1.5, 1.0)
        target /= target.sum()
        np.testing.assert_allclose(joint.sum(axis=1), original, atol=1e-15)
        np.testing.assert_allclose(joint.sum(axis=0), target, atol=1e-15)
        changed = 1 - np.trace(joint)
        total_variation = .5 * np.abs(target - original).sum()
        self.assertAlmostEqual(changed, .1)
        self.assertAlmostEqual(changed, total_variation)
        self.assertAlmostEqual(changed, probability["minimum_changed_probability"])

    def test_no_eligible_all_eligible_and_identity_factor_are_exact_noops(self):
        weights = np.array([1., 3.])
        strata = np.array(["dacl/full/00", "dacl/full/00"])
        base = np.array([0, 1, 1, 0], dtype=np.int64)
        for eligible, factor in ((np.array([False, False]), 1.5),
                                 (np.array([True, True]), 1.5),
                                 (np.array([True, False]), 1.0)):
            rng = np.random.default_rng(59)
            state = copy.deepcopy(rng.bit_generator.state)
            paired = paired_epoch(base, weights, eligible, strata, rng, factor)
            np.testing.assert_array_equal(paired, base)
            self.assertEqual(rng.bit_generator.state, state)

    def test_original_tags_scope_and_exact_paired_draw_counts(self):
        items, full, auxiliary, weights = self.fixture()
        original_items = copy.deepcopy(items)
        original_auxiliary = copy.deepcopy(auxiliary)
        original_weights = weights.copy()
        eligible = eligible_rows(items, full, auxiliary)
        np.testing.assert_array_equal(eligible, [True, False, True, False, False, False, False, False])
        strata = coupling_strata(items, full)
        base = np.tile(np.arange(len(items), dtype=np.int64), 100)
        paired = paired_epoch(base, weights, eligible, strata, np.random.default_rng(59))
        changed = base != paired
        self.assertGreater(int(changed.sum()), 0)
        np.testing.assert_array_equal(strata[base], strata[paired])
        self.assertEqual(draw_counts(base, strata), draw_counts(paired, strata))
        self.assertTrue(np.isin(strata[base[changed]], ELIGIBLE_STRATA).all())
        self.assertFalse(eligible[base[changed]].any())
        self.assertTrue(eligible[paired[changed]].all())
        np.testing.assert_array_equal(base[eligible[base]], paired[eligible[base]])
        protected = ~np.isin(strata[base], ELIGIBLE_STRATA)
        np.testing.assert_array_equal(base[protected], paired[protected])
        targets = np.asarray([item["targets"][:2] for item in items])
        np.testing.assert_array_equal(targets[base], targets[paired])
        self.assertEqual(items, original_items)
        self.assertEqual(auxiliary, original_auxiliary)
        np.testing.assert_array_equal(weights, original_weights)

    def test_one_rng_stream_reproduces_all_six_epochs_and_actual_hash_recipe(self):
        items, full, auxiliary, weights = self.fixture()
        eligible = eligible_rows(items, full, auxiliary)
        strata = coupling_strata(items, full)
        bases = [np.roll(np.tile(np.arange(len(items), dtype=np.int64), 50), epoch)
                 for epoch in range(6)]
        rng_a, rng_b = np.random.default_rng(59), np.random.default_rng(59)
        paired_a = [paired_epoch(base, weights, eligible, strata, rng_a) for base in bases]
        paired_b = [paired_epoch(base, weights, eligible, strata, rng_b) for base in bases]
        for a, b, base in zip(paired_a, paired_b, bases):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(draw_sha256(base), hashlib.sha256(base.astype("<i8").tobytes()).hexdigest())
        self.assertNotEqual(rng_a.bit_generator.state, np.random.default_rng(59).bit_generator.state)
        self.assertEqual(draw_sha256(np.array([0, 1, 2], dtype=">i8")),
                         hashlib.sha256(bytes.fromhex("000000000000000001000000000000000200000000000000")).hexdigest())

    def test_scope_tampering_unknown_or_heldout_rows_and_invalid_arrays_rejected(self):
        items, full, auxiliary, weights = self.fixture()
        for change in ("unknown_spalling", "unknown_crack", "auxiliary_mismatch", "heldout", "missing_auxiliary"):
            rows, tags = copy.deepcopy(items), copy.deepcopy(auxiliary)
            if change == "unknown_spalling":
                rows[0]["targets"][1] = -1
            elif change == "unknown_crack":
                rows[0]["targets"][0] = -1
            elif change == "auxiliary_mismatch":
                tags["d0"][AUX_CLASSES.index("Spalling")] = 1
            elif change == "heldout":
                rows[0]["split"] = "val"
            else:
                del tags["d0"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                eligible_rows(rows, full, tags)
        eligible = eligible_rows(items, full, auxiliary)
        strata = coupling_strata(items, full)
        illegal_eligible = eligible.copy()
        illegal_eligible[-1] = True
        with self.assertRaises(ValueError):
            paired_epoch(np.arange(8), weights, illegal_eligible, strata, np.random.default_rng(59))
        for invalid in (np.array([True]), np.array([.5]), np.array([-1]), np.array([8])):
            with self.subTest(draws=invalid), self.assertRaises(ValueError):
                paired_epoch(invalid, weights, eligible, strata, np.random.default_rng(59))
        with self.assertRaises(ValueError):
            paired_epoch(np.arange(8), np.full(8, np.nan), eligible, strata, np.random.default_rng(59))
        with self.assertRaises(ValueError):
            paired_epoch(np.arange(8), weights, eligible.astype(np.uint8), strata, np.random.default_rng(59))


if __name__ == "__main__":
    unittest.main()
