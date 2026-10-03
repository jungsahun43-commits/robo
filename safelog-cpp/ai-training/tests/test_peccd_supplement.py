import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prepare_peccd_supplement import (
    class_order, parse_labels, safe_member, validate_download, screen_source,
    stratified_select, unknown_pixel_target, author_classes,
)


ORDER = ["Scaling", "Simple Crack", "Spalling", "Deep Crack", "Holes", "Multi-Branched Crack"]


class PeccdTests(unittest.TestCase):
    def test_missing_native_numeric_order_is_never_inferred_from_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing/conflicting author numeric class order"):
                author_classes(Path(directory))

    def test_mapping_uses_evidenced_order_scaling_and_holes_do_not_become_targets(self):
        order = class_order(ORDER)
        targets, _ = parse_labels("0 .5 .5 .2 .2\n4 .5 .5 .1 .1", order)
        self.assertEqual(targets, [0, 0, -1, -1, -1, -1, -1])
        targets, _ = parse_labels("3 .5 .5 .2 .2\n2 .4 .4 .1 .1", order)
        self.assertEqual(targets, [1, 1, -1, -1, -1, -1, -1])

    def test_invalid_vocabulary_or_native_boxes_fail_closed(self):
        with self.assertRaises(ValueError):
            class_order(["crack", "spall", "rust", "hole", "wet", "unknown"])
        for line in ("6 .5 .5 .2 .2", "2 nan .5 .2 .2", "2 .01 .5 .2 .2", "2 .5 .5 0 .2", "2.0 .5 .5 .2 .2", "2 .5 .5"):
            with self.assertRaises(ValueError):
                parse_labels(line, class_order(ORDER))

    def test_portable_archive_paths_reject_traversal_and_wrong_prefix(self):
        self.assertEqual(safe_member("SyrianPostEarthquakeCrackDataset/1.jpg"), "SyrianPostEarthquakeCrackDataset/1.jpg")
        for name in ("SyrianPostEarthquakeCrackDataset/sub/..\\..\\outside.jpg", "/SyrianPostEarthquakeCrackDataset/1.jpg", "C:/SyrianPostEarthquakeCrackDataset/1.jpg", "other/1.jpg", "SyrianPostEarthquakeCrackDataset/x:payload.jpg"):
            with self.assertRaises(ValueError):
                safe_member(name)

    def test_partial_archive_rejected_before_any_hash_or_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.rar"
            path.write_bytes(b"partial")
            with patch("scripts.prepare_peccd_supplement.read", return_value={"verified": True}), patch("scripts.prepare_peccd_supplement.sha") as hash_file:
                with self.assertRaises(ValueError):
                    validate_download(path, Path(directory) / "download.json")
                hash_file.assert_not_called()

    def test_all_pixel_classes_unknown_even_when_photo_labels_known(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.npz"
            unknown_pixel_target(path)
            with np.load(path) as record:
                self.assertEqual(record["mask"].shape, (7, 80, 80))
                self.assertEqual(record["known"].tolist(), [0] * 7)
                self.assertEqual(int(record["mask"].sum()), 0)

    def test_near_old_images_and_conflicting_groups_excluded_and_source_groups_collapsed(self):
        def row(name, digest, fingerprint, targets):
            return {"source_image": name, "pixel_sha256": digest, "dhash": fingerprint, "targets": targets + [-1] * 5, "suspected_derived_filename": False}
        items = [row("a.jpg", "one", 0, [1, 0]), row("b.jpg", "two", 1, [1, 0]),
                 row("c.jpg", "three", (1 << 64) - 1, [0, 1]), row("d.jpg", "four", (1 << 64) - 2, [0, 1])]
        eligible, pending, audit = screen_source(items, [{"pixel_sha256": "old"}], np.asarray([0], dtype=np.uint64))
        self.assertEqual([item["source_image"] for item in eligible], ["c.jpg"])
        self.assertEqual(len(pending), 2)
        self.assertEqual(audit["within_source_collapsed_photos"], 1)
        items[3]["targets"] = [1, 0] + [-1] * 5
        eligible, pending, audit = screen_source(items, [], np.asarray([], dtype=np.uint64))
        self.assertEqual([item["source_image"] for item in eligible], ["a.jpg"])
        self.assertTrue(any("within_near_group_conflicting_photo_labels_review_pending" in item["reasons"] for item in pending))

    def test_fixed_seed_stratified_selection_is_bounded_reproducible_without_scores(self):
        rows = [{"source_image": f"{a}{b}-{i:04d}.jpg", "targets": [a, b] + [-1] * 5} for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)) for i in range(200)]
        selected = stratified_select(rows)
        self.assertEqual(len(selected), 500)
        self.assertEqual(selected, stratified_select(list(reversed(rows))))
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            self.assertEqual(sum(item["targets"][:2] == [a, b] for item in selected), 125)
        with self.assertRaises(ValueError):
            stratified_select(rows, 501)


if __name__ == "__main__":
    unittest.main()
