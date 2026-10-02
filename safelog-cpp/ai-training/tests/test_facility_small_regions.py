import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from scripts import prepare_facility_small_regions as preparation
from scripts.prepare_facility_small_regions import (
    context_box, crop_targets, coarse_targets, choose_replacement,
    validate_manifest, NAMES,
)


class SmallRegionTests(unittest.TestCase):
    def test_verified_orientation_allowed_unverified_skipped_and_missing_record_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            source = root / "data/dacl10k/dev/images/train/native.jpg"
            source.parent.mkdir(parents=True)
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (40, 60)).save(source, exif=exif)
            with Image.open(source) as handle:
                corrected = ImageOps.exif_transpose(handle).convert("RGB")
            corrected.save(root / "data/parent.jpg")
            width, height = corrected.size
            digest = hashlib.sha256(f"{width}x{height}:".encode() + corrected.tobytes()).hexdigest()
            annotation = source.parent.parent.parent / "annotations/train/native.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(json.dumps({"imageWidth": width, "imageHeight": height, "shapes": []}), encoding="utf-8")
            preparation_path = root / "data/dacl10k-yolo/PREPARATION.json"
            preparation_path.parent.mkdir(parents=True)
            preparation_path.write_text(json.dumps({"max_side": 1280}), encoding="utf-8")
            parent = {"image": "data/parent.jpg", "domain": "dacl", "targets": [0] * 7}
            rotated = {"parent": {"split": "train", "exif_orientation_corrected": True,
                                   "source": "dacl10k/dev/images/train/native.jpg", "pixel_sha256": digest}}
            with patch.object(preparation, "ROOT", root):
                checks = {}
                self.assertIsNotNone(preparation.source_masks(parent, rotated, checks))
                self.assertTrue(checks[parent["image"]]["verified"])
                damaged = copy.deepcopy(rotated)
                damaged["parent"]["pixel_sha256"] = "unverified"
                self.assertIsNone(preparation.source_masks(parent, damaged, checks))
                self.assertFalse(checks[parent["image"]]["verified"])
                with self.assertRaises(ValueError):
                    preparation.source_masks(parent, {})
                with self.assertRaises(ValueError):
                    preparation.source_masks(parent, {"parent": {"split": "val", "exif_orientation_corrected": True}})

    def test_context_center_is_positive_even_when_centroid_is_in_hole(self):
        mask = np.zeros((100, 140), dtype=bool)
        mask[0:8, 0] = True
        mask[0, 0:8] = True
        box, (x, y) = context_box(mask)
        self.assertTrue(mask[y, x])
        self.assertGreaterEqual(box[2] - box[0], 20)
        self.assertGreaterEqual(box[3] - box[1], 20)
        self.assertTrue(0 <= box[0] <= x < box[2] <= 140)
        self.assertTrue(0 <= box[1] <= y < box[3] <= 100)
        self.assertEqual((box, (x, y)), context_box(mask))

    def test_crop_partial_labels_are_recomputed_and_unknowns_preserved(self):
        crack = np.zeros((40, 40), dtype=bool)
        crack[2:6, 2:6] = True
        spall = np.zeros_like(crack)
        spall[20:25, 20:25] = True
        masks = [crack, spall] + [None] * 5
        targets = crop_targets(masks, (0, 0, 10, 10), [1, 1, -1, -1, -1, -1, -1])
        self.assertEqual(targets, [1, 0, -1, -1, -1, -1, -1])
        pixel, known, conflicts = coarse_targets(masks, (0, 0, 10, 10), targets)
        self.assertEqual(known.tolist(), [1, 1, 0, 0, 0, 0, 0])
        self.assertGreater(int(pixel[0].sum()), 0)
        self.assertEqual(int(pixel[1:].sum()), 0)
        self.assertEqual(conflicts, 0)
        crack[2, 2] = False
        self.assertEqual(crop_targets(masks, (0, 0, 10, 10), [1, 1, -1, -1, -1, -1, -1])[0], -1)
        crack[2, 2] = True
        with self.assertRaises(ValueError):
            crop_targets(masks, (0, 0, 10, 10), [0, 1, -1, -1, -1, -1, -1])

    def test_only_one_positive_child_chosen_with_smallest_class_first(self):
        masks = [np.zeros((100, 100), dtype=bool) for _ in range(7)]
        masks[0][10:14, 10:14] = True
        masks[1][60:65, 60:65] = True
        parent = {"targets": [1, 1, 0, 0, 0, 0, 0]}
        items = [
            {"targets": [0] * 7, "box_in_parent": [0, 0, 50, 50]},
            {"targets": [1, 1, 0, 0, 0, 0, 0], "box_in_parent": [0, 0, 50, 50]},
            {"targets": [0, 1, 0, 0, 0, 0, 0], "box_in_parent": [50, 50, 100, 100]},
        ]
        chosen = choose_replacement(parent, masks, [0, 1, 2], items)
        self.assertEqual(chosen[:2], (1, 0))
        self.assertEqual(chosen[-1][:2], [1, 0])
        self.assertEqual(items[0]["targets"], [0] * 7)

    def test_manifest_rejects_heldout_parent_order_and_invented_labels(self):
        parent = {"image": "data/train.jpg", "domain": "damsegment", "targets": [1, 0, -1, -1, -1, -1, -1]}
        child = {"image": "data/detail.jpg", "parent_image": parent["image"], "parent_split": "train", "domain": "damsegment", "targets": [1, 0, -1, -1, -1, -1, -1], "box_in_parent": [0, 0, 20, 20]}
        manifest = {"split": "train", "classes": NAMES, "full_count": 1, "items": [{**parent, "pixel_target": "data/mask.npz"}, child]}
        self.assertEqual(list(validate_manifest(manifest, [parent])), [parent["image"]])
        changes = [lambda m: m["items"][1].update(parent_split="val"),
                   lambda m: m["items"][1].update(parent_image="data/test.jpg"),
                   lambda m: m["items"][0].update(image="data/different.jpg"),
                   lambda m: m["items"][1]["targets"].__setitem__(2, 0),
                   lambda m: m["items"][1]["targets"].__setitem__(1, 1)]
        for change in changes:
            damaged = copy.deepcopy(manifest)
            change(damaged)
            with self.assertRaises(ValueError):
                validate_manifest(damaged, [parent])


if __name__ == "__main__":
    unittest.main()
