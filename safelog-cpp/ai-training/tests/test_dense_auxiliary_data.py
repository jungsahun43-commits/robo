"""Native19 masks preserve independent classes, registration and unknown truth."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.prepare_facility_dense_auxiliary import (AUX_CLASSES, ROOT, native_source,
    polygon_points, prepare_row, rasterize_dense, registered_native_size, safe_path,
    sha, validate_arrays, validate_source_manifest, write_new)


def fixture_document(label="Spalling", points=None):
    return {"imageWidth": 128, "imageHeight": 96, "split": "train", "dacl10k_version": "v2",
        "imagePath": "images/train/example.jpg", "imageName": "example.jpg", "shapes": [{"label": label,
            "shape_type": "polygon", "points": points or [[16, 12], [80, 12], [80, 72], [16, 72]]}]}


def targets_for(document):
    labels = {s["label"] for s in document["shapes"]}
    return [int(c in labels) for c in AUX_CLASSES]


class DenseDataTests(unittest.TestCase):
    def fixture_files(self):
        temporary = tempfile.TemporaryDirectory(prefix="dense19-data-test-", dir=ROOT / "runs")
        root = Path(temporary.name).resolve()
        self.assertTrue(root.is_relative_to((ROOT / "runs").resolve()))
        self.addCleanup(temporary.cleanup)
        image = root / "data/dacl10k-yolo/images/train/dacl-example.jpg"
        annotation = root / "data/dacl10k/dacl10k_v2_devphase/annotations/train/example.json"
        source = root / "data/dacl10k/dacl10k_v2_devphase/images/train/example.jpg"
        output = root / "data/facility-dense-auxiliary-training"
        for p in (image.parent, annotation.parent, source.parent, output / "masks"): p.mkdir(parents=True)
        Image.new("RGB", (128, 96), "grey").save(image)
        Image.new("RGB", (128, 96), "grey").save(source)
        doc = fixture_document(); write_new(annotation, doc)
        row = {"image": image.relative_to(root).as_posix(), "annotation": annotation.relative_to(root).as_posix(),
            "annotation_sha256": sha(annotation), "targets": targets_for(doc), "split": "train", "domain": "dacl"}
        return root, output, row, doc

    def test_overlapping_native_classes_remain_independent(self):
        doc = fixture_document(); second = deepcopy(doc["shapes"][0]); second["label"] = "Rockpocket"; doc["shapes"].append(second)
        masks, known, reasons = rasterize_dense(doc, targets_for(doc), (128, 96))
        spall, rock = AUX_CLASSES.index("Spalling"), AUX_CLASSES.index("Rockpocket")
        self.assertTrue(np.array_equal(masks[spall], masks[rock])); self.assertGreater(int(masks[spall].sum()), 0)
        self.assertTrue(known.all()); self.assertFalse(any(reasons))
        self.assertEqual(int(masks[AUX_CLASSES.index("WConccor")].sum()), 0)

    def test_same_class_polygons_are_unioned(self):
        doc = fixture_document(); second = deepcopy(doc["shapes"][0]); second["points"] = [[96, 8], [120, 8], [120, 24], [96, 24]]
        first, _, _ = rasterize_dense(doc, targets_for(doc), (128, 96))
        doc["shapes"].append(second); union, known, _ = rasterize_dense(doc, targets_for(doc), (128, 96))
        self.assertGreater(int(union.sum()), int(first.sum())); self.assertTrue(known.all())

    def test_bad_geometry_masks_entire_channel_unknown(self):
        corrupt = [[[False, 12], [80, 12], [80, 72]], [[float("nan"), 12], [80, 12], [80, 72]],
                   [[-1, 12], [80, 12], [80, 72]], [[16, 12], [129, 12], [80, 72]],
                   [[10, 10], [20, 20], [30, 30]], [[10, 10], [20, 20]]]
        for points in corrupt:
            with self.subTest(points=points):
                doc = fixture_document(); shape = deepcopy(doc["shapes"][0]); shape["points"] = points; doc["shapes"].append(shape)
                masks, known, reasons = rasterize_dense(doc, targets_for(doc), (128, 96)); k = AUX_CLASSES.index("Spalling")
                self.assertEqual(int(known[k]), 0); self.assertEqual(int(masks[k].sum()), 0)
                self.assertIn("invalid_native_polygon", reasons[k]); self.assertEqual(int(known.sum()), 18)

    def test_photo_positive_without_foreground_becomes_unknown(self):
        doc = fixture_document(); targets = targets_for(doc); doc["shapes"] = []
        masks, known, reasons = rasterize_dense(doc, targets, (128, 96)); k = AUX_CLASSES.index("Spalling")
        self.assertEqual(int(known[k]), 0); self.assertFalse(masks[k].any())
        self.assertIn("raster_presence_photo_tag_conflict", reasons[k]); self.assertEqual(targets[k], 1)

    def test_negative_photo_tag_never_acquires_foreground(self):
        doc = fixture_document(); targets = [0] * 19
        masks, known, reasons = rasterize_dense(doc, targets, (128, 96)); k = AUX_CLASSES.index("Spalling")
        self.assertFalse(masks.any()); self.assertEqual(int(known[k]), 0)
        self.assertIn("native_presence_photo_tag_conflict", reasons[k]); validate_arrays(masks, known, targets)

    def test_unknown_label_and_invalid_dimensions_fail_closed(self):
        for doc in (fixture_document("unknown"), {**fixture_document(), "imageWidth": True},
                    {**fixture_document(), "shapes": None}):
            with self.assertRaises(ValueError): rasterize_dense(doc, [0] * 19, (128, 96))

    def test_source_contract_rejects_holdout_crop_bool_or_reordered_classes(self):
        root, output, row, doc = self.fixture_files()
        manifest = {"split": "train", "classes": list(AUX_CLASSES), "items": [row]}
        self.assertEqual(validate_source_manifest(manifest, 1), [row])
        for broken in ({**manifest, "split": "val"}, {**manifest, "classes": list(reversed(AUX_CLASSES))},
                       {**manifest, "items": [{**row, "parent_image": "parent"}]},
                       {**manifest, "items": [{**row, "domain": "codebrim"}]},
                       {**manifest, "items": [{**row, "targets": [True] + row["targets"][1:]}]}):
            with self.assertRaises(ValueError): validate_source_manifest(broken, 1)

    def test_input_paths_reject_traversal_holdout_and_absolute(self):
        root, output, row, doc = self.fixture_files(); prefix = "data/dacl10k-yolo/images/train"
        self.assertEqual(safe_path(root, row["image"], prefix), root / row["image"])
        for path in ("../escape.jpg", "data/dacl10k-yolo/images/test/a.jpg", "C:/escape.jpg",
                     "data/dacl10k-yolo/images/train/../a.jpg", row["image"].replace("/", "\\")):
            with self.assertRaises(ValueError): safe_path(root, path, prefix)

    def test_native_registration_checks_exif_corrected_dimensions(self):
        root, output, row, doc = self.fixture_files(); annotation = root / row["annotation"]
        source = native_source(doc, annotation, root)
        self.assertEqual(registered_native_size(source, doc)[0], (128, 96))
        exif = Image.Exif(); exif[274] = 6; Image.new("RGB", (96, 128)).save(source, exif=exif)
        self.assertEqual(registered_native_size(source, doc), ((128, 96), 6))
        with self.assertRaises(ValueError): registered_native_size(source, {**doc, "imageHeight": 128})

    def test_native_registration_rejects_test_or_unmatched_basename(self):
        root, output, row, doc = self.fixture_files(); annotation = root / row["annotation"]
        for broken in ({**doc, "split": "test"}, {**doc, "imagePath": "images/test/example.jpg"},
                       {**doc, "imageName": "other.jpg"}, {**doc, "dacl10k_version": "v1"}):
            with self.assertRaises(ValueError): native_source(broken, annotation, root)

    def test_written_npz_schema_preserves_photo_tags_and_input_bytes(self):
        root, output, row, doc = self.fixture_files(); original_row = deepcopy(row)
        originals = {p: (sha(p), p.stat().st_mtime_ns) for p in (root / row["image"], root / row["annotation"])}
        result, cells, before = prepare_row((0, row, root, output))
        self.assertEqual(row, original_row); self.assertEqual(result["targets"], row["targets"])
        self.assertEqual(result["annotation_sha256"], row["annotation_sha256"])
        with np.load(root / result["mask"], allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"masks", "known"}); validate_arrays(data["masks"], data["known"], row["targets"])
            self.assertEqual(data["known"].tolist(), result["known"])
        self.assertEqual(cells.shape, (19, 3)); self.assertTrue((cells.sum(1) == 6400).all())
        self.assertEqual(originals, {p: (sha(p), p.stat().st_mtime_ns) for p in originals})
        with self.assertRaises(FileExistsError): prepare_row((0, row, root, output))

    def test_same_raster_recipe_as_original_seven_channel_producer(self):
        from scripts import prepare_facility_spatial as original
        root, output, row, doc = self.fixture_files(); classes = ["concrete_crack", "concrete_spalling", "rust_stain", "exposed_rebar", "wet_surface", "efflorescence", "surface_cavity"]
        parent = {"image": row["image"], "domain": "dacl"}; record = {"source": "dacl10k/dacl10k_v2_devphase/images/train/example.jpg"}
        item = {"targets": [0, 1, 0, 0, 0, 0, 0]}; previous = original.ROOT
        try:
            original.ROOT = root
            masks7, known7, conflicts = original.masks_for(item, parent, record, classes)
        finally:
            original.ROOT = previous
        masks19, known19, _ = rasterize_dense(doc, row["targets"], (128, 96))
        self.assertTrue(np.array_equal(masks7[1], masks19[AUX_CLASSES.index("Spalling")]))
        self.assertEqual(int(known7[1]), int(known19[AUX_CLASSES.index("Spalling")]))
        self.assertEqual(conflicts, 0)

    def test_json_output_is_utf8_lf_and_never_overwrites(self):
        root, output, row, doc = self.fixture_files(); path = output / "format.json"
        write_new(path, {"text": "위치 정답"}); raw = path.read_bytes()
        self.assertNotIn(b"\r", raw); self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(json.loads(raw), {"text": "위치 정답"})
        with self.assertRaises(FileExistsError): write_new(path, {})


if __name__ == "__main__":
    unittest.main()
