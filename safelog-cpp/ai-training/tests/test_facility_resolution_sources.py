"""Synthetic TRAIN provenance and privacy guards; never open real datasets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageOps

from scripts.audit_facility_resolution_sources import (
    AUX_CLASSES, CLASSES, corrected_dimensions, digest, inspect_reference,
    native_targets, preserve_inputs, public_summary, run, select_train_references,
)


class ResolutionSourceAuditTests(unittest.TestCase):
    def mapping(self):
        records = [{"stem": "a", "split": "train"}, {"stem": "b", "split": "train"},
                   {"stem": "heldout-record-not-opened", "split": "test"}]
        auxiliary = {"split": "train", "classes": AUX_CLASSES,
                     "items": [{"image": f"data/private/images/train/{stem}.jpg", "split": "train",
                                "domain": "dacl", "targets": [0] * 19} for stem in ("a", "b")]}
        spatial = {"split": "train", "classes": CLASSES, "full_count": 2,
                   "items": [{"image": f"data/private/images/train/{stem}.jpg",
                              "domain": "dacl", "targets": [0] * 7} for stem in ("a", "b")]}
        return records, auxiliary, spatial

    @staticmethod
    def document(width=6, height=4):
        return {"imageWidth": width, "imageHeight": height,
                "shapes": [{"label": "Crack", "shape_type": "polygon",
                            "points": [[0, 0], [2, 0], [1, 2]]}]}

    def reference(self, root, exif=False):
        source = root / "data/native/images/train/private-case.jpg"
        processed = root / "data/processed/images/train/private-case.jpg"
        annotation = root / "data/native/annotations/train/private-case.json"
        for path in (source, processed, annotation): path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (6, 4), "gray")
        if exif:
            metadata = Image.Exif(); metadata[274] = 6
            image.save(source, exif=metadata)
        else: image.save(source)
        Image.new("RGB", (3, 2), "gray").save(processed)
        annotation.write_text(json.dumps(self.document(4, 6) if exif else self.document()), encoding="utf-8")
        with Image.open(source) as handle:
            corrected = ImageOps.exif_transpose(handle).convert("RGB")
        import hashlib
        pixel_hash = hashlib.sha256(f"{corrected.width}x{corrected.height}:".encode() + corrected.tobytes()).hexdigest()
        auxiliary_targets = [int(label == "Crack") for label in AUX_CLASSES]
        return ("private-case", {"split": "train", "source": "native/images/train/private-case.jpg",
                                 "file_sha256": digest(source), "pixel_sha256": pixel_hash},
                {"annotation": annotation.relative_to(root).as_posix(), "annotation_sha256": digest(annotation),
                 "targets": auxiliary_targets},
                {"image": processed.relative_to(root).as_posix(), "targets": [1, 0, 0, 0, 0, 0, 0]})

    def test_native_polygon_mapping_preserves_nineteen_tags_and_combines_only_crack_acrack(self):
        doc = self.document()
        doc["shapes"].extend([{"label": "ACrack", "points": [[0, 0], [1, 0], [1, 1]]},
                              {"label": "Hollowareas", "points": [[1, 1], [2, 1], [2, 2]]}])
        aux, core, counts = native_targets(doc)
        self.assertEqual(sum(aux), 3)
        self.assertEqual(core, [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(counts, {"Crack": 1, "ACrack": 1, "Hollowareas": 1})
        for mutation in ("unknown_label", "nan", "short_polygon", "wrong_shape", "bool_dimension"):
            invalid = copy.deepcopy(doc)
            if mutation == "unknown_label": invalid["shapes"][0]["label"] = "safe"
            elif mutation == "nan": invalid["shapes"][0]["points"][0][0] = float("nan")
            elif mutation == "short_polygon": invalid["shapes"][0]["points"].pop()
            elif mutation == "wrong_shape": invalid["shapes"][0]["shape_type"] = "rectangle"
            else: invalid["imageWidth"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): native_targets(invalid)

    def test_reference_selection_ignores_heldout_metadata_but_checks_exact_train_sets_order_types(self):
        records, auxiliary, spatial = self.mapping()
        result = select_train_references(records, auxiliary, spatial, expected_count=2)
        self.assertEqual([row[0] for row in result], ["a", "b"])
        for mutation in ("heldout_aux", "duplicate_source", "reordered_aux", "unknown_core", "bool_aux", "wrong_classes"):
            r, a, s = map(copy.deepcopy, (records, auxiliary, spatial))
            if mutation == "heldout_aux": a["items"][0]["split"] = "val"
            elif mutation == "duplicate_source": r[1]["stem"] = "a"
            elif mutation == "reordered_aux": a["items"].reverse()
            elif mutation == "unknown_core": s["items"][0]["targets"][0] = -1
            elif mutation == "bool_aux": a["items"][0]["targets"][0] = False
            else: s["classes"] = list(reversed(CLASSES))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select_train_references(r, a, s, expected_count=2)

    def test_actual_source_file_annotation_hash_and_label_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference = self.reference(root)
            row = inspect_reference(reference, root)
            self.assertEqual(row["source_size"], [6, 4])
            self.assertEqual(row["processed_size"], [3, 2])
            self.assertEqual(row["core_targets"], [1, 0, 0, 0, 0, 0, 0])
            for mutation in ("source_hash", "annotation_hash", "core_targets", "source_escape", "heldout_path"):
                changed = copy.deepcopy(reference)
                if mutation == "source_hash": changed[1]["file_sha256"] = "a" * 64
                elif mutation == "annotation_hash": changed[2]["annotation_sha256"] = "b" * 64
                elif mutation == "core_targets": changed[3]["targets"][0] = 0
                elif mutation == "source_escape": changed[1]["source"] = "../../outside.jpg"
                else: changed[1]["source"] = "native/images/test/private-case.jpg"
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): inspect_reference(changed, root)

    def test_exif_correction_checks_actual_decoded_pixel_hash_against_native_dimensions(self):
        self.assertEqual(corrected_dimensions((6, 4), 6), (4, 6))
        self.assertEqual(corrected_dimensions((6, 4), 2), (6, 4))
        with self.assertRaises(ValueError): corrected_dimensions((6, 4), 9)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference = self.reference(root, exif=True)
            row = inspect_reference(reference, root)
            self.assertTrue(row["exif_corrected_pixel_hash_verified"])
            self.assertEqual(row["source_size"], [4, 6])
            changed = copy.deepcopy(reference); changed[1]["pixel_sha256"] = "c" * 64
            with self.assertRaises(ValueError): inspect_reference(changed, root)

    def test_content_size_or_mtime_change_fails_preservation_check(self):
        before = {"private.json": {"sha256": "a", "size_bytes": 1, "mtime_ns": 2}}
        preserve_inputs(before, copy.deepcopy(before))
        for field, value in (("sha256", "b"), ("size_bytes", 2), ("mtime_ns", 3)):
            after = copy.deepcopy(before); after["private.json"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): preserve_inputs(before, after)

    def test_existing_or_unconfined_outputs_fail_before_any_real_input_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report = root / "reports/existing.json"
            report.parent.mkdir(); report.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError): run(root, report, root / "runs/new.json")
            self.assertEqual(report.read_text(encoding="utf-8"), "preserve")
            with self.assertRaises(ValueError): run(root, root / "reports/new.json", root / "data/private.json")
            self.assertFalse((root / "reports/new.json").exists())

    def test_public_summary_excludes_individual_ids_paths_and_annotation_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); row = inspect_reference(self.reference(root), root)
            summary = public_summary([row], {"records": {"sha256": "a" * 64}},
                                     {"items": [row], "full_count": 1, "audit": {"grid_size": 80}},
                                     {"items": []}, {"max_side": 1280})
            public = json.dumps(summary)
            for sensitive in (row["stem"], row["source_image"], row["processed_image"], row["native_annotation"], str(root)):
                self.assertNotIn(sensitive, public)
            self.assertFalse(summary["individual_paths_or_annotations_published"])
            self.assertFalse(summary["current_resolution_comparison"]["new_native_source_feeding_performed"])
            self.assertEqual(summary["label_changes"], 0)


if __name__ == "__main__":
    unittest.main()
