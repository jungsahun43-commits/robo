"""Synthetic native ROI geometry/provenance tests; never open actual source photos."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from scripts import prepare_facility_native_roi as prep
from scripts.audit_facility_resolution_sources import inspect_reference


class NativeRoiDataTests(unittest.TestCase):
    def asymmetric(self, size=(11, 7)):
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        rgb = np.stack(((x * 31 + y * 7) % 256, (x * 17 + y * 47) % 256,
                        (x * 59 + y * 13 + (x % 2) * 91) % 256), axis=-1).astype(np.uint8)
        return Image.fromarray(rgb, "RGB")

    def fixture(self, root, *, exif=False, fullsize=True):
        stem = "private-case"
        source = root / "data/native/images/train/private-case.jpg"
        processed = root / "data/processed/images/train/private-case.jpg"
        annotation = root / "data/native/annotations/train/private-case.json"
        for path in (source, processed, annotation): path.parent.mkdir(parents=True, exist_ok=True)
        image = self.asymmetric((1537, 1025) if fullsize else (6, 4))
        if exif:
            metadata = Image.Exif(); metadata[274] = 6
            image.save(source, quality=95, exif=metadata)
        else:
            image.save(source, quality=95)
        image.close()
        with Image.open(source) as handle:
            native = ImageOps.exif_transpose(handle).convert("RGB")
        pixel = prep.pixel_sha(native)
        reduced = native.copy(); reduced.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        reduced.save(processed, quality=95)
        width, height = native.size; reduced_size = reduced.size
        native.close(); reduced.close()
        document = {"imageWidth": width, "imageHeight": height,
                    "shapes": [{"label": "Crack", "shape_type": "polygon",
                                "points": [[0, 0], [2, 0], [1, 2]]}]}
        annotation.write_text(json.dumps(document), encoding="utf-8")
        core = {"image": processed.relative_to(root).as_posix(), "targets": [1, 0, 0, 0, 0, 0, 0],
                "domain": "dacl", "pixel_target": "data/masks/00000.npz"}
        aux = {"image": core["image"], "targets": [int(c == "Crack") for c in prep.AUX_CLASSES],
               "split": "train", "domain": "dacl", "annotation": annotation.relative_to(root).as_posix(),
               "annotation_sha256": prep.file_sha(annotation)}
        record = {"stem": stem, "split": "train", "source": source.relative_to(root / "data").as_posix(),
                  "file_sha256": prep.file_sha(source), "pixel_sha256": pixel}
        masks = root / "data/masks"; masks.mkdir()
        np.savez_compressed(masks / "00000.npz", mask=np.zeros((7, 80, 80), dtype=np.uint8), known=np.ones(7, dtype=np.uint8))
        rows = [core]; details = []
        for index, targets in enumerate(([1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 1, 0, 0, 0, 0, 0]), 1):
            original = root / f"data/crops/private-case-{index}.jpg"; original.parent.mkdir(exist_ok=True)
            Image.new("RGB", (9, 7), (index * 40, 30, 50)).save(original)
            box = [0, 0, reduced_size[0], reduced_size[1]] if not fullsize else [index, index, 426 + index, 284 + index]
            detail = {"image": original.relative_to(root).as_posix(), "targets": list(targets), "domain": "dacl",
                      "parent_id": stem, "parent_image": core["image"], "parent_split": "train",
                      "box_in_parent": box, "augmentation": "publisher mask/polygon derived crop"}
            row = {**detail, "pixel_target": f"data/masks/{index:05d}.npz"}
            np.savez_compressed(masks / f"{index:05d}.npz", mask=np.zeros((7, 80, 80), dtype=np.uint8), known=np.ones(7, dtype=np.uint8))
            details.append(detail); rows.append(row)
        base = {"split": "train", "classes": list(prep.CLASSES), "full_count": 1, "items": rows,
                "audit": {"grid_size": 80, "per_label_pixel_cells": {
                    label: {"positive": 0, "negative": 25600} for label in prep.CLASSES}, "policy": "original"}}
        auxiliary = {"split": "train", "classes": list(prep.AUX_CLASSES), "items": [aux]}
        records = [record, {"stem": "heldout-not-opened", "split": "test", "source": "must-never-be-opened.jpg"}]
        documents = {"spatial": base, "detail": {"split": "train", "classes": list(prep.CLASSES), "items": details},
                     "preparation": {"max_side": 1280}, "records": records, "auxiliary": auxiliary}
        for name, relative in prep.MANIFESTS.items():
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(documents[name]), encoding="utf-8")
        reference = (stem, record, aux, core)
        inspected = inspect_reference(reference, root)
        inventory = {"schema": "facility_resolution_source_inventory_v1", "local_only": True,
                     "split": "train", "all_inputs_preserved": True, "rows": [inspected]}
        inventory_path = root / prep.SOURCE_INVENTORY; inventory_path.parent.mkdir(parents=True)
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        audit = {"schema": "facility_resolution_source_audit_v1", "status": "complete",
                 "all_input_sha256_size_mtime_preserved": True, "classes": list(prep.CLASSES),
                 "native_classes": list(prep.AUX_CLASSES), "preparation_max_side": 1280, "train_photos": 1,
                 "private_inventory_sha256": prep.file_sha(inventory_path),
                 "manifest_sha256": {name: prep.file_sha(root / relative) for name, relative in prep.MANIFESTS.items()}}
        for key in ("verified_source_file_hashes", "verified_annotation_hashes", "verified_core7_photo_targets",
                    "verified_native19_photo_targets", "verified_exif_corrected_annotation_dimensions"):
            audit[key] = 1
        audit_path = root / prep.SOURCE_AUDIT; audit_path.parent.mkdir()
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return base, reference, inspected, documents, audit, inventory

    def pairs(self, base):
        manifests = {v: copy.deepcopy(base) for v in ("control", "native")}
        for variant, manifest in manifests.items():
            for index in range(base["full_count"], len(base["items"])):
                manifest["items"][index]["image"] = f"{prep.PAIR_ROOT}/{variant}/images/roi-{index:05d}.png"
            manifest["native_roi"] = {"schema": "facility_native_roi_manifest_v1", "variant": variant,
                                      "recipe": copy.deepcopy(prep.RECIPE), "base_manifest_sha256": "a" * 64}
        return manifests["control"], manifests["native"]

    def test_asymmetric_float_box_matches_exact_lanczos_recipe_without_rounding(self):
        image = self.asymmetric(); box = [1, 1, 4, 3]; processed_size = (5, 4)
        left, right, projected = prep.render_pair(image, processed_size, box)
        self.assertEqual(projected, [2.2, 1.75, 8.8, 5.25])
        expected_left = image.resize(processed_size, Image.Resampling.LANCZOS, reducing_gap=None).resize(
            (640, 640), Image.Resampling.LANCZOS, box=tuple(box), reducing_gap=None)
        expected_right = image.resize((640, 640), Image.Resampling.LANCZOS, box=tuple(projected), reducing_gap=None)
        rounded = image.resize((640, 640), Image.Resampling.LANCZOS, box=tuple(round(v) for v in projected), reducing_gap=None)
        self.assertEqual(left.tobytes(), expected_left.tobytes())
        self.assertEqual(right.tobytes(), expected_right.tobytes())
        self.assertNotEqual(right.tobytes(), rounded.tobytes())
        self.assertNotEqual(left.tobytes(), right.tobytes())
        self.assertEqual(left.size, right.size); self.assertEqual(right.size, (640, 640))
        cached = image.resize(processed_size, Image.Resampling.LANCZOS, reducing_gap=None)
        cached_left, cached_right, _ = prep.render_pair(image, processed_size, box, pre_downsampled_rgb=cached)
        self.assertEqual(cached_left.tobytes(), left.tobytes()); self.assertEqual(cached_right.tobytes(), right.tobytes())
        self.assertEqual(cached.size, processed_size)  # Caller cache is not closed by a crop.
        for value in (cached, cached_left, cached_right): value.close()
        for value in (image, left, right, expected_left, expected_right, rounded): value.close()

    def test_equal_dimensions_are_exactly_equal_and_eligibility_requires_both_axes(self):
        with self.asymmetric() as image:
            left, right, projected = prep.render_pair(image, image.size, [1, 1, 10, 6])
            self.assertEqual(left.tobytes(), right.tobytes()); self.assertEqual(projected, [1., 1., 10., 6.])
            left.close(); right.close()
        self.assertFalse(prep.eligible((11, 7), (11, 7)))
        self.assertTrue(prep.eligible((11, 7), (11, 6)))
        self.assertFalse(prep.eligible((11, 7), (10, 8)))
        for size in ((True, 7), (11, 0), (11,), (11., 7)):
            with self.subTest(size=size), self.assertRaises(ValueError): prep.eligible(size, (5, 4))

    def test_noninteger_out_of_bounds_and_degenerate_boxes_are_rejected(self):
        with self.asymmetric() as image:
            for box in ([-1, 0, 4, 3], [0, 0, 6, 3], [0, 0, 4, 5], [2, 1, 2, 3],
                        [0, 3, 2, 1], [1., 1, 4, 3], [True, 1, 4, 3], [0, 0, 4], [0, 0, float("nan"), 3]):
                with self.subTest(box=box), self.assertRaises(ValueError): prep.render_pair(image, (5, 4), box)
            with self.assertRaises(ValueError): prep.render_pair(image, (12, 7), [0, 0, 5, 4])
            with self.assertRaises(ValueError): prep.render_pair(image.convert("L"), (5, 4), [0, 0, 4, 3])

    def test_pair_preserves_all_targets_unknown_rows_masks_full_rows_metadata_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            base, *_ = self.fixture(Path(directory)); left, right = self.pairs(base)
            before = copy.deepcopy((base, left, right)); result = prep.validate_pair(base, left, right)
            self.assertEqual(result["changed_rows"], 3); self.assertEqual(result["changed_dacl_parents"], 1)
            self.assertTrue(result["labels_and_masks_unchanged"]); self.assertEqual((base, left, right), before)
            self.assertEqual(right["items"][3]["targets"][0], -1)
            for mutation in ("full_image", "targets", "mask", "box", "parent", "augmentation", "order",
                             "one_arm", "audit", "unknown_aux", "wrong_recipe", "escape", "val", "bool_label",
                             "base_digest", "one_provenance"):
                c, n = copy.deepcopy(left), copy.deepcopy(right)
                if mutation == "full_image": c["items"][0]["image"] = "data/new.png"
                elif mutation == "targets": n["items"][1]["targets"][0] = 0
                elif mutation == "mask": n["items"][1]["pixel_target"] = "data/new.npz"
                elif mutation == "box": n["items"][1]["box_in_parent"][0] += 1
                elif mutation == "parent": n["items"][1]["parent_split"] = "val"
                elif mutation == "augmentation": n["items"][1]["augmentation"] = "new truth"
                elif mutation == "order": n["items"][1], n["items"][2] = n["items"][2], n["items"][1]
                elif mutation == "one_arm": n["items"][1]["image"] = base["items"][1]["image"]
                elif mutation == "audit": n["audit"]["per_label_pixel_cells"][prep.CLASSES[0]]["positive"] += 1
                elif mutation == "unknown_aux": n["items"][1]["auxiliary_targets"] = [0] * 19
                elif mutation == "wrong_recipe": n["native_roi"]["recipe"]["reducing_gap"] = 2
                elif mutation == "escape": n["items"][1]["image"] = "../private.png"
                elif mutation == "val": n["split"] = "val"
                elif mutation == "bool_label": n["items"][1]["targets"][0] = True
                elif mutation == "base_digest": n["native_roi"]["base_manifest_sha256"] = "b" * 64
                elif mutation == "one_provenance": n.pop("native_roi")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): prep.validate_pair(base, c, n)

    def test_source_audit_binding_checks_train_membership_manifests_inventory_and_crop_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, _, _, documents, audit, inventory = self.fixture(root)
            hashes = {name: prep.snapshot(root / relative) for name, relative in prep.MANIFESTS.items()}
            sha = prep.file_sha(root / prep.SOURCE_INVENTORY)
            refs, _ = prep.validate_audit(audit, inventory, documents, hashes, sha)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0][1]["split"], "train")
            for mutation in ("unprepared", "manifest_digest", "inventory_digest", "inventory_split",
                             "heldout_aux", "heldout_source", "core_label", "row_order", "source_mapping"):
                a, i, d = map(copy.deepcopy, (audit, inventory, documents))
                if mutation == "unprepared": a["status"] = "running"
                elif mutation == "manifest_digest": a["manifest_sha256"]["spatial"] = "0" * 64
                elif mutation == "inventory_digest": a["private_inventory_sha256"] = "0" * 64
                elif mutation == "inventory_split": i["rows"][0]["split"] = "test"
                elif mutation == "heldout_aux": d["auxiliary"]["items"][0]["split"] = "val"
                elif mutation == "heldout_source": d["records"][0]["split"] = "val"
                elif mutation == "core_label": d["spatial"]["items"][0]["targets"][0] = -1
                elif mutation == "row_order": d["detail"]["items"].reverse()
                elif mutation == "source_mapping": i["rows"][0]["source_image"] = "data/heldout.jpg"
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): prep.validate_audit(a, i, d, hashes, sha)

    def test_exif_corrected_native_raster_and_annotation_dimensions_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, reference, audited, *_ = self.fixture(root, exif=True, fullsize=False)
            result = prep.inspect_and_render(reference, audited, [], root, 1280)
            self.assertTrue(result["exif_corrected"]); self.assertEqual(result["source_size"], [4, 6])
            self.assertFalse(result["eligible"]); self.assertEqual(result["changed_rows"], [])
            broken = copy.deepcopy(reference); broken[1]["pixel_sha256"] = "0" * 64
            with self.assertRaises(ValueError): prep.inspect_and_render(broken, audited, [], root, 1280)

    def test_parent_source_annotation_processed_hash_dimension_and_bounds_drift_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, reference, audited, *_ = self.fixture(root)
            for mutation in ("source_hash", "annotation_hash", "native_pixels", "core_tags", "aux_tags",
                             "source_dimensions", "processed_dimensions", "heldout", "box"):
                ref = copy.deepcopy(reference); ledger = copy.deepcopy(audited)
                crops = [(1, copy.deepcopy(base["items"][1]))]
                if mutation == "source_hash": ref[1]["file_sha256"] = "0" * 64
                elif mutation == "annotation_hash": ref[2]["annotation_sha256"] = "0" * 64
                elif mutation == "native_pixels": ref[1]["pixel_sha256"] = "0" * 64
                elif mutation == "core_tags": ref[3]["targets"][0] = 0
                elif mutation == "aux_tags": ref[2]["targets"][0] = 1
                elif mutation == "source_dimensions": ledger["source_size"][0] -= 1
                elif mutation == "processed_dimensions": ledger["processed_size"][0] -= 1
                elif mutation == "heldout": ref[1]["split"] = "test"
                elif mutation == "box": crops[0][1]["box_in_parent"] = [0, 0, 2000, 3]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    prep.inspect_and_render(ref, ledger, crops, root, 1280)

    def test_end_to_end_synthetic_pair_is_lossless_private_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, _, _, documents, *_ = self.fixture(root)
            original_mask_sha = {r["pixel_target"]: prep.file_sha(root / r["pixel_target"]) for r in base["items"]}
            before = prep.file_sha(root / prep.MANIFESTS["spatial"])
            with patch.object(prep, "render_pair", wraps=prep.render_pair) as renders:
                summary = prep.run(root, workers=1)
            self.assertEqual(renders.call_count, 3)
            caches = [call.kwargs["pre_downsampled_rgb"] for call in renders.call_args_list]
            self.assertTrue(all(value is caches[0] for value in caches))
            self.assertEqual(summary["changed_rows"], 3); self.assertEqual(summary["derived_pngs"], 6)
            self.assertEqual(summary["control_parent_resizes"], 1); self.assertEqual(summary["worker_count"], 1)
            self.assertGreater(summary["preparation_elapsed_minutes"], 0)
            self.assertEqual(summary["new_photo_targets"], 0); self.assertEqual(summary["label_changes"], 0)
            self.assertEqual(summary["changed_original_target_counts"][prep.CLASSES[0]], {"positive": 1, "negative": 1, "unknown": 1})
            self.assertEqual(prep.file_sha(root / prep.MANIFESTS["spatial"]), before)
            manifests = {v: prep.read(root / prep.PAIR_ROOT / v / "train.json") for v in ("control", "native")}
            prep.validate_pair(base, manifests["control"], manifests["native"])
            ledger = prep.read(root / prep.LEDGER_PATH)
            self.assertTrue(ledger["local_only"]); self.assertEqual(ledger["input_snapshots_before"], ledger["input_snapshots_after"])
            self.assertEqual(summary["private_ledger_sha256"], prep.file_sha(root / prep.LEDGER_PATH))
            lookup = {r["image"]: r["targets"] for r in documents["auxiliary"]["items"]}
            for row in ledger["changed_rows"]:
                self.assertEqual(prep.file_sha(root / row["pixel_target"]), original_mask_sha[row["pixel_target"]])
                for variant in ("control", "native"):
                    item = row[variant]
                    self.assertEqual(prep.file_sha(root / item["image"]), item["sha256"])
                    with Image.open(root / item["image"]) as image:
                        self.assertEqual(image.format, "PNG"); self.assertEqual(image.size, (640, 640))
                        self.assertEqual(image.info, {}); self.assertEqual(len(image.getexif()), 0)
                        self.assertEqual(prep.pixel_sha(image), item["pixel_sha256"])
                    self.assertEqual(lookup.get(item["image"], [-1] * 19), [-1] * 19)
            expected_bytes = {v: sum(row[v]["size_bytes"] for row in ledger["changed_rows"]) for v in ("control", "native")}
            self.assertEqual(summary["derived_png_bytes_by_variant"], expected_bytes)
            self.assertEqual(summary["derived_png_bytes_total"], sum(expected_bytes.values()))
            serialized = json.dumps(summary)
            for private in ("private-case", "heldout-not-opened", str(root), "data/native/images", "data/masks",
                            "row_index", "original_image", "parent_image", "box_in_parent", "native_box"):
                self.assertNotIn(private, serialized)
            with self.assertRaises(ValueError): prep.run(root, workers=1)

    def test_input_mutation_during_preparation_rejects_without_publishing_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            actual = prep.snapshot; calls = {}
            def changed_snapshot(path):
                path = Path(path); calls[path] = calls.get(path, 0) + 1
                if path.name == "00001.npz" and calls[path] == 2:
                    with path.open("ab") as stream: stream.write(b"synthetic mutation")
                return actual(path)
            with patch.object(prep, "snapshot", side_effect=changed_snapshot), self.assertRaises(ValueError):
                prep.run(root, workers=1)
            self.assertFalse((root / prep.REPORT_PATH).exists())
            self.assertFalse((root / prep.LEDGER_PATH).exists())
            self.assertTrue((root / prep.PAIR_ROOT).exists())  # Partial evidence retained, never silently overwritten.

    def test_unsafe_paths_workers_and_existing_output_are_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("../outside", "/outside", "data/../outside", "C:/outside", "data\\outside"):
                with self.subTest(value=value), self.assertRaises(ValueError): prep.safe_path(root, value, exists=False)
            for workers in (0, 5, True, 1.5):
                with self.subTest(workers=workers), self.assertRaises(ValueError): prep.run(root, workers)
            (root / prep.PAIR_ROOT).mkdir(parents=True)
            with self.assertRaises(ValueError): prep.run(root, 4)


if __name__ == "__main__":
    unittest.main()
