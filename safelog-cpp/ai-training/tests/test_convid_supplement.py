import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.prepare_convid_supplement import (
    canonical_bytes, download_original, fixed_picks, inspect_photo, official_index,
    positive_targets, SOURCE_URL,
)
from scripts.prepare_peccd_supplement import screen_source, unknown_pixel_target


IDS = ("00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002")


def metadata():
    folders = [{"name": name, "id": identity} for name, identity in zip(("crack", "Spalling"), IDS)]
    files = {}
    for i, folder in enumerate(folders, 10):
        identity = f"00000000-0000-0000-0000-{i:012d}"
        files[folder["name"]] = [{"id": identity, "filename": "photo.jpg", "folder_id": folder["id"], "status": "COMPLETED", "size": 3,
                                  "content_details": {"sha256_hash": "a" * 64, "content_type": "image/jpeg", "size": 3,
                                                      "download_url": f"https://data.mendeley.com/public-files/datasets/fx3rthfjhy/files/{identity}/file_downloaded"}}]
    return folders, files


class ConvidTests(unittest.TestCase):
    def test_exact_author_folder_asserts_only_one_positive_and_six_unknown(self):
        self.assertEqual(positive_targets("crack"), [1, -1, -1, -1, -1, -1, -1])
        self.assertEqual(positive_targets("Spalling"), [-1, 1, -1, -1, -1, -1, -1])
        for invalid in ("Crack", "spalling", "Scaling", "VOID", "Honeycomb"):
            with self.assertRaises(ValueError):
                positive_targets(invalid)

    def test_official_index_requires_exact_folder_identity_sha_and_public_url(self):
        folders, files = metadata()
        document = official_index(folders, files)
        self.assertEqual(document["dataset_url"], SOURCE_URL)
        self.assertEqual(len(document["records"]), 2)
        self.assertEqual(canonical_bytes(document), canonical_bytes(dict(reversed(list(document.items())))))
        for field, invalid in (("folder_id", IDS[1]), ("status", "PARTIAL"), ("filename", "../x.jpg")):
            mutated = copy.deepcopy(files)
            mutated["crack"][0][field] = invalid
            with self.assertRaises(ValueError):
                official_index(folders, mutated)
        for field, invalid in (("sha256_hash", "no-official-hash"), ("download_url", "https://elsewhere.example/x"), ("content_type", "application/zip")):
            mutated = copy.deepcopy(files)
            mutated["crack"][0]["content_details"][field] = invalid
            with self.assertRaises(ValueError):
                official_index(folders, mutated)

    def test_picks_are_source_only_fixed_seed_100_per_folder_not_refilled(self):
        rows = [{"folder": folder, "id": str(i)} for folder in ("crack", "Spalling") for i in range(150)]
        picked = fixed_picks(rows)
        self.assertEqual(picked, fixed_picks(list(reversed(rows))))
        self.assertEqual(len(picked), 200)
        self.assertEqual(sum(row["folder"] == "crack" for row in picked), 100)
        self.assertEqual(sum(row["folder"] == "Spalling" for row in picked), 100)
        # A source shortfall stays a shortfall; the other source is never oversampled.
        short = fixed_picks([row for row in rows if row["folder"] == "Spalling" or int(row["id"]) < 3])
        self.assertEqual(len(short), 103)

    def test_download_checks_individual_bytes_hash_and_never_uses_partial(self):
        _, files = metadata()
        row = official_index(*metadata())["records"][0]
        body = b"verified source"
        row.update({"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def raise_for_status(self): pass
            def iter_content(self, *_): return iter([body])
        with tempfile.TemporaryDirectory() as directory, patch("scripts.prepare_convid_supplement.requests.get", return_value=Response()):
            path = download_original((row, Path(directory)))
            self.assertEqual(path.read_bytes(), body)
            with patch("scripts.prepare_convid_supplement.requests.get") as get:
                self.assertEqual(download_original((row, Path(directory))), path)
                get.assert_not_called()
            row["sha256"] = "b" * 64
            with self.assertRaises(ValueError):
                download_original((row, Path(directory)))
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_exif_full_photo_corrected_before_hash_without_changing_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (IDS[0] + ".jpg")
            image = Image.new("RGB", (40, 20), "red")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            row = {"id": IDS[0], "folder": "crack", "filename": "native.jpg", "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            result = inspect_photo(row, Path(directory))
            self.assertEqual(result["source_size"], [20, 40])
            self.assertTrue(result["exif_orientation_corrected"])
            self.assertEqual(result["targets"], [1] + [-1] * 6)

    def test_whole_photo_overlap_excludes_group_and_collapses_no_label_propagation(self):
        def row(name, digest, fingerprint, folder):
            return {"source_image": name, "pixel_sha256": digest, "dhash": fingerprint,
                    "targets": positive_targets(folder), "suspected_derived_filename": False}
        rows = [row("b.jpg", "b", 0, "crack"), row("a.jpg", "a", 1, "crack"), row("c.jpg", "c", (1 << 64) - 1, "Spalling")]
        kept, pending, audit = screen_source(rows, [], np.array([], dtype=np.uint64), "convid-near:")
        self.assertEqual([r["source_image"] for r in kept], ["a.jpg", "c.jpg"])
        self.assertTrue(kept[0]["group_id"].startswith("convid-near:"))
        self.assertEqual(kept[0]["targets"], [1] + [-1] * 6)
        kept, pending, _ = screen_source(rows, [{"pixel_sha256": "b"}], np.array([], dtype=np.uint64), "convid-near:")
        self.assertEqual([r["source_image"] for r in kept], ["c.jpg"])
        self.assertEqual(len(pending), 2)

    def test_all_spatial_targets_unknown_despite_known_folder_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.npz"
            unknown_pixel_target(path)
            with np.load(path, allow_pickle=False) as values:
                self.assertEqual(values["mask"].shape, (7, 80, 80))
                self.assertEqual(values["known"].tolist(), [0] * 7)
                self.assertEqual(int(values["mask"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
