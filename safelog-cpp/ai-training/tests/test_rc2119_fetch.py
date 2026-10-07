"""Corrupt/untrusted archives must not become verified training sources."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.fetch_rc2119 import (PINNED, ROOT, download_archive, extract_new, inventory,
                                  official_url, safe_name, validate_metadata, verified_archive)


@contextmanager
def workspace_temp():
    root = (ROOT / "runs/rc2119-unit-temp").resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as base:
        if not Path(base).resolve().is_relative_to(root):
            raise ValueError("Test cleanup target leaves intended temporary root")
        yield base


class AcquisitionTests(unittest.TestCase):
    def make_zip(self, base, names):
        path = Path(base) / "sample.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, b"example")
        return path

    def test_paths_reject_windows_and_traversal(self):
        for value in ("../x", "/x", "C:/x", "a\\x", "a//x", "CON.txt", "x.", "a/.. /x", "x:stream"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_name(value)
        self.assertEqual(safe_name("imagedata/12.png"), "imagedata/12.png")

    def test_case_collision_fails_before_extraction(self):
        with workspace_temp() as base:
            archive = self.make_zip(base, ["a/IMAGE.png", "a/image.png"])
            with self.assertRaises(ValueError):
                inventory(archive)

    def test_symlink_fails_before_extraction(self):
        with workspace_temp() as base:
            path = Path(base) / "link.zip"
            info = zipfile.ZipInfo("link"); info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(info, "../outside")
            with self.assertRaises(ValueError):
                inventory(path)

    def test_file_parent_collision_fails(self):
        with workspace_temp() as base:
            path = self.make_zip(base, ["a", "a/image.png"])
            with self.assertRaises(ValueError):
                inventory(path)

    def test_safe_full_extract_and_no_overwrite(self):
        with workspace_temp() as base:
            path = self.make_zip(base, ["imagedata/12.png"])
            root = Path(base) / "source"; dest = root / "image"
            rows = inventory(path); files = extract_new(path, dest, rows, root)
            self.assertEqual((dest / "imagedata/12.png").read_bytes(), b"example")
            self.assertEqual(files[0]["sha256"], hashlib.sha256(b"example").hexdigest())
            with self.assertRaises(ValueError):
                extract_new(path, dest, rows, root)
            with self.assertRaises(ValueError):
                extract_new(path, Path(base) / "outside", rows, root)

    def test_wrong_digest_and_truncation_rejected(self):
        with workspace_temp() as base:
            path = Path(base) / "a.zip"; path.write_bytes(b"bad")
            for pin in (("id", 3, "0" * 64, "x"), ("id", 4, hashlib.sha256(b"bad").hexdigest(), "x")):
                with self.subTest(pin=pin), self.assertRaises(ValueError):
                    verified_archive(path, pin)

    def test_failed_download_never_creates_original(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): yield b"too long"
        with workspace_temp() as base, patch("scripts.fetch_rc2119.requests.get", return_value=Response()):
            path = Path(base) / "a.zip"
            with self.assertRaises(ValueError):
                download_archive(path, ("id", 2, "0" * 64, "x"))
            self.assertFalse(path.exists())

    def test_metadata_binds_all_three_official_files(self):
        rows = [{"filename": name, "id": pin[0], "size": pin[1], "status": "COMPLETED",
                 "content_details": {"size": pin[1], "sha256_hash": pin[2], "content_type": "application/zip",
                                     "download_url": official_url(pin[0])}} for name, pin in PINNED.items()]
        document = {"id": "2vkm6k4cfg", "version": 1, "data_licence": {"short_name": "CC BY 4.0"}, "files": rows}
        self.assertEqual(len(validate_metadata(document)), 3)
        rows[0]["content_details"]["download_url"] = "https://example.com/other.zip"
        with self.assertRaises(ValueError):
            validate_metadata(document)


if __name__ == "__main__":
    unittest.main()
