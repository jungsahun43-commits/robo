import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.build_facility_review_workbench import ROOT, safe_reference, verify_asset, render_html


class WorkbenchBoundaryTests(unittest.TestCase):
    def test_references_are_local_existing_files_and_changed_bytes_reject(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as folder:
            root = Path(folder)
            photo = root / "synthetic-photo.txt"
            photo.write_bytes(b"synthetic asset, no source photo")
            digest = hashlib.sha256(photo.read_bytes()).hexdigest()
            self.assertEqual(safe_reference(photo.name, root), photo.resolve())
            verify_asset(photo.name, digest, root, {})
            photo.write_bytes(b"changed synthetic bytes")
            with self.assertRaises(ValueError): verify_asset(photo.name, digest, root, {})
            for value in ("https://example.com/a.jpg", "javascript:alert(1)", "//example.com/a.jpg", "../../../../../../outside", "synthetic-photo.txt?x=1", "synthetic-photo.txt#x", "bad\\name"):
                with self.subTest(value=value), self.assertRaises(ValueError): safe_reference(value, root)

    def test_html_escapes_source_fields_and_starts_with_hidden_predictions(self):
        case = {"case_id": "test-case", "domain": "dacl", "image_url": "synthetic.jpg", "image_sha256": "a"*64,
                "original_targets": [0]*7, "probabilities": [.1]*7,
                "source_tags": ['</script><script>alert(1)</script>'], "selected_for": [], "annotation": {"kind": "none"}}
        package = {"cases": [case], "weights_sha256": "b"*64, "package_content_sha256": "c"*64}
        page = render_html(package)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertIn('id="show-model" type="checkbox">', page)
        self.assertNotIn('id="show-model" type="checkbox" checked', page)
        self.assertEqual(page.count('<fieldset data-task='), 2)


if __name__ == "__main__":
    unittest.main()
