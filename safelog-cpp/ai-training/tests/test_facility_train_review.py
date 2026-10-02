import unittest
from scripts.build_facility_train_review import CLASSES, local_path, validate_training_manifest, verify_cache, sample_cases, render_html
from pathlib import Path


class TrainReviewTests(unittest.TestCase):
    def item(self, name, domain="dacl", target=0):
        return {"image": name, "domain": domain, "targets": [target, 0, -1, -1, -1, -1, -1]}

    def test_heldout_cache_and_modified_targets_are_rejected(self):
        item = self.item("train.jpg")
        cache = {"split": "train", "weights_sha256": "a"*64, "images": ["train.jpg"], "targets": [item["targets"]], "probabilities": [[.8]*7]}
        self.assertIn("train.jpg", verify_cache(cache, [item], CLASSES, "a"*64))
        for split in ("val", "test"):
            with self.assertRaises(ValueError):
                verify_cache({**cache, "split": split}, [item], CLASSES, "a"*64)
        with self.assertRaises(ValueError):
            verify_cache({**cache, "targets": [[1, 0, -1, -1, -1, -1, -1]]}, [item], CLASSES, "a"*64)
        with self.assertRaises(ValueError):
            verify_cache(cache, [item], CLASSES, "b"*64)

    def test_heldout_original_and_crop_parent_are_rejected(self):
        item = self.item("train.jpg")
        canonical = {"train.jpg": item}
        manifest = {"split": "train", "classes": CLASSES, "items": [item], "full_count": 1}
        self.assertEqual(len(validate_training_manifest(manifest, canonical)[0]), 1)
        with self.assertRaises(ValueError):
            validate_training_manifest({**manifest, "items": [self.item("val.jpg")]}, canonical)
        crop = {**self.item("crop.jpg"), "parent_image": "val.jpg", "parent_split": "train", "box_in_parent": [0,0,4,4]}
        with self.assertRaises(ValueError):
            validate_training_manifest({**manifest, "items": [item, crop]}, canonical, {"split": "train", "items": [crop]})
        crop["parent_image"] = "train.jpg"; crop["parent_split"] = "test"
        with self.assertRaises(ValueError):
            validate_training_manifest({**manifest, "items": [item, crop]}, canonical, {"split": "train", "items": [crop]})

    def test_sampling_has_errors_correct_comparisons_no_duplicates_and_unknown_not_normal(self):
        originals, scores = [], {}
        for domain in ("dacl", "damsegment", "codebrim"):
            for target, probability, kind in ((0,.9,"FP"),(1,.1,"FN"),(0,.1,"TN"),(1,.9,"TP")):
                for index in range(5):
                    item = self.item(f"{domain}-{kind}-{index}.jpg", domain, target)
                    item["targets"][1] = -1
                    originals.append(item); scores[item["image"]] = [probability]*7
        thresholds = {c:.5 for c in CLASSES[:2]}
        cases, counts = sample_cases(originals, scores, thresholds, 2, 1, 49)
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({c["image"] for c in cases}), len(cases))
        self.assertEqual({r["outcome"] for r in counts}, {"FP", "FN", "TP", "TN"})
        self.assertEqual({r["task"] for r in counts}, {"concrete_crack"})
        self.assertEqual((cases, counts), sample_cases(originals, scores, thresholds, 2, 1, 49))
        self.assertTrue(all(c["original_targets"][1] == -1 for c in cases))

    def test_html_escapes_untrusted_source_labels_and_paths(self):
        case = {"case_id": "case1", "domain": "dacl", "image": '<img src=x onerror=alert(1)>', "image_url": 'image.jpg" onerror="alert(1)',
                "original_targets": [0]*7, "probabilities": [.1]*7, "selected_for": [], "source_tags": ['</script><script>alert(1)</script>'],
                "image_sha256": "a"*64, "annotation": {"kind":"publisher_polygons", "imageWidth": 10,"imageHeight":10,
                    "shapes":[{"label": '<script>alert(2)</script>', "points":[[0,0],[0,1],[1,1]]}]}}
        package={"cases":[case],"thresholds":{},"sampling":[],"weights_sha256":"b"*64,"package_content_sha256":"c"*64}
        output=render_html(package)
        self.assertNotIn('<script>alert(1)</script>', output)
        self.assertNotIn('<script>alert(2)</script>', output)
        self.assertIn('&lt;script&gt;alert(2)&lt;/script&gt;', output)
        self.assertNotIn('src="image.jpg" onerror=', output)

    def test_paths_cannot_leave_workspace(self):
        root=Path.cwd()
        with self.assertRaises(ValueError):local_path(root, "../outside.png")
        with self.assertRaises(ValueError):local_path(root, str((root / "photo.jpg").resolve()))
        with self.assertRaises(ValueError):local_path(root, "data\\photo.jpg")


if __name__ == "__main__":
    unittest.main()
