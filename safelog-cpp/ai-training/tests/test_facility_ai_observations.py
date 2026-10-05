from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from safelog_ai.review_feedback import CLASSES, canonical_sha256, validate_feedback
from safelog_ai.review_observations import validate_ai_observations, summarize_ai_observations
from scripts.build_facility_review_workbench import ROOT, main, render_html, safe_reference


class AIObservationTests(unittest.TestCase):
    def fixture(self):
        package = {"schema": "facility_train_review_v1", "split": "train", "classes": CLASSES,
                   "weights_sha256": "b"*64, "cases": [{"case_id": "synthetic-case", "domain": "dacl", "image_sha256": "a"*64,
                   "original_targets": [1, 0, -1, -1, -1, -1, -1], "annotation": {"kind": "publisher_photo_category_only"}}]}
        package["package_content_sha256"] = canonical_sha256(package)
        doc = {"schema": "facility_ai_train_observations_v1", "source_package_sha256": package["package_content_sha256"],
               "weights_sha256": package["weights_sha256"], "observer_type": "ai", "review_status": "unconfirmed",
               "created_utc": "2026-10-05T00:00:00Z", "expert_confirmed_labels": 0, "label_changes": 0, "training_applied": 0,
               "cases": [{"case_id": "synthetic-case", "image_sha256": "a"*64, "annotation_sha256": None,
               "input_photo_viewed": True, "annotation_metadata_read": True, "publisher_mask_viewed": False,
               "facts": ["합성 관찰: 선의 재질을 사진에서 확정하지 못함"], "hypotheses": ["이음매와 혼동 가능성"],
               "questions": ["추가 각도나 원본 정의 확인 필요"], "tasks": [{"task": "concrete_crack", "visual_judgement": "uncertain", "reason": "annotation_uncertain"}]}]}
        return package, doc

    def test_ai_observations_are_separate_from_human_feedback_and_targets(self):
        package, doc = self.fixture(); before = deepcopy(package)
        actual = validate_ai_observations(doc, package)
        actual["cases"][0]["facts"].append("사본만 수정")
        self.assertNotEqual(actual, doc); self.assertEqual(package, before)
        with self.assertRaises(ValueError): validate_feedback(doc, package)

    def test_confirmation_training_or_label_claims_are_rejected(self):
        package, doc = self.fixture()
        for key, value in (("review_status", "confirmed"), ("observer_type", "domain_expert"),
                           ("label_changes", 1), ("training_applied", True), ("expert_confirmed_labels", False)):
            altered = deepcopy(doc); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_ai_observations(altered, package)

    def test_wrong_package_model_asset_digest_and_invented_mask_reject(self):
        package, doc = self.fixture()
        mutations = [lambda d: d.update(source_package_sha256="c"*64), lambda d: d.update(weights_sha256="c"*64),
                     lambda d: d["cases"][0].update(image_sha256="c"*64), lambda d: d["cases"][0].update(annotation_sha256="c"*64),
                     lambda d: d["cases"][0].update(publisher_mask_viewed=True)]
        for mutate in mutations:
            altered = deepcopy(doc); mutate(altered)
            with self.assertRaises(ValueError): validate_ai_observations(altered, package)

    def test_unviewed_duplicate_cases_tasks_and_extra_targets_reject(self):
        package, doc = self.fixture()
        mutations = [lambda d: d["cases"][0].update(input_photo_viewed=False),
                     lambda d: d["cases"].append(deepcopy(d["cases"][0])),
                     lambda d: d["cases"][0]["tasks"].append(deepcopy(d["cases"][0]["tasks"][0])),
                     lambda d: d["cases"][0].update(original_targets=[0]*7),
                     lambda d: d["cases"][0].update(case_id="heldout-case")]
        for mutate in mutations:
            altered = deepcopy(doc); mutate(altered)
            with self.assertRaises(ValueError): validate_ai_observations(altered, package)

    def test_public_summary_excludes_case_ids_and_free_text(self):
        package, doc = self.fixture(); summary = summarize_ai_observations(doc, package)
        encoded = json.dumps(summary, ensure_ascii=False)
        self.assertNotIn("synthetic-case", encoded); self.assertNotIn(doc["cases"][0]["facts"][0], encoded)
        self.assertEqual(summary["observed_cases"], 1); self.assertEqual(summary["observed_task_opinions"], 1)
        self.assertEqual(summary["human_review_opinions_generated"], 0); self.assertEqual(summary["expert_confirmed_labels"], 0)

    def test_read_only_notes_are_escaped_and_do_not_prefill_human_fields(self):
        package, doc = self.fixture()
        package["cases"][0].update(image_url="test.svg", probabilities=[.1]*7, selected_for=[], source_tags=[])
        doc["cases"][0]["facts"] = ['</section><script>alert(1)</script>']
        page = render_html(package, doc)
        self.assertNotIn('<script>alert(1)</script>', page)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', page)
        self.assertIn('id="show-ai" type="checkbox" >', page)
        self.assertIn('class="note" maxlength="4000" aria-label="균열 관찰 메모"></textarea>', page)
        self.assertNotIn('id="show-ai" type="checkbox" checked', page)

    def test_new_output_rebases_links_preserves_package_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as folder:
            root = Path(folder); source = root / "source"; source.mkdir(); photo = source / "synthetic.svg"
            photo.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
            package, doc = self.fixture()
            digest = hashlib.sha256(photo.read_bytes()).hexdigest()
            package["cases"][0].update(image_url=photo.name, image_sha256=digest, probabilities=[.1]*7, source_tags=[], selected_for=[])
            package.pop("package_content_sha256"); package["package_content_sha256"] = canonical_sha256(package)
            doc.update(source_package_sha256=package["package_content_sha256"])
            doc["cases"][0]["image_sha256"] = digest
            packagepath = source / "TRAIN-REVIEW.json"; aipath = root / "ai.json"
            packagepath.write_text(json.dumps(package), encoding="utf-8"); before = packagepath.read_bytes()
            aipath.write_text(json.dumps(doc), encoding="utf-8"); output = root / "new-screen/review-workbench.html"
            args = ["--package", str(packagepath), "--ai-observations", str(aipath), "--output", str(output)]
            with redirect_stdout(StringIO()): main(args)
            self.assertEqual(packagepath.read_bytes(), before)
            self.assertIn('src="../source/synthetic.svg"', output.read_text(encoding="utf-8"))
            self.assertEqual(safe_reference("../source/synthetic.svg", output.parent), photo.resolve())
            with self.assertRaises(ValueError): main(args)

    def test_invalid_ai_attachment_creates_no_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as folder:
            root = Path(folder); package, doc = self.fixture(); doc["weights_sha256"] = "c"*64
            packagepath = root / "TRAIN-REVIEW.json"; aipath = root / "ai.json"; output = root / "not-created/review.html"
            packagepath.write_text(json.dumps(package), encoding="utf-8"); aipath.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError): main(["--package", str(packagepath), "--ai-observations", str(aipath), "--output", str(output)])
            self.assertFalse(output.parent.exists())


if __name__ == "__main__": unittest.main()
