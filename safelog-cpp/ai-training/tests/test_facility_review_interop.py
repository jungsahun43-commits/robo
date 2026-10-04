"""The browser importer and Python aggregator must accept the same opinions."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from safelog_ai.review_feedback import CLASSES, TASKS, canonical_sha256, read_json, validate_feedback

ROOT = Path(__file__).resolve().parents[1]


class FeedbackInteropTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for browser/Python interoperability")
    def test_same_validation_contract_in_browser_and_cli(self):
        package = {"schema": "facility_train_review_v1", "split": "train", "classes": CLASSES,
                   "weights_sha256": "b"*64, "cases": [{"case_id": "synthetic-case", "domain": "dacl", "original_targets": [1, 0, -1, -1, -1, -1, -1]}]}
        package["package_content_sha256"] = canonical_sha256(package)
        doc = {"schema": "facility_train_review_feedback_v2", "source_package_sha256": package["package_content_sha256"], "weights_sha256": package["weights_sha256"],
               "exported_utc": "2026-10-04T12:00:00.000500Z", "reviewer": {"reviewer_id": "synthetic-observer", "name": "합성 테스트", "role": "team_observer", "expertise": ""},
               "cases": [{"case_id": "synthetic-case", "tasks": [{"task": task, "judgement": "unreviewed", "reason": "unreviewed", "note": "", "evidence": "", "reviewed_at": None} for task in TASKS]}]}
        observed = deepcopy(doc)
        observed["cases"][0]["tasks"][0].update(judgement="uncertain", reason="capture_quality", note="합성 의견", reviewed_at="2026-10-04T12:00:00.000500+00:00")
        fixtures = []
        def add(name, value, expected):
            raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=True).encode("utf-8")
            fixtures.append({"name": name, "bytes_base64": base64.b64encode(raw).decode("ascii"), "expected": expected})
        add("empty-valid", doc, True); add("uncertain-microsecond-valid", observed, True)
        expert = deepcopy(observed); expert["reviewer"].update(role="domain_expert", expertise="합성 분야"); expert["cases"][0]["tasks"][0]["evidence"] = "합성 근거"
        add("expert-self-declaration-valid", expert, True)
        mutations = {
            "wrong-model": lambda d: d.update(weights_sha256="c"*64),
            "padded-identity": lambda d: d["reviewer"].update(reviewer_id=" observer "),
            "newline-identity": lambda d: d["reviewer"].update(reviewer_id="team\nA"),
            "newline-name": lambda d: d["reviewer"].update(name="team\rA"),
            "control-expertise": lambda d: d["reviewer"].update(expertise="a\x00b"),
            "lone-surrogate": lambda d: d["reviewer"].update(name="\ud800"),
            "later-microsecond": lambda d: d["cases"][0]["tasks"][0].update(reviewed_at="2026-10-04T12:00:00.000501Z"),
            "non-UTC": lambda d: d.update(exported_utc="2026-10-04T12:00:00+09:00"),
            "invalid-calendar": lambda d: d.update(exported_utc="2026-02-30T12:00:00Z"),
            "too-many-fraction-digits": lambda d: d.update(exported_utc="2026-10-04T12:00:00.0005001Z"),
            "missing-note": lambda d: d["cases"][0]["tasks"][0].update(note=" "),
            "extra-label-vector": lambda d: d["cases"][0]["tasks"][0].update(targets=[1, 0]),
        }
        for name, mutate in mutations.items():
            changed = deepcopy(observed); mutate(changed); add(name, changed, False)
        raw = json.dumps(observed).encode("utf-8")
        add("BOM", b"\xef\xbb\xbf" + raw, False)
        add("invalid-UTF8", raw.replace(b'"synthetic-observer"', b'"bad-\xff"'), False)
        add("duplicate-escaped-key", raw.replace(b'"schema":', b'"\\u0073chema":"duplicate","schema":', 1), False)
        metadata = {"source_package_sha256": package["package_content_sha256"], "weights_sha256": package["weights_sha256"], "case_ids": ["synthetic-case"], "domains": {"synthetic-case": "dacl"}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); suite = root / "fixtures.json"
            suite.write_text(json.dumps({"metadata": metadata, "fixtures": fixtures}), encoding="utf-8")
            actual = json.loads(subprocess.check_output([shutil.which("node"), str(ROOT / "tests/review_workbench_interop.cjs"), str(suite)], text=True))
            for fixture, browser in zip(fixtures, actual):
                path = root / "opinion.json"; path.write_bytes(base64.b64decode(fixture["bytes_base64"]))
                try: validate_feedback(read_json(path), package); accepted = True
                except ValueError: accepted = False
                with self.subTest(name=fixture["name"]):
                    self.assertEqual(accepted, fixture["expected"])
                    self.assertEqual(browser["accepted"], accepted)
            self.assertEqual(len(actual), len(fixtures))


if __name__ == "__main__":
    unittest.main()
