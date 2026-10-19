"""One bounded publisher specification TXT. Fixtures are synthetic and offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
import urllib.error
import urllib.request
import uuid

AI = Path(__file__).resolve().parents[1]
REPO = AI.parents[1]
WORK = REPO.parents[1]
OUT = WORK / "outputs"
POLICY = AI / "reports/WOOD_V162_SPECIFICATION_PROTOCOL.json"
DATA = AI / "data/industrial-wood-v162-specification"
RUN = AI / "runs/industrial-wood-v162-specification"
LOCK = AI / "operation.lock"
GIT = "C:/Program Files/Git/cmd/git.exe"


class GuardStop(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise GuardStop(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(_):
        raise GuardStop("nonfinite JSON")
    return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def age(value, limit):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "timezone required")
    seconds = (datetime.now(timezone.utc) - stamp).total_seconds()
    require(-5 <= seconds <= limit, "stale/future receipt")


def headers_record(headers, policy):
    items = list(headers.raw_items())
    raw = json.dumps(items, ensure_ascii=False).encode("utf-8")
    if len(raw) <= policy["max_saved_header_bytes"]:
        return {"items": items, "truncated": False, "serialized_bytes": len(raw)}
    return {"prefix": raw[:policy["max_saved_header_bytes"]].decode("utf-8", "backslashreplace"),
            "truncated": True, "serialized_bytes": len(raw), "full_header_SHA256": sha(raw)}


def header_gate(status, url, headers, policy):
    require(status == 200 and url == policy["url"], "status or URL mismatch")
    require(len(headers.get_all("Content-Type", [])) == 1, "single MIME required")
    media = headers["Content-Type"].split(";", 1)[0].strip().lower()
    require(media in policy["allowed_media_types"], "unexpected MIME")
    require(len(headers.get_all("Content-Encoding", [])) <= 1
            and headers.get("Content-Encoding", "identity").strip().lower() == "identity",
            "identity encoding required")
    lengths = headers.get_all("Content-Length", [])
    require(len(lengths) == 1 and re.fullmatch(r"[0-9]+", lengths[0]),
            "single decimal Content-Length required")
    require(int(lengths[0]) == policy["exact_body_bytes"], "exact430 length required")


def body_gate(raw, policy):
    require(len(raw) == policy["exact_body_bytes"], "received byte length mismatch")
    require(hashlib.md5(raw).hexdigest() == policy["publisher_MD5"], "publisher MD5 mismatch")
    raw.decode("utf-8", errors="strict")
    return {"bytes": len(raw), "MD5": hashlib.md5(raw).hexdigest(),
            "local_SHA256": sha(raw), "strict_UTF8": True, "ready": False}


def source_gate(policy):
    raw = (AI / policy["frozen_API_relative_path"]).read_bytes()
    require(len(raw) == policy["frozen_API_bytes"]
            and sha(raw) == policy["frozen_API_SHA256"], "preserved API identity changed")
    record = strict_json(raw)
    require(record["id"] == policy["record_id"] and record["doi"] == policy["doi"],
            "source record identity")
    files = [x for x in record["files"] if x["key"] == policy["file_key"]]
    require(len(files) == 1, "one specification declaration required")
    item = files[0]
    require(item["id"] == policy["file_id"] and item["links"]["self"] == policy["url"]
            and item["size"] == policy["exact_body_bytes"]
            and item["checksum"] == "md5:" + policy["publisher_MD5"],
            "frozen publisher file declaration mismatch")


def publication_gate(path, policy):
    receipt = strict_json(path.read_bytes())
    require(receipt.get("success") is True and receipt.get("ordinary_nonforce") is True,
            "successful ordinary preregistration publication required")
    require(receipt.get("destination") == policy["publication_destination"]
            and receipt.get("branch") == policy["publication_branch"], "wrong destination")
    age(receipt["verified_UTC"], policy["publication_max_age_seconds"])
    commit = receipt.get("verified_commit", "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "verified commit required")
    files = receipt.get("files", [])
    require(len(files) == 3 and {x["path"] for x in files} == set(policy["publication_paths"]),
            "exact three preregistration paths required")
    for item in files:
        raw = (REPO / item["path"]).read_bytes()
        require(len(raw) == item["bytes"] and sha(raw) == item["SHA256"],
                "published local bytes changed")
        result = subprocess.run([GIT, "-C", str(REPO), "-c", "safe.directory=" + str(REPO),
                                 "show", commit + ":" + item["path"]],
                                check=True, capture_output=True, timeout=20)
        require(sha(result.stdout) == item["SHA256"], "published commit blob mismatch")
    return commit


def preflight_gate(path, policy):
    record = strict_json(path.read_bytes())
    age(record["UTC"], policy["preflight_max_age_seconds"])
    require(record.get("cim_succeeded") is True and record.get("related") == []
            and record.get("operation_locks") == [] and record.get("PID23948_present") is False,
            "CIM/process/lock gate")
    require(record.get("scope") == "AI root/runs and WORK outputs only, not whole source tree",
            "preflight lock scope")
    entry = record["last_v85_optimizer_entry"]
    require(entry["actual_update"] == 450 and entry["epoch"] == 30, "v85 ledger changed")
    require(not LOCK.exists() and not list((AI / "runs").rglob("operation.lock"))
            and not list(OUT.rglob("operation.lock")), "operation lock appeared")
    require(shutil.disk_usage(WORK).free >= policy["reserve_bytes"]
            + policy["data_run_budget_bytes"], "storage gate")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy):
        super().__init__()
        self.policy, self.observation = policy, None
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.observation = {"status": code, "target_not_requested": newurl,
                            "headers": headers_record(headers, self.policy)}
        raise GuardStop("redirect forbidden before next request/body")


def run(args, policy):
    require(os.name == "nt", "managed Windows workspace required")
    require(not DATA.exists() and not RUN.exists(), "preserve prior attempt; no rerun")
    source_gate(policy)
    commit = publication_gate(args.publication_receipt, policy)
    preflight_gate(args.preflight, policy)
    own = {"PID": os.getpid(), "token": uuid.uuid4().hex, "stage": "v162", "UTC": utc()}
    save(LOCK, own)
    redirect = NoRedirect(policy)
    attempts, saved = 0, 0
    observation = None
    try:
        RUN.mkdir()
        DATA.mkdir()
        save(RUN / "started.json", {"UTC": utc(), "publication_commit": commit,
                                   "script_SHA256": sha(Path(__file__).read_bytes()),
                                   "policy_SHA256": sha(POLICY.read_bytes())})
        save(DATA / "source-protection.json", {"UTC": utc(),
             "source_family_containers_members_records_copies": "UNASSIGNED",
             "original_TEST_publisher_VAL_reserved": True, "ready": False})
        request_headers = {"Accept": "text/plain, application/octet-stream",
                           "Accept-Encoding": "identity", "User-Agent": "SafeLog-spec-v162/1.0"}
        save(DATA / "request-headers.json", {"url": policy["url"], "headers": request_headers})
        request = urllib.request.Request(policy["url"], headers=request_headers, method="GET")
        opener = urllib.request.build_opener(redirect)
        attempts = 1
        with opener.open(request, timeout=policy["timeout_seconds"]) as response:
            observation = {"status": response.status, "url": response.geturl(),
                           "headers": headers_record(response.headers, policy)}
            save(DATA / "response-headers.json", observation)
            require(not observation["headers"]["truncated"], "header cap exceeded")
            header_gate(response.status, response.geturl(), response.headers, policy)
            with (DATA / "specification.body").open("xb") as stream:
                while True:
                    chunk = response.read(min(431 - saved, 1024))
                    if not chunk:
                        break
                    require(saved + len(chunk) <= policy["exact_body_bytes"], "body exceeds430")
                    stream.write(chunk)
                    saved += len(chunk)
        summary = body_gate((DATA / "specification.body").read_bytes(), policy)
        logical_bytes = sum(p.stat().st_size for root in (DATA, RUN)
                            for p in root.rglob("*") if p.is_file())
        require(logical_bytes + 16384 <= policy["data_run_budget_bytes"], "output budget")
        save(RUN / "complete.json", {"UTC": utc(), "success": True,
             "specification_HTTP_attempts": attempts, "integrity": summary,
             "text_not_code_executed_or_GT_role_mapping_applied": True,
             "photos_masks_archives_pixels_features_scores_training_runtime_deletion": 0,
             "ready": False})
        print(json.dumps(summary))
    except BaseException as exc:
        if isinstance(exc, urllib.error.HTTPError):
            observation = {"status": exc.code, "headers": headers_record(exc.headers, policy)}
        if redirect.observation is not None:
            observation = redirect.observation
        if RUN.is_dir():
            save(RUN / "failure.json", {"UTC": utc(), "error_type": type(exc).__name__,
                 "error": str(exc), "HTTP_attempts": attempts, "saved_body_bytes": saved,
                 "response_observation": observation, "partials_preserved_no_retry": True})
        raise
    finally:
        try:
            if LOCK.exists() and strict_json(LOCK.read_bytes()) == own:
                LOCK.unlink()
        except (OSError, ValueError, GuardStop):
            pass


class SpecificationFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = strict_json(POLICY.read_bytes())

    def test_synthetic_text_length_digest_guards(self):
        raw = b"Synthetic fixture only; not publisher specification.\n"
        raw += b" " * (430 - len(raw))
        hypothetical = dict(self.policy, publisher_MD5=hashlib.md5(raw).hexdigest())
        self.assertFalse(body_gate(raw, hypothetical)["ready"])
        for bad in (raw[:-1], b"X" + raw[1:]):
            with self.assertRaises(GuardStop):
                body_gate(bad, hypothetical)
        invalid_utf8 = b"\xff" * 430
        with self.assertRaises(UnicodeError):
            body_gate(invalid_utf8, dict(hypothetical, publisher_MD5=hashlib.md5(invalid_utf8).hexdigest()))

    def test_HTTP_header_and_redirect_guards(self):
        from email.message import Message
        def headers(**kw):
            message = Message()
            for key, value in {"Content-Type": "text/plain", "Content-Length": "430", **kw}.items():
                message[key] = value
            return message
        header_gate(200, self.policy["url"], headers(), self.policy)
        for bad in (headers(**{"Content-Length": "431"}),
                    headers(**{"Content-Type": "text/html"}),
                    headers(**{"Content-Encoding": "gzip"})):
            with self.assertRaises(GuardStop):
                header_gate(200, self.policy["url"], bad, self.policy)
        guard = NoRedirect(self.policy)
        with self.assertRaises(GuardStop):
            guard.redirect_request(None, None, 302, "", headers(), "https://example.invalid/")
        self.assertEqual(guard.observation["status"], 302)
        self.assertFalse(guard.observation["headers"]["truncated"])

    def test_header_cap_JSON_and_time_guards(self):
        from email.message import Message
        message = Message()
        message["X-Fixture"] = "A" * 40000
        record = headers_record(message, self.policy)
        self.assertTrue(record["truncated"])
        self.assertIn("full_header_SHA256", record)
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}'):
            with self.assertRaises(GuardStop):
                strict_json(raw)
        with self.assertRaises(GuardStop):
            age("2020-01-01T00:00:00Z", 120)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--publication-receipt", type=Path)
    args = parser.parse_args()
    if args.fixtures:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SpecificationFixtures)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    require(args.preflight is not None and args.publication_receipt is not None,
            "fresh preflight and exact publication receipts required")
    run(args, strict_json(POLICY.read_bytes()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
