"""Bounded publisher-record metadata only. --fixtures never opens the network.

The live path requires an approved exact preregistration publication receipt and
a fresh external read-only CIM preflight. It never follows publisher file links.
"""
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
import unicodedata
import unittest
import urllib.error
import urllib.request
import uuid

AI = Path(__file__).resolve().parents[1]
REPO = AI.parents[1]
WORK = REPO.parents[1]
OUT = WORK / "outputs"
POLICY_PATH = AI / "reports/WOOD_V161_METADATA_PROTOCOL.json"
DATA = AI / "data/industrial-wood-v161-metadata"
RUN = AI / "runs/industrial-wood-v161-metadata"
LOCK = AI / "operation.lock"
GIT = Path("C:/Program Files/Git/cmd/git.exe")


class GuardStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuardStop(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise GuardStop("nonfinite JSON number")

    try:
        return json.loads(raw.decode("utf-8", errors="strict"),
                          object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, ValueError) as exc:
        raise GuardStop("invalid strict UTF-8 JSON") from exc


def age_seconds(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "UTC timestamp required")
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    require(age >= -5, "future timestamp")
    return age


def save_once(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def header_gate(status, url, headers, policy):
    require(status == 200 and url == policy["url"], "status or URL mismatch")
    require(len(headers.get_all("Content-Type", [])) == 1
            and len(headers.get_all("Content-Encoding", [])) <= 1,
            "ambiguous MIME or encoding headers")
    require(headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            == "application/json", "JSON MIME required")
    require(headers.get("Content-Encoding", "identity").strip().lower()
            == "identity", "identity encoding required")
    lengths = headers.get_all("Content-Length", [])
    require(len(lengths) == 1 and re.fullmatch(r"[0-9]+", lengths[0]),
            "one decimal Content-Length required")
    size = int(lengths[0])
    require(0 < size <= policy["max_body_bytes"], "body budget exceeded")
    return size


def capture_headers(headers, policy):
    items = list(headers.raw_items())
    raw = json.dumps(items, ensure_ascii=False).encode("utf-8")
    if len(raw) <= policy["max_saved_header_bytes"]:
        return {"items": items, "truncated": False, "serialized_bytes": len(raw)}
    return {"serialized_prefix": raw[:policy["max_saved_header_bytes"]].decode(
                "utf-8", errors="backslashreplace"),
            "truncated": True, "serialized_bytes": len(raw),
            "full_serialized_SHA256": hashlib.sha256(raw).hexdigest()}


def record_gate(raw, policy):
    body = strict_json(raw)
    require(isinstance(body, dict), "record object required")
    require(type(body.get("id")) is int and body["id"] == policy["record_id"],
            "record ID mismatch")
    require(body.get("doi") == policy["doi"], "record DOI mismatch")
    metadata = body.get("metadata")
    require(isinstance(metadata, dict)
            and metadata.get("title") == policy["title"], "record title mismatch")
    creators = metadata.get("creators")
    require(isinstance(creators, list) and len(creators) == 3, "creators mismatch")
    names = []
    for creator in creators:
        require(isinstance(creator, dict) and isinstance(creator.get("name"), str),
                "creator name required")
        normalized = unicodedata.normalize("NFKD", creator["name"]).lower()
        names.append(sorted(re.findall(r"[a-z]+", normalized)))
    expected = sorted(sorted(x) for x in policy["expected_creator_name_token_sets"])
    require(sorted(names) == expected, "creator identity mismatch")
    files = body.get("files")
    require(isinstance(files, list) and len(files) == 13, "file declaration count")
    keys = []
    for item in files:
        require(isinstance(item, dict) and isinstance(item.get("key"), str),
                "file declaration object/key")
        require(type(item.get("size")) is int and item["size"] > 0,
                "declared positive byte length required")
        require(isinstance(item.get("checksum"), str)
                and re.fullmatch(r"md5:[0-9a-f]{32}", item["checksum"]),
                "publisher MD5 declaration format")
        keys.append(item["key"])
    require(len(set(keys)) == 13 and set(keys) == set(policy["expected_file_keys"]),
            "file declaration keys mismatch")
    spec = next(x for x in files if x["key"] == "Semantic Map Specification.txt")
    expected_spec = policy["specification_publisher_declaration_only"]
    require(spec["size"] == expected_spec["size"]
            and spec["checksum"] == expected_spec["checksum"],
            "specification declaration mismatch")
    return {
        "record_id": body["id"], "DOI": body["doi"],
        "metadata_version": metadata.get("version"),
        "service_revision": body.get("revision"),
        "license_declaration": metadata.get("license"),
        "creators_declaration": creators,
        "declared_files": [{"key": x["key"], "size": x["size"],
                            "checksum": x["checksum"]} for x in files],
        "ready_for_training": False,
        "actual_files_GT_roles_groups_rights_fullgraph_budget_audited": False,
    }


def validate_publication(path, policy):
    receipt = strict_json(path.read_bytes())
    require(receipt.get("success") is True and receipt.get("ordinary_nonforce") is True,
            "successful ordinary preregistration publication required")
    require(receipt.get("destination") == policy["publication_destination"]
            and receipt.get("branch") == policy["publication_branch"],
            "publication destination mismatch")
    commit = receipt.get("verified_commit", "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "verified commit required")
    require(age_seconds(receipt["verified_UTC"])
            <= policy["publication_verification_max_age_seconds"],
            "fresh remote verification required")
    approved = receipt.get("files", [])
    require(len(approved) == 3
            and {x["path"] for x in approved} == set(policy["publication_paths"]),
            "exact three published paths required")
    for item in approved:
        local = REPO / item["path"]
        require(local.stat().st_size == item["bytes"]
                and sha256(local) == item["SHA256"], "published local bytes changed")
        result = subprocess.run(
            [str(GIT), "-C", str(REPO), "-c", "safe.directory=" + str(REPO),
             "show", commit + ":" + item["path"]],
            check=True, capture_output=True, timeout=20)
        require(hashlib.sha256(result.stdout).hexdigest() == item["SHA256"],
                "published commit blob mismatch")
    return commit


def validate_preflight(path, policy):
    preflight = strict_json(path.read_bytes())
    require(age_seconds(preflight["UTC"]) <= policy["preflight_max_age_seconds"],
            "fresh preflight required")
    require(preflight.get("cim_succeeded") is True
            and preflight.get("related") == []
            and preflight.get("operation_locks") == []
            and preflight.get("PID23948_present") is False,
            "CIM/process/lock preflight failed")
    require(preflight.get("scope") ==
            "AI root/runs and WORK outputs only, not whole source tree", "lock scope")
    entry = preflight["last_v85_optimizer_entry"]
    require(entry["actual_update"] == 450 and entry["epoch"] == 30,
            "v85 actual ledger changed")
    locks = list((AI / "runs").rglob("operation.lock"))
    locks += list(OUT.rglob("operation.lock"))
    require(not LOCK.exists() and not locks, "operation lock appeared")
    require(shutil.disk_usage(WORK).free >=
            policy["reserve_bytes"] + policy["data_run_budget_bytes"], "storage gate")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy=None):
        super().__init__()
        self.policy = policy
        self.observation = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if self.policy is not None:
            self.observation = {"status": code, "target_not_requested": newurl,
                                "headers": capture_headers(headers, self.policy)}
        raise GuardStop("redirect forbidden before target request/body")


def live_run(args, policy):
    require(os.name == "nt", "Windows managed workspace required")
    require(not DATA.exists() and not RUN.exists(), "preserve prior attempt; no rerun")
    commit = validate_publication(args.publication_receipt, policy)
    validate_preflight(args.preflight, policy)
    token = uuid.uuid4().hex
    own = {"PID": os.getpid(), "token": token, "stage": "v161", "UTC": now()}
    save_once(LOCK, own)
    attempts = 0
    received = 0
    response_headers = None
    redirect_guard = NoRedirect(policy)
    try:
        RUN.mkdir()
        DATA.mkdir()
        save_once(RUN / "started.json", {"UTC": now(), "PID": os.getpid(),
                  "publication_commit": commit, "policy_SHA256": sha256(POLICY_PATH),
                  "script_SHA256": sha256(Path(__file__)), "request_limit": 1})
        save_once(DATA / "source-protection.json", {
            "UTC": now(), "record_id": policy["record_id"],
            "source_family_all_possible_containers_members_records_copies": "UNASSIGNED",
            "original_TEST_publisher_VAL_reserved": True,
            "pixel_read_role_assignment_graph_ready": False})
        headers = {"Accept": "application/json", "Accept-Encoding": "identity",
                   "User-Agent": "SafeLog-metadata-v161/1.0"}
        save_once(DATA / "request-headers.json", {"URL": policy["url"], "headers": headers})
        opener = urllib.request.build_opener(redirect_guard)
        request = urllib.request.Request(policy["url"], headers=headers, method="GET")
        attempts = 1
        with opener.open(request, timeout=policy["timeout_seconds"]) as response:
            response_headers = capture_headers(response.headers, policy)
            save_once(DATA / "response-headers.json", {
                "status": response.status, "URL": response.geturl(),
                "headers": response_headers})
            require(not response_headers["truncated"], "response header cap exceeded")
            declared = header_gate(response.status, response.geturl(), response.headers, policy)
            with (DATA / "publisher-record.body").open("xb") as stream:
                while True:
                    chunk = response.read(min(16384, declared - received + 1))
                    if not chunk:
                        break
                    require(received + len(chunk) <= declared, "body exceeds declared length")
                    stream.write(chunk)
                    received += len(chunk)
            require(received == declared, "body length mismatch")
        raw_path = DATA / "publisher-record.body"
        declarations = record_gate(raw_path.read_bytes(), policy)
        allocated_logical = sum(p.stat().st_size for root in (DATA, RUN)
                                for p in root.rglob("*") if p.is_file())
        require(allocated_logical + 32768 <= policy["data_run_budget_bytes"], "output budget")
        save_once(RUN / "complete.json", {
            "UTC": now(), "success": True, "publisher_metadata_HTTP_attempts": attempts,
            "body_bytes": received, "body_SHA256": sha256(raw_path),
            "declarations_not_actual_archive_audit": declarations,
            "files_or_links_followed_pixels_training_deletions": 0,
            "ready_for_training": False})
        print(json.dumps({"metadata_body_bytes": received,
                          "SHA256": sha256(raw_path), "ready": False}))
    except BaseException as exc:
        if isinstance(exc, urllib.error.HTTPError):
            response_headers = capture_headers(exc.headers, policy)
        if redirect_guard.observation is not None:
            response_headers = redirect_guard.observation
        if RUN.is_dir() and not (RUN / "failure.json").exists():
            save_once(RUN / "failure.json", {
                "UTC": now(), "error_type": type(exc).__name__, "error": str(exc),
                "publisher_metadata_HTTP_attempts": attempts, "saved_body_bytes": received,
                "response_headers_if_error": response_headers,
                "partials_preserved_no_retry": True, "ready": False})
        raise
    finally:
        try:
            if LOCK.exists() and strict_json(LOCK.read_bytes()) == own:
                LOCK.unlink()
        except (OSError, GuardStop):
            pass  # Preserve an unreadable or changed lock; never delete by PID alone.


class MetadataFixtures(unittest.TestCase):
    """Synthetic metadata, no source files, HTTP or model operations."""
    @classmethod
    def setUpClass(cls):
        cls.policy = strict_json(POLICY_PATH.read_bytes())

    def fixture(self):
        p = self.policy
        files = [{"key": key, "size": 1, "checksum": "md5:" + "0" * 32}
                 for key in p["expected_file_keys"]]
        spec = p["specification_publisher_declaration_only"]
        files[files.index(next(x for x in files if x["key"] == spec["key"]))] = dict(spec)
        return {"id": p["record_id"], "doi": p["doi"], "files": files,
                "metadata": {"title": p["title"], "creators": [
                    {"name": "Kodytek, Pavel"}, {"name": "Bodzas, Alexandra"},
                    {"name": "Bilik, Petr"}], "license": {"id": "fixture-only"}}}

    def test_valid_declarations_stay_unready(self):
        result = record_gate(json.dumps(self.fixture()).encode(), self.policy)
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(result["license_declaration"]["id"], "fixture-only")

    def test_strict_JSON_identity_and_file_guards(self):
        for raw in (b'{"id":1,"id":2}', b'{"x":NaN}', b'\xff'):
            with self.assertRaises(GuardStop):
                strict_json(raw)
        for mutation in ("id", "checksum", "duplicate_file"):
            body = self.fixture()
            if mutation == "id":
                body["id"] = 1
            elif mutation == "checksum":
                body["files"][0]["checksum"] = "md5:invalid"
            else:
                body["files"][0] = body["files"][1]
            with self.assertRaises(GuardStop):
                record_gate(json.dumps(body).encode(), self.policy)

    def test_header_budget_and_redirect_guards(self):
        from email.message import Message
        base = {"Content-Type": "application/json; charset=utf-8", "Content-Length": "100"}
        def msg(fields):
            result = Message()
            for key, value in fields.items():
                result[key] = value
            return result
        self.assertEqual(header_gate(200, self.policy["url"], msg(base), self.policy), 100)
        for fields in ({**base, "Content-Length": "262145"},
                       {"Content-Type": "application/json"},
                       {**base, "Content-Encoding": "gzip"}):
            with self.assertRaises(GuardStop):
                header_gate(200, self.policy["url"], msg(fields), self.policy)
        with self.assertRaises(GuardStop):
            NoRedirect().redirect_request(None, None, 302, "", None, "https://example.invalid/")

    def test_redirect_headers_preserved(self):
        from email.message import Message
        headers = Message()
        headers["Location"] = "https://example.invalid/blocked"
        handler = NoRedirect(self.policy)
        with self.assertRaises(GuardStop):
            handler.redirect_request(None, None, 302, "", headers,
                                     "https://example.invalid/blocked")
        self.assertEqual(handler.observation["status"], 302)
        self.assertEqual(handler.observation["headers"]["items"],
                         [("Location", "https://example.invalid/blocked")])


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixtures", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--publication-receipt", type=Path)
    args = parser.parse_args()
    if args.fixtures:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetadataFixtures)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    require(args.preflight is not None and args.publication_receipt is not None,
            "publication and fresh preflight receipts required")
    live_run(args, strict_json(POLICY_PATH.read_bytes()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
