"""One preregistered Zenodo record GET; no archive, pixel or author-code reads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

AI = Path(__file__).resolve().parents[1]
POLICY = AI / "reports/PAGG_V230_METADATA_PROTOCOL.json"
DATA = AI / "data/pagg-v230-record-metadata"
RUN = AI / "runs/pagg-v230-record-metadata"
CORE_SHA = "f72eb1deea38ac4f615e2b6c53d30fc67a5d7a9b3b7f3f708bb839d0140be999"


def load_core():
    path = AI / "scripts/inspect_fresh_concrete_v227.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORE_SHA:
        raise RuntimeError("preregistered standard-library guard dependency changed")
    spec = importlib.util.spec_from_file_location("v227_pinned_metadata_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(raw, policy, core):
    value = core.strict(raw)
    core.need(isinstance(value, dict) and type(value.get("id")) is int
              and value["id"] == policy["record_id"], "record ID mismatch")
    core.need(value.get("doi") == policy["DOI"], "record DOI mismatch")
    meta = value.get("metadata", {})
    core.need(isinstance(meta, dict) and meta.get("title") == policy["title"], "record title mismatch")
    creators = meta.get("creators", [])
    core.need(isinstance(creators, list) and len(creators) == 1, "curator identity count")
    tokens = re.findall(r"[a-z]+", creators[0].get("name", "").lower())
    core.need(sorted(tokens) == ["xingzhuang", "zhao"], "curator name mismatch")
    files = value.get("files")
    core.need(isinstance(files, list) and len(files) == 1, "one declared source archive required")
    item = files[0]
    core.need(isinstance(item, dict) and item.get("key") == "Zenodo.zip"
              and item.get("checksum") == policy["publisher_archive_checksum"], "archive name/checksum declaration")
    size = item.get("size")
    core.need(type(size) is int and policy["declared_archive_min_bytes"] <= size
              <= policy["declared_archive_max_bytes"], "archive size declaration outside bound")
    link = item.get("links", {}).get("self", "")
    parsed = urllib.parse.urlsplit(link)
    core.need(parsed.scheme == "https" and parsed.hostname == "zenodo.org"
              and parsed.username is None and parsed.password is None and parsed.port in (None, 443)
              and (parsed.path.endswith("/Zenodo.zip") or parsed.path.endswith("/Zenodo.zip/content")),
              "unexpected declared file link")
    return {"record_id": value["id"], "DOI": value["doi"], "version_declaration": meta.get("version"),
            "curator_declaration": creators[0], "license_declaration": meta.get("license"),
            "access_right_declaration": meta.get("access_right"), "service_revision": value.get("revision"),
            "file": {"name": item["key"], "publisher_bytes": size,
                     "publisher_checksum": item["checksum"], "publisher_link_not_requested": link},
            "role": "UNASSIGNED; originalTEST and publisherVAL HELD", "source_accepted": False,
            "actual_files_GT_rights_chain_roles_capture_groups_graph_minima_budget_audited": False,
            "ready": False}


def run(args, policy, core):
    core.need(not DATA.exists() and not RUN.exists(), "prior attempt preserved; no retry")
    commit = core.publication(args.publication_receipt, policy)
    observed = core.preflight(policy)
    own = {"PID": os.getpid(), "token": uuid.uuid4().hex, "stage": "v230", "UTC": core.utc()}
    core.save(core.LOCK, own)
    attempts = 0
    try:
        DATA.mkdir()
        RUN.mkdir()
        core.save(RUN / "preflight.json", observed)
        core.save(RUN / "started.json", {"UTC": core.utc(), "publication_commit": commit,
                  "script_SHA256": core.sha(Path(__file__)), "policy_SHA256": core.sha(POLICY)})
        core.save(DATA / "source-protection.json", {"role": "UNASSIGNED", "originalTEST_publisherVAL_HELD": True,
                  "source_accepted": False, "ready": False, "archive_pixels_training": 0})
        headers = {"Accept": "application/json", "Accept-Encoding": "identity",
                   "User-Agent": "SafeLog-record-metadata-v230/1.0"}
        core.save(DATA / "request.json", {"URL": policy["metadata_url"], "headers": headers})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), core.NoRedirect())
        request = urllib.request.Request(policy["metadata_url"], headers=headers, method="GET")
        attempts = 1
        with opener.open(request, timeout=30) as response:
            selected = {key: response.headers.get_all(key, []) for key in
                        ("Content-Type", "Content-Length", "Content-Encoding", "ETag", "Last-Modified")}
            core.save(DATA / "response-headers.json", {"status": response.status,
                      "URL": response.geturl(), "selected": selected})
            core.need(response.status == 200 and response.geturl() == policy["metadata_url"], "HTTP status/URL")
            core.need(len(selected["Content-Type"]) == 1 and len(selected["Content-Encoding"]) <= 1
                      and response.headers["Content-Type"].split(";", 1)[0].strip().lower() == "application/json"
                      and response.headers.get("Content-Encoding", "identity").lower() == "identity", "JSON MIME/encoding")
            lengths = selected["Content-Length"]
            core.need(len(lengths) <= 1, "ambiguous Content-Length")
            declared = None
            if lengths:
                core.need(re.fullmatch(r"[0-9]+", lengths[0]), "invalid Content-Length")
                declared = int(lengths[0])
                core.need(0 < declared <= policy["max_metadata_body_bytes"], "declared body cap")
            with (DATA / "publisher-record.body").open("xb") as target:
                count = core.copy_bounded(response, target, policy["max_metadata_body_bytes"])
                target.flush()
                os.fsync(target.fileno())
            core.need(declared is None or declared == count, "metadata length mismatch")
        body = DATA / "publisher-record.body"
        summary = record(body.read_bytes(), policy, core)
        summary.update({"UTC": core.utc(), "metadata_bytes": body.stat().st_size,
                        "metadata_SHA256": core.sha(body), "HTTP_attempts": attempts})
        core.need(sum(p.stat().st_size for root in (DATA, RUN) for p in root.rglob("*") if p.is_file())
                  + 65536 <= policy["metadata_job_budget_bytes"], "metadata output budget")
        core.save(DATA / "record-declarations.json", summary)
        core.save(RUN / "complete.json", {"UTC": core.utc(), "success": True, "HTTP_attempts": attempts,
                  "archive_photos_masks_features_training": 0, "source_accepted": False, "ready": False})
        print(json.dumps(summary))
    except BaseException as error:
        if RUN.is_dir():
            body = DATA / "publisher-record.body"
            core.save(RUN / "failure.json", {"UTC": core.utc(), "type": type(error).__name__, "error": str(error),
                      "HTTP_status": error.code if isinstance(error, urllib.error.HTTPError) else None,
                      "HTTP_attempts": attempts, "saved_body_bytes": body.stat().st_size if body.exists() else 0,
                      "partials_preserved_no_retry": True, "ready": False})
        raise
    finally:
        if core.LOCK.exists() and core.strict(core.LOCK.read_bytes()) == own:
            core.LOCK.unlink()


def self_check(policy, core):
    example = {"id": policy["record_id"], "doi": policy["DOI"],
               "metadata": {"title": policy["title"], "creators": [{"name": "Zhao, Xingzhuang"}]},
               "files": [{"key": "Zenodo.zip", "checksum": policy["publisher_archive_checksum"],
                          "size": policy["declared_archive_min_bytes"],
                          "links": {"self": "https://zenodo.org/api/records/18010179/files/Zenodo.zip/content"}}]}
    core.need(record(json.dumps(example).encode(), policy, core)["ready"] is False, "offline identity guard")
    for change in ({"id": 0}, {"doi": "wrong"}, {"files": []}):
        try:
            record(json.dumps(dict(example, **change)).encode(), policy, core)
        except core.Stop:
            pass
        else:
            raise AssertionError("invalid record identity accepted")
    print(json.dumps({"offline_guard_checks": "passed", "HTTP": 0, "source_data": 0, "training": 0}))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--publication-receipt", type=Path)
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    core = load_core()
    if args.self_check:
        self_check(policy, core)
    else:
        core.need(args.publication_receipt is not None, "exact3 publication receipt required")
        run(args, policy, core)


if __name__ == "__main__":
    main()
