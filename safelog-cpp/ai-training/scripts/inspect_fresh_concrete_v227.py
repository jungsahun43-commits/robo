"""One preregistered anonymous publisher file-list GET; never dataset bytes.

The offline self-check makes no requests. Live execution requires a receipt for
the exact three preregistered blobs and performs fresh process/lock/pin checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import uuid

AI = Path(__file__).resolve().parents[1]
REPO = AI.parents[1]
WORK = REPO.parents[1]
POLICY = AI / "reports/FRESH_CONCRETE_V227_METADATA_PROTOCOL.json"
DATA = AI / "data/fresh-concrete-v227-metadata"
RUN = AI / "runs/fresh-concrete-v227-metadata"
LOCK = AI / "operation.lock"
GIT = "C:/Program Files/Git/cmd/git.exe"


class Stop(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise Stop(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise Stop("nonfinite JSON number: " + value)

    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs,
                      parse_constant=invalid)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8388608), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    need(len(raw) <= 65536, "receipt budget")
    with path.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    need(path.read_bytes() == raw, "receipt readback")


def age(value, maximum):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(stamp.tzinfo is not None, "timezone required")
    delta = (datetime.now(timezone.utc) - stamp).total_seconds()
    need(-5 <= delta <= maximum, "stale or future receipt")


def publication(receipt_path, policy):
    receipt = strict(receipt_path.read_bytes())
    need(receipt.get("success") is True and receipt.get("ordinary_nonforce") is True,
         "verified ordinary preregistration required")
    need(receipt.get("destination") == policy["destination"]
         and receipt.get("ref") == policy["ref"], "publication destination")
    commit = receipt.get("verified_commit", "")
    need(re.fullmatch(r"[0-9a-f]{40}", commit), "publication commit")
    age(receipt["verified_UTC"], 1800)
    rows = receipt["files"]
    need(len(rows) == 3 and {x["path"] for x in rows} == set(policy["publication_paths"]),
         "exact three preregistered paths")
    for row in rows:
        path = REPO / row["path"]
        need(path.stat().st_size == row["bytes"] and sha(path) == row["SHA256"],
             "changed preregistered file")
        blob = subprocess.run([GIT, "-c", "safe.directory=" + str(REPO), "-C", str(REPO),
                               "show", commit + ":" + row["path"]],
                              capture_output=True, check=True, timeout=20).stdout
        need(hashlib.sha256(blob).hexdigest() == row["SHA256"], "published blob mismatch")
    return commit


def preflight(policy):
    command = r"""$ErrorActionPreference='Stop'
$p=@(Get-CimInstance Win32_Process -ErrorAction Stop)
$r=@($p | Where-Object { $_.ProcessId -ne PYPID -and (($_.Name -match '^(python|pythonw|safelog.*|nvidia-smi)\.exe$') -or $_.ProcessId -eq 23948) } | ForEach-Object { [ordered]@{pid=$_.ProcessId;name=$_.Name;creation=$_.CreationDate;command_line=$_.CommandLine} })
[ordered]@{UTC=[DateTime]::UtcNow.ToString('o');cim_succeeded=$true;related=$r} | ConvertTo-Json -Depth 5 -Compress
""".replace("PYPID", str(os.getpid()))
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, timeout=45,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    need(result.returncode == 0, "CIM failed; process status UNKNOWN")
    observed = strict(result.stdout.decode("utf-8-sig").encode())
    need(observed.get("cim_succeeded") is True and observed.get("related") == [],
         "related process present or process state UNKNOWN")
    age(observed["UTC"], 180)
    locks = ([LOCK] if LOCK.exists() else [])
    locks += list((AI / "runs").rglob("operation.lock"))
    locks += list((WORK / "outputs").rglob("operation.lock"))
    need(not locks, "foreign operation lock present")
    observed["operation_locks"] = []
    observed["lock_scope"] = "AI root/runs and WORK outputs only; not entire source tree"
    observed["protected10"] = []
    for row in policy["protected10"]:
        path = WORK / row["path"]
        need(path.stat().st_size == row["bytes"] and sha(path) == row["SHA256"],
             "protected identity changed")
        observed["protected10"].append(row)
    ledger = AI / "runs/industrial-corrosion-context-v85/optimizer.jsonl"
    with ledger.open("rb") as source:
        source.seek(max(0, ledger.stat().st_size - 8192))
        last = strict(source.read().splitlines()[-1])
    need(last["actual_update"] == 450 and last["epoch"] == 30, "v85 ledger identity changed")
    observed["v85_last_optimizer_entry"] = {"actual_update": 450, "epoch": 30}
    observed["free_bytes"] = shutil.disk_usage(WORK).free
    need(observed["free_bytes"] >= policy["reserve_bytes"] + policy["metadata_job_budget_bytes"],
         "metadata storage budget")
    age(observed["UTC"], 180)
    return observed


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Stop("redirect rejected before next request: " + str(code))


def copy_bounded(response, target, cap):
    count = 0
    while True:
        block = response.read(min(16384, cap + 1 - count))
        if not block:
            break
        target.write(block)
        count += len(block)
        need(count <= cap, "metadata body exceeds cap; partial preserved")
    return count


def file_list(raw, policy):
    rows = strict(raw)
    need(isinstance(rows, list) and len(rows) == 1, "unexpected publisher root file list")
    item = rows[0]
    need(isinstance(item, dict), "publisher file record required")
    identity = item.get("id", "")
    need(isinstance(identity, str) and re.fullmatch(
        r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", identity), "file UUID")
    need(item.get("filename") == policy["expected_file_name"]
         and item.get("status") == "COMPLETED", "filename/status mismatch")
    details = item.get("content_details", {})
    size, checksum = details.get("size"), details.get("sha256_hash", "")
    need(type(size) is int and policy["archive_declaration_min_bytes"] <= size
         <= policy["archive_declaration_max_bytes"], "archive size declaration outside bound")
    need(isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum),
         "publisher SHA256 declaration required")
    url = "https://data.mendeley.com/public-files/datasets/r8zrzk5w8j/files/" + identity + "/file_downloaded"
    need(details.get("download_url") == url, "unexpected publisher file URL")
    return {"filename": item["filename"], "publisher_bytes": size,
            "publisher_SHA256": checksum, "publisher_url_not_requested": url,
            "role": "UNASSIGNED; originalTEST and publisherVAL HELD",
            "source_accepted": False, "ready_for_data_acquisition_or_training": False}


def live(receipt_path, policy):
    need(os.name == "nt", "managed Windows workspace required")
    need(not DATA.exists() and not RUN.exists(), "prior attempt preserved; no retry")
    commit = publication(receipt_path, policy)
    observed = preflight(policy)
    own = {"PID": os.getpid(), "token": uuid.uuid4().hex, "stage": "v227", "UTC": utc()}
    save(LOCK, own)
    attempts = 0
    try:
        RUN.mkdir()
        DATA.mkdir()
        save(RUN / "preflight.json", observed)
        save(RUN / "started.json", {"UTC": utc(), "publication_commit": commit,
             "script_SHA256": sha(Path(__file__)), "policy_SHA256": sha(POLICY)})
        save(DATA / "source-protection.json", {"role": "UNASSIGNED", "originalTEST_publisherVAL_HELD": True,
             "source_accepted": False, "ready": False, "photos_masks_code_training": 0})
        headers = {"Accept": "application/vnd.mendeley-public-dataset.1+json",
                   "Accept-Encoding": "identity", "User-Agent": "SafeLog-metadata-v227/1.0"}
        save(DATA / "request.json", {"URL": policy["metadata_url"], "headers": headers})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        req = urllib.request.Request(policy["metadata_url"], headers=headers, method="GET")
        attempts = 1
        with opener.open(req, timeout=30) as response:
            need(response.status == 200 and response.geturl() == policy["metadata_url"], "HTTP status/URL")
            selected = {k: response.headers.get_all(k, []) for k in
                        ("Content-Type", "Content-Length", "Content-Encoding", "ETag", "Last-Modified")}
            save(DATA / "response-headers.json", {"status": response.status, "selected": selected})
            mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            need(len(response.headers.get_all("Content-Type", [])) == 1
                 and len(response.headers.get_all("Content-Encoding", [])) <= 1, "ambiguous MIME/encoding")
            need(mime in ("application/json", "application/vnd.mendeley-public-dataset.1+json"), "JSON MIME")
            need(response.headers.get("Content-Encoding", "identity").lower() == "identity", "identity encoding")
            lengths = response.headers.get_all("Content-Length", [])
            need(len(lengths) <= 1, "ambiguous Content-Length")
            declared = None
            if lengths:
                need(re.fullmatch(r"[0-9]+", lengths[0]), "invalid Content-Length")
                declared = int(lengths[0])
                need(0 < declared <= policy["max_metadata_body_bytes"], "declared body cap")
            with (DATA / "publisher-files.body").open("xb") as target:
                count = copy_bounded(response, target, policy["max_metadata_body_bytes"])
                target.flush()
                os.fsync(target.fileno())
            need(declared is None or count == declared, "body length mismatch")
        body = DATA / "publisher-files.body"
        selected = file_list(body.read_bytes(), policy)
        selected.update({"UTC": utc(), "metadata_body_bytes": body.stat().st_size,
                         "metadata_body_SHA256": sha(body), "HTTP_attempts": attempts})
        need(sum(p.stat().st_size for root in (DATA, RUN) for p in root.rglob("*") if p.is_file())
             + 65536 <= policy["metadata_job_budget_bytes"], "metadata output budget")
        save(DATA / "publisher-file-declarations.json", selected)
        save(RUN / "complete.json", {"UTC": utc(), "success": True, "HTTP_attempts": attempts,
                                    "ready": False, "source_archive_training": 0})
        print(json.dumps(selected))
    except BaseException as error:
        if RUN.is_dir():
            body = DATA / "publisher-files.body"
            save(RUN / "failure.json", {"UTC": utc(), "type": type(error).__name__,
                 "error": str(error), "HTTP_attempts": attempts,
                 "HTTP_status": error.code if isinstance(error, urllib.error.HTTPError) else None,
                 "saved_body_bytes": body.stat().st_size if body.exists() else 0,
                 "partials_preserved_no_retry": True, "ready": False})
        raise
    finally:
        if LOCK.exists() and strict(LOCK.read_bytes()) == own:
            LOCK.unlink()


def self_check(policy):
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        try:
            strict(raw)
        except Stop:
            pass
        else:
            raise AssertionError("invalid JSON accepted")
    target = io.BytesIO()
    need(copy_bounded(io.BytesIO(b"abc"), target, 3) == 3, "exact bounded read")
    try:
        copy_bounded(io.BytesIO(b"abcd"), io.BytesIO(), 3)
    except Stop:
        pass
    else:
        raise AssertionError("oversize accepted")
    try:
        NoRedirect().redirect_request(None, None, 302, "", None, "https://example.invalid")
    except Stop:
        pass
    else:
        raise AssertionError("redirect accepted")
    for raw in (b"[]", b"{}", b"[{}]"):
        try:
            file_list(raw, policy)
        except Stop:
            pass
        else:
            raise AssertionError("incomplete file declaration accepted")
    print(json.dumps({"offline_guard_checks": "passed", "HTTP": 0,
                      "source_data": 0, "training": 0, "ready": False}))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--publication-receipt", type=Path)
    args = parser.parse_args()
    policy = strict(POLICY.read_bytes())
    if args.self_check:
        self_check(policy)
    else:
        need(args.publication_receipt is not None, "publication receipt required")
        live(args.publication_receipt, policy)


if __name__ == "__main__":
    main()
