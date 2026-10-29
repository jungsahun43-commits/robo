"""One pinned-revision CiF file metadata GET; no parquet/image/GT acquisition."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request
import uuid

AI = Path(__file__).resolve().parents[1]
POLICY = AI / "reports/CIF_V232_METADATA_PROTOCOL.json"
DATA = AI / "data/cif-v232-file-metadata"
RUN = AI / "runs/cif-v232-file-metadata"
CORE_SHA = "f72eb1deea38ac4f615e2b6c53d30fc67a5d7a9b3b7f3f708bb839d0140be999"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def core_module():
    path = AI / "scripts/inspect_fresh_concrete_v227.py"
    need(hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA, "pinned guard dependency changed")
    spec = importlib.util.spec_from_file_location("v227_pinned_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def inventory(info, policy):
    need(isinstance(info, dict) and info.get("id") == policy["repo_id"], "dataset identity")
    need(info.get("private") is False and info.get("gated") is False
         and info.get("disabled", False) is False, "public ungated active record required")
    revision = info.get("sha", "")
    need(isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision)
         and revision.startswith(policy["observed_short_revision"]), "observed revision mismatch")
    need(info.get("cardData", {}).get("license") == "cdla-permissive-2.0", "license declaration changed")
    siblings = info.get("siblings")
    need(isinstance(siblings, list) and 0 < len(siblings) <= policy["max_files"], "file inventory cap")
    groups, seen, rows = {}, set(), []
    for item in siblings:
        need(isinstance(item, dict), "file object required")
        name = item.get("rfilename", "")
        need(isinstance(name, str) and 0 < len(name) <= 160 and name not in seen,
             "duplicate/invalid file identity")
        seen.add(name)
        size, blob = item.get("size"), item.get("blobId", "")
        need(type(size) is int and 0 < size <= policy["max_declared_shard_bytes"]
             and isinstance(blob, str) and re.fullmatch("[0-9a-f]{40}", blob), "size/Git blob metadata")
        row = {"path": name, "publisher_bytes": size, "Git_blob_declaration": blob,
               "actual_file_checksum_verified": False}
        if name in ("README.md", ".gitattributes"):
            need(size <= 65536, "root metadata size cap")
        else:
            match = re.fullmatch(r"data/((?:train|val|test)_(?:full|tiled))-(\d{5})-of-(\d{5})\.parquet", name)
            need(match is not None, "unexpected inventory path; no path guessing")
            role, index, count = match[1], int(match[2]), int(match[3])
            need(0 <= index < count <= policy["max_files"], "shard index bounds")
            lfs = item.get("lfs")
            need(isinstance(lfs, dict), "publisher LFS checksum metadata required")
            checksum = lfs.get("sha256", lfs.get("oid", ""))
            need(isinstance(checksum, str) and re.fullmatch("[0-9a-f]{64}", checksum)
                 and lfs.get("size") == size, "publisher LFS size/checksum mismatch")
            row.update({"split_declaration": role, "publisher_LFS_SHA256": checksum,
                        "role": "UNASSIGNED" if role.startswith("train_") else "HELD"})
            groups.setdefault(role, []).append((index, count, size))
        rows.append(row)
    expected = {r + "_" + k for r in ("train", "val", "test") for k in ("full", "tiled")}
    need(set(groups) == expected, "six original split families required")
    counts = {}
    for role, members in groups.items():
        total = members[0][1]
        need(len(members) == total and all(x[1] == total for x in members)
             and sorted(x[0] for x in members) == list(range(total)), "incomplete shard inventory")
        counts[role] = {"files": total, "publisher_bytes": sum(x[2] for x in members)}
    total = sum(row["publisher_bytes"] for row in rows)
    need(policy["min_total_declared_bytes"] <= total <= policy["max_total_declared_bytes"],
         "declared repository size outside primary-page bound")
    return rows, {"repo_id": info["id"], "full_revision": revision,
                  "license_declaration": info["cardData"]["license"],
                  "files": len(rows), "split_file_counts_bytes": counts, "total_declared_bytes": total,
                  "source_accepted": False, "ready": False, "parquet_image_annotation_requests": 0}


def live(args, policy, core):
    need(not DATA.exists() and not RUN.exists(), "existing attempt preserved; no source retry")
    commit = core.publication(args.publication_receipt, policy)
    preflight = core.preflight(policy)
    command = "$ErrorActionPreference='Stop'; [int64](Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreePhysicalMemory * 1024"
    ram = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                         capture_output=True, check=True, timeout=30,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    preflight["free_RAM_bytes"] = int(ram.stdout.decode().strip())
    need(preflight["free_RAM_bytes"] >= policy["RAM_budget_bytes"] + policy["RAM_reserve_bytes"], "metadata RAM budget")
    core.age(preflight["UTC"], 180)
    own = {"PID": os.getpid(), "token": uuid.uuid4().hex, "stage": "v232", "UTC": core.utc()}
    core.save(core.LOCK, own)
    attempts = 0
    try:
        DATA.mkdir()
        RUN.mkdir()
        core.save(RUN / "preflight.json", preflight)
        core.save(RUN / "started.json", {"UTC": core.utc(), "publication_commit": commit,
                  "script_SHA256": core.sha(Path(__file__)), "policy_SHA256": core.sha(POLICY)})
        core.save(DATA / "source-protection.json", {"role": "UNASSIGNED", "originalTEST_publisherVAL_HELD": True,
                  "source_accepted": False, "ready": False, "no_tile_original_double_count": True})
        headers = {"Accept": "application/json", "Accept-Encoding": "identity",
                   "User-Agent": "SafeLog-CiF-file-metadata-v232/1.0"}
        core.save(DATA / "request.json", {"URL": policy["metadata_url"], "headers": headers})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), core.NoRedirect())
        request = urllib.request.Request(policy["metadata_url"], headers=headers, method="GET")
        attempts = 1
        target = DATA / "publisher-file-metadata.body"
        with opener.open(request, timeout=policy["timeout_seconds"]) as response:
            selected = {key: response.headers.get_all(key, []) for key in
                        ("Content-Type", "Content-Length", "Content-Encoding", "Link", "ETag")}
            core.save(DATA / "response-headers.json", {"status": response.status,
                      "final_URL": response.geturl(), "headers": selected})
            need(response.status == 200 and response.geturl() == policy["metadata_url"], "exact200 URL required")
            need(len(selected["Content-Type"]) == 1 and response.headers.get_content_type() == "application/json"
                 and len(selected["Content-Encoding"]) <= 1
                 and response.headers.get("Content-Encoding", "identity").lower() == "identity", "JSON MIME/encoding")
            need(not any('rel="next"' in x or "rel=next" in x for x in selected["Link"]), "unresolved pagination")
            need(len(selected["Content-Length"]) <= 1, "duplicate response length")
            length = response.headers.get("Content-Length")
            if length is not None:
                need(length.isdecimal() and 0 < int(length) <= policy["max_metadata_body_bytes"], "metadata length cap")
            with target.open("xb") as output:
                count = core.copy_bounded(response, output, policy["max_metadata_body_bytes"])
                output.flush()
                os.fsync(output.fileno())
            need(length is None or count == int(length), "short metadata body")
        rows, summary = inventory(core.strict(target.read_bytes()), policy)
        ledger = DATA / "original-file-inventory.jsonl"
        with ledger.open("xb") as output:
            written = 0
            for row in sorted(rows, key=lambda x: x["path"]):
                raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                written += len(raw)
                need(written <= policy["max_inventory_jsonl_bytes"], "inventory output cap")
                output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        summary.update({"UTC": core.utc(), "HTTP_attempts": attempts, "metadata_bytes": target.stat().st_size,
                        "metadata_SHA256": core.sha(target), "inventory_SHA256": core.sha(ledger),
                        "file_checksums_are_declarations_not_acquired_data_hashes": True})
        core.save(DATA / "record-declarations.json", summary)
        need(sum(p.stat().st_size for root in (DATA, RUN) for p in root.iterdir() if p.is_file()) + 65536
             <= policy["metadata_job_budget_bytes"], "metadata stage budget")
        core.save(RUN / "complete.json", {"success": True, **summary})
        print(json.dumps(summary))
    except BaseException as error:
        if RUN.is_dir():
            core.save(RUN / "failure.json", {"UTC": core.utc(), "type": type(error).__name__, "error": str(error),
                      "HTTP_status": error.code if isinstance(error, urllib.error.HTTPError) else None,
                      "HTTP_attempts": attempts, "partials_preserved_no_retry": True, "ready": False})
        raise
    finally:
        if core.LOCK.exists() and core.strict(core.LOCK.read_bytes()) == own:
            core.LOCK.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    args = parser.parse_args()
    core = core_module()
    policy = core.strict(POLICY.read_bytes())
    live(args, policy, core)


if __name__ == "__main__":
    main()
