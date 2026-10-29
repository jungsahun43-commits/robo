"""Bounded ZIP directory/README metadata only; no image or mask decoding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib

AI = Path(__file__).resolve().parents[1]
WORK = AI.parents[3]
POLICY = AI / "reports/FRESH_CONCRETE_V228_ZIP_METADATA_PROTOCOL.json"
DATA = AI / "data/fresh-concrete-v228-zip-metadata"
RUN = AI / "runs/fresh-concrete-v228-zip-metadata"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    need(len(raw) <= 2097152, "metadata receipt cap")
    with path.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    need(path.read_bytes() == raw, "metadata readback")


def safe_name(name):
    need(isinstance(name, str) and 0 < len(name) <= 512 and "\\" not in name
         and ":" not in name and "\x00" not in name, "unsafe ZIP member name")
    path = PurePosixPath(name)
    need(not path.is_absolute() and all(x and x not in (".", "..") and not x.endswith((".", " "))
         for x in name.rstrip("/").split("/")), "unsafe ZIP member path")
    return name


def eocd(tail, total, policy):
    offset = tail.rfind(b"PK\x05\x06")
    need(offset >= 0 and len(tail) - offset >= 22, "ZIP end directory missing")
    row = struct.unpack_from("<4s4H2LH", tail, offset)
    need(offset + 22 + row[7] == len(tail), "ZIP comment/end mismatch")
    need(row[1] == row[2] == 0 and row[3] == row[4], "multi-disk ZIP rejected")
    need(0 < row[4] <= policy["max_members"] and row[4] != 65535, "member count/ZIP64 rejected")
    size, start = row[5], row[6]
    absolute = total - len(tail) + offset
    need(0 < size <= policy["max_directory_bytes"] and start + size == absolute,
         "directory bounds/ZIP64 rejected")
    return start, size, row[4]


def directory(raw, count, total):
    rows, cursor, seen = [], 0, set()
    while cursor < len(raw):
        need(cursor + 46 <= len(raw), "short central record")
        row = struct.unpack_from("<4s6H3L5H2L", raw, cursor)
        need(row[0] == b"PK\x01\x02" and row[13] == 0 and not row[3] & 1,
             "invalid/encrypted/multi-disk member")
        length = 46 + row[10] + row[11] + row[12]
        need(cursor + length <= len(raw), "central field bounds")
        name = raw[cursor + 46:cursor + 46 + row[10]].decode("utf-8" if row[3] & 2048 else "cp437")
        safe_name(name)
        need(name.casefold() not in seen, "duplicate/case-colliding member")
        seen.add(name.casefold())
        mode = row[15] >> 16
        need(stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR), "special/symlink member rejected")
        need(row[16] < total and row[8] != 4294967295 and row[9] != 4294967295,
             "member bounds/ZIP64 rejected")
        rows.append({"name": name, "flags": row[3], "method": row[4], "CRC32": row[7],
                     "compressed_bytes": row[8], "uncompressed_bytes": row[9],
                     "local_header_offset": row[16]})
        cursor += length
    need(cursor == len(raw) and len(rows) == count, "directory count mismatch")
    return rows


def readme_payload(header, payload, entry, policy):
    need(len(header) == 30, "local header size")
    h = struct.unpack("<4s5H3L2H", header)
    need(h[0] == b"PK\x03\x04" and h[2] == entry["flags"] and h[3] == entry["method"],
         "README local header mismatch")
    name_bytes = payload[:h[9]]
    name = name_bytes.decode("utf-8" if h[2] & 2048 else "cp437")
    need(name == entry["name"], "README name mismatch")
    need(len(payload) == h[9] + h[10] + entry["compressed_bytes"], "README payload bounds")
    compressed = payload[h[9] + h[10]:]
    if entry["method"] == 0:
        raw = compressed
    else:
        need(entry["method"] == 8, "unsupported README compression")
        decoder = zlib.decompressobj(-15)
        raw = decoder.decompress(compressed, policy["max_readme_bytes"] + 1)
        need(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
             "README deflate bound/trailing data")
    need(len(raw) == entry["uncompressed_bytes"] <= policy["max_readme_bytes"]
         and zlib.crc32(raw) & 4294967295 == entry["CRC32"], "README length/CRC mismatch")
    raw.decode("utf-8-sig", "strict")
    return raw


def allowed_delivery(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    return (parsed.scheme == "https" and parsed.port in (None, 443)
            and parsed.username is None and parsed.password is None and len(url) <= 8192
            and (host == "data.mendeley.com" or re.fullmatch(
                r"prod-dcd-datasets-public-files(?:-[a-z0-9-]+)?\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com", host)))


class DeliveryRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, state):
        self.state, self.redirects = state, 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirects += 1
        need(self.redirects <= 1 and code in (301, 302, 303, 307, 308)
             and allowed_delivery(newurl), "unapproved delivery redirect")
        self.state["HTTP_requests"] += 1
        self.state["delivery_redirects"].append({"status": code, "observed_target": newurl})
        need(self.state["HTTP_requests"] <= 8, "HTTP request cap")
        return urllib.request.Request(newurl, headers=dict(req.header_items()), method="GET")


def live(args, policy, core):
    need(not DATA.exists() and not RUN.exists(), "prior attempt preserved; no retry")
    commit = core.publication(args.publication_receipt, policy)
    observed = core.preflight(policy)
    source = AI / "data/fresh-concrete-v227-metadata/publisher-files.body"
    need(core.sha(source) == policy["source_metadata_SHA256"], "source metadata identity changed")
    own = {"PID": os.getpid(), "token": uuid.uuid4().hex, "stage": "v228", "UTC": core.utc()}
    core.save(core.LOCK, own)
    state = {"HTTP_requests": 0, "range_GETs": 0, "saved_fragment_bytes": 0,
             "delivery_redirects": [], "ETag": None}
    try:
        DATA.mkdir()
        RUN.mkdir()
        save(RUN / "preflight.json", observed)
        save(RUN / "started.json", {"UTC": core.utc(), "publication_commit": commit,
             "script_SHA256": core.sha(Path(__file__)), "policy_SHA256": core.sha(POLICY)})
        save(DATA / "source-protection.json", {"role": "UNASSIGNED", "originalTEST_publisherVAL_HELD": True,
             "source_accepted": False, "ready": False, "image_mask_annotation_decoding": 0})

        def get_range(start, end, label):
            need(type(start) is type(end) is int and 0 <= start <= end < policy["archive_bytes"],
                 "range bounds")
            need(end - start + 1 <= policy["max_range_bytes"] and state["range_GETs"] < 4,
                 "range request cap")
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                       "Accept": "application/zip, application/octet-stream",
                       "User-Agent": "SafeLog-ZIP-metadata-v228/1.0"}
            if state["ETag"] is not None:
                headers["If-Match"] = state["ETag"]
            state["range_GETs"] += 1
            state["HTTP_requests"] += 1
            save(DATA / (label + "-request.json"), {"URL": policy["archive_url"], "headers": headers})
            redirect = DeliveryRedirect(state)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), redirect)
            req = urllib.request.Request(policy["archive_url"], headers=headers, method="GET")
            path = DATA / (label + ".fragment")
            with opener.open(req, timeout=30) as response:
                selected = {k: response.headers.get_all(k, []) for k in
                            ("Content-Type", "Content-Length", "Content-Range", "Content-Encoding", "ETag")}
                save(DATA / (label + "-headers.json"), {"status": response.status,
                     "final_URL": response.geturl(), "headers": selected})
                need(response.status == 206 and allowed_delivery(response.geturl()),
                     "range not supported or unexpected delivery; no full body read")
                need(selected["Content-Range"] == [f"bytes {start}-{end}/{policy['archive_bytes']}"]
                     and selected["Content-Length"] == [str(end - start + 1)], "range header mismatch")
                need(len(selected["Content-Type"]) == 1 and len(selected["Content-Encoding"]) <= 1
                     and response.headers.get("Content-Encoding", "identity").lower() == "identity",
                     "MIME/encoding ambiguity")
                need(response.headers["Content-Type"].split(";", 1)[0].strip().lower()
                     in ("application/zip", "application/octet-stream"), "archive MIME")
                tags = selected["ETag"]
                need(len(tags) == 1 and len(tags[0]) <= 512 and tags[0].startswith('"')
                     and tags[0].endswith('"') and not tags[0].startswith("W/"), "strong ETag required")
                if state["ETag"] is None:
                    state["ETag"] = tags[0]
                need(state["ETag"] == tags[0], "archive revision changed between ranges")
                with path.open("xb") as target:
                    count = core.copy_bounded(response, target, end - start + 1)
                    target.flush()
                    os.fsync(target.fileno())
                need(count == end - start + 1, "short range body")
            state["saved_fragment_bytes"] += path.stat().st_size
            need(state["saved_fragment_bytes"] <= policy["max_total_fragment_bytes"], "fragment budget")
            save(DATA / (label + "-integrity.json"), {"bytes": path.stat().st_size,
                 "SHA256": core.sha(path), "range": [start, end], "whole_archive_SHA256_verified": False})
            return path.read_bytes()

        total = policy["archive_bytes"]
        tail = get_range(total - 65557, total - 1, "zip-tail")
        start, size, count = eocd(tail, total, policy)
        if start >= total - len(tail):
            central = tail[start - total + len(tail):start - total + len(tail) + size]
        else:
            central = get_range(start, start + size - 1, "zip-directory")
        rows = directory(central, count, total)
        save(DATA / "directory-metadata.json", {"entries": rows, "count": count,
             "directory_range": [start, start + size - 1], "whole_archive_SHA256_verified": False})
        readmes = [row for row in rows if PurePosixPath(row["name"]).name.casefold() == "readme.txt"]
        need(len(readmes) == 1, "one author README.txt required")
        entry = readmes[0]
        need(entry["method"] in (0, 8) and 0 < entry["uncompressed_bytes"] <= policy["max_readme_bytes"]
             and 0 < entry["compressed_bytes"] <= policy["max_readme_bytes"], "README budget/type")
        offset = entry["local_header_offset"]
        header = get_range(offset, offset + 29, "readme-local-header")
        local = struct.unpack("<4s5H3L2H", header)
        payload_size = local[9] + local[10] + entry["compressed_bytes"]
        need(0 < payload_size <= policy["max_readme_payload_bytes"]
             and offset + 30 + payload_size <= start, "README record bounds")
        payload = get_range(offset + 30, offset + 29 + payload_size, "readme-compressed-payload")
        text = readme_payload(header, payload, entry, policy)
        with (DATA / "author-README.txt").open("xb") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        summary = {"UTC": core.utc(), **state, "directory_entries": count, "README_member": entry["name"],
             "README_bytes": len(text), "README_SHA256": hashlib.sha256(text).hexdigest(),
             "whole_archive_SHA256_verified": False, "source_accepted": False, "ready": False,
             "images_masks_annotations_decoded": 0, "features_optimizer_CAL_TEST": 0,
             "archive_fragments_may_include_incidental_compressed_member_bytes": True}
        need(sum(p.stat().st_size for root in (DATA, RUN) for p in root.rglob("*") if p.is_file())
             + 65536 <= policy["metadata_job_budget_bytes"], "metadata output budget")
        save(RUN / "complete.json", {"success": True, **summary})
        print(json.dumps(summary))
    except BaseException as error:
        if RUN.is_dir():
            save(RUN / "failure.json", {"UTC": core.utc(), "type": type(error).__name__,
                 "error": str(error), "HTTP_status": error.code if isinstance(error, urllib.error.HTTPError) else None,
                 **state, "partials_preserved_no_retry": True, "ready": False})
        raise
    finally:
        if core.LOCK.exists() and core.strict(core.LOCK.read_bytes()) == own:
            core.LOCK.unlink()


def self_check(policy):
    for name in ("../bad", "/absolute", "C:stream", "x\\bad", "dir./bad", "dir /bad"):
        try:
            safe_name(name)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe path accepted")
    sample = io.BytesIO()
    with zipfile.ZipFile(sample, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dataset/README.txt", "Offline parser guard check only.\n")
        archive.writestr("dataset/train/example.bin", b"not source imagery")
    raw = sample.getvalue()
    start, size, count = eocd(raw, len(raw), policy)
    rows = directory(raw[start:start + size], count, len(raw))
    entry = rows[0]
    offset = entry["local_header_offset"]
    header = raw[offset:offset + 30]
    h = struct.unpack("<4s5H3L2H", header)
    payload = raw[offset + 30:offset + 30 + h[9] + h[10] + entry["compressed_bytes"]]
    need(readme_payload(header, payload, entry, policy).startswith(b"Offline"), "README parser check")
    try:
        readme_payload(header, payload, dict(entry, CRC32=0), policy)
    except RuntimeError:
        pass
    else:
        raise AssertionError("CRC corruption accepted")
    need(not allowed_delivery("http://data.mendeley.com/x")
         and not allowed_delivery("https://user:password@data.mendeley.com/x")
         and not allowed_delivery("https://example.com/x"), "delivery host guard")
    print(json.dumps({"offline_guard_checks": "passed", "HTTP": 0, "source_data": 0, "training": 0}))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--publication-receipt", type=Path)
    args = parser.parse_args()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if args.self_check:
        self_check(policy)
        return
    dependency = AI / "scripts/inspect_fresh_concrete_v227.py"
    need(hashlib.sha256(dependency.read_bytes()).hexdigest() == policy["dependency_SHA256"],
         "preregistered helper dependency changed")
    spec = importlib.util.spec_from_file_location("v227_preregistered_core", dependency)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    need(args.publication_receipt is not None, "exact3 publication receipt required")
    live(args, policy, core)


if __name__ == "__main__":
    main()
