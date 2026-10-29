"""Bounded remote ZIP/ZIP64 directory and small documentation metadata only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

AI = Path(__file__).resolve().parents[1]
POLICY = AI / "reports/PAGG_V231_ZIP_METADATA_PROTOCOL.json"
DATA = AI / "data/pagg-v231-zip-metadata"
RUN = AI / "runs/pagg-v231-zip-metadata"
CORE_SHA = "f72eb1deea38ac4f615e2b6c53d30fc67a5d7a9b3b7f3f708bb839d0140be999"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def core_module():
    path = AI / "scripts/inspect_fresh_concrete_v227.py"
    need(hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA, "pinned guard dependency changed")
    spec = importlib.util.spec_from_file_location("v227_pinned_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def delivery(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    return (parsed.scheme == "https" and parsed.port in (None, 443)
            and parsed.username is None and parsed.password is None and len(url) <= 8192
            and (host == "zenodo.org" or host.endswith(".zenodo.org")))


class Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, state):
        self.state, self.count = state, 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.count += 1
        self.state["redirects"].append({"status": code, "observed_target_not_yet_followed": newurl})
        need(self.count <= 1 and code in (301, 302, 303, 307, 308) and delivery(newurl),
             "unapproved publisher delivery redirect")
        self.state["HTTP_requests"] += 1
        need(self.state["HTTP_requests"] <= 32, "HTTP request cap")
        return urllib.request.Request(newurl, headers=dict(req.header_items()), method="GET")


class RangeReader(io.RawIOBase):
    def __init__(self, size, fetch):
        self.size, self.fetch, self.position, self.cache = size, fetch, 0, []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        need(whence in (0, 1, 2) and 0 <= value <= self.size, "ZIP seek bounds")
        self.position = value
        return value

    def read(self, length=-1):
        if length is None or length < 0:
            length = self.size - self.position
        length = min(length, self.size - self.position)
        need(0 <= length <= 16777216, "ZIP metadata read cap; full source read forbidden")
        if length == 0:
            return b""
        start, end = self.position, self.position + length - 1
        for base, body in self.cache:
            if base <= start and end < base + len(body):
                self.position += length
                return body[start - base:end - base + 1]
        # A small aligned block avoids separate requests for the local header,
        # filename and adjacent small document bytes. Incidental compressed
        # member bytes stay UNASSIGNED and are never decoded as imagery.
        if length < 4096:
            first = start // 4096 * 4096
            last = min(self.size - 1, (end // 4096 + 1) * 4096 - 1)
        else:
            first, last = start, end
        body = self.fetch(first, last)
        need(len(body) == last - first + 1, "short range cache")
        self.cache.append((first, body))
        self.position += length
        return body[start - first:end - first + 1]


def safe_member(info):
    name = info.filename
    need(info.orig_filename == name and 0 < len(name) <= 512 and "\\" not in name and ":" not in name and "\x00" not in name,
         "unsafe ZIP member name")
    path = PurePosixPath(name)
    need(not path.is_absolute() and all(x and x not in (".", "..") and not x.endswith((".", " "))
         for x in name.rstrip("/").split("/")), "unsafe ZIP member path")
    need(not info.flag_bits & 1 and stat.S_IFMT(info.external_attr >> 16)
         in (0, stat.S_IFREG, stat.S_IFDIR), "encrypted/special/symlink member")
    return name


def directory_bounds(stream, policy):
    # Bound the eager stdlib central-directory parser before it allocates ZipInfo objects.
    stream.seek(max(0, stream.size - 65558))
    tail = stream.read()
    pos = tail.rfind(b"PK\x05\x06")
    need(pos >= 0 and len(tail) - pos >= 22, "ZIP end record missing")
    end = struct.unpack_from("<4s4H2LH", tail, pos)
    need(len(tail) - pos == 22 + end[7] and end[1:3] == (0, 0), "ZIP end/multidisk boundary")
    count, size, offset = end[4], end[5], end[6]
    if count == 65535 or size == 4294967295 or offset == 4294967295:
        end_offset = stream.size - len(tail) + pos
        need(end_offset >= 20, "ZIP64 locator missing")
        stream.seek(end_offset - 20)
        locator = struct.unpack("<4sLQL", stream.read(20))
        need(locator[0] == b"PK\x06\x07" and locator[1] == 0 and locator[3] == 1,
             "ZIP64 locator/multidisk boundary")
        stream.seek(locator[2])
        large = struct.unpack("<4sQ2H2L4Q", stream.read(56))
        need(large[0] == b"PK\x06\x06" and large[1] == 44 and large[4:6] == (0, 0)
             and large[6] == large[7] and locator[2] + 56 == end_offset - 20,
             "ZIP64 record boundary")
        count, size, offset = large[7], large[8], large[9]
    else:
        need(end[3] == count, "ZIP per-disk member mismatch")
    need(0 < count <= policy["max_members"] and 0 < size <= policy["max_central_directory_bytes"]
         and offset + size <= stream.size, "central directory count/byte cap")
    stream.seek(0)
    return count


def inspect_archive(stream, policy, directory_target, document_target):
    declared_count = directory_bounds(stream, policy)
    with zipfile.ZipFile(stream, "r", allowZip64=True) as archive:
        infos = archive.infolist()
        need(len(infos) == declared_count, "member count mismatch")
        seen, total_names, docs, directory_bytes = set(), 0, [], 0
        for info in infos:
            name = safe_member(info)
            key = name.casefold()
            need(key not in seen, "duplicate/case-colliding ZIP member")
            seen.add(key)
            total_names += len(name)
            need(total_names <= policy["max_total_name_characters"], "member-name memory budget")
            row = {"name": name, "compressed_bytes": info.compress_size, "uncompressed_bytes": info.file_size,
                   "CRC32": info.CRC, "local_header_offset": info.header_offset, "method": info.compress_type}
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode()
            directory_bytes += len(encoded)
            need(directory_bytes <= policy["max_directory_jsonl_bytes"], "directory ledger budget")
            directory_target.write(encoded)
            parts = [x.casefold() for x in PurePosixPath(name).parts]
            if (not info.is_dir() and parts[-1] in policy["document_basenames"]
                    and not set(parts[:-1]) & {"test", "val", "valid", "validation", "cal"}
                    and 0 < info.file_size <= policy["max_document_bytes"]
                    and 0 < info.compress_size <= policy["max_document_bytes"]
                    and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                docs.append(info)
        selected = sorted(docs, key=lambda x: x.filename.casefold())[:policy["max_documents"]]
        documents = []
        for number, info in enumerate(selected, 1):
            with archive.open(info, "r") as source:
                raw = source.read(policy["max_document_bytes"] + 1)
                need(len(raw) == info.file_size <= policy["max_document_bytes"] and source.read(1) == b"",
                     "documentation size/CRC boundary")
            raw.decode("utf-8-sig", "strict")
            document_target(number, raw)
            documents.append({"member": info.filename, "bytes": len(raw),
                              "SHA256": hashlib.sha256(raw).hexdigest(), "CRC32": info.CRC})
        return {"directory_members": len(infos), "directory_ledger_bytes": directory_bytes,
                "eligible_small_document_count": len(docs), "documents": documents}


def live(args, policy, core):
    need(not DATA.exists() and not RUN.exists(), "prior attempt preserved; no retry")
    commit = core.publication(args.publication_receipt, policy)
    observed = core.preflight(policy)
    metadata = AI / "data/pagg-v230-record-metadata/publisher-record.body"
    need(core.sha(metadata) == policy["source_metadata_SHA256"], "publisher record identity changed")
    ram_command = "$ErrorActionPreference='Stop'; [int64](Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreePhysicalMemory * 1024"
    ram = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ram_command],
                         capture_output=True, check=True, timeout=30,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    observed["free_RAM_bytes"] = int(ram.stdout.decode().strip())
    need(observed["free_RAM_bytes"] >= policy["RAM_budget_bytes"] + policy["RAM_reserve_bytes"], "metadata RAM budget")
    core.age(observed["UTC"], 180)
    own = {"PID": os.getpid(), "token": uuid.uuid4().hex, "stage": "v231", "UTC": core.utc()}
    core.save(core.LOCK, own)
    state = {"range_GETs": 0, "HTTP_requests": 0, "saved_fragment_bytes": 0, "redirects": [], "ETag": None}
    started = time.monotonic()
    try:
        DATA.mkdir()
        RUN.mkdir()
        core.save(RUN / "preflight.json", observed)
        core.save(RUN / "started.json", {"UTC": core.utc(), "publication_commit": commit,
                  "script_SHA256": core.sha(Path(__file__)), "policy_SHA256": core.sha(POLICY)})
        core.save(DATA / "source-protection.json", {"role": "UNASSIGNED", "originalTEST_publisherVAL_HELD": True,
                  "source_accepted": False, "ready": False, "image_mask_annotation_decoding": 0})

        def fetch(start, end):
            remaining_seconds = policy["wall_seconds"] - (time.monotonic() - started)
            need(remaining_seconds > 0
                 and state["range_GETs"] < policy["max_range_GETs"], "metadata request/time budget")
            length = end - start + 1
            need(0 <= start <= end < policy["archive_bytes"] and length <= policy["max_range_bytes"]
                 and state["saved_fragment_bytes"] + length <= policy["max_total_fragment_bytes"], "range budget")
            state["range_GETs"] += 1
            state["HTTP_requests"] += 1
            label = "range-%02d" % state["range_GETs"]
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                       "Accept": "application/zip, application/octet-stream", "User-Agent": "SafeLog-ZIP-metadata-v231/1.0"}
            if state["ETag"] is not None:
                headers["If-Match"] = state["ETag"]
            core.save(DATA / (label + "-request.json"), {"URL": policy["archive_url"], "headers": headers})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirect(state))
            request = urllib.request.Request(policy["archive_url"], headers=headers, method="GET")
            path = DATA / (label + ".fragment")
            with opener.open(request, timeout=min(30, remaining_seconds)) as response:
                selected = {key: response.headers.get_all(key, []) for key in
                            ("Content-Type", "Content-Length", "Content-Range", "Content-Encoding", "ETag")}
                core.save(DATA / (label + "-headers.json"), {"status": response.status,
                          "final_URL": response.geturl(), "headers": selected})
                need(response.status == 206 and delivery(response.geturl()), "range unsupported; no full body read")
                need(selected["Content-Range"] == [f"bytes {start}-{end}/{policy['archive_bytes']}"]
                     and selected["Content-Length"] == [str(length)], "range identity/length")
                need(len(selected["Content-Type"]) == 1 and len(selected["Content-Encoding"]) <= 1
                     and response.headers.get("Content-Encoding", "identity").lower() == "identity"
                     and response.headers["Content-Type"].split(";", 1)[0].strip().lower()
                     in ("application/zip", "application/octet-stream"), "archive MIME/encoding")
                tags = selected["ETag"]
                need(len(tags) == 1 and 0 < len(tags[0]) <= 512 and tags[0].startswith('"')
                     and tags[0].endswith('"') and not tags[0].startswith("W/"), "strong ETag required")
                if state["ETag"] is None:
                    state["ETag"] = tags[0]
                need(state["ETag"] == tags[0], "archive revision changed")
                with path.open("xb") as target:
                    count = 0
                    while True:
                        need(time.monotonic() - started <= policy["wall_seconds"], "metadata elapsed budget")
                        block = response.read(min(16384, length + 1 - count))
                        if not block:
                            break
                        target.write(block)
                        count += len(block)
                        need(count <= length, "range body exceeds declared length; partial preserved")
                    target.flush()
                    os.fsync(target.fileno())
                need(count == length, "short metadata range")
            state["saved_fragment_bytes"] += path.stat().st_size
            core.save(DATA / (label + "-integrity.json"), {"range": [start, end], "bytes": path.stat().st_size,
                      "SHA256": core.sha(path), "whole_archive_MD5_verified": False})
            return path.read_bytes()

        reader = RangeReader(policy["archive_bytes"], fetch)
        def document_target(number, raw):
            with (DATA / ("author-document-%02d.txt" % number)).open("xb") as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
        ledger = DATA / "directory-metadata.jsonl"
        with ledger.open("xb") as target:
            summary = inspect_archive(reader, policy, target, document_target)
            target.flush()
            os.fsync(target.fileno())
        summary.update({"UTC": core.utc(), **state, "directory_SHA256": core.sha(ledger),
                        "whole_archive_MD5_verified": False, "source_accepted": False, "ready": False,
                        "image_mask_annotation_decoding_features_optimizer_CAL_TEST": 0,
                        "fragments_may_include_incidental_compressed_member_bytes": True})
        need(sum(p.stat().st_size for root in (DATA, RUN) for p in root.rglob("*") if p.is_file())
             + 65536 <= policy["metadata_job_budget_bytes"], "metadata output budget")
        core.save(RUN / "complete.json", {"success": True, **summary})
        print(json.dumps(summary))
    except BaseException as error:
        if RUN.is_dir():
            actual_fragment_bytes = sum(p.stat().st_size for p in DATA.glob("range-*.fragment"))
            core.save(RUN / "failure.json", {"UTC": core.utc(), "type": type(error).__name__, "error": str(error),
                      "HTTP_status": error.code if isinstance(error, urllib.error.HTTPError) else None,
                      **state, "actual_fragment_bytes_including_partial": actual_fragment_bytes,
                      "partials_preserved_no_retry": True, "ready": False})
        raise
    finally:
        if core.LOCK.exists() and core.strict(core.LOCK.read_bytes()) == own:
            core.LOCK.unlink()


def self_check(policy):
    sample = io.BytesIO()
    with zipfile.ZipFile(sample, "w", zipfile.ZIP_DEFLATED) as archive:
        with archive.open("dataset/README.md", "w", force_zip64=True) as target:
            target.write(b"Offline parser check; not original source evidence.\n")
        archive.writestr("dataset/train/example.bin", b"not source imagery")
    raw = sample.getvalue()
    pos = raw.rfind(b"PK\x05\x06")
    e = struct.unpack_from("<4s4H2LH", raw, pos)
    large_end = struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, e[4], e[4], e[5], e[6])
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, pos, 1)
    end = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 65535, 65535, 4294967295, 4294967295, 0)
    for data in (raw, raw[:pos] + large_end + locator + end):
        def fetch(start, finish):
            return data[start:finish + 1]
        reader = RangeReader(len(data), fetch)
        docs = []
        result = inspect_archive(reader, policy, io.BytesIO(), lambda number, body: docs.append(body))
        need(result["directory_members"] == 2 and len(docs) == 1 and docs[0].startswith(b"Offline"),
             "ZIP/ZIP64 metadata parser check")
    for name in ("../bad", "/absolute", "C:stream", "dir./bad", "test\\bad"):
        try:
            safe_member(zipfile.ZipInfo(name))
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe path accepted")
    need(not delivery("http://zenodo.org/file") and not delivery("https://u:p@zenodo.org/file")
         and not delivery("https://example.com/file"), "delivery guard")
    print(json.dumps({"offline_ZIP_ZIP64_guard_checks": "passed", "HTTP": 0, "source_data": 0, "training": 0}))


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
    else:
        need(args.publication_receipt is not None, "exact3 publication receipt required")
        live(args, policy, core_module())


if __name__ == "__main__":
    main()
