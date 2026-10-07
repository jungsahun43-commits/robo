"""Retrieve the three pinned RC2119 V1 archives; never modify older datasets.

Acquisition is not training eligibility. Label coverage, provenance and overlap
must be audited separately before any image becomes a training example.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import unicodedata
import uuid
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://data.mendeley.com/datasets/2vkm6k4cfg/1"
API_URL = "https://data.mendeley.com/public-api/datasets/2vkm6k4cfg"
PINNED = {
    "imagedata.zip": ("e06f002e-47e7-477d-87e0-87bd02054c95", 153509549,
                      "d11697ed550fac8b42799f9bb39c00ca70736d1b5c88fcfd280c558c465c851d", "image"),
    "jsondata.zip": ("ffcd179b-3772-43c4-ac5b-2236e075657a", 2193484,
                     "db2242e4dae9a36afd52cf0af7ad66654b4ddf8195eec75cb5780c5c432d3dd1", "json"),
    "maskdata.zip": ("1380fe8e-8f6e-49d5-b217-08f1fe7729c5", 4504373,
                     "bd2c96de2b6a8eb90291062438661ddcf6b9a91f2bae500d98755423c0c1b1e8", "mask"),
}
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": SOURCE_URL}
MAX_FILES = 50000
MAX_TOTAL_BYTES = 2_000_000_000
MAX_MEMBER_BYTES = 100_000_000
RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def official_url(identity):
    return f"https://data.mendeley.com/public-files/datasets/2vkm6k4cfg/files/{identity}/file_downloaded"


def validate_metadata(document):
    if (document.get("id") != "2vkm6k4cfg" or document.get("version") != 1
            or document.get("data_licence", {}).get("short_name") != "CC BY 4.0"):
        raise ValueError("Wrong publisher dataset, version or licence")
    files = document.get("files", [])
    if len(files) != len(PINNED) or len({f.get("filename") for f in files}) != len(files):
        raise ValueError("Publisher file inventory differs from the three pinned archives")
    for item in files:
        pin = PINNED.get(item.get("filename"))
        if pin is None:
            raise ValueError("Unpinned archive")
        identity, size, digest, _ = pin
        details = item.get("content_details", {})
        if (item.get("id") != identity or item.get("status") != "COMPLETED"
                or type(item.get("size")) is not int or item["size"] != size
                or type(details.get("size")) is not int or details["size"] != size
                or details.get("sha256_hash") != digest or details.get("content_type") != "application/zip"
                or details.get("download_url") != official_url(identity)):
            raise ValueError("Publisher metadata does not match pinned identity/size/SHA/URL")
    return files


def verified_archive(path, pin):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != pin[1] or sha(path) != pin[2]:
        raise ValueError("Archive bytes must match the complete pinned publisher original")


def download_archive(path, pin):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verified_archive(path, pin)
        return
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    digest, count = hashlib.sha256(), 0
    with requests.get(official_url(pin[0]), headers=HEADERS, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        with temporary.open("xb") as stream:
            for block in response.iter_content(1024 * 1024):
                if not block:
                    continue
                count += len(block)
                if count > pin[1]:
                    raise ValueError("Download exceeds pinned publisher byte count")
                digest.update(block)
                stream.write(block)
    if count != pin[1] or digest.hexdigest() != pin[2]:
        raise ValueError("Incomplete/corrupt archive; partial file never becomes an original")
    # rename, rather than replace, refuses to overwrite another completed file.
    temporary.rename(path)
    verified_archive(path, pin)


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("Unsafe ZIP member name")
    clean = value[:-1] if value.endswith("/") else value
    pieces = clean.split("/")
    if (PurePosixPath(clean).is_absolute() or any(p in ("", ".", "..") for p in pieces)
            or any(p.endswith((" ", ".")) or RESERVED.fullmatch(p) for p in pieces)
            or any(ord(c) < 32 or c in '<>"|?*' for c in clean)):
        raise ValueError("ZIP member is not a portable safe relative path")
    return clean


def inventory(archive):
    """Validate every member before extracting any member; CRC checks all data."""
    rows, seen, total = [], set(), 0
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if not infos or len(infos) > MAX_FILES:
            raise ValueError("Unexpected ZIP member count")
        for info in infos:
            name = safe_name(info.filename)
            key = unicodedata.normalize("NFC", name).casefold()
            kind = stat.S_IFMT(info.external_attr >> 16)
            directory = info.is_dir()
            if (key in seen or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (kind == stat.S_IFDIR and not directory) or info.flag_bits & 1
                    or not 0 <= info.file_size <= MAX_MEMBER_BYTES):
                raise ValueError("ZIP contains duplicate paths, links, encryption or oversized members")
            seen.add(key)
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("ZIP exceeds bounded decompression budget")
            rows.append({"path": name, "directory": directory, "bytes": info.file_size, "crc32": info.CRC})
        # A file cannot also be an ancestor directory of another member.
        file_names = {unicodedata.normalize("NFC", r["path"]).casefold() for r in rows if not r["directory"]}
        for row in rows:
            parts = row["path"].split("/")
            if any("/".join(parts[:i]).casefold() in file_names for i in range(1, len(parts))):
                raise ValueError("ZIP member has a file as parent")
        if source.testzip() is not None:
            raise ValueError("ZIP CRC integrity check failed")
    return rows


def extract_new(archive, destination, rows, allowed_root):
    destination, allowed_root = Path(destination).resolve(), Path(allowed_root).resolve()
    if destination == allowed_root or not destination.is_relative_to(allowed_root):
        raise ValueError("Extraction destination leaves the dataset source directory")
    if destination.exists():
        raise ValueError("Never overwrite an extracted source directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir(exist_ok=False)
    actual = []
    with zipfile.ZipFile(archive) as source:
        for info, row in zip(source.infolist(), rows, strict=True):
            if safe_name(info.filename) != row["path"] or info.file_size != row["bytes"]:
                raise ValueError("Archive inventory changed")
            target = temporary / row["path"]
            if not target.resolve().is_relative_to(temporary.resolve()):
                raise ValueError("Extraction path escape")
            if row["directory"]:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with source.open(info) as stream, target.open("xb") as output:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block); count += len(block); output.write(block)
            if count != row["bytes"]:
                raise ValueError("Extracted member byte count changed")
            actual.append({**row, "sha256": digest.hexdigest()})
    temporary.rename(destination)
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    base = ROOT / "data/rc2119-training"
    metadata = base / "metadata/dataset.json"
    if not metadata.exists():
        response = requests.get(API_URL, headers=HEADERS, timeout=45)
        response.raise_for_status()
        document = response.json(); validate_metadata(document); write_new(metadata, document)
    document = json.loads(metadata.read_text(encoding="utf-8"))
    validate_metadata(document)
    evidence = {"schema": "rc2119_acquisition_v1", "source": SOURCE_URL, "version": 1,
                "license": "CC BY 4.0", "metadata_sha256": sha(metadata), "archives": [],
                "training_eligibility_asserted": False}
    for name, pin in PINNED.items():
        archive = base / "archives" / name
        if args.download:
            print(f"Downloading/verifying {name} ({pin[1]} bytes)", flush=True)
            download_archive(archive, pin)
        verified_archive(archive, pin)
        rows = inventory(archive)
        record = {"filename": name, "sha256": pin[2], "bytes": pin[1], "crc_verified": True,
                  "regular_files": sum(not r["directory"] for r in rows), "uncompressed_bytes": sum(r["bytes"] for r in rows)}
        if args.extract:
            destination = base / "source" / pin[3]
            ledger = base / "metadata" / (pin[3] + "-extracted.json")
            if destination.exists():
                if not ledger.is_file():
                    raise ValueError("Existing source lacks its extraction proof; never overwrite")
                previous = json.loads(ledger.read_text(encoding="utf-8"))
                if previous["archive_sha256"] != pin[2] or previous["inventory"] != rows:
                    raise ValueError("Existing source archive/inventory differs")
                expected = {r["path"]: r for r in previous["files"]}
                observed = {p.relative_to(destination).as_posix(): p for p in destination.rglob("*") if p.is_file()}
                if set(expected) != set(observed) or any(p.is_symlink() for p in destination.rglob("*")):
                    raise ValueError("Existing extraction contains unexpected files/links")
                for path, p in observed.items():
                    if p.stat().st_size != expected[path]["bytes"] or sha(p) != expected[path]["sha256"]:
                        raise ValueError("Existing extracted bytes changed")
            else:
                files = extract_new(archive, destination, rows, base / "source")
                write_new(ledger, {"archive_sha256": pin[2], "inventory": rows, "files": files})
            record["extraction_ledger_sha256"] = sha(ledger)
        evidence["archives"].append(record)
        print(json.dumps(record), flush=True)
    report = base / "acquisition.json"
    if report.exists():
        if json.loads(report.read_text(encoding="utf-8")) != evidence:
            raise ValueError("Existing acquisition proof differs; preserve it")
    else:
        write_new(report, evidence)


if __name__ == "__main__":
    main()
