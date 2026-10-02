"""Read the pinned author ZIP's legacy 4-GiB offsets without changing its bytes."""
from collections import Counter
from pathlib import Path
import shutil
import stat
import struct
import zipfile


def header_matches(raw, offset, info, central_start):
    if not 0 <= offset < central_start: return False
    raw.seek(offset); header = raw.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04": return False
    flags, method = struct.unpack_from("<HH", header, 6)
    length = struct.unpack_from("<H", header, 26)[0]
    name = raw.read(length).decode("utf-8" if flags & 2048 else "cp437")
    return name == info.orig_filename and method == info.compress_type


def correct_legacy_offsets(bundle, raw):
    raw.seek(bundle.start_dir)
    if raw.read(4) != b"PK\x01\x02": raise ValueError("Central directory not at the expected location")
    corrections = Counter()
    for info in bundle.infolist():
        matches = [offset for offset in (info.header_offset, info.header_offset - 2**32, info.header_offset + 2**32)
                   if header_matches(raw, offset, info, bundle.start_dir)]
        if len(matches) != 1: raise ValueError(f"Ambiguous or invalid ZIP header: {info.filename}")
        corrections[str(matches[0] - info.header_offset)] += 1
        info.header_offset = matches[0]
    ordered = sorted(bundle.infolist(), key=lambda info: info.header_offset)
    if len({i.header_offset for i in ordered}) != len(ordered): raise ValueError("Overlapping member headers")
    for index, info in enumerate(ordered):
        # Rebuild CPython's overlap guard with corrected absolute offsets. Keep
        # CRC, compression and uncompressed-size verification in ZipExtFile.
        info._end_offset = ordered[index+1].header_offset if index+1 < len(ordered) else bundle.start_dir
    return dict(corrections)


def extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle, Path(archive).open("rb") as raw:
        corrections = correct_legacy_offsets(bundle, raw)
        targets = []
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination): raise ValueError("ZIP path escapes the extraction folder")
            if stat.S_ISLNK(info.external_attr >> 16): raise ValueError("Symlink member rejected")
            if info.is_dir() and info.file_size: raise ValueError("Directory contains unexpected data")
            targets.append((info, target))
        count = 0
        for info, target in targets:
            if info.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, 4*1024**2)
            if target.stat().st_size != info.file_size: raise ValueError("Extracted size differs from author metadata")
            count += 1
            if count % 1000 == 0: print(f"CODEBRIM CRC-checked extracted files {count}", flush=True)
    return {"header_corrections_bytes": corrections, "crc_checked_files": count,
            "archive_modified": False, "policy": "Unique matching filename/compression local header at offset or +/-4GiB; reconstructed overlap bounds; standard ZipExtFile CRC and size checks; all paths confined"}
