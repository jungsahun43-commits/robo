"""Resume official CODEBRIM with at most four verified HTTP range streams."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import urllib.request

from download_dataset import safe_extract

ROOT = Path(__file__).resolve().parents[1]


def read_range(url, start, end):
    response = urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "SafeLog-Academic-Research/1.0", "Range": f"bytes={start}-{end}"}), timeout=45)
    expected = f"bytes {start}-{end}/"
    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
        response.close()
        raise ValueError("Server did not honor the exact requested byte range")
    return response


def part(url, path, start, end):
    length = end - start + 1
    for attempt in range(4):
        offset = path.stat().st_size if path.exists() else 0
        if offset == length: return
        if offset > length: raise ValueError("Oversized existing part")
        try:
            with read_range(url, start + offset, end) as response, path.open("ab") as stream:
                while chunk := response.read(1024**2): stream.write(chunk)
            if path.stat().st_size != length: raise ValueError("Incomplete part")
            print(f"Verified range length {path.name}: {length} bytes", flush=True)
            return
        except Exception:
            if attempt == 3: raise
            time.sleep(2)


def main():
    pinned = json.loads((ROOT / "datasets/codebrim_source.json").read_text(encoding="utf-8"))
    folder = ROOT / "data/codebrim"
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in (("metadata.json", pinned["metadata_url"]), ("license.md", pinned["license_url"])):
        path = folder / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "SafeLog-Academic-Research/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response: value = response.read()
            path.write_bytes(value)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    source = next(f for f in metadata["files"] if f["key"] == "CODEBRIM_classification_dataset.zip")
    if source["size"] != pinned["bytes"] or source["checksum"] != "md5:" + pinned["md5"]:
        raise ValueError("Publisher archive differs from the pinned author source")
    total, url = source["size"], source["links"]["self"]
    target = ROOT / "data/facility-archives/codebrim.zip"
    parts = target.parent / "codebrim-parts"
    parts.mkdir(parents=True, exist_ok=True)
    manifest = parts / "ranges.json"
    if manifest.exists():
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        if plan["url"] != url or plan["total"] != total: raise ValueError("Download source changed")
    else:
        prefix = target.stat().st_size if target.exists() else 0
        if prefix > total: raise ValueError("Oversized prefix")
        width = max(1, math.ceil((total - prefix) / 4))
        ranges = [[start, min(total - 1, start + width - 1)] for start in range(prefix, total, width)]
        plan = {"url": url, "total": total, "prefix": prefix, "ranges": ranges}
        manifest.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if not target.exists(): target.touch()
    if target.stat().st_size not in (plan["prefix"], total): raise ValueError("Prefix changed during resume")
    if target.stat().st_size != total:
        futures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, (start, end) in enumerate(plan["ranges"]):
                futures.append(pool.submit(part, url, parts / f"part-{index}.bin", start, end))
            while not all(f.done() for f in futures):
                size = plan["prefix"] + sum(p.stat().st_size for p in parts.glob("part-*.bin"))
                print(f"CODEBRIM download {size / total:.1%} ({size / 1024**2:.0f}/{total / 1024**2:.0f} MiB)", flush=True)
                time.sleep(30)
            for future in futures: future.result()
        candidate = parts / "assembled.zip"
        with candidate.open("wb") as output:
            with target.open("rb") as source_file: shutil.copyfileobj(source_file, output, 4*1024**2)
            for index in range(len(plan["ranges"])):
                with (parts / f"part-{index}.bin").open("rb") as source_file:
                    shutil.copyfileobj(source_file, output, 4*1024**2)
        check = candidate
    else: check = target
    with check.open("rb") as stream: digest = hashlib.file_digest(stream, "md5").hexdigest()
    expected = source["checksum"].removeprefix("md5:")
    if check.stat().st_size != total or digest != expected: raise ValueError("Official CODEBRIM checksum failed")
    if check != target: check.replace(target)
    with target.open("rb") as stream: digest_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    target.with_suffix(".zip.complete").write_text(url, encoding="utf-8")
    print(f"CODEBRIM official MD5 verified; SHA256={digest_sha}", flush=True)
    destination = ROOT / "data/codebrim/source"
    marker = destination / "EXTRACTED.json"
    if not marker.exists():
        safe_extract(target, destination)
        marker.write_text(json.dumps({"md5": digest, "sha256": digest_sha}), encoding="utf-8")
    print("CODEBRIM verified and extracted", flush=True)


if __name__ == "__main__": main()
