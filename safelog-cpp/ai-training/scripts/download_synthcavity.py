"""Fetch the publisher's two synthcavity archives; verify MD5 before extraction."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

from download_facility_data import download
from download_dataset import safe_extract

ROOT = Path(__file__).resolve().parents[1]


def fetch(item):
    target = ROOT / "data/facility-archives" / item["name"]
    download(item["url"], target, expected_md5=item["md5"])
    if target.stat().st_size != item["bytes"]:
        raise ValueError(f"Publisher file size mismatch: {target}")
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        print(f"Archive inventory {target.name}: {len(names)} members; {names[:8]}", flush=True)
    destination = ROOT / "data/synthcavity" / target.stem
    marker = destination / "EXTRACTED.json"
    if not marker.exists():
        safe_extract(target, destination)
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        marker.write_text(json.dumps({**item, "sha256": digest}, indent=2), encoding="utf-8")
    return json.loads(marker.read_text(encoding="utf-8"))


def main():
    source = json.loads((ROOT / "datasets/synthcavity_source.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, source["files"]))
    (ROOT / "data/synthcavity/SOURCE.json").write_text(json.dumps({**source, "verified_files": results}, indent=2), encoding="utf-8")
    print("Both publisher archives verified and extracted", flush=True)


if __name__ == "__main__": main()
