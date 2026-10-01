"""Download author-hosted facility data with checksum and safe extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from download_dataset import safe_extract

ROOT = Path(__file__).resolve().parents[1]


def download(url: str, target: Path, expected_hash: str | None = None, expected_md5: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_suffix(target.suffix + ".complete")
    if not marker.exists():
        for attempt in range(4):
            offset = target.stat().st_size if target.exists() else 0
            headers = {"User-Agent": "SafeLog-Academic-Research/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    append = response.status == 206 and offset > 0
                    written = offset if append else 0
                    total = int(response.headers.get("Content-Length", "0")) + written
                    next_notice = written + 128 * 1024**2
                    with target.open("ab" if append else "wb") as stream:
                        while chunk := response.read(1024**2):
                            stream.write(chunk)
                            written += len(chunk)
                            if written >= next_notice:
                                print(f"{target.name}: {written / 1024**2:.0f}/{total / 1024**2:.0f} MiB", flush=True)
                                next_notice = written + 128 * 1024**2
                    if total and written != total:
                        raise RuntimeError(f"Incomplete download: {written}/{total}")
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
        marker.write_text(url, encoding="utf-8")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_hash and digest != expected_hash:
        marker.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {target}: {digest}")
    if expected_md5:
        with target.open("rb") as stream:
            md5 = hashlib.file_digest(stream, "md5").hexdigest()
        if md5 != expected_md5:
            marker.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 mismatch for {target}: {md5}")
    print(f"Checksum checked: {target.name} SHA256={digest}", flush=True)


def main() -> int:
    sources = json.loads((ROOT / "datasets/facility_sources.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sources)
    args = parser.parse_args()
    source = sources[args.dataset]
    destination = ROOT / "data" / args.dataset
    destination.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "data" / "facility-archives" / f"{args.dataset}.zip"
    download(source["url"], archive, source.get("sha256"), source.get("md5"))
    if not (destination / "SOURCE.json").exists():
        print("Extracting verified archive...", flush=True)
        safe_extract(archive, destination)
        (destination / "SOURCE.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Ready: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
