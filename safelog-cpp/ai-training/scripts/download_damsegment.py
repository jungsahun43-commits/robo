"""Reproduce the publisher-verified DamSegment download; no account required."""
import json
from pathlib import Path
from download_facility_data import download
from download_dataset import safe_extract

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / "datasets/damsegment_source.json").read_text(encoding="utf-8"))
    destination = ROOT / "data/damsegment/source"
    for item in source["files"]:
        archive = ROOT / "data/facility-archives" / item["local_archive"]
        download(item["url"], archive, expected_hash=item["sha256"])
        if archive.stat().st_size != item["bytes"]:
            raise ValueError("Publisher file size mismatch")
        safe_extract(archive, destination)
    (destination.parent / "SOURCE.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(f"Verified source ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
