"""Original-author G-Link archive; no image viewing, extraction or model execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

OUT = ROOT / 'data/industrial-glinks-source-v35'
URL = 'https://zenodo.org/api/records/15693611/files/g-link_anomaly.zip/content'
SIZE = 306973826
MD5 = '7faf736f2da8e3fb98b12f73bbf35889'
ARCHIVE = OUT / 'publisher-original-v1.zip'
METADATA = ROOT / 'data/industrial-source-candidates-v35/glinks-zenodo-metadata.txt'


def metadata():
    value = read(METADATA)
    assert value['id'] == 15693611
    assert value['metadata']['license']['id'] == 'cc-by-nc-sa-4.0'
    assert value['metadata']['access_right'] == 'open'
    original = next(row for row in value['files'] if row['key'] == 'g-link_anomaly.zip')
    assert original['size'] == SIZE and original['checksum'] == 'md5:' + MD5
    return value


def fetch():
    value = metadata()
    OUT.mkdir(exist_ok=True)
    assert not ARCHIVE.exists()
    assert shutil.disk_usage(ROOT).free >= SIZE + 1024 ** 3
    reservation = OUT / 'intent-before-original-images.json'
    assert not reservation.exists()
    new(reservation, {'UTC': datetime.now(timezone.utc).isoformat(), 'source': 'Zenodo15693611 v1 original variant',
                      'source_metadata_SHA256': sha(METADATA), 'license': 'CC BY-NC-SA4.0; cite authors and dataset DOI10.5281/zenodo.15693611',
                      'download_file': 'g-link_anomaly.zip', 'expected_bytes': SIZE, 'publisher_MD5': MD5,
                      'preserve_original_publisher_TEST': True, 'publisher_TEST_or_duplicates_never_training': True,
                      'training_intent': 'NORMAL publisher TRAIN only after physical/near-group audit and deterministic FIT/VAL/CAL protocol; no current/new TEST mask as training target',
                      'other_variants_not_extra_photos': True, 'no_image_pixels_viewed_or_model_scores': True,
                      'scope': 'visible forged steel coupling-link surface condition only; no internal/load-bearing certification',
                      'full_training_protocol_fixed': False})
    part = ARCHIVE.with_suffix('.partial.zip')
    assert not part.exists()
    with requests.get(URL, stream=True, timeout=(20, 40)) as response:
        response.raise_for_status()
        assert response.url.startswith('https://zenodo.org/')
        expected = response.headers.get('Content-Length')
        if expected is not None: assert int(expected) == SIZE
        with part.open('xb') as output:
            size = 0; announced = 0
            for chunk in response.iter_content(1024 ** 2):
                if chunk:
                    size += len(chunk); assert size <= SIZE
                    output.write(chunk)
                    if size - announced >= 32 * 1024 ** 2:
                        print({'downloaded': size, 'expected': SIZE}, flush=True); announced = size
    assert part.stat().st_size == SIZE
    md5 = hashlib.md5()
    with part.open('rb') as source:
        while chunk := source.read(1024 ** 2): md5.update(chunk)
    assert md5.hexdigest() == MD5
    part.rename(ARCHIVE)
    new(OUT / 'download.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'url': URL, 'bytes': SIZE,
                               'publisher_MD5_verified': md5.hexdigest(), 'local_SHA256': sha(ARCHIVE),
                               'SHA256_not_author_supplied': True, 'access': 'public_no_login_personal_information_or_contact_submission',
                               'other_variants_downloaded': 0, 'training_or_image_decoding': 0})
    inspect()


def inspect():
    metadata(); assert ARCHIVE.stat().st_size == SIZE
    assert not (OUT / 'archive-members-before-pixels.json').exists()
    with zipfile.ZipFile(ARCHIVE) as archive:
        rows = []; names = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in member.filename
            assert path.parts and ':' not in path.parts[0]
            assert not stat.S_ISLNK(member.external_attr >> 16)
            assert member.filename not in names; names.add(member.filename)
            rows.append({'member': member.filename, 'bytes': member.file_size,
                         'compressed_bytes': member.compress_size, 'CRC32': f'{member.CRC:08x}', 'is_directory': member.is_dir()})
        total = sum(row['bytes'] for row in rows)
        assert total < 8 * 1024 ** 3, 'Unexpected expanded size; do not extract'
        assert archive.testzip() is None
    files = [row for row in rows if not row['is_directory']]
    counts = Counter(PurePosixPath(row['member']).suffix.lower() for row in files)
    prefixes = Counter('/'.join(PurePosixPath(row['member']).parts[:3]) for row in files)
    audit = {'UTC': datetime.now(timezone.utc).isoformat(), 'local_archive_SHA256': sha(ARCHIVE),
             'publisher_MD5_verified_at_download': MD5, 'CRC_all_members_passed': True,
             'safe_paths_no_symlinks_or_duplicates': True, 'files': len(files), 'uncompressed_bytes': total,
             'extensions': dict(counts), 'prefix_counts': dict(prefixes), 'members': rows,
             'no_archive_code_executed': True, 'image_decoding_or_viewing': 0,
             'pixel_duplicate_audit': False, 'training_features_scores_optimizer': 0,
             'publisher_TEST_reserved_before_image_viewing': True}
    new(OUT / 'archive-members-before-pixels.json', audit)
    report = {key: val for key, val in audit.items() if key != 'members'}
    new(ROOT / 'reports/industrial-glinks-v35-archive-audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('fetch', 'inspect'))
    args = parser.parse_args(); globals()[args.stage]()
