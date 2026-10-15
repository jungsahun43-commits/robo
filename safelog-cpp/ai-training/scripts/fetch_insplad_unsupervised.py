"""Obtain ONLY the author's unsupervised subarchive via bounded public ZIP range.

The outer 6.4GB ZIP is never downloaded. Raw-deflate output is streamed into
the original 1.17GB inner ZIP, checked against its published container CRC.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import sys
import zipfile
import zlib
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha

SOURCE = ROOT / 'data/industrial-insplad-v43-source'
OUT = ROOT / 'data/industrial-insplad-unsupervised-v55-source'
POLICY = ROOT / 'reports/industrial-insplad-unsupervised-v55-download-policy.json'
ARCHIVE = OUT / 'publisher-unsupervised-anomaly-detection.zip'
NAME = 'unsupervised_anomaly_detection.zip'


def freeze():
    assert not POLICY.exists() and not ARCHIVE.exists() and not OUT.exists()
    assert shutil.disk_usage(ROOT).free>1170967719+5*1024**3//4
    OUT.mkdir()
    listing = read(SOURCE / 'outer-zip-directory.json')
    member = next(row for row in listing['rows'] if row['name'] == NAME)
    assert member['bytes'] == 1170967719 and member['packed'] == 1166006282 and member['compression'] == 8
    assert member['CRC32'] == '94d7f2b0'
    new(POLICY, {'UTC': datetime.now(timezone.utc).isoformat(), 'fixed_before_subarchive_download': True,
        'script_SHA256': sha(Path(__file__)), 'source_outer_metadata_SHA256': sha(SOURCE / 'outer-zip-directory.json'),
        'public_file_list_SHA256': sha(SOURCE / 'publisher-root-files.json'), 'member': member,
        'source': 'author andreluizbvs/InsPLAD links Mendeley5n3fjgvfyz V1 DOI10.17632/5n3fjgvfyz.1',
        'license': 'Mendeley V1 and pinned author LICENSE both CC BY-NC3.0, noncommercial academic research/citation. Old project page separately says NC-SA4.0; do not silently relabel original distribution or other sources.',
        'paper': 'DOI10.1080/01431161.2023.2283900; author/source/version/change attribution retained',
        'scope': 'publisher unsupervised cropped powerline asset normalTRAIN candidate and untouched publisher evaluation; archive names/counts/overlap not yet inspected. Do not assume new realdefects inTRAIN or subtype annotations. No pixelGT or whole equipment health.',
        'original_TEST': 'reserve all publisherTEST and corresponding full-frame/crop/near duplicates BEFORE pixel viewing or train split; no anomaly/detection archive combined as new photos',
        'download': 'public author distribution URL, verify206 exactContentRange, stream original outer member rawdeflate into innerZIP directly; no packed copy or full outer ZIP',
        'checksums': 'verify original inner uncompressed size+outer-member CRC32, save compressed-range and inner local SHA256; outer publisherSHA remains unverified without all6.4GB',
        'storage': '1170967719bytes innerZIP plus metadata; require innerbytes+1.25GiB reserve before network. Inspect expanded bytes before extracting any photos, no old source/cache deletion.',
        'reason': 'v54 fixed vari-grip candidate rejected with FPR12.61/FNR20.93 and bird-nest typedFN21.74percent. Seek actualauthorTRAIN support, not TESTcut adjustment. No newdata/learning efficacy promised.',
        'reservation': 'all publisherTEST orVAL and near/frame groups across source and existing6417InsPLAD plus8995oldTEST must remain evaluation-only; no newdata added before separate audit/split policy.',
        'new_photos_pixel_decodes_features_scores_optimizer': 0, 'training_protocol_fixed': False})
    print({'policy_SHA256': sha(POLICY), 'downloaded_inner_zip_expected_bytes': member['bytes']}, flush=True)


def fetch():
    policy = read(POLICY)
    assert sha(Path(__file__)) == policy['script_SHA256']
    import subprocess
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for path in (Path(__file__).resolve(),POLICY):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+path.relative_to(ROOT).as_posix()])).hexdigest()==sha(path)
    assert sha(SOURCE / 'outer-zip-directory.json') == policy['source_outer_metadata_SHA256']
    assert sha(SOURCE / 'publisher-root-files.json') == policy['public_file_list_SHA256']
    assert not ARCHIVE.exists() and not ARCHIVE.with_suffix('.partial.zip').exists()
    member = policy['member']
    assert shutil.disk_usage(ROOT).free > member['bytes'] + 5 * 1024 ** 3 // 4
    file = read(SOURCE / 'publisher-root-files.json')[0]['content_details']
    url = file['download_url']

    def response(start, count):
        result = requests.get(url, headers={'Range': f'bytes={start}-{start + count - 1}', 'Accept-Encoding': 'identity'},
                              stream=True, timeout=(20, 45))
        if result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {start}-{start + count - 1}/{file["size"]}':
            result.close()
            raise ValueError('Expected exact public bounded Range, never full outer ZIP')
        return result

    with response(member['header_offset'], 4096) as result:
        local = result.content
    assert local[:4] == b'PK\x03\x04'
    fields = struct.unpack_from('<4s5H3I2H', local)
    assert fields[3] == 8 and fields[2] == member['flags']
    name_length, extra_length = fields[-2:]
    assert name_length + extra_length + 30 <= len(local)
    assert local[30:30 + name_length].decode('utf-8') == NAME
    start = member['header_offset'] + 30 + name_length + extra_length
    new(OUT / 'selected-range-before-download.json', {'source_start': start, 'packed_bytes': member['packed'],
        'unpacked_bytes': member['bytes'], 'outer_member_CRC32': member['CRC32'], 'local_header_SHA256': hashlib.sha256(local).hexdigest()})
    partial = ARCHIVE.with_suffix('.partial.zip')
    compressed_hash, inner_hash = hashlib.sha256(), hashlib.sha256()
    inflater = zlib.decompressobj(-15)
    packed = unpacked = crc = 0
    try:
        with response(start, member['packed']) as result, partial.open('xb') as target:
            for chunk in result.iter_content(1024 ** 2):
                if not chunk:
                    continue
                packed += len(chunk)
                assert packed <= member['packed']
                compressed_hash.update(chunk)
                incoming = chunk
                while incoming:
                    data = inflater.decompress(incoming, 2 * 1024 ** 2)
                    incoming = inflater.unconsumed_tail
                    unpacked += len(data)
                    assert unpacked <= member['bytes']
                    crc = zlib.crc32(data, crc)
                    inner_hash.update(data)
                    target.write(data)
                if packed // (128 * 1024 ** 2) != (packed - len(chunk)) // (128 * 1024 ** 2):
                    print({'compressed_range_received': packed, 'expected': member['packed']}, flush=True)
            tail = inflater.flush()
            unpacked += len(tail)
            crc = zlib.crc32(tail, crc)
            inner_hash.update(tail)
            target.write(tail)
        assert inflater.eof and not inflater.unused_data and packed == member['packed'] and unpacked == member['bytes']
        assert f'{crc:08x}' == member['CRC32']
        partial.rename(ARCHIVE)
        new(OUT / 'download.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'public_distribution_URL': url,
            'source_start': start, 'compressed_received': packed, 'inner_zip_bytes': unpacked,
            'outer_member_CRC32_verified': f'{crc:08x}', 'compressed_range_local_SHA256': compressed_hash.hexdigest(),
            'inner_zip_local_SHA256': inner_hash.hexdigest(), 'outer_publisher_SHA256_not_verified': file['sha256_hash'],
            'full_outer_archive_downloaded': False, 'standalone_photos_extracted_decoded_or_models_run': 0})
    except BaseException as error:
        new(OUT / 'download-failure.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'error': repr(error),
            'compressed_received': packed, 'partial_unpacked_bytes': unpacked, 'partial_preserved': True, 'optimizer': 0})
        raise
    inspect_archive()


def inspect_archive():
    assert not (OUT / 'unsupervised-archive-members.json').exists()
    receipt = read(OUT / 'download.json')
    assert sha(ARCHIVE) == receipt['inner_zip_local_SHA256']
    rows = []
    names = set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in entry.filename and ':' not in entry.filename
            assert not stat.S_ISLNK(entry.external_attr >> 16) and not entry.flag_bits & 1
            assert entry.filename not in names
            names.add(entry.filename)
            rows.append({'member': entry.filename, 'bytes': entry.file_size, 'packed': entry.compress_size,
                         'CRC32': f'{entry.CRC:08x}', 'directory': entry.is_dir()})
        expanded = sum(row['bytes'] for row in rows)
        assert expanded < 3 * 1024 ** 3
        assert archive.testzip() is None
    files = [row for row in rows if not row['directory']]
    result = {'UTC': datetime.now(timezone.utc).isoformat(), 'inner_zip_local_SHA256': receipt['inner_zip_local_SHA256'],
        'files': len(files), 'expanded_bytes': expanded, 'all_member_CRCS_verified': True, 'safe_paths_no_symlinks_or_encryption': True,
        'counts_by_parent': dict(Counter(str(PurePosixPath(row['member']).parent) for row in files)),
        'photo_pixel_decodes_features_scores_optimizer': 0, 'source_TEST_not_used_for_training': True,
        'publisher_split_still_needs_metadata_reservation_and_original_frame_group_audit': True}
    new(OUT / 'unsupervised-archive-members.json', {'rows': rows, 'audit': result})
    new(ROOT / 'reports/industrial-insplad-unsupervised-v55-archive-audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'fetch', 'inspect_archive'))
    globals()[parser.parse_args().stage]()
