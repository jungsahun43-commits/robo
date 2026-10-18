"""Bounded ZIP directory metadata only; no member contents or image decoding."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from email.message import Message
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import urllib.error
import urllib.request
import inspect_hrcds_metadata_v121 as common

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
WORK = REPO.parents[1]
SCRIPT = 'scripts/inspect_masonry_directory_v148.py'
POLICY = ROOT / 'reports/industrial-masonry-v148-directory-policy.json'
DOC = 'reports/MASONRY_V148_DIRECTORY_PROTOCOL_KO.md'
OUT = ROOT / 'data/industrial-masonry-v148-directory'
RUN = ROOT / 'runs/industrial-masonry-v148-directory'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def identity_headers(headers):
    if headers.get('Content-Encoding', 'identity').strip().lower() not in ('', 'identity'):
        raise ValueError('Non-identity encoding before body')


def validate_range(response, url, size, etag, version, first, last):
    identity_headers(response.headers)
    if response.status != 206 or response.geturl() != url:
        raise ValueError('Exact206/same URL required before body')
    if response.headers.get('Content-Range') != f'bytes {first}-{last}/{size}':
        raise ValueError('Exact Content-Range required before body')
    if response.headers.get('Content-Length') != str(last - first + 1):
        raise ValueError('Exact Content-Length required before body')
    if response.headers.get('ETag') != etag or response.headers.get('x-amz-version-id') != version:
        raise ValueError('HEAD ETag/storage version changed before body')


class Reader:
    def __init__(self, policy, folder, opener=None):
        self.policy, self.folder = policy, folder
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.body_bytes = 0
        self.requests = []
        self.head_count = 0
        self.get_count = 0

    def open(self, request, name):
        try:
            response = self.opener.open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            common.write_json(self.folder / (name + '-headers.json'), {'UTC': common.utc(), 'method': request.get_method(), 'url': request.full_url, 'status': exc.code, 'headers': {k: v for k, v in exc.headers.items() if k.lower() not in ('set-cookie', 'cookie')}, 'body_bytes_read': 0})
            self.requests.append({'method': request.get_method(), 'name': name, 'status': exc.code, 'body_bytes': 0})
            exc.close()
            raise RuntimeError(f'HTTP {exc.code}; no redirect/retry/fallback') from None
        common.write_json(self.folder / (name + '-headers.json'), {'UTC': common.utc(), 'method': request.get_method(), 'url': request.full_url, 'final_url': response.geturl(), 'status': response.status, 'request_headers': dict(request.header_items()), 'headers': {k: v for k, v in response.headers.items() if k.lower() not in ('set-cookie', 'cookie')}})
        self.requests.append({'method': request.get_method(), 'name': name, 'status': response.status, 'body_bytes': 0})
        return response

    def head(self, item):
        self.head_count += 1
        if self.head_count > self.policy['HEAD_max']:
            raise ValueError('HEAD count cap')
        request = urllib.request.Request(item['url'], method='HEAD', headers={'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-masonry-directory-v148/1.0'})
        with self.open(request, item['key'] + '-HEAD') as response:
            identity_headers(response.headers)
            if response.status != 200 or response.geturl() != item['url']:
                raise ValueError('Exact200/same HEAD URL; no redirect')
            if response.headers.get('Content-Length') != str(item['bytes']):
                raise ValueError('HEAD size differs from frozen publisher declaration')
            if response.headers.get('Accept-Ranges', '').strip().lower() != 'bytes':
                raise ValueError('Explicit byte Range support required')
            etag = response.headers.get('ETag', '')
            if not re.fullmatch(r'"[\x21\x23-\x7e]+"', etag) or etag.startswith('W/'):
                raise ValueError('Quoted strong HEAD ETag required; not archive hash')
            return {'etag': etag, 'version': response.headers.get('x-amz-version-id'), 'last_modified_not_capture_date': response.headers.get('Last-Modified')}

    def get(self, item, state, first, length, label):
        last = first + length - 1
        if first < 0 or length <= 0 or last >= item['bytes']:
            raise ValueError('Out-of-archive byte interval')
        if self.body_bytes + length > self.policy['network_body_cap_bytes']:
            raise ValueError('Global network cap before request')
        self.get_count += 1
        if self.get_count > self.policy['GET_max']:
            raise ValueError('GET count cap')
        name = item['key'] + '-' + label
        request = urllib.request.Request(item['url'], headers={'Accept-Encoding': 'identity', 'Range': f'bytes={first}-{last}', 'If-Range': state['etag'], 'User-Agent': 'SafeLog-masonry-directory-v148/1.0'})
        with self.open(request, name) as response:
            validate_range(response, item['url'], item['bytes'], state['etag'], state['version'], first, last)
            raw_path = self.folder / (name + '.bin')
            got = 0
            with raw_path.open('xb') as target:
                while got < length:
                    block = response.read(min(16384, length - got))
                    if not block:
                        raise ValueError('Truncated range; partial preserved')
                    target.write(block)
                    got += len(block)
                    self.body_bytes += len(block)
                    self.requests[-1]['body_bytes'] += len(block)
                overrun = response.read(1)
                if overrun:
                    target.write(overrun)
                    self.body_bytes += 1
                    self.requests[-1]['body_bytes'] += 1
                    raise ValueError('Overlong range body')
            common.write_json(self.folder / (name + '-receipt.json'), {'bytes': got, 'start': first, 'end': last, 'archive_declared_size': item['bytes'], 'local_SHA256': common.sha(raw_path), 'ETag_not_archive_SHA_or_MD5': state['etag'], 'member_CRC_archive_SHA_verified': False})
            return raw_path.read_bytes()


def end_record(raw, size):
    if len(raw) != 22:
        raise ValueError('Fixed EOCD record length')
    sig, disk, cd_disk, disk_n, total_n, cd_size, cd_offset, comment = struct.unpack('<4s4H2LH', raw)
    if sig != b'PK\x05\x06' or comment != 0:
        raise ValueError('No exact comment0 EOCD at final22; stop without tail scan/member bytes')
    if disk != 0 or cd_disk != 0 or disk_n != total_n:
        raise ValueError('Multidisk/inconsistent EOCD rejected')
    return {'entries': total_n, 'cd_size': cd_size, 'cd_offset': cd_offset, 'metadata_start': size - 22,
            'zip64_markers': total_n == 0xffff or cd_size == 0xffffffff or cd_offset == 0xffffffff}


def directory_location(reader, item, state, policy):
    end = end_record(reader.get(item, state, item['bytes'] - 22, 22, 'EOCD22'), item['bytes'])
    gap = end['metadata_start'] - end['cd_offset'] - end['cd_size']
    if end['zip64_markers'] or gap != 0:
        locator_pos = item['bytes'] - 42
        locator = reader.get(item, state, locator_pos, 20, 'ZIP64-locator20')
        sig, disk, record_pos, disks = struct.unpack('<4sLQL', locator)
        if sig != b'PK\x06\x07' or disk != 0 or disks != 1 or not 0 <= record_pos <= locator_pos - 56:
            raise ValueError('ZIP64 locator/offset/single disk guard')
        fixed = reader.get(item, state, record_pos, 56, 'ZIP64-fixed56')
        sig, rec_size, made, needed, disk, cd_disk, disk_n, total_n, cd_size, cd_offset = struct.unpack('<4sQ2H2L4Q', fixed)
        if sig != b'PK\x06\x06' or not 44 <= rec_size <= policy['zip64_record_size_cap']:
            raise ValueError('ZIP64 signature or record-size cap')
        if record_pos + 12 + rec_size != locator_pos or disk != 0 or cd_disk != 0 or disk_n != total_n:
            raise ValueError('ZIP64 record boundary/disk/count guard')
        if rec_size > 44:
            reader.get(item, state, record_pos + 56, rec_size - 44, 'ZIP64-extension')
        for key, actual, marker in [('entries', total_n, 0xffff), ('cd_size', cd_size, 0xffffffff), ('cd_offset', cd_offset, 0xffffffff)]:
            if end[key] != marker and end[key] != actual:
                raise ValueError('ZIP32/ZIP64 non-sentinel declaration mismatch')
        end.update(entries=total_n, cd_size=cd_size, cd_offset=cd_offset, metadata_start=record_pos, ZIP64=True)
    else:
        end['ZIP64'] = False
    if not 1 <= end['entries'] <= policy['entry_count_cap'] or not 46 <= end['cd_size'] <= policy['central_directory_cap_bytes']:
        raise ValueError('Central count/length cap')
    if not 0 <= end['cd_offset'] or end['cd_offset'] + end['cd_size'] != end['metadata_start']:
        raise ValueError('Central-directory exact bounds; no prefix/trailer fallback')
    return end


def parse_directory(raw, count):
    rows, position, seen = [], 0, set()
    while position < len(raw):
        if len(raw) - position < 46:
            raise ValueError('Truncated central fixed header')
        fields = struct.unpack_from('<4s6H3L5H2L', raw, position)
        sig, made, need, flags, method, tim, dat, crc, packed, plain, name_n, extra_n, comment_n, disk, internal, external, local = fields
        end = position + 46 + name_n + extra_n + comment_n
        if sig != b'PK\x01\x02' or end > len(raw) or name_n == 0 or flags & 1:
            raise ValueError('Central entry signature/size/name/encryption guard')
        name_raw = raw[position + 46:position + 46 + name_n]
        name = name_raw.decode('utf-8' if flags & 0x800 else 'cp437', errors='strict')
        parts = name.rstrip('/').split('/')
        if name in seen or '\x00' in name or '\\' in name or name.startswith('/') or any(x in ('', '.', '..') or ':' in x for x in parts):
            raise ValueError('Duplicate/unsafe/noncanonical member name; no rename')
        seen.add(name)
        extra = raw[position + 46 + name_n:position + 46 + name_n + extra_n]
        extra_pos, z64 = 0, None
        while extra_pos < len(extra):
            if len(extra) - extra_pos < 4:
                raise ValueError('Malformed extra field')
            tag, n = struct.unpack_from('<HH', extra, extra_pos)
            extra_pos += 4
            if extra_pos + n > len(extra):
                raise ValueError('Extra field exceeds entry')
            if tag == 1:
                if z64 is not None:
                    raise ValueError('Duplicate ZIP64 extra')
                z64 = extra[extra_pos:extra_pos + n]
            extra_pos += n
        values, zpos = [plain, packed, local, disk], 0
        for index, width in enumerate((8, 8, 8, 4)):
            marker = 0xffff if index == 3 else 0xffffffff
            if values[index] == marker:
                if z64 is None or zpos + width > len(z64):
                    raise ValueError('Missing/truncated ZIP64 entry field')
                values[index] = int.from_bytes(z64[zpos:zpos + width], 'little')
                zpos += width
        plain, packed, local, disk = values
        if disk != 0:
            raise ValueError('Central entry on another disk')
        rows.append({'name': name, 'name_raw_hex': name_raw.hex(), 'is_directory': name.endswith('/'), 'compressed_bytes_declared': packed, 'expanded_bytes_declared': plain, 'CRC32_declared_not_verified': f'{crc:08x}', 'local_header_offset': local, 'compression_method': method, 'flags': flags, 'name_role_tokens_observed_not_final_roles': sorted({x.lower() for x in parts if x.lower() in ('train', 'training', 'test', 'testing', 'val', 'validation', 'cal', 'calibration')})})
        position = end
        if len(rows) > count:
            raise ValueError('Entry count exceeded')
    if position != len(raw) or len(rows) != count:
        raise ValueError('Exact central size/count mismatch')
    return rows


def fixtures(folder):
    folder.mkdir(exist_ok=False)
    class FakeResponse:
        def __init__(self, status, cr):
            self.status, self.reads = status, 0
            self.headers = Message()
            for k, v in {'Content-Range': cr, 'Content-Length': '22', 'ETag': '"fixture"'}.items():
                self.headers[k] = v
        def geturl(self): return 'https://fixture.invalid/archive'
        def read(self, n):
            self.reads += 1
            raise RuntimeError('Body must not be read')
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class FakeOpener:
        def __init__(self, response): self.response = response
        def open(self, *args, **kwargs): return self.response
    checks = []
    for label, status, cr in [('reject-full200-before-body', 200, 'bytes 78-99/100'), ('reject-wrong-range-before-body', 206, 'bytes 77-98/100')]:
        response = FakeResponse(status, cr)
        reader = Reader({'GET_max': 1, 'network_body_cap_bytes': 22}, folder, FakeOpener(response))
        try:
            reader.get({'key': label, 'url': response.geturl(), 'bytes': 100}, {'etag': '"fixture"', 'version': None}, 78, 22, 'test')
        except ValueError:
            assert response.reads == 0 and reader.body_bytes == 0
            checks.append({'name': label, 'pass': True, 'body_reads': 0})
        else: raise AssertionError('Fixture accepted invalid response')
    raw = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0, 0xffff, 0xffff, 0xffffffff, 0xffffffff, 0)
    assert end_record(raw, 100)['zip64_markers']
    try:
        end_record(raw[:-2] + b'\x01\x00', 100)
    except ValueError: pass
    else: raise AssertionError('Nonzero comment accepted')
    class FakeDirectoryReader:
        def get(self, item, state, first, length, label):
            records = {
                'EOCD22': (222, raw),
                'ZIP64-locator20': (202, struct.pack('<4sLQL', b'PK\x06\x07', 0, 146, 1)),
                'ZIP64-fixed56': (146, struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, 1, 1, 46, 100)),
            }
            offset, data = records[label]
            assert first == offset and len(data) == length
            return data
    location = directory_location(FakeDirectoryReader(), {'bytes': 244}, {}, {'zip64_record_size_cap': 4096, 'entry_count_cap': 2000, 'central_directory_cap_bytes': 1048576})
    assert location['ZIP64'] and location['cd_offset'] == 100 and location['cd_size'] == 46
    name = b'train/x.jpg'
    extra = struct.pack('<HH3Q', 1, 24, 20, 10, 0)
    central = struct.pack('<4s6H3L5H2L', b'PK\x01\x02', 45, 45, 0, 8, 0, 0, 123, 0xffffffff, 0xffffffff, len(name), len(extra), 0, 0, 0, 0, 0xffffffff) + name + extra
    entry = parse_directory(central, 1)[0]
    assert entry['expanded_bytes_declared'] == 20 and entry['compressed_bytes_declared'] == 10 and entry['local_header_offset'] == 0
    checks.append({'name': 'ZIP64-markers-record-entry-and-comment-stop', 'pass': True, 'real_source_bytes': 0})
    common.write_json(folder / 'new-fixtures.json', checks)
    print(json.dumps({'new_fixtures': len(checks), 'passed': True, 'HTTP': 0, 'old_fixture_reruns': 0}))


def main(host_path):
    policy = common.strict_json(POLICY.read_text(encoding='utf-8'))
    if common.sha(ROOT / SCRIPT) != policy['script_sha256'] or common.sha(Path(common.__file__)) != policy['utility_sha256']:
        raise ValueError('Frozen code mismatch')
    for row in policy['preserved_inputs']:
        if common.sha(ROOT / row['path']) != row['SHA256']:
            raise ValueError('Preserved source/protection input changed')
    for rel in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), DOC):
        if hashlib.sha256(common.git('show', 'HEAD:safelog-cpp/ai-training/' + rel)).hexdigest() != common.sha(ROOT / rel):
            raise ValueError('Prerequisite not committed exactly')
    lines = common.git('status', '--porcelain').decode('utf-8').splitlines()
    expected_notes = policy['preserved_untracked_local_documents']
    if set(lines) != {'?? ' + name for name in expected_notes}:
        raise ValueError('Tracked work must be clean; only fixed local notes may be untracked')
    for name, digest in expected_notes.items():
        if common.sha(REPO / name) != digest:
            raise ValueError('Preserved local document changed')
    head = common.git('rev-parse', 'HEAD').decode().strip()
    if head != common.git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]:
        raise ValueError('Ordinary prerequisite feature push unconfirmed')
    host_path = Path(host_path).resolve(strict=True)
    if host_path.parent != WORK / 'outputs':
        raise ValueError('Host snapshot outside fixed outputs')
    host = common.strict_json(host_path.read_text(encoding='utf-8-sig'))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(host['UTC'].replace('Z', '+00:00'))).total_seconds()
    if not 0 <= age <= 300 or not host['cim_succeeded'] or host['related'] or host['operation_locks'] or host['PID23948_present_in_CIM']:
        raise ValueError('Fresh host/lock/process guard')
    if host['last_v85_optimizer_entry']['actual_update'] != 450 or host['last_v85_optimizer_entry']['epoch'] != 30 or common.process_exists(23948):
        raise ValueError('Actual ledger/read-only PID guard')
    if list(ROOT.glob('operation.lock')) or list((ROOT / 'runs').rglob('operation.lock')):
        raise ValueError('Existing operation lock')
    if shutil.disk_usage(ROOT).free < policy['reserve_bytes'] + policy['storage_budget_bytes']:
        raise ValueError('Storage/reserve preflight')
    if OUT.exists() or RUN.exists():
        raise ValueError('Stage already started; no rerun/overwrite')
    OUT.mkdir()
    RUN.mkdir()
    token = {'pid': os.getpid(), 'stage': 'masonry-v148', 'UTC': common.utc()}
    common.write_json(RUN / 'started.json', {'UTC': common.utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': common.sha(POLICY), 'host_receipt': str(host_path), 'host_SHA256': common.sha(host_path), 'tracked_clean': True, 'known_untracked_local_notes': list(expected_notes)})
    lock = RUN / 'operation.lock'
    common.write_json(lock, token)
    reader = Reader(policy, OUT)
    try:
        common.write_json(OUT / 'all-containers-members-copies-protected-before-requests.json', {'UTC': common.utc(), 'source_DOI': policy['source_DOI'], 'files': policy['archives'], 'all_containers_possible_members_and_copies': 'UNASSIGNED protected', 'actual_photo_TEST_VALHELD_final_roles_groups_not_fixed': True})
        results = []
        for item in policy['archives']:
            state = reader.head(item)
            end = directory_location(reader, item, state, policy)
            raw = reader.get(item, state, end['cd_offset'], end['cd_size'], 'central-directory')
            rows = parse_directory(raw, end['entries'])
            for row in rows:
                if row['local_header_offset'] >= end['cd_offset'] or row['compressed_bytes_declared'] > end['cd_offset'] - row['local_header_offset']:
                    raise ValueError('Member declarations outside pre-directory bounds')
            list_path = OUT / (item['key'] + '-member-list.json')
            common.write_json(list_path, rows)
            common.write_json(OUT / (item['key'] + '-all-listed-identities-protected.json'), {'UTC': common.utc(), 'member_list_SHA256': common.sha(list_path), 'all_exact_names_and_possible_copies': 'UNASSIGNED protected; no content access or final role permission', 'original_TEST_VALHELD_not_resolved_by_name_tokens': True})
            results.append({'key': item['key'], 'HEAD_state': state, 'directory_layout': end, 'central_directory_local_SHA256': hashlib.sha256(raw).hexdigest(), 'member_list_SHA256': common.sha(list_path), 'members': len(rows), 'files_by_name_only': sum(not row['is_directory'] for row in rows), 'sum_declared_expanded_bytes_not_actual_photos': sum(row['expanded_bytes_declared'] for row in rows), 'actual_archive_SHA_MD5_member_CRC_verified': False})
        stored = sum(p.stat().st_size for base in (OUT, RUN) for p in base.rglob('*') if p.is_file())
        if stored > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve')
        result = {'UTC': common.utc(), 'HEAD_prerequisite': head, 'policy_SHA256': common.sha(POLICY), 'archives': results, 'HEAD_requests': reader.head_count, 'GET_requests': reader.get_count, 'actual_network_body_bytes': reader.body_bytes, 'requests': reader.requests, 'stored_bytes_before_completion': stored, 'ZIP_metadata_bytes_acquired_not_whole_source_reception_zero': True, 'source_photos_masks_annotations_full_archives_PDF_code_weights_acquired': 0, 'member_content_pixels_QA_raster_features_scores_gradient_optimizer_CAL_TEST_delivery': 0, 'source_roles_physicalgroups_fullgraph_minima_GT_unknown_not_verified': True, 'ready_for_training': False, 'prior_completed_helpers_HTTP_fixtures_rerun': 0}
        common.write_json(OUT / 'directory-complete.json', result)
        print(json.dumps({'complete': True, 'HEAD': reader.head_count, 'GET': reader.get_count, 'body_bytes': reader.body_bytes, 'member_counts': [r['members'] for r in results], 'ready': False}))
    except Exception as exc:
        common.write_json(RUN / 'failure.json', {'UTC': common.utc(), 'error_type': type(exc).__name__, 'error': str(exc), 'HEAD_attempted': reader.head_count, 'GET_attempted': reader.get_count, 'actual_network_body_bytes': reader.body_bytes, 'requests': reader.requests, 'partials_preserved': True, 'retry_redirect_full_archive_or_alternate_endpoint': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and common.strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fixture-output')
    mode.add_argument('--host-receipt')
    args = parser.parse_args()
    if args.fixture_output:
        fixtures(Path(args.fixture_output))
    else:
        main(args.host_receipt)
