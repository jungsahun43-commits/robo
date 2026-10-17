"""Preregistered HEAD and ZIP directory ranges only; never read member contents."""
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_hrcds_metadata_v121 import git, process_exists, sha, strict_json, utc, write_json

SCRIPT = 'scripts/inspect_mdmcs_zip_directory_v124.py'
POLICY = ROOT / 'reports/industrial-mdmcs-v124-directory-policy.json'
OUT = ROOT / 'data/industrial-mdmcs-v124-directory'
RUN = ROOT / 'runs/industrial-mdmcs-v124-directory'


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def check_headers(status, url, headers, expected_url, total, etag=None, interval=None):
    if url != expected_url or status != (200 if interval is None else 206):
        raise ValueError('Unexpected HTTP status/URL; response body not read')
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('Encoded response rejected before body')
    observed = headers.get('ETag', '')
    if not re.fullmatch(r'"[^"\x00-\x20\x7f]*"', observed):
        raise ValueError('Strong quoted ETag required before body')
    if etag is not None and observed != etag:
        raise ValueError('ETag changed; response body not read')
    expected_length = total if interval is None else interval[1] - interval[0] + 1
    if headers.get('Content-Length', '') != str(expected_length):
        raise ValueError('Exact Content-Length required before body')
    if interval is None:
        if headers.get('Accept-Ranges', '').lower() != 'bytes':
            raise ValueError('HEAD must declare byte ranges; no probe fallback')
    elif headers.get('Content-Range', '') != f'bytes {interval[0]}-{interval[1]}/{total}':
        raise ValueError('Exact Content-Range required before body')
    return observed


def parse_eocd(raw, total, limits):
    if len(raw) != 22:
        raise ValueError('Only the preregistered 22-byte EOCD is accepted')
    signature, disk, cd_disk, disk_count, count, size, offset, comment = struct.unpack('<4s4H2IH', raw)
    if signature != b'PK\x05\x06' or disk != 0 or cd_disk != 0 or disk_count != count or comment != 0:
        raise ValueError('Ordinary single-disk zero-comment EOCD required; no larger-tail retry')
    if count in (0, 65535) or size == 0xffffffff or offset == 0xffffffff:
        raise ValueError('Empty/ZIP64 directory rejected')
    if count > limits['max_members'] or not 0 < size <= limits['max_central_directory_bytes']:
        raise ValueError('Directory/member count bound exceeded')
    if offset <= 0 or offset + size != total - 22:
        raise ValueError('Directory must end immediately before EOCD; no other ranges')
    return {'members': count, 'bytes': size, 'offset': offset, 'EOCD_comment_bytes': 0, 'single_disk': True}


def safe_name(encoded, flags):
    name = encoded.decode('utf-8' if flags & 0x800 else 'cp437', errors='strict')
    if not name or '\\' in name or ':' in name or name.startswith('/') or any(ord(c) < 32 for c in name):
        raise ValueError('Unsafe or ambiguous archive path')
    parts = name.rstrip('/').split('/')
    if any(p in ('', '.', '..') or p.endswith((' ', '.')) for p in parts):
        raise ValueError('Unsafe relative archive path')
    if any(re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', p) for p in parts):
        raise ValueError('Windows reserved archive path')
    return name


def parse_directory(raw, eocd, limits):
    rows, seen, cursor, expanded = [], set(), 0, 0
    role_tokens = {'train', 'training', 'val', 'validation', 'cal', 'calibration', 'test', 'testing', 'not_used'}
    while cursor < len(raw):
        start = cursor
        if len(raw) - cursor < 46:
            raise ValueError('Truncated central header')
        fields = struct.unpack_from('<4s6H3I5H2I', raw, cursor)
        signature, made, needed, flags, method, mtime, mdate, crc, packed, unpacked, nlen, xlen, clen, disk, iattr, attrs, local = fields
        if signature != b'PK\x01\x02' or flags & (1 | 64 | 8192) or disk != 0:
            raise ValueError('Encrypted/nonordinary central entry')
        if needed >= 45 or method not in (0, 8) or max(packed, unpacked, local) == 0xffffffff:
            raise ValueError('ZIP64/unsupported entry rejected')
        cursor += 46
        if nlen == 0 or cursor + nlen + xlen + clen > len(raw):
            raise ValueError('Central record exceeds directory')
        encoded = raw[cursor:cursor + nlen]
        name = safe_name(encoded, flags)
        extra = raw[cursor + nlen:cursor + nlen + xlen]
        ep = 0
        while ep < len(extra):
            if ep + 4 > len(extra):
                raise ValueError('Malformed extra field')
            tag, length = struct.unpack_from('<HH', extra, ep)
            ep += 4
            if ep + length > len(extra) or tag in (0x0001, 0x7075):
                raise ValueError('ZIP64/alternate Unicode path/malformed extra field')
            ep += length
        if name.casefold() in seen or ((attrs >> 16) & 0o170000) == 0o120000:
            raise ValueError('Duplicate path or symbolic link')
        seen.add(name.casefold())
        if unpacked > limits['max_declared_member_bytes'] or local >= eocd['offset']:
            raise ValueError('Member size/offset bound failed')
        expanded += unpacked
        if expanded > limits['max_declared_expanded_bytes']:
            raise ValueError('Declared expanded size bound exceeded')
        cursor += nlen + xlen + clen
        tokens = sorted(set(PurePosixPath(name).parts[:-1]) & role_tokens)
        rows.append({'name': name, 'filename_bytes_base64': base64.b64encode(encoded).decode('ascii'),
            'central_record_SHA256': hashlib.sha256(raw[start:cursor]).hexdigest(),
            'compressed_bytes_declared': packed, 'expanded_bytes_declared': unpacked,
            'CRC32_declared_not_verified': f'{crc:08x}', 'local_header_offset_declared': local,
            'compression_method': method, 'flags': flags, 'is_directory_name': name.endswith('/'),
            'explicit_lowercase_path_role_tokens_only': tokens,
            'role': 'UNASSIGNED_PROTECTED_NO_MEMBER_CONTENT_READ',
            'physical_group_or_photo_TEST_id_inferred': False})
        if len(rows) > limits['max_members']:
            raise ValueError('Too many central entries')
    if cursor != len(raw) or len(rows) != eocd['members']:
        raise ValueError('Central directory count/length mismatch')
    return rows, expanded


def fixtures(limits):
    def entry(name, flags=0):
        name = name.encode('ascii')
        return struct.pack('<4s6H3I5H2I', b'PK\x01\x02', 20, 20, flags, 8, 0, 0, 0, 3, 5,
                           len(name), 0, 0, 0, 0, 0, 10) + name
    raw = entry('test/a.jpg')
    e = {'offset': 50, 'bytes': len(raw), 'members': 1}
    total = 50 + len(raw) + 22
    eo = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 1, 1, len(raw), 50, 0)
    assert parse_eocd(eo, total, limits)['members'] == 1
    assert parse_directory(raw, e, limits)[0][0]['role'].startswith('UNASSIGNED')
    for invalid in (entry('../a.jpg'), entry('a.jpg', 1), entry('a.jpg') + entry('a.jpg')):
        try:
            parse_directory(invalid, e, limits)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe directory fixture accepted')
    headers = {'ETag': '"fixture"', 'Content-Length': '22', 'Content-Range': 'bytes 78-99/100'}
    for status, wrong_etag in ((200, '"fixture"'), (206, '"other"')):
        try:
            check_headers(status, 'https://fixture.invalid/', headers, 'https://fixture.invalid/', 100, wrong_etag, (78, 99))
        except ValueError:
            pass
        else:
            raise AssertionError('Full-response/changed-ETag guard failed')
    return {'ordinary_EOCD_and_directory': True, 'unsafe_encrypted_duplicate_paths_rejected': True,
            'full_response_and_changed_ETag_rejected_before_body': True}


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert sha(ROOT / SCRIPT) == policy['script_sha256']
    for name, digest in policy['preserved_inputs_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/MDMCS_V124_DIRECTORY_PROTOCOL_KO.md',
                 'scripts/inspect_hrcds_metadata_v121.py'):
        assert hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + name)).hexdigest() == sha(ROOT / name), name
    assert not git('status', '--porcelain').strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    assert not process_exists(23948)
    assert not list(ROOT.glob('operation.lock')) and not list((ROOT / 'runs').rglob('operation.lock'))
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_budget_bytes']
    source = strict_json((ROOT / policy['frozen_V2_files']).read_text(encoding='utf-8'))
    assert len(source) == 1 and source[0]['id'] == policy['file_id']
    assert source[0]['size'] == policy['expected_archive_bytes']
    assert source[0]['content_details']['sha256_hash'] == policy['publisher_archive_SHA256_not_locally_verified']
    assert source[0]['content_details']['download_url'] == policy['url']
    OUT.mkdir(exist_ok=False)
    RUN.mkdir(exist_ok=False)
    lock = RUN / 'operation.lock'
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'HEAD-and-central-directory-only'})
    requests, received = [], 0
    try:
        write_json(RUN / 'directory-guard-fixtures.json', fixtures(policy['limits']))
        opener = urllib.request.build_opener(RejectRedirect())

        def request(name, method, interval=None, etag=None):
            nonlocal received
            headers = {'User-Agent': 'SafeLog-academic-directory-review/1.0', 'Accept-Encoding': 'identity'}
            if interval is not None:
                headers.update({'Range': f'bytes={interval[0]}-{interval[1]}', 'If-Range': etag})
            req = urllib.request.Request(policy['url'], headers=headers, method=method)
            try:
                response = opener.open(req, timeout=45)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                receipt = {'UTC': utc(), 'method': method, 'requested_URL': policy['url'], 'response_URL': response.url,
                    'status': response.code, 'headers': {k: v for k, v in response.headers.items()
                    if k.lower() not in ('cookie', 'set-cookie')}, 'requested_range': interval,
                    'If_Range': etag, 'body_bytes_read_by_helper': 0}
                write_json(OUT / (name + '-headers.json'), receipt)
                requests.append(receipt)
                observed = check_headers(response.code, response.url, response.headers, policy['url'],
                                         policy['expected_archive_bytes'], etag, interval)
                if interval is None:
                    return observed, None
                needed = interval[1] - interval[0] + 1
                if received + needed > policy['aggregate_range_body_byte_limit']:
                    raise ValueError('Aggregate body limit rejected before read')
                partial = OUT / (name + '.partial')
                count = 0
                with partial.open('xb') as output:
                    while count < needed:
                        chunk = response.read(min(65536, needed - count))
                        if not chunk:
                            raise ValueError('Truncated exact Range; partial preserved')
                        output.write(chunk)
                        count += len(chunk)
                        received += len(chunk)
                        receipt['body_bytes_read_by_helper'] = count
                final = OUT / (name + '.bin')
                partial.rename(final)
                write_json(OUT / (name + '-receipt.json'), dict(receipt, local_SHA256=sha(final),
                    archive_or_member_CRC_verified=False))
                return observed, final.read_bytes()

        etag, _ = request('publisher-HEAD', 'HEAD')
        total = policy['expected_archive_bytes']
        _, tail = request('publisher-EOCD22', 'GET', (total - 22, total - 1), etag)
        eocd = parse_eocd(tail, total, policy['limits'])
        write_json(OUT / 'EOCD-declarations.json', eocd)
        _, directory = request('publisher-central-directory', 'GET',
            (eocd['offset'], eocd['offset'] + eocd['bytes'] - 1), etag)
        rows, expanded = parse_directory(directory, eocd, policy['limits'])
        write_json(OUT / 'central-directory-declarations.json', rows)
        write_json(OUT / 'all-members-protected-before-content.json', {
            'all_members_UNASSIGNED_protected': True, 'members': [r['name'] for r in rows],
            'explicit_TEST_path_members_also_protected': [r['name'] for r in rows
                if set(r['explicit_lowercase_path_role_tokens_only']) & {'test', 'testing'}],
            'all_possible_copies_remain_protected_no_content_read': True,
            'photo_roles_grouping_and_copy_identity_verified': False})
        stored = sum(p.stat().st_size for base in (OUT, RUN) for p in base.rglob('*') if p.is_file())
        assert stored < policy['storage_budget_bytes'] - 1048576
        result = {'UTC': utc(), 'status': 'DIRECTORY_METADATA_COMPLETE', 'preregistered_HEAD': head,
            'policy_SHA256': sha(POLICY), 'requests': requests, 'actual_range_body_bytes': received,
            'strong_ETag': etag, 'directory_members': len(rows), 'declared_expanded_bytes_not_acquired': expanded,
            'new_V1_V2_page_or_API_requests': 0, 'full_archive_or_member_content_acquired': False,
            'source_pixels_QA_features_scores_gradient_optimizer_CAL_TEST_API_CPP_ZIP_APK': 0,
            'archive_SHA_or_member_CRC_locally_verified': False,
            'final_photo_roles_native_GT_groups_graph_minima_verified': False,
            'ready_for_training': False, 'stored_bytes_before_final_report': stored,
            'disk_free': shutil.disk_usage(ROOT).free}
        write_json(OUT / 'directory-complete.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'actual_range_body_bytes', 'directory_members', 'ready_for_training')}), flush=True)
    except BaseException as error:
        failure = {'UTC': utc(), 'status': 'DIRECTORY_METADATA_STOPPED', 'error': repr(error),
            'preregistered_HEAD': head, 'policy_SHA256': sha(POLICY), 'requests': requests,
            'actual_range_body_bytes': received, 'originals_partials_headers_preserved': True,
            'source_pixels_gradient_optimizer_CAL_TEST': 0, 'training_failure': False,
            'whole_stage_restart_retry_or_full_download_allowed': False, 'ready_for_training': False}
        write_json(RUN / 'failure.json', failure)
        print(json.dumps({'status': failure['status'], 'error': repr(error), 'range_body_bytes': received}), flush=True)
        raise
    finally:
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
