"""One preregistered publisher root-list GET. No binary, pixel or author code."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[3]
OUTPUTS = WORK / 'outputs'
SCRIPT = 'scripts/inspect_ceramics_metadata_v151.py'
POLICY = ROOT / 'reports/industrial-ceramics-v151-metadata-policy.json'
OUT = ROOT / 'data/industrial-ceramics-v151-metadata'
RUN = ROOT / 'runs/industrial-ceramics-v151-metadata'
GIT = 'C:/Program Files/Git/cmd/git.exe'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant')
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite JSON number')
        return number
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write('\n')


def git(*args):
    return subprocess.check_output([GIT, '-C', str(ROOT.parent.parent),
        '-c', 'safe.directory=' + str(ROOT.parent.parent),
        '-c', 'http.sslBackend=openssl', '-c', 'credential.interactive=false', *args])


def validate_headers(status, final_url, expected_url, mime, encoding, length, cap):
    valid_mime = mime.lower() == 'application/json' or (mime.lower().startswith('application/') and mime.lower().endswith('+json'))
    if status != 200 or final_url != expected_url or not valid_mime:
        raise ValueError('Exact200/URL/JSON MIME required before body')
    if encoding.lower() not in ('', 'identity'):
        raise ValueError('Identity encoding required')
    if length is not None and (not length.isdecimal() or int(length) > cap):
        raise ValueError('Invalid/excessive Content-Length before body')


def file_identities(value):
    if not isinstance(value, list) or not 0 < len(value) < 1000:
        raise ValueError('Unknown/empty/potentially truncated root list')
    result = []
    seen = set()
    names = set()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError('Unknown file entry')
        identifier, name = row.get('id'), row.get('filename')
        size = row.get('size')
        details = row.get('content_details')
        digest = details.get('sha256_hash') if isinstance(details, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Missing/duplicate publisher file ID')
        if not isinstance(name, str) or not name or name in names or type(size) is not int or size < 0 or row.get('status') != 'COMPLETED':
            raise ValueError('Incomplete file declaration')
        if not isinstance(digest, str) or re.fullmatch('[0-9a-fA-F]{64}', digest) is None:
            raise ValueError('Missing/malformed publisher SHA declaration')
        seen.add(identifier)
        names.add(name)
        result.append({'id': identifier, 'filename': name, 'bytes': size,
            'publisher_SHA256': digest, 'role': 'UNASSIGNED_ALL_CONTAINED_RECORDS_COPIES'})
    return result


def fixtures(path):
    path = Path(path)
    path.mkdir(exist_ok=False)
    invalid_headers = [(302, 'u', 'application/json', None), (200, 'other', 'application/json', None), (200, 'u', 'text/html', None), (200, 'u', 'application/json', '513')]
    for status, final, mime, length in invalid_headers:
        try:
            validate_headers(status, final, 'u', mime, 'identity', length, 512)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid header allowed before body')
    for text in ('{"id":1,"id":2}', '{"a":NaN}', '{"a":1e999}'):
        try:
            strict_json(text)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid strict JSON accepted')
    row = {'id': 'fixture', 'filename': 'f.zip', 'size': 1, 'status': 'COMPLETED', 'content_details': {'sha256_hash': 'a' * 64}}
    assert len(file_identities([row])) == 1
    for rows in ([dict(row, content_details={})], [row, row], [row, dict(row, id='other')], [row] * 1000):
        try:
            file_identities(rows)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid publisher list accepted')
    write(path / 'fixtures.json', {'UTC': utc(), 'new_fixture_cases': 3, 'passed': True,
        'header_rejection_body_reads': 0, 'duplicate_nonfinite_JSON_rejected': True,
        'list_truncation_duplicate_missingSHA_rejected': True, 'network_requests': 0,
        'script_SHA256': sha(ROOT / SCRIPT)})
    print(json.dumps({'new_fixtures': 3, 'passed': True, 'network': 0}))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main(host_path):
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    host = strict_json(Path(host_path).read_text(encoding='utf-8-sig'))
    if (datetime.now(timezone.utc) - datetime.fromisoformat(host['UTC'].replace('Z', '+00:00'))).total_seconds() > 300:
        raise ValueError('Fresh host snapshot required')
    if host['related'] or host['operation_locks'] or host['PID23948_present_in_CIM'] or host['last_v85_optimizer_entry']['actual_update'] != 450 or host['last_v85_optimizer_entry']['epoch'] != 30:
        raise ValueError('Host process/locks/ledger gate')
    if sha(ROOT / SCRIPT) != policy['script_sha256'] or sha(ROOT / policy['provenance_memo']) != policy['provenance_memo_sha256']:
        raise ValueError('Frozen script/provenance mismatch')
    fixture = strict_json(Path(policy['new_fixture_receipt']).read_text(encoding='utf-8'))
    if fixture.get('new_fixture_cases') != 3 or fixture.get('passed') is not True or fixture.get('network_requests') != 0 or fixture.get('script_SHA256') != sha(ROOT / SCRIPT):
        raise ValueError('Current-code fixture receipt required')
    for rel in policy['prerequisite_files']:
        if hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + rel)).hexdigest() != sha(ROOT / rel):
            raise ValueError('Prerequisite differs from committed HEAD')
    expected_notes = set(policy['preserved_untracked_notes'])
    if set(git('status', '--porcelain').decode().splitlines()) != expected_notes:
        raise ValueError('Unexpected repository changes')
    for rel, digest in policy['preserved_untracked_notes_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Frozen local retention note mismatch')
    head = git('rev-parse', 'HEAD').decode().strip()
    if head != git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]:
        raise ValueError('Normal prerequisite push not confirmed')
    if list(ROOT.glob('operation.lock')) or list((ROOT / 'runs').rglob('operation.lock')) or list(OUTPUTS.rglob('operation.lock')):
        raise ValueError('Operation lock appeared')
    if OUT.exists() or RUN.exists() or shutil.disk_usage(ROOT).free < policy['reserve_bytes'] + policy['storage_budget_bytes']:
        raise ValueError('No-rerun/storage gate')
    OUT.mkdir()
    RUN.mkdir()
    write(RUN / 'started.json', {'UTC': utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': sha(POLICY), 'host_receipt': str(host_path)})
    token = {'pid': os.getpid(), 'stage': 'ceramics-v151', 'UTC': utc()}
    lock = RUN / 'operation.lock'
    write(lock, token)
    received = 0
    try:
        write(OUT / 'all-source-containers-records-copies-protected-before-request.json', {'UTC': utc(),
            'DOI': policy['dataset_DOI'], 'related_record_observed_not_requested': 'wmptdnw356/1',
            'all_possible_containers_records_copies': 'UNASSIGNED', 'photo_TEST_VALHELD_final_roles_fixed': False,
            'no_source_binary_member_or_pixel_permission': True})
        url = policy['url']
        request = urllib.request.Request(url, headers={'User-Agent': 'SafeLog-public-metadata-v151/1.0',
            'Accept-Encoding': 'identity', 'Accept': 'application/vnd.mendeley-public-dataset.1+json'})
        opener = urllib.request.build_opener(NoRedirect())
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            write(OUT / 'publisher-root-files-headers.json', {'UTC': utc(), 'URL': url, 'status': error.code,
                'headers': {k: v for k, v in error.headers.items() if k.lower() not in ('cookie', 'set-cookie')}, 'body_read': False})
            error.close()
            raise RuntimeError('HTTP error; no body/retry/fallback') from None
        with response:
            write(OUT / 'publisher-root-files-headers.json', {'UTC': utc(), 'URL': url, 'final_URL': response.geturl(), 'status': response.status,
                'headers': {k: v for k, v in response.headers.items() if k.lower() not in ('cookie', 'set-cookie')}})
            length = response.headers.get('Content-Length')
            cap = policy['response_limit_bytes']
            validate_headers(response.status, response.geturl(), url, response.headers.get_content_type(),
                response.headers.get('Content-Encoding', 'identity'), length, cap)
            raw = OUT / 'publisher-root-files.json'
            with raw.open('xb') as target:
                while True:
                    block = response.read(min(16384, cap + 1 - received))
                    if not block:
                        break
                    target.write(block)
                    received += len(block)
                    if received > cap:
                        raise ValueError('Bound exceeded; partial preserved')
            if length is not None and received != int(length):
                raise ValueError('Actual/header length mismatch')
        values = strict_json(raw.read_text(encoding='utf-8-sig'))
        identities = file_identities(values)
        write(OUT / 'listed-files-all-records-copies-UNASSIGNED.json', {'UTC': utc(), 'declarations': identities,
            'whole_source_tree_or_photo_roles_groups_nativeGT_verified': False, 'download_links_followed': False})
        write(OUT / 'publisher-root-files-receipt.json', {'UTC': utc(), 'bytes': received, 'local_SHA256': sha(raw),
            'publisher_checksums_are_declarations_not_local_archive_verification': True, 'actual_requests': 1})
        result = {'UTC': utc(), 'prerequisite_HEAD': head, 'policy_SHA256': sha(POLICY), 'script_SHA256': sha(ROOT / SCRIPT),
            'dataset_DOI': policy['dataset_DOI'], 'GET': 1, 'body_bytes': received, 'local_raw_metadata_SHA256': sha(raw),
            'root_file_count': len(identities), 'publisher_declared_bytes_sum': sum(item['bytes'] for item in identities),
            'data_licence_page_declaration': 'CC BY-NC 3.0', 'actual_archive_SHA_CRC_photos_masks_annotations_code_weights': 0,
            'pixels_QA_raster_features_our_scores_gradient_optimizer_CAL_TEST_API_CPU_GPU_CPP_ZIP_APK': 0,
            'ready_for_training': False, 'whole_source_tree_roles_groups_GTunknown_minima_budget_verified': False}
        stored = sum(p.stat().st_size for folder in (OUT, RUN) for p in folder.rglob('*') if p.is_file())
        if stored + 65536 > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve')
        result['stored_before_complete_bytes'] = stored
        write(OUT / 'metadata-complete.json', result)
        print(json.dumps({'complete': True, 'GET': 1, 'body_bytes': received, 'root_files': len(identities),
            'declared_bytes': result['publisher_declared_bytes_sum'], 'ready': False}))
    except Exception as error:
        write(RUN / 'failure.json', {'UTC': utc(), 'error_type': type(error).__name__, 'error': str(error),
            'body_bytes_received': received, 'partial_originals_preserved': True, 'retry_redirect_fallback': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures')
    parser.add_argument('--host-receipt')
    arguments = parser.parse_args()
    if arguments.fixtures:
        fixtures(arguments.fixtures)
    elif arguments.host_receipt:
        main(arguments.host_receipt)
    else:
        parser.error('Select fixture-only or one metadata attempt with fresh host receipt')
