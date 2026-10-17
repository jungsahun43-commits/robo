"""Inspect only the exact public Location recorded by the v124 publisher HEAD."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_hrcds_metadata_v121 import git, process_exists, sha, strict_json, utc, write_json
from scripts.inspect_mdmcs_zip_directory_v124 import RejectRedirect, check_headers, parse_eocd, parse_directory

SCRIPT = 'scripts/inspect_mdmcs_location_directory_v125.py'
POLICY = ROOT / 'reports/industrial-mdmcs-v125-location-directory-policy.json'
OUT = ROOT / 'data/industrial-mdmcs-v125-location-directory'
RUN = ROOT / 'runs/industrial-mdmcs-v125-location-directory'


def validate_location(receipt, original_url, exact_location):
    parsed = urllib.parse.urlsplit(exact_location)
    if receipt['method'] != 'HEAD' or receipt['status'] != 302 or receipt['body_bytes_read_by_helper'] != 0:
        raise ValueError('Frozen publisher HEAD provenance mismatch')
    if receipt['requested_URL'] != original_url or receipt['response_URL'] != original_url:
        raise ValueError('Not the original publisher endpoint')
    locations = [v for k, v in receipt['headers'].items() if k.lower() == 'location']
    if locations != [exact_location] or parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Only the exact observed public HTTPS Location is allowed')
    if parsed.netloc != 'prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com':
        raise ValueError('Unexpected observed storage host')
    return exact_location


def provenance_fixture(policy):
    receipt = {'method': 'HEAD', 'status': 302, 'body_bytes_read_by_helper': 0,
        'requested_URL': policy['original_publisher_url'], 'response_URL': policy['original_publisher_url'],
        'headers': {'Location': policy['url']}}
    assert validate_location(receipt, policy['original_publisher_url'], policy['url']) == policy['url']
    for changed in (dict(receipt, status=200), dict(receipt, requested_URL='https://fixture.invalid/')):
        try:
            validate_location(changed, policy['original_publisher_url'], policy['url'])
        except ValueError:
            pass
        else:
            raise AssertionError('Unverified Location provenance accepted')
    return {'exact_publisher_302_provenance_required': True, 'changed_status_or_origin_rejected': True,
            'v124_completed_fixtures_rerun': False}


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert sha(ROOT / SCRIPT) == policy['script_sha256']
    for name, digest in policy['preserved_inputs_sha256'].items():
        assert sha(ROOT / name) == digest, name
    tracked = (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/MDMCS_V125_LOCATION_DIRECTORY_PROTOCOL_KO.md',
               'scripts/inspect_hrcds_metadata_v121.py', 'scripts/inspect_mdmcs_zip_directory_v124.py')
    for name in tracked:
        assert hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + name)).hexdigest() == sha(ROOT / name), name
    assert not git('status', '--porcelain').strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    assert not process_exists(23948)
    assert not list(ROOT.glob('operation.lock')) and not list((ROOT / 'runs').rglob('operation.lock'))
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_budget_bytes']
    frozen = strict_json((ROOT / policy['frozen_publisher_HEAD']).read_text(encoding='utf-8'))
    validate_location(frozen, policy['original_publisher_url'], policy['url'])
    OUT.mkdir(exist_ok=False)
    RUN.mkdir(exist_ok=False)
    lock = RUN / 'operation.lock'
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'observed-Location-directory-only'})
    requests, received = [], 0
    try:
        write_json(RUN / 'new-location-provenance-fixture.json', provenance_fixture(policy))
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
                            raise ValueError('Truncated Range; partial preserved')
                        output.write(chunk)
                        count += len(chunk)
                        received += len(chunk)
                        receipt['body_bytes_read_by_helper'] = count
                final = OUT / (name + '.bin')
                partial.rename(final)
                write_json(OUT / (name + '-receipt.json'), dict(receipt, local_SHA256=sha(final),
                    archive_or_member_CRC_verified=False))
                return observed, final.read_bytes()

        etag, _ = request('storage-HEAD', 'HEAD')
        total = policy['expected_archive_bytes']
        _, tail = request('storage-EOCD22', 'GET', (total - 22, total - 1), etag)
        eocd = parse_eocd(tail, total, policy['limits'])
        write_json(OUT / 'EOCD-declarations.json', eocd)
        _, directory = request('storage-central-directory', 'GET',
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
            'new_original_publisher_or_V1_V2_metadata_requests': 0, 'old_v124_job_or_fixtures_rerun': False,
            'full_archive_or_member_content_acquired': False,
            'source_pixels_QA_features_scores_gradient_optimizer_CAL_TEST_API_CPP_ZIP_APK': 0,
            'archive_SHA_or_member_CRC_locally_verified': False,
            'final_photo_roles_native_GT_groups_graph_minima_verified': False,
            'ready_for_training': False, 'stored_bytes_before_final_report': stored,
            'disk_free': shutil.disk_usage(ROOT).free}
        write_json(OUT / 'directory-complete.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'actual_range_body_bytes', 'directory_members', 'ready_for_training')}), flush=True)
    except BaseException as error:
        failure = {'UTC': utc(), 'status': 'LOCATION_DIRECTORY_METADATA_STOPPED', 'error': repr(error),
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
