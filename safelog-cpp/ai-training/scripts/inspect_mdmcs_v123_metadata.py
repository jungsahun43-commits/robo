"""V2 public metadata only; compare publisher identities with frozen V1 metadata."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.inspect_hrcds_metadata_v121 import NoRedirect, git, process_exists, sha, strict_json, utc, write_json
from scripts.continue_hrcds_metadata_v122 import recover_state

SCRIPT = 'scripts/inspect_mdmcs_v123_metadata.py'
POLICY = ROOT / 'reports/industrial-mdmcs-v123-metadata-policy.json'
OUT = ROOT / 'data/industrial-mdmcs-v123-metadata'
RUN = ROOT / 'runs/industrial-mdmcs-v123-metadata'


def identities(rows):
    if not isinstance(rows, list) or len(rows) >= 1000:
        raise ValueError('Unknown or potentially truncated file list')
    result = []
    for row in rows:
        digest = row['content_details']['sha256_hash']
        if not isinstance(row['id'], str) or not isinstance(row['filename'], str):
            raise ValueError('Unknown file identity')
        if type(row['size']) is not int or row['size'] < 0 or row['status'] != 'COMPLETED':
            raise ValueError('Incomplete/unknown publisher file declaration')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Missing publisher checksum declaration')
        result.append({'id': row['id'], 'filename': row['filename'], 'bytes': row['size'], 'publisher_SHA256': digest})
    return sorted(result, key=lambda r: (r['id'], r['filename']))


def fixtures():
    row = {'id': 'fixture', 'filename': 'a.zip', 'size': 3, 'status': 'COMPLETED',
           'content_details': {'sha256_hash': 'a' * 64, 'download_expiry_time': 'old'}}
    new = strict_json(json.dumps(row))
    new['content_details']['download_expiry_time'] = 'new'
    assert identities([row]) == identities([new])
    new['content_details']['sha256_hash'] = 'b' * 64
    assert identities([row]) != identities([new])
    del new['content_details']['sha256_hash']
    try:
        identities([new])
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError('Missing checksum must not imply identity equality')
    return {'expiry_not_content_identity': True, 'changed_checksum_detected': True, 'missing_checksum_rejected': True}


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert sha(ROOT / SCRIPT) == policy['script_sha256']
    for name, expected in policy['preserved_inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    tracked = (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/MDMCS_V123_METADATA_PROTOCOL_KO.md',
               'scripts/inspect_hrcds_metadata_v121.py', 'scripts/continue_hrcds_metadata_v122.py')
    for name in tracked:
        assert hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + name)).hexdigest() == sha(ROOT / name), name
    assert not git('status', '--porcelain').strip()
    head = git('rev-parse', 'HEAD').decode().strip()
    assert head == git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    assert not process_exists(23948)
    assert not list(ROOT.glob('operation.lock')) and not list((ROOT / 'runs').rglob('operation.lock'))
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_budget_bytes']
    OUT.mkdir(exist_ok=False)
    RUN.mkdir(exist_ok=False)
    lock = RUN / 'operation.lock'
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'V2-metadata-only'})
    try:
        tests = fixtures()
        write_json(RUN / 'publisher-identity-fixtures.json', tests)
        old_files = strict_json((ROOT / policy['preserved_V1_file_list']).read_text(encoding='utf-8'))
        old_identities = identities(old_files)
        opener = urllib.request.build_opener(NoRedirect())
        received = 0
        receipts = []
        values = {}
        for item in policy['requests']:
            url = item['url']
            request = urllib.request.Request(url, headers={'User-Agent': 'SafeLog-academic-metadata-review/1.0',
                'Accept-Encoding': 'identity', 'Accept': 'text/html' if item['kind'] == 'html' else 'application/vnd.mendeley-public-dataset.1+json'})
            partial = OUT / (item['name'] + '.partial')
            count = 0
            digest = hashlib.sha256()
            with opener.open(request, timeout=45) as response:
                assert response.status == 200 and response.url == url
                content_type = response.headers.get('Content-Type', '').lower()
                assert ('text/html' in content_type if item['kind'] == 'html' else 'json' in content_type)
                write_json(OUT / (item['name'] + '-headers.json'), {'UTC': utc(), 'URL': url, 'status': response.status,
                    'headers': {k: v for k, v in response.headers.items() if k.lower() not in ('cookie', 'set-cookie')}})
                with partial.open('xb') as output:
                    while True:
                        chunk = response.read(min(65536, item['byte_limit'] - count + 1))
                        if not chunk:
                            break
                        output.write(chunk)
                        count += len(chunk)
                        received += len(chunk)
                        digest.update(chunk)
                        if count > item['byte_limit'] or received > policy['aggregate_response_byte_limit']:
                            raise ValueError('Preregistered metadata byte bound exceeded; partial preserved')
            final = OUT / (item['name'] + ('.html' if item['kind'] == 'html' else '.json'))
            partial.rename(final)
            receipt = {'UTC': utc(), 'URL': url, 'bytes': count, 'local_SHA256': digest.hexdigest(),
                       'metadata_only': True, 'source_archive_checksum_verified': False}
            write_json(OUT / (item['name'] + '-receipt.json'), receipt)
            receipts.append(receipt)
            if item['kind'] == 'html':
                state, render_time = recover_state(final.read_text(encoding='utf-8'))
                snapshot = state['dataset']['snapshot']
                assert snapshot['id'] == policy['dataset_id'] and snapshot['version'] == 2
                assert snapshot['name'] == policy['expected_title']
                assert snapshot['licence']['short_name'] == 'CC BY 4.0'
                write_json(OUT / 'publisher-initial-state.json', state)
                write_json(OUT / 'render-time-not-capture-date.json', {'ignored_page_render_initialTime': render_time,
                    'photo_group_or_capture_date_inferred': False, 'JavaScript_executed': False})
            else:
                values[item['name']] = strict_json(final.read_text(encoding='utf-8'))
        new_files = values['publisher-root-files']
        new_identities = identities(new_files)
        write_json(OUT / 'V1-V2-publisher-file-identity-comparison.json', {
            'V1_frozen_list_SHA256': sha(ROOT / policy['preserved_V1_file_list']),
            'V1': old_identities, 'V2': new_identities,
            'publisher_declared_identity_equal': old_identities == new_identities,
            'local_archives_acquired_or_compared': False,
            'same_family_not_additional_independent_photos': True})
        write_json(OUT / 'all-listed-V2-root-files-protected-before-binary.json', {
            'files': [dict(row, role='UNASSIGNED_PROTECTED_BINARY_NOT_ACQUIRED_ALL_CONTAINED_RECORDS') for row in new_identities],
            'V1_protection_retained': True, 'photo_TEST_ids_and_final_roles_fixed': False, 'binary_content_accessed': False})
        folders = values['publisher-folders']
        stored = sum(p.stat().st_size for base in (OUT, RUN) for p in base.rglob('*') if p.is_file())
        assert stored < policy['storage_budget_bytes'] - 1048576
        result = {'UTC': utc(), 'preregistered_HEAD': head, 'policy_SHA256': sha(POLICY), 'responses': receipts,
            'actual_new_V2_metadata_response_bytes': received, 'new_V1_HTTP_requests': 0,
            'V2_root_file_count': len(new_identities), 'V2_declared_root_bytes': sum(r['bytes'] for r in new_identities),
            'folders_kind': type(folders).__name__, 'folders_count_if_list': len(folders) if isinstance(folders, list) else None,
            'publisher_declared_V1_V2_identity_equal': old_identities == new_identities,
            'source_archives_or_photos_locally_compared': False, 'complete_source_tree_or_native_GT_verified': False,
            'stored_bytes_before_final_report': stored, 'publisher_identity_fixture_checks': tests,
            'source_photos_masks_archives_PDF_code_weights_acquired': 0,
            'source_pixels_visual_QA_encoder_scores_gradients_optimizer_CAL_TEST': 0,
            'final_photo_TEST_roles_minima_groups_and_model_protocol_fixed': False, 'ready_for_training': False,
            'disk_free': shutil.disk_usage(ROOT).free}
        write_json(OUT / 'metadata-complete.json', result)
        write_json(ROOT / 'reports/industrial-mdmcs-v123-metadata-result.json', result)
        print(json.dumps({'V2_metadata_complete': True, 'received': received, 'root_files': len(new_identities),
            'root_bytes': result['V2_declared_root_bytes'], 'publisher_V1_V2_identity_equal': old_identities == new_identities,
            'source_pixels_optimizer': 0}), flush=True)
    except BaseException as error:
        write_json(RUN / ('failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'),
            {'UTC': utc(), 'error': repr(error), 'partial_originals_preserved': True, 'source_pixels_optimizer': 0,
             'whole_stage_restart_allowed': False})
        raise
    finally:
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
