"""Recover only frozen v121 page JSON, then fetch two never-attempted metadata URLs."""
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
from scripts.inspect_hrcds_metadata_v121 import (
    ScriptText, NoRedirect, finite_float, git, no_constant, no_duplicates,
    process_exists, sha, strict_json, utc, write_json,
)

POLICY = ROOT / 'reports/industrial-hrcds-v122-metadata-continuation-policy.json'
OUT = ROOT / 'data/industrial-hrcds-v122-metadata'
RUN = ROOT / 'runs/industrial-hrcds-v122-metadata'
SCRIPT = 'scripts/continue_hrcds_metadata_v122.py'


def recover_state(html):
    parser = ScriptText()
    parser.feed(html)
    parts = [v for v in parser.parts if v.lstrip().startswith('window.INITIAL_STATE =')]
    if len(parts) != 1:
        raise ValueError('Expected exactly one observed state script')
    text = parts[0].lstrip()[len('window.INITIAL_STATE ='):].lstrip()
    decoder = json.JSONDecoder(object_pairs_hook=no_duplicates, parse_constant=no_constant, parse_float=finite_float)
    value, end = decoder.raw_decode(text)
    tail = text[end:]
    match = re.fullmatch(r'\s*\n\s*window\.initialTime\s*=\s*([0-9]{1,16})\s*;?\s*', tail)
    if not match or not isinstance(value, dict):
        raise ValueError('Only observed decimal initialTime suffix permitted; no JavaScript executed')
    return value, int(match.group(1))


def fixtures():
    prefix = '<script>window.INITIAL_STATE = {"dataset":{}}'
    state, timestamp = recover_state(prefix + '\nwindow.initialTime = 123</script>')
    assert state == {'dataset': {}} and timestamp == 123
    for value in (prefix + '\nwindow.initialTime = run()</script>',
                  '<script>window.INITIAL_STATE = {"dataset":{},"x":1e999}\nwindow.initialTime = 123</script>'):
        try:
            recover_state(value)
        except ValueError:
            pass
        else:
            raise AssertionError('Unexpected suffix or non-finite value accepted')
    return {'observed_metadata_suffix': True, 'executable_suffix_rejected': True, 'overflow_number_rejected': True}


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert sha(ROOT / SCRIPT) == policy['script_sha256']
    for name, expected in policy['preserved_inputs_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/HRCDS_V122_METADATA_CONTINUATION_KO.md',
                 'scripts/inspect_hrcds_metadata_v121.py', 'reports/industrial-hrcds-v121-metadata-policy.json'):
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
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'metadata-only-continuation'})
    try:
        tests = fixtures()
        write_json(RUN / 'metadata-recovery-fixtures.json', tests)
        page = ROOT / policy['preserved_page']
        state, render_time = recover_state(page.read_text(encoding='utf-8'))
        snapshot = state['dataset']['snapshot']
        assert snapshot['id'] == policy['dataset_id'] and snapshot['version'] == 1
        assert snapshot['licence']['short_name'] == 'CC BY 4.0'
        write_json(OUT / 'publisher-initial-state.json', state)
        write_json(OUT / 'reused-page-receipt.json', {
            'path': policy['preserved_page'], 'local_SHA256': sha(page), 'bytes': page.stat().st_size,
            'new_page_HTTP_requests': 0, 'ignored_page_render_initialTime_not_capture_date': render_time,
            'external_JavaScript_executed': False})
        opener = urllib.request.build_opener(NoRedirect())
        responses = []
        values = {}
        received = 0
        for request_spec in policy['requests']:
            url = request_spec['url']
            request = urllib.request.Request(url, headers={'User-Agent': 'SafeLog-academic-metadata-review/1.0',
                'Accept-Encoding': 'identity', 'Accept': 'application/vnd.mendeley-public-dataset.1+json'})
            partial = OUT / (request_spec['name'] + '.partial')
            count = 0
            digest = hashlib.sha256()
            with opener.open(request, timeout=45) as response:
                assert response.status == 200 and response.url == url and 'json' in response.headers.get('Content-Type', '').lower()
                write_json(OUT / (request_spec['name'] + '-headers.json'), {'UTC': utc(), 'URL': url, 'status': response.status,
                    'headers': {k: v for k, v in response.headers.items() if k.lower() not in ('cookie', 'set-cookie')}})
                with partial.open('xb') as output:
                    while True:
                        chunk = response.read(min(65536, request_spec['byte_limit'] - count + 1))
                        if not chunk:
                            break
                        output.write(chunk)
                        count += len(chunk)
                        received += len(chunk)
                        digest.update(chunk)
                        if count > request_spec['byte_limit'] or received > policy['aggregate_response_byte_limit']:
                            raise ValueError('Response bound exceeded; partial preserved')
            final = OUT / (request_spec['name'] + '.json')
            partial.rename(final)
            receipt = {'UTC': utc(), 'URL': url, 'bytes': count, 'local_SHA256': digest.hexdigest(),
                       'publisher_source_file_checksum_verified': False, 'metadata_only': True}
            write_json(OUT / (request_spec['name'] + '-receipt.json'), receipt)
            responses.append(receipt)
            values[request_spec['name']] = strict_json(final.read_text(encoding='utf-8'))
        files = values['publisher-root-files']
        folders = values['publisher-folders']
        assert isinstance(files, list) and len(files) < 1000
        protected = []
        for item in files:
            assert isinstance(item['id'], str) and isinstance(item['filename'], str)
            assert isinstance(item['size'], int) and item['size'] >= 0
            protected.append({'id': item['id'], 'filename': item['filename'], 'bytes': item['size'],
                              'role': 'UNASSIGNED_PROTECTED_BINARY_NOT_ACQUIRED', 'content_details': item.get('content_details')})
        write_json(OUT / 'all-listed-root-files-protected-before-binary.json', {
            'purpose': 'Protect all containers/contained records pending original TEST metadata; not final photo TEST allocation',
            'files': protected, 'binary_content_accessed': False})
        stored = sum(p.stat().st_size for base in (OUT, RUN) for p in base.rglob('*') if p.is_file())
        assert stored < policy['storage_budget_bytes'] - 1048576
        result = {'UTC': utc(), 'preregistered_HEAD': head, 'policy_SHA256': sha(POLICY), 'responses': responses,
            'reused_v121_page_bytes': page.stat().st_size, 'new_page_HTTP_requests': 0, 'actual_new_API_response_bytes': received,
            'actual_combined_v121_v122_metadata_response_bytes': received + page.stat().st_size,
            'root_file_count': len(files), 'declared_root_file_bytes': sum(f['bytes'] for f in protected),
            'folders_response_kind': type(folders).__name__, 'folders_response_count_if_list': len(folders) if isinstance(folders, list) else None,
            'stored_bytes_before_final_report': stored, 'complete_tree_inventory': False, 'nested_folders_followed': False,
            'source_photos_masks_archives_PDF_code_weights_acquired': 0, 'source_pixels_visual_QA_features_scores_gradients_optimizer_CAL_TEST': 0,
            'full_source_checksum_CRC_native_GT_roles_physical_groups_verified': False, 'ready_for_training': False,
            'metadata_recovery_fixture_checks': tests, 'remaining_disk_free': shutil.disk_usage(ROOT).free,
            'v121_failure_preserved_not_reclassified_as_pass': True}
        write_json(OUT / 'metadata-complete.json', result)
        write_json(ROOT / 'reports/industrial-hrcds-v122-metadata-result.json', result)
        print(json.dumps({'metadata_complete': True, 'root_files': len(files), 'root_bytes': result['declared_root_file_bytes'],
                          'received': received, 'reused_page': page.stat().st_size, 'source_pixels_optimizer': 0}), flush=True)
    except BaseException as error:
        write_json(RUN / ('failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'),
                   {'UTC': utc(), 'error': repr(error), 'partial_originals_preserved': True,
                    'source_pixels_optimizer': 0, 'whole_stage_restart_allowed': False})
        raise
    finally:
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
