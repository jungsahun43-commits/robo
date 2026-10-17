"""Preregistered public metadata only. Never follow dataset/download/code URLs."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
POLICY = ROOT / 'reports/industrial-hrcds-v121-metadata-policy.json'
OUT = ROOT / 'data/industrial-hrcds-v121-metadata'
RUN = ROOT / 'runs/industrial-hrcds-v121-metadata'
GIT = 'C:/Program Files/Git/cmd/git.exe'
SCRIPT = 'scripts/inspect_hrcds_metadata_v121.py'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def no_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite JSON number')
    return result


def strict_json(text):
    return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=no_constant, parse_float=finite_float)


class ScriptText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.active = True
            self.parts.append('')

    def handle_data(self, text):
        if self.active:
            self.parts[-1] += text

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False


def initial_state(html):
    parser = ScriptText()
    parser.feed(html)
    candidates = []
    for text in parser.parts:
        match = re.fullmatch(r'\s*(?:window\.)?(?:INITIAL_STATE|__INITIAL_STATE__)\s*=\s*(\{.*\})\s*;?\s*', text, re.S)
        if match:
            value = strict_json(match.group(1))
            if isinstance(value, dict) and 'dataset' in value:
                candidates.append(value)
    if len(candidates) != 1:
        raise ValueError('Exactly one strict JSON INITIAL_STATE required; no JavaScript execution or fallback')
    return candidates[0]


def self_test():
    good = '<script>window.INITIAL_STATE = {"dataset":{"snapshot":{"id":"fixture"}}};</script>'
    assert initial_state(good)['dataset']['snapshot']['id'] == 'fixture'
    for bad in ('<script>window.INITIAL_STATE={"dataset":{}};alert(1)</script>',
                '<script>window.INITIAL_STATE={"dataset":{},"dataset":{}};</script>'):
        try:
            initial_state(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe/duplicate fixture accepted')
    return {'strict_assignment': True, 'trailing_javascript_rejected': True, 'duplicate_keys_rejected': True}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirect forbidden; no alternate endpoint followed: ' + newurl)


def process_exists(pid):
    """Read-only Windows OpenProcess; no os.kill or process modification."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if handle:
        kernel.CloseHandle(handle)
        return True
    error = ctypes.get_last_error()
    if error == 87:
        return False
    raise OSError(error, 'Process existence could not be established read-only')


def git(*args):
    command = [GIT, '-C', str(REPO), '-c', 'safe.directory=' + REPO.as_posix(),
               '-c', 'http.sslBackend=openssl', '-c', 'credential.interactive=false', *args]
    return subprocess.check_output(command)


def main():
    policy = strict_json(POLICY.read_text(encoding='utf-8'))
    assert policy['script_sha256'] == sha(ROOT / SCRIPT)
    for name in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), 'reports/HRCDS_V121_METADATA_PROTOCOL_KO.md'):
        assert hashlib.sha256(git('show', 'HEAD:safelog-cpp/ai-training/' + name)).hexdigest() == sha(ROOT / name)
    assert not git('status', '--porcelain').strip(), 'Repository must be clean before metadata HTTP'
    head = git('rev-parse', 'HEAD').decode().strip()
    remote = git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    assert head == remote, 'Prerequisite ordinary feature push must be remotely confirmed'
    assert policy['source_photo_mask_download_or_decode_allowed'] is False
    assert not process_exists(23948), 'Preserved training.lock PID is unexpectedly alive'
    for path in [*ROOT.glob('operation.lock'), *(ROOT / 'runs').rglob('operation.lock')]:
        raise RuntimeError('Existing operation lock: ' + str(path))
    assert shutil.disk_usage(ROOT).free >= policy['reserve_bytes'] + policy['storage_budget_bytes']
    OUT.mkdir(exist_ok=False)
    RUN.mkdir(exist_ok=False)
    lock = RUN / 'operation.lock'
    write_json(lock, {'pid': os.getpid(), 'UTC': utc(), 'argv': sys.argv, 'kind': 'metadata-only'})
    try:
        tests = self_test()
        write_json(RUN / 'strict-json-fixtures.json', tests)
        opener = urllib.request.build_opener(NoRedirect())
        received = 0
        receipts = []
        values = {}
        for item in policy['requests']:
            url = item['url']
            request = urllib.request.Request(url, headers={
                'User-Agent': 'SafeLog-academic-metadata-review/1.0', 'Accept-Encoding': 'identity',
                'Accept': 'text/html' if item['kind'] == 'html' else 'application/vnd.mendeley-public-dataset.1+json'})
            partial = OUT / (item['name'] + '.partial')
            with opener.open(request, timeout=45) as response:
                assert response.status == 200 and response.url == url
                content_type = response.headers.get('Content-Type', '').lower()
                assert ('text/html' in content_type if item['kind'] == 'html' else 'json' in content_type)
                header = {'UTC': utc(), 'URL': url, 'status': response.status,
                          'headers': {k: v for k, v in response.headers.items() if k.lower() not in ('set-cookie', 'cookie')}}
                write_json(OUT / (item['name'] + '-headers.json'), header)
                count = 0
                digest = hashlib.sha256()
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
                            raise ValueError('Preregistered response byte bound exceeded; partial preserved')
            final = OUT / (item['name'] + ('.html' if item['kind'] == 'html' else '.json'))
            partial.rename(final)
            receipt = {'URL': url, 'bytes': count, 'local_SHA256': digest.hexdigest(),
                       'publisher_checksum_verified': False, 'metadata_only': True}
            write_json(OUT / (item['name'] + '-receipt.json'), receipt)
            receipts.append(receipt)
            if item['kind'] == 'html':
                state = initial_state(final.read_text(encoding='utf-8'))
                snapshot = state['dataset']['snapshot']
                assert snapshot['id'] == policy['dataset_id'] and snapshot['version'] == policy['dataset_version']
                assert snapshot['licence']['short_name'] == policy['expected_data_license']
                write_json(OUT / 'publisher-initial-state.json', state)
            else:
                values[item['name']] = strict_json(final.read_text(encoding='utf-8'))
        files = values['publisher-root-files']
        folders = values['publisher-folders']
        assert isinstance(files, list) and len(files) < 1000, 'Unknown/truncated file-list response; stop without pagination'
        protected_files = []
        for item in files:
            assert isinstance(item['id'], str) and isinstance(item['filename'], str)
            assert isinstance(item['size'], int) and item['size'] >= 0
            protected_files.append({'id': item['id'], 'filename': item['filename'], 'bytes': item['size'],
                                    'role': 'UNASSIGNED_PROTECTED_BINARY_NOT_ACQUIRED',
                                    'content_details': item.get('content_details')})
        write_json(OUT / 'all-listed-root-files-protected-before-binary.json', {
            'purpose': 'Protect all listed containers/contents pending original TEST identities; this is not a final TEST split',
            'files': protected_files, 'role_assignment_or_source_content_access': False})
        stored = sum(p.stat().st_size for base in (OUT, RUN) for p in base.rglob('*') if p.is_file())
        assert stored < policy['storage_budget_bytes'] - 1024 * 1024
        result = {'UTC': utc(), 'preregistered_HEAD': head, 'policy_SHA256': sha(POLICY),
                  'responses': receipts, 'actual_response_bytes': received, 'stored_bytes_before_final_report': stored,
                  'root_file_count': len(files), 'declared_root_file_bytes': sum(f['bytes'] for f in protected_files),
                  'folders_response_kind': type(folders).__name__,
                  'folders_response_count_if_list': len(folders) if isinstance(folders, list) else None,
                  'complete_tree_inventory': False, 'nested_folders_followed': False,
                  'local_publisher_metadata_frozen': True, 'full_source_checksum_or_CRC_verified': False,
                  'source_photos_masks_archives_PDF_code_weights_acquired': 0,
                  'source_pixels_visual_QA_features_scores_gradients_optimizer_CAL_TEST': 0,
                  'actual_training_TEST_id_or_final_role_assignment': False, 'ready_for_training': False,
                  'strict_JSON_fixture_checks': tests, 'remaining_reserve_bytes': shutil.disk_usage(ROOT).free}
        write_json(OUT / 'metadata-complete.json', result)
        write_json(ROOT / 'reports/industrial-hrcds-v121-metadata-result.json', result)
        print(json.dumps({'metadata_complete': True, 'root_files': len(files),
                          'root_bytes': result['declared_root_file_bytes'], 'received': received,
                          'source_pixels_optimizer': 0}, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(RUN / ('failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'),
                   {'UTC': utc(), 'error': repr(error), 'partial_originals_preserved': True,
                    'source_pixels_optimizer': 0, 'whole_stage_restart_allowed': False})
        raise
    finally:
        # Remove only this process's own operation lock; all metadata/failures remain.
        if lock.exists() and strict_json(lock.read_text(encoding='utf-8'))['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
