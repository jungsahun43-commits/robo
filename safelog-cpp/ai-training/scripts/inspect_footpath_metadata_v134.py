"""Preregistered Zenodo metadata and two small static text files; no image access."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import inspect_hrcds_metadata_v121 as common

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'reports/industrial-footpath-v134-metadata-policy.json'
SCRIPT = 'scripts/inspect_footpath_metadata_v134.py'
DOC = 'reports/FOOTPATH_V134_METADATA_PROTOCOL_KO.md'
OUT = ROOT / 'data/industrial-footpath-v134-metadata'
RUN = ROOT / 'runs/industrial-footpath-v134-metadata'
received = 0


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def get_once(name, url, cap, expected_size=None, expected_md5=None, kind='json'):
    global received
    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-public-metadata-v134/1.0'})
    opener = urllib.request.build_opener(RejectRedirect())
    try:
        response = opener.open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        common.write_json(OUT / (name + '-headers.json'), {'UTC': common.utc(), 'url': url, 'status': exc.code, 'headers': dict(exc.headers), 'body_read': False})
        exc.close()
        raise RuntimeError(f'{name}: HTTP guard stopped at {exc.code}; body not read') from None
    with response:
        headers = dict(response.headers)
        common.write_json(OUT / (name + '-headers.json'), {'UTC': common.utc(), 'url': url, 'final_url': response.geturl(), 'status': response.status, 'headers': headers})
        if response.status != 200 or response.geturl() != url:
            raise ValueError('Exact URL/status required; redirects forbidden')
        if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
            raise ValueError('Content encoding not identity')
        mime = response.headers.get_content_type()
        allowed = ('application/json',) if kind == 'json' else ('text/plain', 'text/markdown', 'application/octet-stream')
        if mime not in allowed:
            raise ValueError('Unexpected MIME; no HTML/image/PDF body read')
        length = response.headers.get('Content-Length')
        if length is not None:
            if not length.isdecimal() or int(length) > cap or (expected_size is not None and int(length) != expected_size):
                raise ValueError('Content-Length guard stopped before body')
        path = OUT / name
        count = 0
        with path.open('xb') as target:
            while True:
                block = response.read(min(16384, cap + 1 - count))
                if not block:
                    break
                target.write(block)
                count += len(block)
                received += len(block)
                if count > cap or received > 1048576:
                    raise ValueError('Body/network cap exceeded; partial preserved')
        if length is not None and count != int(length):
            raise ValueError('Actual length differs from header')
        if expected_size is not None and count != expected_size:
            raise ValueError('Actual text length differs from publisher metadata')
        raw = path.read_bytes()
        if expected_md5 is not None and hashlib.md5(raw).hexdigest() != expected_md5:
            raise ValueError('Publisher MD5 mismatch')
        text = raw.decode('utf-8-sig', errors='strict')
        if '\x00' in text:
            raise ValueError('NUL in supposed static text')
        common.write_json(OUT / (name + '-receipt.json'), {'bytes': count, 'local_SHA256': common.sha(path), 'publisher_MD5_checked': expected_md5, 'MIME': mime, 'executed_or_links_followed': False})
        return text


def main():
    policy = common.strict_json(POLICY.read_text(encoding='utf-8'))
    if common.sha(ROOT / SCRIPT) != policy['script_sha256'] or common.sha(Path(common.__file__)) != policy['utility_sha256']:
        raise ValueError('Frozen script/utility SHA mismatch')
    for relative in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), DOC):
        if hashlib.sha256(common.git('show', 'HEAD:safelog-cpp/ai-training/' + relative)).hexdigest() != common.sha(ROOT / relative):
            raise ValueError('Prerequisite tracked file differs from HEAD')
    if common.git('status', '--porcelain').strip():
        raise ValueError('Clean repository required')
    head = common.git('rev-parse', 'HEAD').decode().strip()
    remote = common.git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]
    if head != remote:
        raise ValueError('Prerequisite ordinary feature push not confirmed remotely')
    if common.process_exists(23948):
        raise ValueError('Preserved training.lock PID unexpectedly alive')
    if shutil.disk_usage(ROOT).free < policy['reserve_bytes'] + policy['storage_budget_bytes']:
        raise ValueError('Reserve/storage gate')
    if OUT.exists() or RUN.exists():
        raise ValueError('Current stage already started; no rerun/overwrite')
    OUT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    common.write_json(RUN / 'started.json', {'UTC': common.utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': common.sha(POLICY)})
    lock = RUN / 'operation.lock'
    token = {'pid': os.getpid(), 'stage': 'footpath-v134', 'UTC': common.utc()}
    common.write_json(lock, token)
    try:
        record = common.strict_json(get_once('publisher-record.json', policy['metadata_url'], policy['api_byte_limit']))
        if record.get('id') != 17154852 or record.get('doi') != '10.5281/zenodo.17154852' or record.get('metadata', {}).get('title') != policy['title']:
            raise ValueError('Publisher record identity mismatch')
        entries = record.get('files')
        if not isinstance(entries, list) or len(entries) != 4:
            raise ValueError('Expected fixed four-file listing')
        files = {item['key']: item for item in entries}
        if set(files) != set(policy['file_MD5']):
            raise ValueError('File list differs from observed publisher list')
        for key, expected in policy['file_MD5'].items():
            item = files[key]
            if item.get('checksum') != 'md5:' + expected or type(item.get('size')) is not int or item['size'] < 1:
                raise ValueError('Publisher file MD5/size metadata differs')
        common.write_json(OUT / 'all-listed-files-protected-before-text.json', {'UTC': common.utc(), 'record': 17154852, 'files': entries, 'all_containers_and_possible_records': 'UNASSIGNED protected; no image TEST IDs/final roles known', 'archive_member_pixels_allowed': False})
        observations = []
        for selection in policy['selected_static_text']:
            key = selection['key']
            item = files[key]
            if item['size'] > policy['text_byte_limit']:
                raise ValueError('Text metadata size exceeds fixed cap')
            if key == 'LICENSE.txt' and item['size'] != 931:
                raise ValueError('LICENSE must match observed exact931 bytes')
            url = item.get('links', {}).get('self')
            parsed = urllib.parse.urlsplit(url or '')
            base = '/api/records/17154852/files/' + urllib.parse.quote(key, safe='') + '/content'
            public = '/records/17154852/files/' + urllib.parse.quote(key, safe='')
            if parsed.scheme != 'https' or parsed.netloc != 'zenodo.org' or parsed.username or parsed.fragment or parsed.path not in (base, public) or parsed.query not in ('', 'download=1'):
                raise ValueError('Publisher text URL outside preregistered same-record paths')
            text = get_once(selection['output'], url, policy['text_byte_limit'], item['size'], policy['file_MD5'][key], 'text')
            observations.append({'key': key, 'bytes': item['size'], 'local_SHA256': common.sha(OUT / selection['output']), 'characters': len(text), 'content_executed_image_decoded_or_links_followed': False})
        stored = sum(p.stat().st_size for folder in (OUT, RUN) for p in folder.rglob('*') if p.is_file())
        if stored > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve gate')
        common.write_json(OUT / 'metadata-complete.json', {'UTC': common.utc(), 'HEAD_prerequisite': head, 'policy_SHA256': common.sha(POLICY), 'network_body_bytes_received': received, 'file_declarations': entries, 'static_text_files_acquired': observations, 'stored_bytes_before_completion': stored, 'source_photos_masks_annotations_archives_acquired': 0, 'pixels_QA_features_scores_gradient_optimizer_CAL_TEST_delivery': 0, 'ready_for_training': False, 'all_old_GT_roles_graphs_models_outputs_preserved': True})
        print(json.dumps({'complete': True, 'network_body_bytes': received, 'static_text_files': len(observations), 'ready': False}))
    except Exception as exc:
        common.write_json(RUN / 'failure.json', {'UTC': common.utc(), 'error_type': type(exc).__name__, 'error': str(exc), 'network_body_bytes_received': received, 'partials_preserved': True, 'retry_or_archive_fallback': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and common.strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()  # Only our current transient lock; never training.lock or another owner.


if __name__ == '__main__':
    main()
