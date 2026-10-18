"""One fixed public ref JSON. No author source, image, or model execution."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
import inspect_hrcds_metadata_v121 as common

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/inspect_turboshaft_revision_v141.py'
POLICY = ROOT / 'reports/industrial-turboshaft-v141-revision-policy.json'
DOC = 'reports/TURBOSHAFT_V141_REVISION_PROTOCOL_KO.md'
OUT = ROOT / 'data/industrial-turboshaft-v141-revision'
RUN = ROOT / 'runs/industrial-turboshaft-v141-revision'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    policy = common.strict_json(POLICY.read_text(encoding='utf-8'))
    if common.sha(ROOT / SCRIPT) != policy['script_sha256'] or common.sha(Path(common.__file__)) != policy['utility_sha256']:
        raise ValueError('Frozen local helper mismatch')
    for rel, expected in policy['frozen_inputs'].items():
        if common.sha(ROOT / rel) != expected:
            raise ValueError('Frozen provenance mismatch')
    memo = common.strict_json((ROOT / policy['provenance_memo']).read_text(encoding='utf-8'))
    if memo['original_linked_repo'] != policy['original_linked_repo']:
        raise ValueError('Original-linked repository mismatch')
    for rel in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), DOC):
        committed = common.git('show', 'HEAD:safelog-cpp/ai-training/' + rel)
        if hashlib.sha256(committed).hexdigest() != common.sha(ROOT / rel):
            raise ValueError('Prerequisite differs from committed HEAD')
    if common.git('status', '--porcelain').strip():
        raise ValueError('Clean repository required')
    head = common.git('rev-parse', 'HEAD').decode().strip()
    if head != common.git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]:
        raise ValueError('Ordinary prerequisite feature push unconfirmed')
    if common.process_exists(23948):
        raise ValueError('Preserved training lock PID unexpectedly alive')
    if shutil.disk_usage(ROOT).free < policy['reserve_bytes'] + policy['storage_budget_bytes']:
        raise ValueError('Storage/reserve preflight')
    if OUT.exists() or RUN.exists():
        raise ValueError('Stage already started; preserve without rerun/overwrite')
    OUT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    common.write_json(RUN / 'started.json', {'UTC': common.utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': common.sha(POLICY)})
    token = {'pid': os.getpid(), 'stage': 'turboshaft-v141', 'UTC': common.utc()}
    lock = RUN / 'operation.lock'
    common.write_json(lock, token)
    received = 0
    try:
        url = policy['url']
        request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'Accept': 'application/vnd.github+json', 'User-Agent': 'SafeLog-public-revision-v141/1.0'})
        try:
            response = urllib.request.build_opener(NoRedirect()).open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            common.write_json(OUT / 'ref-headers.json', {'UTC': common.utc(), 'url': url, 'status': exc.code, 'headers': dict(exc.headers), 'body_read': False})
            exc.close()
            raise RuntimeError(f'HTTP {exc.code}; no body/retry/fallback') from None
        with response:
            common.write_json(OUT / 'ref-headers.json', {'UTC': common.utc(), 'url': url, 'final_url': response.geturl(), 'status': response.status, 'headers': dict(response.headers)})
            if response.status != 200 or response.geturl() != url or response.headers.get_content_type() != 'application/json':
                raise ValueError('Exact200/URL/JSON MIME before body')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise ValueError('Non-identity encoding')
            length = response.headers.get('Content-Length')
            cap = policy['response_limit_bytes']
            if length is not None and (not length.isdecimal() or int(length) > cap):
                raise ValueError('Content length/cap before body')
            raw = OUT / 'author-repo-ref.json'
            with raw.open('xb') as target:
                while True:
                    block = response.read(min(16384, cap + 1 - received))
                    if not block:
                        break
                    target.write(block)
                    received += len(block)
                    if received > cap:
                        raise ValueError('Body cap exceeded; partial preserved')
            if length is not None and received != int(length):
                raise ValueError('Actual/header length mismatch')
        value = common.strict_json(raw.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict) or value.get('ref') != 'refs/heads/main' or value.get('url') != url:
            raise ValueError('Fixed repository/ref identity mismatch')
        obj = value.get('object')
        if not isinstance(obj, dict) or obj.get('type') != 'commit' or not isinstance(obj.get('sha'), str) or not re.fullmatch('[0-9a-f]{40}', obj['sha']):
            raise ValueError('Commit identity schema mismatch')
        result = {'UTC': common.utc(), 'HEAD_prerequisite': head, 'policy_SHA256': common.sha(POLICY), 'network_GET': 1, 'network_body_bytes_received': received, 'raw_local_SHA256': common.sha(raw), 'original_linked_repo': policy['original_linked_repo'], 'ref': value['ref'], 'observed_commit_SHA_not_dataset_revision_or_file_checksum': obj['sha'], 'author_code_or_nested_links_acquired_or_executed': False, 'old_HTTP_helper_fixture_reruns': 0, 'source_photos_masks_annotations_archives_PDF_code_weights_acquired': 0, 'pixels_QA_features_our_scores_gradient_optimizer_CAL_TEST_delivery': 0, 'ready_for_training': False}
        common.write_json(OUT / 'ref-receipt.json', {'bytes': received, 'local_SHA256': common.sha(raw), 'repository_commit_not_file_or_archive_checksum': True, 'nested_links_followed': False})
        stored = sum(p.stat().st_size for folder in (OUT, RUN) for p in folder.rglob('*') if p.is_file())
        if stored + 8192 > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve')
        result['stored_bytes_before_completion'] = stored
        common.write_json(OUT / 'revision-complete.json', result)
        print(json.dumps({'complete': True, 'bytes': received, 'observed_commit_SHA': obj['sha'], 'author_code_acquired': 0, 'ready': False}))
    except Exception as exc:
        common.write_json(RUN / 'failure.json', {'UTC': common.utc(), 'error_type': type(exc).__name__, 'error': str(exc), 'network_body_bytes_received': received, 'partials_preserved': True, 'retry_redirect_fallback': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and common.strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()


if __name__ == '__main__':
    main()
