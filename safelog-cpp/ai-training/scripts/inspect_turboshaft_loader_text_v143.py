"""Fetch one preregistered immutable author text; never import or execute it."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import inspect_hrcds_metadata_v121 as common

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/inspect_turboshaft_loader_text_v143.py'
POLICY = ROOT / 'reports/industrial-turboshaft-v143-loader-text-policy.json'
DOC = 'reports/TURBOSHAFT_V143_LOADER_TEXT_PROTOCOL_KO.md'
OUT = ROOT / 'data/industrial-turboshaft-v143-loader-text'
RUN = ROOT / 'runs/industrial-turboshaft-v143-loader-text'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    policy = common.strict_json(POLICY.read_text(encoding='utf-8'))
    if common.sha(ROOT / SCRIPT) != policy['script_sha256'] or common.sha(Path(common.__file__)) != policy['utility_sha256']:
        raise ValueError('Frozen local helper mismatch')
    for rel, expected in policy['frozen_inputs'].items():
        if common.sha(ROOT / rel) != expected:
            raise ValueError('Frozen provenance/failure changed')
    ref = common.strict_json((ROOT / policy['preserved_ref']).read_text(encoding='utf-8'))
    headers = common.strict_json((ROOT / policy['preserved_headers']).read_text(encoding='utf-8'))
    failure = common.strict_json((ROOT / policy['preserved_failure']).read_text(encoding='utf-8'))
    if ref.get('ref') != 'refs/heads/main' or ref.get('url') != policy['ref_internal_url'] or ref.get('object') != policy['expected_ref_object']:
        raise ValueError('Preserved exact ref/object differs; no old HTTP retry')
    if headers.get('status') != 200 or headers.get('url') != policy['previous_request_url'] or headers.get('final_url') != policy['previous_request_url']:
        raise ValueError('Preserved transport receipt mismatch')
    if failure.get('error') != 'Fixed repository/ref identity mismatch' or failure.get('network_body_bytes_received') != 382:
        raise ValueError('Original v141 failure must remain unchanged')
    for rel in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), DOC):
        if hashlib.sha256(common.git('show', 'HEAD:safelog-cpp/ai-training/' + rel)).hexdigest() != common.sha(ROOT / rel):
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
        raise ValueError('Stage already started; no rerun/overwrite')
    OUT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    common.write_json(RUN / 'started.json', {'UTC': common.utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': common.sha(POLICY)})
    token = {'pid': os.getpid(), 'stage': 'turboshaft-v143', 'UTC': common.utc()}
    lock = RUN / 'operation.lock'
    common.write_json(lock, token)
    received = 0
    try:
        url = policy['url']
        request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-loader-text-v143/1.0'})
        try:
            response = urllib.request.build_opener(NoRedirect()).open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            common.write_json(OUT / 'author-text-headers.json', {'UTC': common.utc(), 'url': url, 'status': exc.code, 'headers': dict(exc.headers), 'body_read': False})
            exc.close()
            raise RuntimeError(f'HTTP {exc.code}; no body/retry/fallback') from None
        with response:
            common.write_json(OUT / 'author-text-headers.json', {'UTC': common.utc(), 'url': url, 'final_url': response.geturl(), 'status': response.status, 'headers': dict(response.headers)})
            if response.status != 200 or response.geturl() != url or response.headers.get_content_type() != 'text/plain':
                raise ValueError('Exact200/URL/text MIME before body')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise ValueError('Non-identity encoding')
            length = response.headers.get('Content-Length')
            cap = policy['response_limit_bytes']
            if length is not None and (not length.isdecimal() or int(length) > cap):
                raise ValueError('Content length/cap before body')
            raw = OUT / 'author-dataset.py.txt'
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
        text = raw.read_text(encoding='utf-8')
        if not text or '\x00' in text:
            raise ValueError('Nonempty UTF8 static text required')
        result = {'UTC': common.utc(), 'HEAD_prerequisite': head, 'policy_SHA256': common.sha(POLICY), 'network_GET': 1, 'network_body_bytes_received': received, 'raw_local_SHA256': common.sha(raw), 'repo_commit_not_dataset_revision_or_author_file_checksum': policy['commit_sha'], 'author_repo_relative_path': policy['author_path'], 'author_static_code_files_acquired': 1, 'author_code_imported_executed_or_dependencies_followed': False, 'v141_original_failure_preserved_not_retroactively_passed': True, 'old_HTTP_helper_fixture_reruns': 0, 'source_photos_masks_annotations_archives_PDF_weights_acquired': 0, 'pixels_QA_raster_features_our_scores_gradient_optimizer_CAL_TEST_delivery': 0, 'GT_contract_or_training_ready': False}
        common.write_json(OUT / 'author-text-receipt.json', {'bytes': received, 'local_SHA256': common.sha(raw), 'UTF8_text_validation_only': True, 'no_import_exec_eval_decode_or_dependency_requests': True})
        stored = sum(p.stat().st_size for folder in (OUT, RUN) for p in folder.rglob('*') if p.is_file())
        if stored + 8192 > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve')
        result['stored_bytes_before_completion'] = stored
        common.write_json(OUT / 'static-text-complete.json', result)
        print(json.dumps({'complete': True, 'bytes': received, 'local_SHA256': common.sha(raw), 'author_code_execution': 0, 'ready': False}))
    except Exception as exc:
        common.write_json(RUN / 'failure.json', {'UTC': common.utc(), 'error_type': type(exc).__name__, 'error': str(exc), 'network_body_bytes_received': received, 'partials_preserved': True, 'retry_redirect_fallback': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and common.strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()


if __name__ == '__main__':
    main()
