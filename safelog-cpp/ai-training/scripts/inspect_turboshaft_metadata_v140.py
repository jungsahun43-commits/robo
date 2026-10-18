"""One new fixed publisher API, metadata only. No archive/member/code execution."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import inspect_hrcds_metadata_v121 as common

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/inspect_turboshaft_metadata_v140.py'
POLICY = ROOT / 'reports/industrial-turboshaft-v140-metadata-policy.json'
DOC = 'reports/TURBOSHAFT_V140_METADATA_PROTOCOL_KO.md'
OUT = ROOT / 'data/industrial-turboshaft-v140-metadata'
RUN = ROOT / 'runs/industrial-turboshaft-v140-metadata'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    policy = common.strict_json(POLICY.read_text(encoding='utf-8'))
    if common.sha(ROOT / SCRIPT) != policy['script_sha256'] or common.sha(Path(common.__file__)) != policy['utility_sha256']:
        raise ValueError('Frozen code mismatch')
    for rel in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), DOC, 'reports/INDUSTRIAL_SOURCE_V139_METADATA_KO.md'):
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
        raise ValueError('Stage already started; preserve without rerun/overwrite')
    OUT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    common.write_json(RUN / 'started.json', {'UTC': common.utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': common.sha(POLICY)})
    lock = RUN / 'operation.lock'
    token = {'pid': os.getpid(), 'stage': 'turboshaft-v140', 'UTC': common.utc()}
    common.write_json(lock, token)
    received = 0
    try:
        url = policy['url']
        opener = urllib.request.build_opener(NoRedirect())
        request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-public-metadata-v140/1.0'})
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            common.write_json(OUT / 'publisher-record-headers.json', {'UTC': common.utc(), 'url': url, 'status': exc.code, 'headers': dict(exc.headers), 'body_read': False})
            exc.close()
            raise RuntimeError(f'HTTP {exc.code}; no body/retry/fallback') from None
        with response:
            common.write_json(OUT / 'publisher-record-headers.json', {'UTC': common.utc(), 'url': url, 'final_url': response.geturl(), 'status': response.status, 'headers': dict(response.headers)})
            if response.status != 200 or response.geturl() != url or response.headers.get_content_type() != 'application/json':
                raise ValueError('Exact200/URL/JSON MIME before body')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise ValueError('Non-identity encoding')
            length = response.headers.get('Content-Length')
            cap = policy['response_limit_bytes']
            if length is not None and (not length.isdecimal() or int(length) > cap):
                raise ValueError('Content length/cap before body')
            raw = OUT / 'publisher-record.json'
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
        if not isinstance(value, dict) or value.get('id') != 17834008 or value.get('doi') != '10.5281/zenodo.17834008':
            raise ValueError('Publisher record identity mismatch')
        metadata = value.get('metadata')
        files = value.get('files')
        if not isinstance(metadata, dict) or not isinstance(files, list) or len(files) != 1:
            raise ValueError('Metadata/file-list schema or fixed count mismatch')
        item = files[0]
        if item.get('key') != policy['expected_key'] or item.get('checksum') != policy['expected_publisher_checksum'] or type(item.get('size')) is not int or item['size'] <= 0:
            raise ValueError('File declaration mismatch; no binary fallback')
        common.write_json(OUT / 'all-listed-containers-protected-before-binary.json', {'UTC': common.utc(), 'files': files, 'containers_and_all_possible_records': 'UNASSIGNED protected; no file/member content permission', 'photo_TEST_or_final_roles_fixed': False})
        common.write_json(OUT / 'publisher-record-receipt.json', {'bytes': received, 'local_SHA256': common.sha(raw), 'publisher_data_checksum_verified_locally': False, 'executed_or_links_followed': False})
        result = {'UTC': common.utc(), 'HEAD_prerequisite': head, 'policy_SHA256': common.sha(POLICY), 'network_body_bytes_received': received, 'record_id': value['id'], 'DOI': value['doi'], 'metadata_version': metadata.get('version'), 'service_revision_not_dataset_version': value.get('revision'), 'publisher_license_metadata': metadata.get('license'), 'creators_as_supplied': metadata.get('creators'), 'file_declarations': files, 'publisher_raw_metadata_local_SHA256': common.sha(raw), 'actual_archive_file_or_photo_identity_verified': False, 'new_raw_metadata_files_acquired': 1, 'source_photos_masks_annotations_archives_PDF_code_weights_acquired': 0, 'pixels_QA_features_our_scores_gradient_optimizer_CAL_TEST_API_CPU_GPU_CPP_ZIP_APK': 0, 'ready_for_training': False}
        stored = sum(p.stat().st_size for folder in (OUT, RUN) for p in folder.rglob('*') if p.is_file())
        if stored > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve')
        result['stored_bytes_before_completion'] = stored
        common.write_json(OUT / 'metadata-complete.json', result)
        print(json.dumps({'complete': True, 'bytes': received, 'license_metadata': metadata.get('license'), 'declared_archive_bytes': item['size'], 'ready': False}))
    except Exception as exc:
        common.write_json(RUN / 'failure.json', {'UTC': common.utc(), 'error_type': type(exc).__name__, 'error': str(exc), 'network_body_bytes_received': received, 'partials_preserved': True, 'retry_redirect_archive_fallback': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and common.strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()


if __name__ == '__main__':
    main()
