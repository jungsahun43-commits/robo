"""Two new public metadata endpoints only; preserved v134 record/text never fetched again."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import inspect_hrcds_metadata_v121 as common

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/inspect_footpath_citation_v135.py'
POLICY = ROOT / 'reports/industrial-footpath-v135-citation-policy.json'
DOC = 'reports/FOOTPATH_V135_CITATION_PROTOCOL_KO.md'
OUT = ROOT / 'data/industrial-footpath-v135-citation'
RUN = ROOT / 'runs/industrial-footpath-v135-citation'
received = 0


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def get_json_once(name, url):
    global received
    opener = urllib.request.build_opener(RejectRedirect())
    request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'SafeLog-public-citation-metadata-v135/1.0'})
    try:
        response = opener.open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        common.write_json(OUT / (name + '-headers.json'), {'UTC': common.utc(), 'url': url, 'status': exc.code, 'headers': dict(exc.headers), 'body_read': False})
        exc.close()
        raise RuntimeError(f'HTTP {exc.code}; body not read; no retry') from None
    with response:
        common.write_json(OUT / (name + '-headers.json'), {'UTC': common.utc(), 'url': url, 'final_url': response.geturl(), 'status': response.status, 'headers': dict(response.headers)})
        if response.status != 200 or response.geturl() != url or response.headers.get_content_type() != 'application/json':
            raise ValueError('Exact200/URL/JSON MIME guard failed before body')
        if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
            raise ValueError('Non-identity encoding')
        length = response.headers.get('Content-Length')
        if length is not None and (not length.isdecimal() or int(length) > 524288):
            raise ValueError('Length cap before body')
        path = OUT / name
        count = 0
        with path.open('xb') as target:
            while True:
                block = response.read(min(16384, 524289 - count))
                if not block:
                    break
                target.write(block)
                count += len(block)
                received += len(block)
                if count > 524288 or received > 1048576:
                    raise ValueError('Cap exceeded; partial preserved')
        if length is not None and count != int(length):
            raise ValueError('Header/actual length mismatch')
        value = common.strict_json(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError('JSON object required')
        common.write_json(OUT / (name + '-receipt.json'), {'bytes': count, 'local_SHA256': common.sha(path), 'author_provided_SHA256': False, 'executed_or_links_followed': False})
        return value


def ident(value):
    raw = value.get('id')
    if type(raw) is int:
        return raw
    if isinstance(raw, str) and raw.isdecimal():
        return int(raw)
    raise ValueError('Invalid numeric record ID')


def main():
    policy = common.strict_json(POLICY.read_text(encoding='utf-8'))
    if common.sha(ROOT / SCRIPT) != policy['script_sha256'] or common.sha(Path(common.__file__)) != policy['utility_sha256']:
        raise ValueError('Frozen code mismatch')
    for name, expected in policy['preserved_inputs_sha256'].items():
        if common.sha(ROOT / name) != expected:
            raise ValueError('Preserved input SHA mismatch: ' + name)
    for relative in (SCRIPT, POLICY.relative_to(ROOT).as_posix(), DOC):
        if hashlib.sha256(common.git('show', 'HEAD:safelog-cpp/ai-training/' + relative)).hexdigest() != common.sha(ROOT / relative):
            raise ValueError('Prerequisite file differs from committed HEAD')
    if common.git('status', '--porcelain').strip():
        raise ValueError('Clean repository required')
    head = common.git('rev-parse', 'HEAD').decode().strip()
    if head != common.git('ls-remote', 'origin', 'refs/heads/feature/ai-engine').decode().split()[0]:
        raise ValueError('Prerequisite ordinary feature push unconfirmed')
    if common.process_exists(23948):
        raise ValueError('Preserved training.lock PID unexpectedly alive')
    if shutil.disk_usage(ROOT).free < policy['reserve_bytes'] + policy['storage_budget_bytes']:
        raise ValueError('Storage/reserve gate')
    current = common.strict_json((ROOT / policy['preserved_current_record']).read_text(encoding='utf-8'))
    license_text = (ROOT / policy['preserved_LICENSE']).read_text(encoding='utf-8')
    if current['id'] != 17154852 or current['links']['versions'] != policy['versions_url'] or 'https://doi.org/10.5281/zenodo.15414238' not in license_text:
        raise ValueError('New endpoints must match preserved author citation/version-link provenance')
    if OUT.exists() or RUN.exists():
        raise ValueError('Stage already started; no rerun/overwrite')
    OUT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    common.write_json(RUN / 'started.json', {'UTC': common.utc(), 'pid': os.getpid(), 'HEAD': head, 'policy_SHA256': common.sha(POLICY)})
    lock = RUN / 'operation.lock'
    token = {'pid': os.getpid(), 'stage': 'footpath-v135', 'UTC': common.utc()}
    common.write_json(lock, token)
    try:
        cited = get_json_once('publisher-cited-record.json', policy['cited_record_url'])
        if ident(cited) != 15414238 or cited.get('doi') != '10.5281/zenodo.15414238':
            raise ValueError('Cited record ID/DOI identity mismatch')
        files = cited.get('files')
        if not isinstance(files, list) or len(files) > 32:
            raise ValueError('Cited file-list schema/cap')
        common.write_json(OUT / 'cited-containers-protected-before-any-file.json', {'UTC': common.utc(), 'files': files, 'all_containers_possible_records': 'UNASSIGNED protected; no binary/file content permitted', 'photo_TEST_roles_fixed': False})
        versions = get_json_once('publisher-version-list.json', policy['versions_url'])
        hits = versions.get('hits')
        if not isinstance(hits, dict) or not isinstance(hits.get('hits'), list) or len(hits['hits']) > 25:
            raise ValueError('Version-list schema/cap')
        total = hits.get('total')
        total_relation = None
        if isinstance(total, dict):
            total_relation = total.get('relation')
            total = total.get('value')
        if type(total) is not int or total < 0:
            raise ValueError('Invalid version total')
        summaries = []
        for item in hits['hits']:
            index = item.get('versions', {}).get('index')
            if index is not None and (type(index) is not int or index < 1):
                raise ValueError('Invalid explicit version index')
            summaries.append({'id': ident(item), 'explicit_version_index': index, 'metadata_version': item.get('metadata', {}).get('version'), 'parent_id': item.get('parent', {}).get('id'), 'DOI': item.get('doi') or item.get('pids', {}).get('doi', {}).get('identifier')})
        if len({item['id'] for item in summaries}) != len(summaries):
            raise ValueError('Duplicate version-list record ID')
        declared_matches = []
        current_files = {f['key']: f for f in current['files']}
        for item in files:
            key = item.get('key')
            prior = current_files.get(key)
            if prior:
                declared_matches.append({'key': key, 'same_declared_size': item.get('size') == prior.get('size'), 'same_declared_checksum': item.get('checksum') == prior.get('checksum'), 'cited_size': item.get('size'), 'current_size': prior.get('size'), 'cited_checksum': item.get('checksum'), 'current_checksum': prior.get('checksum'), 'actual_file_content_compared': False})
        same_concept = cited.get('conceptdoi') == current.get('conceptdoi') and str(cited.get('conceptrecid')) == str(current.get('conceptrecid'))
        result = {'UTC': common.utc(), 'HEAD_prerequisite': head, 'policy_SHA256': common.sha(POLICY), 'network_body_bytes_received': received, 'cited_record_id': ident(cited), 'current_record_id': current['id'], 'cited_conceptdoi': cited.get('conceptdoi'), 'current_conceptdoi': current.get('conceptdoi'), 'same_publisher_concept_family_confirmed': same_concept, 'version_list_total': total, 'version_list_total_relation': total_relation, 'version_list_complete_as_returned': total_relation in (None, 'eq') and total == len(summaries), 'version_summaries': summaries, 'declared_file_identity_comparisons': declared_matches, 'actual_archive_file_or_photo_identity_verified': False, 'new_metadata_files_acquired': 2, 'source_file_content_pixels_QA_features_gradient_optimizer_CAL_TEST_delivery': 0, 'ready_for_training': False}
        stored = sum(p.stat().st_size for folder in (OUT, RUN) for p in folder.rglob('*') if p.is_file())
        if stored > policy['storage_budget_bytes'] or shutil.disk_usage(ROOT).free < policy['reserve_bytes']:
            raise ValueError('Post-run storage/reserve gate')
        result['stored_bytes_before_completion'] = stored
        common.write_json(OUT / 'citation-complete.json', result)
        print(json.dumps({'complete': True, 'network_body_bytes': received, 'same_concept': same_concept, 'versions': summaries, 'ready': False}))
    except Exception as exc:
        common.write_json(RUN / 'failure.json', {'UTC': common.utc(), 'error_type': type(exc).__name__, 'error': str(exc), 'network_body_bytes_received': received, 'partials_preserved': True, 'retry_or_archive_fallback': False, 'training_error': False})
        raise
    finally:
        if lock.exists() and common.strict_json(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()


if __name__ == '__main__':
    main()
