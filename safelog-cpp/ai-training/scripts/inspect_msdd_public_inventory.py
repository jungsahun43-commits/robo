"""Read public MSDD archive-tree JSON, never the 9.78GB archive or image pixels."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.fetch_equipment_expansion import sha

OUT = ROOT / 'data/industrial-structured-metal-v42-source'
LIMIT = 64 * 1024 ** 2


def flatten_tree(value):
    """Author public preview schema, treated strictly as metadata, not paths to extract."""
    result = []
    seen = set()

    def visit(node, depth=0):
        if depth > 12 or not isinstance(node, dict):
            raise ValueError('Unexpected archive tree nesting')
        if node.get('type') == 'file':
            path = node['path']
            parts = PurePosixPath(path)
            if (parts.is_absolute() or '..' in parts.parts or '\\' in path or ':' in path
                    or parts.name != node['name'] or path in seen):
                raise ValueError('Unsafe or duplicate public archive metadata path')
            if not re.fullmatch('[a-f0-9]{32}', node['id']) or not isinstance(node['size'], int) or node['size'] < 0:
                raise ValueError('Unexpected public archive ID/size')
            seen.add(path)
            result.append({key: node[key] for key in ('name', 'path', 'id', 'size', 'type')})
        else:
            if not isinstance(node.get('children'), list):
                raise ValueError('Expected public directory children')
            for child in node['children']:
                visit(child, depth + 1)

    visit(value)
    return result


def main():
    assert not (OUT / 'complete-archive-inventory.json').exists()
    original = read(OUT / 'public-file-root-response-r2.json')['data']
    assert len(original) == 1
    item = original[0]
    assert item['fileName'] == 'MSDD.rar' and item['size'] == 9785811363 and item['version'] == 'V1'
    url = 'https://www.scidb.cn/api/sdb-download-service/downloadFileMole?fileId=' + item['id']
    new(OUT / 'complete-preview-intent.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'public_preview_url': url, 'observed_contract': 'author public ZipFileTree component GET then JSON.parse; no author JavaScript execution',
        'bound': LIMIT, 'previous_16MiB_JSON_tree_partial_preserved': True,
        'public_archive_metadata_only': True, 'photo_or_annotation_contents_decoded': False,
        'archive_publisher_MD5_not_yet_verified': item['md5'], 'whole_archive_download_forbidden': True})
    path = OUT / 'public-archive-preview-complete.json'
    size = 0
    complete = False
    try:
        with requests.get(url, headers={'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 45)) as response:
            response.raise_for_status()
            with path.open('xb') as target:
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue
                    if size == 0 and chunk.lstrip()[:1] != b'{':
                        raise ValueError('Not archive-tree JSON; do not download the archive')
                    if size + len(chunk) > LIMIT:
                        raise ValueError('Archive preview exceeds metadata-only bound')
                    size += len(chunk)
                    target.write(chunk)
        complete = True
        value = read(path)
        rows = flatten_tree(value)
        counts = dict(Counter(PurePosixPath(row['path']).parts[0] for row in rows))
        extensions = dict(Counter(PurePosixPath(row['path']).suffix.lower() for row in rows))
        captures = set()
        dates = Counter()
        for row in rows:
            match = re.match(r'(\d{8})_(\d{6})_Gige\[(\d+)\]', row['name'])
            if match:
                captures.add(match[1] + '_' + match[2])
                dates[match[1]] += 1
        split_files = [row for row in rows if PurePosixPath(row['path']).suffix.lower() in ('.txt', '.csv', '.json')]
        report = {'UTC': datetime.now(timezone.utc).isoformat(), 'public_preview_bytes': size,
            'public_preview_SHA256': sha(path), 'source_archive_name': item['fileName'], 'source_archive_bytes': item['size'],
            'publisher_archive_MD5': item['md5'], 'archive_MD5_verified': False,
            'listed_files': len(rows), 'counts_by_root': counts, 'extensions': extensions,
            'uncompressed_member_bytes_by_root': {key: sum(row['size'] for row in rows if PurePosixPath(row['path']).parts[0] == key) for key in counts},
            'date_proxy_file_counts': dict(dates), 'unique_capture_timestamp_proxies': len(captures),
            'possible_author_split_or_description_files': split_files,
            'author_capture_proxy_not_physical_identity_proof': True, 'raw_illuminations_not_independent_parts': True,
            'photo_or_annotation_contents_downloaded': 0, 'new_encoder_features_scores_optimizer': 0,
            'TEST_split_or_training_protocol_fixed': False,
            'public_child_ids_not_proven_individual_download_support': True,
            'source_is_multilight_and_pseudoRGB_not_smartphone_input': True}
        new(OUT / 'complete-archive-inventory.json', {'rows': rows, 'audit': report})
        report['inventory_SHA256'] = sha(OUT / 'complete-archive-inventory.json')
        new(ROOT / 'reports/industrial-structured-metal-v42-archive-metadata.json', report)
        print({key: value for key, value in report.items() if key != 'possible_author_split_or_description_files'}, flush=True)
        print({'possible_split_files': split_files[:15]}, flush=True)
    except BaseException as error:
        new(OUT / 'complete-preview-failure.json', {'UTC': datetime.now(timezone.utc).isoformat(),
            'error': repr(error), 'received_bytes': size, 'HTTP_complete': complete, 'partial_preserved': True,
            'image_decode_model_execution': 0})
        raise


if __name__ == '__main__':
    main()
