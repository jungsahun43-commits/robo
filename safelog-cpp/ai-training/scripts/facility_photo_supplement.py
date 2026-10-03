"""Validate bounded, TRAIN-only building photo-label supplements.

Native boxes assert photo presence, not segmentation inside each rectangle.
The source has no factory/generalization evaluation in this experiment.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import numpy as np

ARCHIVE_SHA = '9acc37537c2a618c22bbda55d1353823d4e2dd82d98ae729d0c30315c72388ec'
PREFIX = PurePosixPath('data/peccd-training')
CONVID_URL = 'https://data.mendeley.com/datasets/fx3rthfjhy/4'
CONVID_INDEX_SHA = '7d10a985b47334894cd0e871a5416a565f20a000610e924f2fb825999e47e211'
CONVID_PICKS_SHA = '5016204d5ad1d8a230c3515d1c077748344dd4671233bac03321c0290124fecc'
ORIGINAL_MANIFESTS = (
    'data/facility-spatial-training/train.json', 'data/dacl10k-yolo/records.json',
    'data/damsegment-training/train.json', 'data/damsegment-training/test.json',
    'data/codebrim-training/train.json', 'data/codebrim-training/val.json', 'data/codebrim-training/test.json',
)


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_convid_preflight(audit, root, prefix):
    picks_path = confined_file(str(prefix / 'SOURCE-PICKS.json'), root, prefix)
    if audit.get('source_pick_sha256') != CONVID_PICKS_SHA or file_sha(picks_path) != CONVID_PICKS_SHA:
        raise ValueError('Frozen source-only picks changed after selection')
    picks = json.loads(picks_path.read_text(encoding='utf-8'))
    if picks.get('seed') != 52 or picks.get('per_folder_limit') != 100:
        raise ValueError('Source-only selection policy changed')
    selected = {str(r['id']): r for r in picks.get('picks', [])}
    if len(selected) != len(picks.get('picks', [])):
        raise ValueError('Frozen source-only picks repeat IDs')
    expected = list(ORIGINAL_MANIFESTS)
    if (root / 'data/s2ds/pixel-audit.json').exists():
        expected.append('data/s2ds/pixel-audit.json')
    snapshots = audit.get('original_manifests_unchanged_sha256', {})
    if set(snapshots) != set(expected) or any(file_sha(root / p) != snapshots[p] for p in expected):
        raise ValueError('Original source split/label manifests changed after preparation')
    public_path = root / 'reports/facility-convid-data-audit.json'
    if file_sha(public_path) != audit.get('audit_file_sha256'):
        raise ValueError('Public source preparation audit changed')
    public = json.loads(public_path.read_text(encoding='utf-8'))
    if public != {k: v for k, v in audit.items() if k not in ('source_index_path', 'audit_file_sha256')}:
        raise ValueError('Local source audit disagrees with published aggregate preparation')
    return selected


def confined_file(raw, root, prefix):
    if not isinstance(raw, str) or chr(92) in raw:
        raise ValueError('Photo supplement paths must be relative POSIX paths')
    path = PurePosixPath(raw)
    resolved = (root / raw).resolve()
    if (path.is_absolute() or '..' in path.parts or not path.is_relative_to(prefix)
            or not resolved.is_relative_to((root / str(prefix)).resolve()) or not resolved.is_file()):
        raise ValueError('Photo supplement files must remain inside their ignored source directory')
    return resolved


def validate_photo_supplement(data, classes, root, original_images):
    audit = data.get('audit', {})
    items = data.get('items', [])
    domain = items[0].get('domain') if items else None
    maximum = 200 if domain == 'convid' else 500
    if (data.get('split') != 'train' or data.get('classes') != classes
            or audit.get('status') != 'prepared' or domain not in ('peccd', 'convid')
            or not 1 <= len(items) <= maximum):
        raise ValueError('Photo supplement must be a verified, bounded building TRAIN preparation')
    root = Path(root).resolve()
    prefix = PurePosixPath('data/convid-training') if domain == 'convid' else PREFIX
    records = picks = None
    if domain == 'peccd':
        if audit.get('source_archive_sha256') != ARCHIVE_SHA:
            raise ValueError('PECCD archive checksum changed')
    else:
        index_path = confined_file(audit.get('source_index_path'), root, prefix)
        digest = file_sha(index_path)
        if audit.get('source_index_sha256') != CONVID_INDEX_SHA or digest != CONVID_INDEX_SHA:
            raise ValueError('ConViD official source index checksum changed')
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('dataset_url') != CONVID_URL or index.get('version') != 4:
            raise ValueError('ConViD source version changed')
        records = {str(r['id']): r for r in index.get('records', [])}
        if len(records) != len(index.get('records', [])):
            raise ValueError('ConViD source IDs repeat')
        picks = validate_convid_preflight(audit, root, prefix)
        if any(records.get(identity) != row for identity, row in picks.items()):
            raise ValueError('Frozen source picks disagree with official source index')
    images, masks, source_ids = set(), set(), set()
    for item in items:
        if item.get('domain') != domain or item.get('source_split') != 'train_only_unsplit':
            raise ValueError('Only one recorded, unsplit building TRAIN source may supplement')
        target = item.get('targets', [])
        if domain == 'peccd':
            if (len(target) != 7 or any(v not in (0, 1) for v in target[:2])
                    or target[2:] != [-1] * 5):
                raise ValueError('PECCD asserts only crack/spalling photo labels; all other labels are unknown')
        else:
            source_id = str(item.get('source_id'))
            record = records.get(source_id)
            if (not record or source_id not in picks or source_id in source_ids or record.get('folder') not in ('crack', 'Spalling')
                    or item.get('source_sha256') != record.get('sha256')
                    or item.get('source_folder') != record['folder']
                    or not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in record['sha256'])):
                raise ValueError('ConViD photo provenance does not match its official source index')
            expected = [-1] * 7
            expected[0 if record['folder'] == 'crack' else 1] = 1
            if target != expected:
                raise ValueError('ConViD folder asserts only one positive; all other photo labels remain unknown')
            source_ids.add(source_id)
        for field in ('image', 'pixel_target'):
            confined_file(item.get(field), root, prefix)
        if domain == 'convid':
            original = confined_file(item.get('source_original'), root, prefix)
            if file_sha(original) != item['source_sha256']:
                raise ValueError('ConViD original image checksum changed')
            digest = file_sha(root / item['image'])
            if item.get('prepared_image_sha256') != digest:
                raise ValueError('Prepared ConViD image checksum changed')
        if item['image'] in images or item['image'] in original_images:
            raise ValueError('Photo supplement repeats an image path')
        images.add(item['image'])
        masks.add(item['pixel_target'])
    if domain == 'convid':
        if any(not 1 <= sum(i['source_folder'] == folder for i in items) <= 100 for folder in ('crack', 'Spalling')):
            raise ValueError('ConViD must retain between one and 100 screened photos from each frozen folder')
    for mask in masks:
        with np.load(root / mask, allow_pickle=False) as fields:
            if (fields['mask'].shape != (7, 80, 80) or fields['known'].shape != (7,)
                    or np.any(fields['mask']) or np.any(fields['known'])):
                raise ValueError('Source boxes must not invent pixel regions or absent pixel truth')
    return items
