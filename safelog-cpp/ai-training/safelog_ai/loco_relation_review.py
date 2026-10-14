"""Normal-TRAIN connector observation review. No implicit training acceptance."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'loco_relation_observations_v1'
COLORS = ('yellow_green', 'blue', 'brown', 'other', 'unknown')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise ValueError(f'Duplicate JSON key: {k}')
            result[k] = v
        return result
    def invalid(value): raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid,
                      parse_float=lambda s: finite_float(s))


def finite_float(value):
    value = float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite number')
    return value


def require(condition, message):
    if not condition: raise ValueError(message)


def exact_keys(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys), 'Unexpected or missing fields')


def integer(value, low, high):
    require(type(value) is int and low <= value <= high, 'Invalid count or position')


def utc(value):
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z', value),
            'Millisecond UTC timestamp required')
    try: result = datetime.fromisoformat(value[:-1] + '+00:00')
    except (ValueError, TypeError) as e: raise ValueError('Invalid timestamp') from e
    require(result.utcoffset().total_seconds() == 0, 'UTC required')
    return result


def check_package(package):
    require(package['schema'] == 'loco_relation_review_package_v1', 'Wrong review package')
    payload = {k: v for k, v in package.items() if k != 'package_sha256'}
    require(canonical_sha(payload) == package['package_sha256'], 'Package hash differs')
    seen = set(); groups = set()
    for row in package['cases']:
        require(row['id'] not in seen, 'Duplicate source ID'); seen.add(row['id'])
        require(row['phase'] == 'fit' and row['publisher_split'] == 'train'
                and '/train/good/' in row['id'] and '/test/' not in row['id'], 'Only normal FIT is allowed')
        require(row['appearance_group_proxy'] not in groups, 'Repeated appearance group')
        groups.add(row['appearance_group_proxy'])
        require(row['label'] == 0 and row['masks_used'] == 0, 'Defect TEST masks are not allowed')
        require(len(row['connector_shapes']) >= 2 and len(row['cable_shapes']) >= 1, 'Missing source objects')
    require(package['training_ready'] is False, 'Review material cannot self-promote to training')


def validate_observations(document, package):
    """Return immutable review observations; never authorizes a training run."""
    check_package(package)
    exact_keys(document, ('schema', 'package_sha256', 'reviewer', 'exported_utc', 'cases'))
    require(document['schema'] == SCHEMA, 'Wrong observation schema')
    require(document['package_sha256'] == package['package_sha256'], 'Wrong source package')
    reviewer = document['reviewer']
    require(isinstance(reviewer, str) and reviewer == reviewer.strip() and 1 <= len(reviewer) <= 80
            and not any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in reviewer),
            'Reviewer name required')
    exported = utc(document['exported_utc'])
    require(isinstance(document['cases'], list), 'Cases must be a list')
    source = {r['id']: r for r in package['cases']}; seen = set()
    for row in document['cases']:
        exact_keys(row, ('id', 'image_sha256', 'annotation_sha256', 'status', 'reviewed_utc',
                         'polygons_accepted', 'groups', 'left_count', 'right_count', 'wire_count',
                         'wire_color', 'left_endpoint', 'right_endpoint', 'left_pin', 'right_pin', 'note'))
        require(row['id'] in source and row['id'] not in seen, 'Unknown or repeated source image')
        seen.add(row['id']); original = source[row['id']]
        for key in ('image_sha256', 'annotation_sha256'):
            require(row[key] == original[key], 'Source image/annotation hash differs')
        require(row['status'] in ('draft', 'confirmed', 'uncertain'), 'Invalid review status')
        require(type(row['polygons_accepted']) is bool, 'Polygon acceptance must be boolean')
        require(isinstance(row['note'], str) and len(row['note']) <= 2000, 'Invalid note')
        require(isinstance(row['groups'], list), 'Groups must be a list')
        shape_ids = {r['shape_id'] for r in original['connector_shapes']}; assigned = set()
        for assignment in row['groups']:
            exact_keys(assignment, ('shape_id', 'side'))
            require(type(assignment['shape_id']) is int and assignment['shape_id'] in shape_ids
                    and assignment['shape_id'] not in assigned, 'Invalid or repeated clamp shape')
            require(assignment['side'] in ('left', 'right'), 'Unknown component group')
            assigned.add(assignment['shape_id'])
        require(row['wire_color'] in COLORS, 'Invalid wire color')
        for key in ('left_count', 'right_count', 'wire_count', 'left_pin', 'right_pin'):
            if row[key] is not None: integer(row[key], 1, 16)
        for key in ('left_endpoint', 'right_endpoint'):
            if row[key] is not None:
                p = row[key]
                require(isinstance(p, list) and len(p) == 2 and all(
                    type(n) in (float, int) and math.isfinite(n) and 0 <= n <= 1 for n in p),
                    'Endpoint outside photo or not finite')
        if row['status'] == 'draft':
            require(row['reviewed_utc'] is None, 'Draft is not a completed review')
            continue
        reviewed = utc(row['reviewed_utc'])
        require(reviewed <= exported, 'Review timestamp is later than export')
        if row['status'] == 'uncertain':
            require(len(row['note'].strip()) >= 2, 'Explain uncertainty')
            continue
        require(row['polygons_accepted'] and assigned == shape_ids, 'All source polygons must be inspected')
        for side in ('left', 'right'):
            count = sum(a['side'] == side for a in row['groups'])
            require(count > 0 and row[f'{side}_count'] == count, 'Observed clamp count differs from grouping')
            integer(row[f'{side}_pin'], 1, count)
            require(row[f'{side}_endpoint'] is not None, 'Both wire attachment points must be marked')
            polygons = [s['points'] for s in original['connector_shapes']
                        if next(a['side'] for a in row['groups'] if a['shape_id'] == s['shape_id']) == side]
            xs = [p[0] for poly in polygons for p in poly]; ys = [p[1] for poly in polygons for p in poly]
            x, y = row[f'{side}_endpoint']
            require(min(xs)-0.03 <= x <= max(xs)+0.03 and min(ys)-0.03 <= y <= max(ys)+0.03,
                    'Endpoint is not near its assigned assembly')
        integer(row['wire_count'], 1, 16)
        require(row['wire_count'] == len(original['cable_shapes']), 'Wire polygon count needs correction/review')
        require(row['wire_color'] != 'unknown', 'Unknown color is not a confirmed label')
    return document


def review_summary(document, package):
    validate_observations(document, package)
    counts = {s: sum(r['status'] == s for r in document['cases']) for s in ('draft', 'confirmed', 'uncertain')}
    return {'review_counts': counts, 'missing_cases': len(package['cases']) - len(document['cases']),
            'training_ready': False, 'training_labels_created': 0, 'optimizer_updates': 0,
            'remaining_gates': ['Independent object/relation review and correction',
                                'Clarify licensing of newly added publisher annotations',
                                'Separate supervised validation annotations and preregister the training protocol']}


def import_review(path, package_path, output):
    package = strict_json(Path(package_path).read_text(encoding='utf-8'))
    document = strict_json(Path(path).read_text(encoding='utf-8'))
    summary = review_summary(document, package)
    result = {'schema': 'loco_relation_import_queue_v1', 'observations': document,
              'summary': summary, 'imported_utc': datetime.now(timezone.utc).isoformat(),
              'observations_sha256': canonical_sha(document)}
    with Path(output).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    return summary
