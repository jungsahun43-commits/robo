"""Build the fixed normal-FIT annotation observation package. No training/evaluation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.loco_relation_review import canonical_sha, check_package, strict_json

DATA = ROOT / 'data/industrial-relation-review-v14'
RUN = ROOT / 'runs/industrial-relation-review-v14'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return strict_json(path.read_text(encoding='utf-8'))


def proposals(shapes):
    """Input-polygon two-means proposal only, never verified assembly labels."""
    xs = [sum(p[0] for p in s['points']) / len(s['points']) for s in shapes]
    centers = [min(xs), max(xs)]
    assert centers[1] - centers[0] > 0.1
    for _ in range(20):
        assignment = [min(range(2), key=lambda j: abs(x-centers[j])) for x in xs]
        assert set(assignment) == {0, 1}
        centers = [sum(x for x, a in zip(xs, assignment) if a == j)/assignment.count(j) for j in range(2)]
    return [{'shape_id': s['shape_id'], 'side': ('left', 'right')[a]} for s, a in zip(shapes, assignment)]


def main():
    assert not RUN.exists(), 'Retain old artifacts; choose a new output for a rebuild.'
    protocol = read(DATA / 'protocol.json')
    receipts = read(DATA / 'download-receipts.json')
    receipts = {r['entry']: r for r in receipts}
    package = {'schema': 'loco_relation_review_package_v1', 'protocol_sha256': sha(DATA / 'protocol.json'),
               'created_utc': datetime.now(timezone.utc).isoformat(), 'training_ready': False,
               'author_url': protocol['author_url'], 'author_repo_commit': protocol['author_repo_commit'],
               'license_note': 'MVTec image CC BY-NC-SA4.0 bundled; separate new annotation rights not expressly stated',
               'derived_groups_verified': False, 'human_relation_labels_collected': 0, 'cases': []}
    report = {'schema': 'loco_relation_review_audit_v1', 'protocol_sha256': package['protocol_sha256'],
              'protocol_commit': '37fb3c8', 'photos': 24, 'actual_optimizer_updates': 0,
              'TEST_annotations_downloaded': 0, 'VAL_CAL_annotations_downloaded': 0,
              'current_TEST_scores_recomputed': False, 'TRAIN_defects_acquired': 0,
              'human_verified_relation_labels': 0, 'training_ready': False}
    for row in protocol['rows']:
        stem = Path(row['image']).stem
        annotation = DATA / f'publisher/train/good/{stem}.json'
        png = DATA / f'publisher/train/good/{stem}.png'
        original = read(annotation)
        receipt = receipts[f'LOCO-Annotations/splicing_connectors/train/good/{stem}.png']
        assert receipt['CRC_verified'] and receipt['local_source_RGB_exact']
        assert sha(png) == receipt['content_sha256']
        assert sha(annotation) == receipts[f'LOCO-Annotations/splicing_connectors/train/good/{stem}.json']['content_sha256']
        assert original['imagePath'] == stem + '.png' and not original.get('imageData')
        w, h = row['size']
        assert [original['imageWidth'], original['imageHeight']] == [w, h]
        assert original['image_text'].strip().lower() == 'normal.'
        shapes = []
        for i, shape in enumerate(original['shapes'], 1):
            assert shape['label'] in ('connector', 'cable') and shape['shape_type'] == 'polygon'
            assert len(shape['points']) >= 3
            points = [[float(x)/w, float(y)/h] for x, y in shape['points']]
            assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)
            shapes.append({'shape_id': i, 'label': shape['label'], 'points': points})
        connectors = [s for s in shapes if s['label'] == 'connector']
        cables = [s for s in shapes if s['label'] == 'cable']
        proposed = proposals(connectors)
        package['cases'].append({
            'id': row['id'], 'phase': row['phase'], 'publisher_split': row['publisher_split'],
            'appearance_group_proxy': row['appearance_group_proxy'], 'label': row['label'], 'masks_used': 0,
            'image': f'images/{stem}.png', 'image_sha256': sha(png), 'local_source_image_sha256': row['image_sha256'],
            'rgb_sha256': row['rgb_sha256'], 'annotation_sha256': sha(annotation), 'width': w, 'height': h,
            'connector_shapes': connectors, 'cable_shapes': cables, 'proposed_groups': proposed,
            'proposal_provenance': 'two-means on publisher polygon centers; not verified labels',
            'human_verified': False})
    package['package_sha256'] = canonical_sha(package)
    check_package(package)
    RUN.mkdir(); ui = RUN / 'review'; ui.mkdir(); (ui / 'images').mkdir(); (ui / 'publisher-annotations').mkdir()
    for path in (ROOT / 'relation-review-ui').iterdir():
        if path.is_file(): shutil.copyfile(path, ui / path.name)
    for row in package['cases']:
        stem = Path(row['image']).stem
        shutil.copyfile(DATA / f'publisher/train/good/{stem}.png', ui / row['image'])
        shutil.copyfile(DATA / f'publisher/train/good/{stem}.json', ui / f'publisher-annotations/{stem}.json')
    for name in ('license.txt', 'readme.txt'):
        shutil.copyfile(DATA / f'publisher/{name}', ui / f'MVTEC_{name}')
    (ui / 'package.json').write_text(json.dumps(package, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    script = 'window.REVIEW_PACKAGE = ' + json.dumps(package, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c') + ';\n'
    (ui / 'package.js').write_text(script, encoding='utf-8')
    report['package_sha256'] = package['package_sha256']
    report['all24_author_images_local_RGB_equal'] = True
    report['all24_author_images_local_file_bytes_equal'] = all(r['image_sha256'] == r['local_source_image_sha256'] for r in package['cases'])
    report['appearance_groups'] = len({r['appearance_group_proxy'] for r in package['cases']})
    report['source_polygon_counts'] = dict(Counter(str(len(r['connector_shapes'])) for r in package['cases']))
    report['source_cable_polygon_counts'] = dict(Counter(str(len(r['cable_shapes'])) for r in package['cases']))
    report['near_group_audit_inherited_from_v13'] = True
    report['physical_acquisition_groups_verified'] = False
    report['inherited_parent_near_training_exposure_undone'] = False
    report['annotation_specific_license_explicit'] = False
    for path in (RUN / 'audit.json', ROOT / 'reports/industrial-loco-relation-review-v14-audit.json'):
        with path.open('x', encoding='utf-8', newline='\n') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print(report)


if __name__ == '__main__': main()
