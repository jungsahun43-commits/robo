"""One declared equal-probability ensemble; validation first, frozen test second."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_facility_target import read, save, sha, dacl_items, TARGETS
from scripts.facility_error_target import operating_point, rates, under_target

MEMBERS = ('facility-presence-target-codebrim', 'facility-presence-target-spatial')
NAME = 'facility-presence-target-ensemble'


def combine(predictions):
    arrays = [np.asarray(value, dtype=float) for value in predictions]
    if len(arrays) != 2 or arrays[0].shape != arrays[1].shape:
        raise ValueError('Exactly two predictions of the same shape required')
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in arrays):
        raise ValueError('Invalid probabilities')
    return (arrays[0] + arrays[1]) / 2


def frozen_members():
    members = []
    for name in MEMBERS:
        run = ROOT / 'runs' / name
        training = read(run / 'TRAINING.json')
        if training['status'] != 'complete' or training['weights_sha256'] != sha(run / 'best.pt'):
            raise ValueError('Member must have complete, unchanged trained weights')
        members.append({'run': name, 'weights_sha256': training['weights_sha256'],
                        'training_sha256': sha(run / 'TRAINING.json'), 'classes': training['classes'],
                        'split_sha256': training['split_sha256'],
                        'additional_validation': training['additional_validation']})
    if members[0]['classes'] != members[1]['classes']:
        raise ValueError('Member classes do not match')
    if any(members[0][k] != members[1][k] for k in ('split_sha256', 'additional_validation')):
        raise ValueError('Member validation sources do not match')
    return members


def select():
    members = frozen_members()
    classes = members[0]['classes']
    points, counts, sources = {}, {}, {}
    domain_predictions = {}
    for domain in ('dacl', 'damsegment', 'codebrim'):
        values, hashes, images = [], [], []
        for member in members:
            path = ROOT / 'runs' / member['run'] / f'validation-{domain}.json'
            value = read(path)
            if value['split'] != 'val' or value['weights_sha256'] != member['weights_sha256']:
                raise ValueError('Saved validation must belong to the frozen member')
            cache_path = path.with_name(f'target-validation-{domain}-grid1.json')
            cache = read(cache_path)
            if cache['signature']['weights_sha256'] != member['weights_sha256'] or cache['signature']['grid'] != 1 or cache['classes'] != classes:
                raise ValueError('Frozen full-photo cache mismatch')
            images.append(cache['signature']['images_sha256'])
            value['probabilities'] = cache['probabilities']
            values.append(value); hashes.append(sha(path))
            hashes.append(sha(cache_path))
        if images[0] != images[1]:
            raise ValueError('Member validation images/order differ')
        target = np.asarray(values[0]['targets'])
        if not np.array_equal(target, values[1]['targets']):
            raise ValueError('Different validation targets/order')
        if any((target[:, classes.index(label)] < 0).any() for label in TARGETS):
            raise ValueError('Unknown target class cannot be scored as a negative')
        scores = combine([v['probabilities'] for v in values])
        if scores.shape != target.shape:
            raise ValueError('Validation score/target shape mismatch')
        domain_predictions[domain] = (target, scores)
        counts[domain] = len(target); sources[domain] = hashes
    for label in TARGETS:
        k = classes.index(label)
        points[label] = operating_point({domain: (t[:, k].astype(bool), p[:, k])
                                        for domain, (t, p) in domain_predictions.items()})
    result = {'run': NAME, 'selection_split': 'val', 'members': members, 'classes': classes,
              'rule': 'Equal mean of two full-photo probabilities; no fitted weights or view selection',
              'validation_counts': counts, 'validation_score_sha256': sources, 'per_class': points,
              'worst_error': max(p['worst_error'] for p in points.values()),
              'target_passed': all(p['target_passed'] for p in points.values()),
              'limitation': 'Repeatedly selected source validation; offline candidate, not active API or field accuracy'}
    save(ROOT / 'runs' / NAME / 'TARGET-SELECTION.json', result)
    save(ROOT / 'reports' / f'{NAME}-target-validation.json', result)
    print(__import__('json').dumps(result, indent=2))


def test(device):
    from scripts.evaluate_facility_target import cached
    run = ROOT / 'runs' / NAME
    selection = read(run / 'TARGET-SELECTION.json')
    if not selection['target_passed']:
        raise ValueError('Validation target not achieved; do not use test for repeat selection')
    if frozen_members() != selection['members']:
        raise ValueError('Frozen ensemble members changed')
    frozen = sha(run / 'TARGET-SELECTION.json')
    dacl, classes = dacl_items('test', 640)
    items = {'dacl': dacl, 'damsegment': read(ROOT / 'data/damsegment-training/test.json')['items']}
    metadata = [read(ROOT / 'runs' / n / 'TRAINING.json') for n in MEMBERS]
    for training in metadata:
        source = training['additional_test']; path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError('Frozen additional test changed')
    if metadata[0]['additional_test'] != metadata[1]['additional_test']:
        raise ValueError('Member test sources differ')
    code = read(ROOT / metadata[0]['additional_test']['path'])
    if code['split'] != 'test' or code['classes'] != classes or classes != selection['classes']:
        raise ValueError('Frozen test classes/split mismatch')
    items['codebrim'] = code['items']
    result = {}
    for domain, records in items.items():
        probabilities = combine([cached(ROOT / 'runs' / n / 'best.pt', records, 1,
                                         run / f'test-{domain}-{n}.json', device) for n in MEMBERS])
        result[domain] = {label: rates(np.array([i['targets'][classes.index(label)] for i in records], dtype=bool),
                                      probabilities[:, classes.index(label)] >= selection['per_class'][label]['threshold'])
                          for label in TARGETS}
    if frozen != sha(run / 'TARGET-SELECTION.json'):
        raise ValueError('Frozen selection changed during test')
    report = {'run': NAME, 'split': 'test', 'selection_sha256': frozen, 'members': selection['members'],
              'domains': result, 'target_passed': under_target([r for d in result.values() for r in d.values()]),
              'limitation': 'Recorded source holdouts; not independent field accuracy or deployed API verification'}
    save(ROOT / 'reports' / f'{NAME}-target-test.json', report)
    print(__import__('json').dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'test')); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    select() if args.stage == 'select' else test(args.device)
