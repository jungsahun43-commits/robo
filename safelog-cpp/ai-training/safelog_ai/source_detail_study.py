"""Pure split and original-photo-weighted validation helpers for actual masks."""
from collections import defaultdict
import numpy as np


def source_only_rows(rows):
    result = {'fit': [], 'validation': []}
    for row in rows:
        if not row['is_NEU'] or not row['active_support']:
            continue
        phase = row['phase']
        if phase not in result:
            continue
        active = row['training_active'] if phase == 'fit' else row['validation_active']
        if not active:
            continue
        if phase == 'fit':
            assert row['source_folder'] in ('NEU_SDI', 'NEU_SPDI')
        else:
            assert row['source_folder'] == 'NEU_Oil' and '/Scratch_' not in row['photo_id']
        assert row['bucket'] in ('real_positive', 'annotation_negative')
        assert row['positive_pixels'] == 0 or row['positive_pixels'] >= 32
        result[phase].append(dict(row))
    assert {r['photo_id'] for r in result['fit']}.isdisjoint(r['photo_id'] for r in result['validation'])
    return result


def photo_bucket_means(rows, losses):
    if len(rows) != len(losses):
        raise ValueError('One loss per crop required')
    grouped = defaultdict(lambda: defaultdict(list))
    for row, loss in zip(rows, losses):
        if not np.isfinite(loss):
            raise ValueError('Finite per-crop losses required')
        grouped[row['bucket']][row['photo_id']].append(float(loss))
    return {bucket: float(np.mean([np.mean(values) for values in photos.values()]))
            for bucket, photos in grouped.items()}
