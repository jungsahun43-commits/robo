"""Use the already audited tube normal views alongside actual source masks."""
from .source_detail_study import source_only_rows


def tube_adaptation_rows(rows):
    parts = source_only_rows(rows)
    for row in rows:
        if row['is_NEU'] or row.get('category') != 'mpdd_tubes':
            continue
        phase = row['phase']
        if phase not in parts:
            continue
        active = row['training_active'] if phase == 'fit' else row['validation_active']
        if active:
            assert row['label'] == 0 and row['positive_pixels'] == 0 and row['active_support']
            parts[phase].append(dict(row))
    assert {r['photo_id'] for r in parts['fit']}.isdisjoint(r['photo_id'] for r in parts['validation'])
    return parts
