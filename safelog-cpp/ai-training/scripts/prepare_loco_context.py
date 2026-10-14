"""Re-audit source grouping and split normal TRAIN into FIT and validation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.loco_context import CATEGORY
from safelog_ai.belt_normal_roi import distance
from scripts.prepare_industrial_foreground import source
from scripts.study_equipment_expansion import MANIFEST, read
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new

DATA = ROOT/'data/industrial-loco-context-v13'; RUN = ROOT/'runs/industrial-loco-context-v13'
REPORT = ROOT/'reports/industrial-loco-context-v13-data-audit.json'


def main():
    assert not DATA.exists() and not RUN.exists(); DATA.mkdir(); RUN.mkdir()
    selected = [r for r in read(MANIFEST)['rows'] if r['category'] == CATEGORY and r['phase'] in ('fit', 'cal', 'test')]
    with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(source, selected))
    roots = list(range(len(rows))); links = []
    def find(i):
        while roots[i] != i: roots[i] = roots[roots[i]]; i = roots[i]
        return i
    for i, a in enumerate(rows):
        for j, b in enumerate(rows[:i]):
            near = distance(a['orientation_phashes'], b['orientation_phashes']) <= 2
            if a['rgb_sha256'] == b['rgb_sha256'] or near:
                x, y = find(i), find(j); roots[max(x, y)] = min(x, y)
                links.append({'a': a['id'], 'b': b['id'], 'exact_RGB': a['rgb_sha256'] == b['rgb_sha256'], 'near_pHash': near})
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(find(i), []).append(row)
    for group in groups.values():
        key = min(r['id'] for r in group)
        for row in group: row['appearance_group_proxy'] = key
        test = [r for r in group if r['phase'] == 'test']; cal = sorted((r for r in group if r['phase'] == 'cal'), key=lambda r: r['id'])
        if test:
            for row in group:
                if row['phase'] != 'test': row['phase'] = 'excluded_orientation_near_TEST'
        elif cal:
            for row in group:
                if row['phase'] == 'fit': row['phase'] = 'excluded_orientation_near_CAL'
            for row in cal[1:]: row['phase'] = 'excluded_duplicate_CAL_group'
        else:
            normal = sorted(group, key=lambda r:r['id'])
            phase = 'validation' if int(hashlib.sha256(('loco13:'+key).encode()).hexdigest(), 16)%5 == 0 else 'fit'
            normal[0]['phase'] = phase
            for row in normal[1:]: row['phase'] = 'excluded_duplicate_TRAIN_group'
    phases = {phase: {r['appearance_group_proxy'] for r in rows if r['phase'] == phase} for phase in ('fit','validation','cal','test')}
    assert all(not phases[a] & phases[b] for i,a in enumerate(phases) for b in list(phases)[i+1:])
    for row in rows:
        if row['phase'] in ('fit','validation','cal'): assert row['label'] == 0 and row['publisher_split'] != 'test'
    counts = dict(Counter(r['phase'] for r in rows))
    assert counts.get('fit',0)>=50 and counts.get('validation',0)>=20 and counts.get('cal',0)>=20
    audit = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': sha(MANIFEST), 'counts': counts,
        'source_RGB_images_masks_SHA_rechecked': len(rows), 'group_proxy': 'exactRGB or any8orientation pHash<=2 connected components',
        'group_count': len(groups), 'near_links': links, 'TEST_near_groups_in_new_FIT_VAL_CAL': False,
        'FIT_VAL_CAL_TEST_groups_disjoint': True, 'VAL': 'one fifth of eligible normal publisher TRAIN appearance groups via fixedSHA256',
        'CAL': 'remaining publisher normal validation after group exclusions; no gradient or epoch selection',
        'physical_acquisition_groups_locally_verified': False, 'mirror_original_equivalence_verified': False,
        'inherited_parent_near_training_exposure_not_undone': True, 'pHash_similarity_does_not_prove_physical_duplicate': True,
        'new_model_predictions_or_training_done': False, 'independent_field_validated': False}
    new(DATA/'manifest.json', {'rows': rows, 'audit': audit}); new(REPORT, audit)
    print({'data_audit': 'passed', 'counts': counts, 'groups': len(groups), 'near_links': len(links)}, flush=True)


if __name__ == '__main__': main()
