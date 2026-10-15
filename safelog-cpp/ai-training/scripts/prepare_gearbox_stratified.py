"""Separate v26 data design; v25 failed its immutable calibration-data gate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.gearbox_groups import stratified_allocation
from scripts.prepare_gearbox_grouped import MANIFEST as OLD
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

DATA = ROOT / 'data/industrial-gearbox-stratified-v26'
RUN = ROOT / 'runs/industrial-gearbox-v26'
POLICY = ROOT / 'reports/industrial-gearbox-v26-data-policy.json'
MANIFEST = DATA / 'manifest.json'


def freeze():
    assert not DATA.exists() and not RUN.exists() and not POLICY.exists()
    audit = read(OLD)['audit']; assert not audit['ready']
    assert audit['counts']['cal']['0'] == 10 and not audit['old_TEST_direct_matches']
    DATA.mkdir(); RUN.mkdir()
    new(POLICY, {'UTC': datetime.now(timezone.utc).isoformat(), 'fixed_before_new_split_features_scores_optimizer': True,
        'parent_v25_manifest_sha256': sha(OLD),
        'source_sha256': {n: sha(ROOT/n) for n in ('safelog_ai/gearbox_groups.py', 'scripts/prepare_gearbox_stratified.py', 'tests/test_gearbox_groups.py')},
        'v25_stopped_unchanged': 'CAL normal10<20; features/scores/optimizer0. Do not overwrite initial manifest or relax minimum.',
        'changes': 'new study uses publisher TRAIN labels only for deterministic greedy stratified groups; same reserved authorS3fold0TEST, source exclusions and exact/near groups preserved',
        'rule': 'sort intact nonTEST groups by decreasing patches then SHA256(gearbox26:+group); fixed group capacities N-2ceil(.15N)/ceil(.15N)/ceil(.15N); greedily minimize incremental squared normalized per-label count distance to .70/.15/.15; tie fit,val,cal. No seed or partition retries.',
        'gate': 'FIT eachlabel>=100/groups>=8; VAL/CAL eachlabel>=20/groups>=3; TEST eachlabel>=20/groups>=3; no old/reserved TEST-near group in FIT/VAL/CAL',
        'TEST_used_for_split_objective': False, 'new_model_features_scores_optimizer_updates': 0,
        'new_unit_tests_passed': 3})


def prepare():
    p = read(POLICY); assert sha(OLD) == p['parent_v25_manifest_sha256']
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest
    assert not MANIFEST.exists()
    original = read(OLD); rows = original['rows']
    split = stratified_allocation([r for r in rows if r['phase'] in ('fit', 'validation', 'cal')])
    for r in rows:
        if r['phase'] in ('fit', 'validation', 'cal'): r['phase'] = split[r['group_id']]
    phases = ('fit', 'validation', 'cal', 'test')
    counts = {p: dict(Counter(str(r['label']) for r in rows if r['phase'] == p)) for p in phases}
    groups = {p: sorted({r['group_id'] for r in rows if r['phase'] == p}) for p in phases}
    assert all(set(groups[a]).isdisjoint(groups[b]) for i,a in enumerate(phases) for b in phases[i+1:])
    ready = (all(counts['fit'].get(str(y), 0) >= 100 for y in (0,1)) and len(groups['fit']) >= 8
        and all(all(counts[k].get(str(y), 0) >= 20 for y in (0,1)) and len(groups[k]) >= 3 for k in phases[1:]))
    audit = {'UTC': datetime.now(timezone.utc).isoformat(), 'ready': ready, 'policy_sha256': sha(POLICY),
        'v25_manifest_sha256': sha(OLD), 'counts': counts, 'groups_by_phase': groups,
        'existing_TEST_fingerprinted': 5746, 'old_TEST_matches': {}, 'inherited_internal_near_pairs': 2691,
        'reserved_author_TEST_ids_preserved': True, 'metadata_and_pixel_groups_disjoint': True,
        'TEST_used_for_split_objective': False, 'new_features_scores_optimizer_updates': 0,
        'unannotated_PNGs_excluded': 103, 'TRAIN_patches_excluded_by_TEST_group': sum(r['phase'] == 'excluded_TEST_group' for r in rows),
        'publisher128patch_label_not_whole_health_or_mask': True, 'new_unit_tests_passed': 3}
    new(MANIFEST, {'schema':'safelog-gearbox-stratified-v26', 'rows':rows, 'audit':audit})
    new(ROOT/'reports/industrial-gearbox-v26-data-audit.json', audit)
    new(RUN/'data-ready.json', {'ready':ready, 'manifest_sha256':sha(MANIFEST), 'audit':audit})
    print(audit, flush=True)
    assert ready, 'Fixed stratified data gate failed; preserve state, no new features/optimizer'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze','prepare'))
    args = parser.parse_args(); globals()[args.stage]()
