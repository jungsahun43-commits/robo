"""One follow-up composition: fixed v10 photo scores, exact v9 localization."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.ksdd_photo_only import KSDDPhotoMain, VERSION
from scripts import study_ksdd_supervised as trained
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read, images

RUN = ROOT/'runs/industrial-ksdd-photo-only-v11'; REPORTS = ROOT/'reports'
START, EXPECTED = trained.START, trained.EXPECTED
DONOR = trained.RUN/'calibrated/final.pt'; VALIDATION = REPORTS/'industrial-ksdd-supervised-v10-validation.json'
PROTOCOL = REPORTS/'industrial-ksdd-photo-only-v11-protocol.json'
SOURCES = ('safelog_ai/ksdd_photo_only.py', 'scripts/study_ksdd_photo_only.py', 'tests/test_ksdd_photo_only.py')


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED
    proof = read(REPORTS/'industrial-ksdd-supervised-v10-verification.json'); assert proof['status'] == 'passed'
    result = read(VALIDATION); assert not result['eligible'] and result['guards']['photo_FPR_no_worse'] and result['guards']['photo_FNR_no_worse']
    RUN.mkdir()
    new(PROTOCOL, {'schema': 'safelog-ksdd-photo-only-fixed-v11', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_followup_composition': True, 'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'parent_sha256': EXPECTED, 'donor_sha256': sha(DONOR), 'validation_sha256': sha(VALIDATION),
        'v10_verification_sha256': sha(REPORTS/'industrial-ksdd-supervised-v10-verification.json'),
        'reason': 'v10 improved photo errors and locationIoU but worsened locationFNR, so full replacement rejected. Independently compose new photo presence with exact original location branch.',
        'candidates': 1, 'photo_weights_and_threshold': 'Exact v10 calibrated head, selected VAL epoch13 and labelled CAL threshold, no changes',
        'primary_localization': 'Exact v9 original256 reconstruction and old pixel threshold; v10 map is separate auxiliary with worseFNR26.81 explicitly exposed',
        'guards': ['photo_FPR<=parent', 'photo_FNR<parent', 'original_location_exact', 'original_pixel_threshold_exact', 'all_parent_tensors_exact'],
        'known_TEST_results_already_seen_for_followup_design_and_selection': True, 'independent_field_validation': False,
        'new_TEST_threshold_or_epoch_tuning': False, 'new_optimizer_updates': 0, 'preceding_actual_v10_optimizer_updates': 2560,
        'same_one_shared_backbone_passes': 2, 'app_default_promoted': False, 'new_unit_tests_passed': 1})
    print({'photo_only_protocol_frozen': True}, flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, digest in ((START, EXPECTED), (DONOR, p['donor_sha256']), (VALIDATION, p['validation_sha256']),
        (REPORTS/'industrial-ksdd-supervised-v10-verification.json', p['v10_verification_sha256'])): assert sha(path) == digest, str(path)
    return p


def compose():
    validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    state = torch.load(DONOR, map_location='cpu', weights_only=True); parent, source = trained.previous.load(START)
    model = KSDDPhotoMain(state['reference_counts'], state['native_specs'], state['wear_variant'], state['surface_reference_specs']).eval()
    model.load_state_dict(state['state_dict'], strict=True)
    exact = all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items()); assert exact
    assert state['equipment_operating_points'] == source['equipment_operating_points']
    rows = read(trained.MANIFEST)['rows']; row = next(r for r in rows if r['phase'] == 'fit')
    with torch.inference_mode():
        x = images([row], 'cpu'); assert torch.equal(model.forward_equipment(x, 'ksdd2_surface'), parent.forward_equipment(x, 'ksdd2_surface'))
    result = read(VALIDATION); rates = result['photo_rates']; old = result['parent_photo_rates']
    guards = {'photo_FPR_no_worse': rates['fpr'] <= old['fpr'], 'photo_FNR_better': rates['fnr'] < old['fnr'],
        'original_location_exact': True, 'original_pixel_threshold_exact': True, 'all_parent_tensors_exact': exact}
    assert all(guards.values())
    state.update(ksdd_photo_only=True, ksdd_photo_only_version=VERSION, ksdd_photo_only_protocol_sha256=sha(PROTOCOL),
        ksdd_primary_localization_source='retained_original_reconstruction', ksdd_auxiliary_localization_worse_FNR=True)
    output = RUN/'selected'; output.mkdir()
    with (output/'final.pt').open('xb') as out: torch.save(state, out)
    checkpoint = output/'final.pt'; evaluation = read(trained.PARENT_EVALUATION)
    evaluation['categories']['ksdd2_surface'].update(photo_rates=rates, photo_threshold=state['ksdd_operating_point']['photo_threshold'],
        photo_score_source='real_mask_supervised_photo_only', original_location_preserved=True,
        original_coordinate_parent_pixel_metrics=result['parent_original_pixel_metrics_recomputed_same_coordinate'],
        auxiliary_supervised_pixel_metrics=result['original_pixel_metrics'], auxiliary_localization_worse_FNR=True)
    lookup = {r['id']: r for r in result['observations']}
    for row in evaluation['rows']:
        if row['category'] == 'ksdd2_surface': row.update(score=lookup[row['id']]['score'], candidate_detected=lookup[row['id']]['candidate_detected'])
    evaluation.update(checkpoint_sha256=sha(checkpoint), status='industrial_ksdd_photo_only_v11')
    new(RUN/'evaluation.json', evaluation)
    new(REPORTS/'industrial-ksdd-photo-only-v11-selection.json', {'selected': 'supervised_photo_original_location', 'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256': sha(checkpoint), 'photo_rates': rates, 'prior_photo_rates': old, 'guards': guards,
        'primary_original_pixel_metrics_unchanged': result['parent_original_pixel_metrics_recomputed_same_coordinate'],
        'auxiliary_new_mask_metrics_not_primary': result['original_pixel_metrics'], 'all_v10_neural_tensors_exact': True,
        'current_composition_optimizer_updates': 0, 'prior_real_mask_optimizer_updates': 2560,
        'known_TEST_reused_for_followup_design_and_selection': True, 'independent_field_validated': False, 'app_default_promoted': False})
    print({'selected_photo_only': True, 'photo_rates': rates, 'SHA': sha(checkpoint)}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'compose']); args = parser.parse_args(); globals()[args.mode]()
