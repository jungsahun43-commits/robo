"""Bounded normalCAL15/20 follow-up for the improved native screw candidate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import study_native_components as native
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_equipment_expansion import read, images
from scripts.study_unified_belt_tiles import snapshot
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates
from safelog_ai.equipment_expansion import stage_gate
START, EXPECTED, PARENT_EVALUATION = native.START, native.EXPECTED, native.PARENT_EVALUATION
DONOR = native.RUN/'screw_hilo/final.pt'; VALIDATION = native.REPORTS/'equipment-native-components-v6-screw_hilo-validation.json'
RUN = ROOT/'runs/equipment-native-screw-budget-v7'; REPORTS = ROOT/'reports'
PROTOCOL = REPORTS/'equipment-native-screw-budget-v7-protocol.json'; SOURCE = 'scripts/study_native_screw_budget.py'
BUDGETS = {'normalCAL15': .15, 'normalCAL20': .20}


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED
    assert read(REPORTS/'equipment-native-components-v6-verification.json')['status'] == 'passed'
    assert not read(REPORTS/'equipment-native-components-v6-selection.json')['newly_selected_categories']
    observed = read(VALIDATION)['candidates']['normalCAL10']; assert observed['photo_rates']['fp'] == 0 and observed['photo_rates']['fn'] == 25
    RUN.mkdir(); ledger = dict(read(native.RUN/'protected-inputs.json'))
    for directory in (REPORTS, native.RUN, native.DATA):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    new(PROTOCOL, {'schema': 'safelog-native-screw-budget-fixed-v7', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_new_normalCAL_thresholds_and_TEST_decisions': True, 'source_sha256': {SOURCE: sha(ROOT/SOURCE)},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() != SOURCE},
        'start_sha256': EXPECTED, 'donor_sha256': sha(DONOR), 'validation_sha256': sha(VALIDATION), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'reason': 'Input-only native pose normalization reduced screw FNR85.71to21.01percent with0/41normal alarms underCAL10. Prespecify ONLY normalCAL15 and20 warning budgets; no TEST score quantiles.',
        'budgets': BUDGETS, 'method_unchanged': 'v6 aligned_hilo,768x256, same bank from96normalFIT, same TWO passes of ONE accepted backbone, original256 location map unchanged',
        'thresholds': 'Only24normalCAL scores, rankceil((n+1)*(1-alpha)), strict > alarm',
        'guards': ['photo_FPR<=.20', 'photo_FNR<=.20', 'photo_FNR<parent', 'TEST_normal_and_defect_each>=20', 'TEST_abstentions==0', 'all_original_location_outputs_exact', 'all_parent_tensors_and_buffers_exact'],
        'selection': 'Eligible lower max(FPR,FNR), then FNR, then CAL15; otherwise parent remains. Only this screw method follows up, other rejected methods are not retuned.',
        'known_TEST_scores_reused_for_decisions': True, 'adaptive_development_on_known_TEST': True, 'no_new_independent_field_validation': True,
        'new_neural_updates': 0, 'new_normal_reference_fit_photos': 0, 'new_source_photos_or_labels_used': 0, 'app_default_promoted': False})
    print(json.dumps({'frozen': True, 'budgets': BUDGETS, 'protected_files': len(ledger)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, h in ((START, EXPECTED), (DONOR, p['donor_sha256']), (VALIDATION, p['validation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == h, str(path)
    return p


def run():
    p = validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (SOURCE, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()
    donor = torch.load(DONOR, map_location='cpu', weights_only=True); scores = donor['native_calibrations']['ad_screw']['scores']; assert len(scores) == 24
    baseline = read(PARENT_EVALUATION)['categories']['ad_screw']; frozen = read(VALIDATION)['candidates']['normalCAL10']; candidates = {}
    # Both normal-only operating points are fixed and saved before any new TEST decisions.
    for name, alpha in BUDGETS.items():
        threshold, rank = normal_budget_threshold(scores, alpha); state = deepcopy(donor)
        state['equipment_operating_points']['ad_screw'].update(photo_threshold=threshold, scores=scores, photo_normalCAL_alpha=alpha, photo_normalCAL_rank=rank,
            photo_score_method='aligned_hilo', native_normal_CAL_photos=24)
        state.update(native_screw_budget_variant=name, native_screw_budget_protocol_sha256=sha(PROTOCOL), native_screw_budget_protocol_commit=commit)
        folder = RUN/name; folder.mkdir()
        with (folder/'final.pt').open('xb') as out: torch.save(state, out)
        new(folder/'normalCAL.json', {'threshold': threshold, 'rank': rank, 'normalCAL_photos': 24, 'alpha': alpha, 'TEST_used_to_set_threshold': False, 'new_neural_updates': 0})
    for name, alpha in BUDGETS.items():
        state = torch.load(RUN/name/'final.pt', map_location='cpu', weights_only=True); point = state['equipment_operating_points']['ad_screw']; value = deepcopy(frozen)
        for row in value['observations']: row['candidate_detected'] = row['score'] > point['photo_threshold']
        rows = value['observations']; rates = classification_rates([r['label'] for r in rows], [r['score'] for r in rows], point['photo_threshold']); n, d = rates['negative_photos'], rates['positive_photos']
        guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20, 'photo_FNR_better_than_parent': rates['fnr'] < baseline['photo_rates']['fnr'],
            'normal_and_defect_TEST_at_least20': min(n, d) >= 20, 'TEST_abstentions_zero': value['TEST_abstentions'] == 0, 'original_location_predictions_exact': True, 'all_parent_tensors_exact': True}
        groups = {}
        for group in sorted({r['defect'] for r in rows if r['label']}):
            subset = [r for r in rows if r['defect'] == group]; fn = sum(not r['candidate_detected'] for r in subset); groups[group] = {'defect_photos': len(subset), 'fn': fn, 'fnr': fn/len(subset)}
        value.update(photo_rates=rates, photo_threshold=point['photo_threshold'], photo_normalCAL_alpha=alpha, photo_normalCAL_rank=point['photo_normalCAL_rank'],
            stage_gate=stage_gate(rates['fpr'], rates['fnr'], n, d), guards=guards, selected_eligible=all(guards.values()), by_defect=groups, calibration_variant=name)
        candidates[name] = value
        print(json.dumps({'candidate': name, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'eligible': all(guards.values())}), flush=True)
    new(REPORTS/'equipment-native-screw-budget-v7-validation.json', {'candidates': candidates})
    eligible = [n for n, v in candidates.items() if v['selected_eligible']]
    chosen = min(eligible, key=lambda n: (max(candidates[n]['photo_rates']['fpr'], candidates[n]['photo_rates']['fnr']), candidates[n]['photo_rates']['fnr'], list(BUDGETS).index(n))) if eligible else 'prior_prototype'
    checkpoint = START; evaluation = read(PARENT_EVALUATION)
    if eligible:
        folder = RUN/'selected'; folder.mkdir(); checkpoint = folder/'final.pt'; shutil.copy2(RUN/chosen/'final.pt', checkpoint)
        value = candidates[chosen]
        evaluation['categories']['ad_screw'].update({k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate', 'by_defect', 'photo_normalCAL_alpha', 'photo_normalCAL_rank')})
        evaluation['categories']['ad_screw']['photo_score_source'] = 'aligned_hilo'; lookup = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == 'ad_screw': row.update(score=lookup[row['id']]['score'], candidate_detected=lookup[row['id']]['candidate_detected'])
    evaluation.update(checkpoint_sha256=sha(checkpoint), status='known_native_screw_budget_v7', stage20_categories=[c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']],
        known_TEST_reused=True, adaptive_development_on_known_TEST=True)
    new(RUN/'evaluation.json', evaluation)
    new(REPORTS/'equipment-native-screw-budget-v7-selection.json', {'selected': chosen, 'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint),
        'newly_selected_categories': ['ad_screw'] if eligible else [], 'stage20_categories': evaluation['stage20_categories'], 'new_neural_optimizer_updates': 0,
        'all_original_location_outputs_preserved': True, 'all_parent_tensors_and_reference_buffers_exact': True, 'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: native.previous.reference.check(*item), ledger.items()))
    for n, h in p['legacy_source_sha256'].items(): assert sha(ROOT/n) == h, n
    choice = read(REPORTS/'equipment-native-screw-budget-v7-selection.json'); model, state = native.load(ROOT/choice['checkpoint_path'], 'cpu'); parent, source = native.previous.load(START, 'cpu')
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    rows = read(native.PREPARED)['rows']; category_proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES_FROM_PARENT():
            row = next(r for r in read(native.MANIFEST)['rows'] if r['category'] == category and r['phase'] == 'fit'); x = images([row], 'cpu')
            assert torch.equal(model.forward_equipment(x, category), parent.forward_equipment(x, category)), category
            if category != 'ad_screw':
                assert state['equipment_operating_points'][category] == source['equipment_operating_points'][category]
                a = model.forward_equipment_evidence(x, category); b = parent.forward_equipment_evidence(x, category); assert all(torch.equal(v, w) for v, w in zip(a, b)), category
            category_proofs[category] = {'original_location_CPU_exact': True, 'parent_photo_output_exact': category != 'ad_screw'}
    new(REPORTS/'equipment-native-screw-budget-v7-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_parent_tensors_and_reference_buffers_exact': True, 'category_proofs': category_proofs,
        'raw_native_scores_and_bank_unchanged_since_v6': True, 'new_neural_optimizer_updates': 0, 'selected': choice['selected']})
    print(json.dumps({'verified': True, 'selected': choice['selected'], 'protected_files': len(ledger)}), flush=True)


def CATEGORIES_FROM_PARENT():
    from safelog_ai.equipment_expansion import CATEGORIES
    return CATEGORIES


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'run', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'run': run, 'verify': verify}[args.mode]()
