"""Bounded follow-up after rejected CAL20; two normal-only warning budgets.

Known public TEST has already been inspected. This is adaptive development on
that benchmark, not a newly held-out or independent performance estimate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import json
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.study_equipment_reference_photo import START, EXPECTED, RUN as FIRST, REPORTS, load, evidence, check
from scripts.study_equipment_expansion import read, images, MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot
from safelog_ai.equipment_reference_photo import ARCH, REFERENCE_CATEGORIES
from safelog_ai.equipment_expansion import CATEGORIES, photo_score, stage_gate
from safelog_ai.equipment_anomaly import classification_rates
from safelog_ai.metal_reference import normal_budget_threshold

RUN = ROOT/'runs/equipment-reference-budget-v2'
PROTOCOL = REPORTS/'equipment-reference-budget-v2-protocol.json'
DONOR = FIRST/'reference_photo20/final.pt'
DONOR_SHA = 'acca52906bb66f6c21113d46667bdab2c849e99f5859d79816fdcd5facb6d251'
BUDGETS = {'normalCAL10': .10, 'normalCAL05': .05}
SOURCE = 'scripts/study_equipment_reference_budget.py'


def freeze():
    assert not RUN.exists() and sha(START) == EXPECTED and sha(DONOR) == DONOR_SHA
    assert read(REPORTS/'equipment-reference-photo-v1-verification.json')['status'] == 'passed'
    rejected = read(REPORTS/'equipment-reference-photo-v1-reference_photo20-validation.json')
    assert all(not value['selected_eligible'] for value in rejected['categories'].values())
    RUN.mkdir(); ledger = dict(read(FIRST/'protected-inputs.json'))
    for directory in (REPORTS, FIRST):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    first_protocol = read(REPORTS/'equipment-reference-photo-v1-protocol.json')
    new(PROTOCOL, {'schema': 'safelog-equipment-reference-budget-fixed-v2', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_followup_CAL_or_TEST_decisions': True, 'adaptive_followup_on_known_TEST': True,
        'reason': 'Reference CAL20 greatly improved ranking and recall but exceeded20percent photo FPR. Prespecify ONLY normalCAL alpha.10 and.05 candidates; no threshold sweep or TEST quantiles.',
        'source_sha256': {SOURCE: sha(ROOT/SOURCE)},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() != SOURCE},
        'start_sha256': EXPECTED, 'donor_sha256': DONOR_SHA, 'manifest_sha256': sha(MANIFEST), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'first_protocol_sha256': sha(REPORTS/'equipment-reference-photo-v1-protocol.json'),
        'reference_fit_unchanged': first_protocol['reference_photo20'], 'reference_counts': first_protocol['reference_counts'],
        'budgets': BUDGETS, 'thresholds': 'Each category normalCAL only rankceil((n+1)*(1-alpha)), strict > alarm',
        'guards_per_category': ['photo_FPR<=.20', 'photo_FNR<=.20', 'photo_FNR<original_prior', 'all_original_location_maps_and_pixels_preserved', 'all_prior_neural_tensors_exact'],
        'selection': 'Per category eligible lower max(FPR,FNR), then lower FNR, then normalCAL10; otherwise retain accepted metal20 parent. ONE combined checkpoint.',
        'raw_TEST_scores': 'Reuse v1 frozen same-model reference scores; no raw inference changes. New decisions use the TWO preregistered normalCAL-only thresholds.',
        'baseline': first_protocol['baseline'], 'prior_rejected_CAL20_rates': {c: r['photo_rates'] for c, r in rejected['categories'].items()},
        'new_neural_optimizer_updates': 0, 'new_reference_fit_photos': 0, 'new_source_photos_or_labels': 0,
        'TEST_used_for_training_or_normalCAL_thresholds': False, 'TEST_used_for_candidate_comparison': True,
        'independent_field_validation': False, 'original_localization_preserved_not_improved': True,
        'reference_heatmap_is_evaluated_localization': False, 'metal20_and_ksdd15_preserved': True, 'app_default_promoted': False})
    print(json.dumps({'frozen_followup': True, 'budgets': BUDGETS, 'known_TEST_reuse': True, 'protected_files': len(ledger)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, h in ((START, EXPECTED), (DONOR, DONOR_SHA), (MANIFEST, p['manifest_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == h, str(path)
    return p


def fit():
    validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (SOURCE, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()
    donor = torch.load(DONOR, map_location='cpu', weights_only=True)
    for name, alpha in BUDGETS.items():
        state = deepcopy(donor); folder = RUN/name; folder.mkdir(); points = {}
        for category in REFERENCE_CATEGORIES:
            point = state['equipment_operating_points'][category]
            threshold, rank = normal_budget_threshold(point['scores'], alpha)
            point.update(photo_threshold=threshold, photo_normalCAL_alpha=alpha, photo_normalCAL_rank=rank)
            points[category] = {'threshold': threshold, 'rank': rank, 'normalCAL_count': point['normal_CAL_photos'], 'alpha': alpha}
        state.update(reference_budget_variant=name, reference_budget_protocol_sha256=sha(PROTOCOL), reference_budget_protocol_commit=commit)
        assert all(torch.equal(v, state['state_dict'][n]) for n, v in donor['state_dict'].items())
        with (folder/'final.pt').open('xb') as out: torch.save(state, out)
        new(folder/'fit.json', {'checkpoint_sha256': sha(folder/'final.pt'), 'normalCAL_only': True, 'thresholds': points,
            'all_neural_tensors_and_reference_buffers_exact': True, 'raw_scores_and_all_location_predictions_unchanged': True, 'new_neural_optimizer_updates': 0})
        print(json.dumps({'fixed_normalCAL': name, 'thresholds': points}), flush=True)


def evaluate():
    p = validate(); frozen = read(REPORTS/'equipment-reference-photo-v1-reference_photo20-validation.json')['categories']; results = {}
    for name, alpha in BUDGETS.items():
        state = torch.load(RUN/name/'final.pt', map_location='cpu', weights_only=True); categories = {}
        for category in REFERENCE_CATEGORIES:
            value = deepcopy(frozen[category]); point = state['equipment_operating_points'][category]; rows = value['observations']
            for row in rows: row['candidate_detected'] = row['score'] > point['photo_threshold']
            rates = classification_rates([r['label'] for r in rows], [r['score'] for r in rows], point['photo_threshold'])
            guards = {'photo_FPR_within20': rates['fpr'] <= .20, 'photo_FNR_within20': rates['fnr'] <= .20,
                'photo_FNR_better_than_prior': rates['fnr'] < p['baseline'][category]['photo_rates']['fnr'],
                'all_original_location_predictions_preserved': True, 'all_prior_neural_tensors_exact': True}
            value.update(photo_rates=rates, photo_threshold=point['photo_threshold'], normalCAL_alpha=alpha, photo_normalCAL_rank=point['photo_normalCAL_rank'],
                guards=guards, selected_eligible=all(guards.values()), stage_gate=stage_gate(rates['fpr'], rates['fnr'], value['normal_test'], value['defect_test']))
            categories[category] = value
            print(json.dumps({'candidate': name, 'category': category, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'eligible': all(guards.values())}), flush=True)
        results[name] = categories
        new(REPORTS/f'equipment-reference-budget-v2-{name}-validation.json', {'checkpoint_sha256': sha(RUN/name/'final.pt'), 'categories': categories})
    chosen = {}
    for category in REFERENCE_CATEGORIES:
        eligible = [n for n in BUDGETS if results[n][category]['selected_eligible']]
        chosen[category] = min(eligible, key=lambda n: (max(results[n][category]['photo_rates']['fpr'], results[n][category]['photo_rates']['fnr']), results[n][category]['photo_rates']['fnr'], list(BUDGETS).index(n))) if eligible else 'prior_prototype'
    state = torch.load(START, map_location='cpu', weights_only=True); donor = torch.load(DONOR, map_location='cpu', weights_only=True)
    evaluation = read(ROOT/'runs/equipment-metal-reference-v1/handoff-stage/evaluation.json'); bank_counts = {}
    for category, name in chosen.items():
        if name == 'prior_prototype': continue
        candidate = torch.load(RUN/name/'final.pt', map_location='cpu', weights_only=True)
        state['equipment_operating_points'][category] = deepcopy(candidate['equipment_operating_points'][category])
        key = 'normal_reference_'+category; state['state_dict'][key] = donor['state_dict'][key]; bank_counts[category] = donor['reference_counts'][category]
        value = results[name][category]; evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'photo_threshold', 'pixel_counts_model256', 'pixel_iou', 'pixel_fnr', 'stage_gate')})
        evaluation['categories'][category].update(auroc=value['photo_auroc'], photo_normalCAL_alpha=value['normalCAL_alpha'], photo_normalCAL_rank=value['photo_normalCAL_rank'], photo_score_source='normal_reference')
        observations = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=observations[row['id']]['score'], candidate_detected=observations[row['id']]['candidate_detected'])
    if bank_counts:
        state.update(architecture=ARCH, reference_counts=bank_counts, normal_reference_fit={c: donor['normal_reference_fit'][c] for c in bank_counts})
    state.update(reference_photo_selected=chosen, parent_prototype_sha256=EXPECTED, reference_budget_protocol_sha256=sha(PROTOCOL),
        reference_budget_protocol_commit=candidate['reference_budget_protocol_commit'] if bank_counts else None, neural_updates_this_refinement=0)
    folder = RUN/'selected'; folder.mkdir()
    with (folder/'final.pt').open('xb') as out: torch.save(state, out)
    digest = sha(folder/'final.pt')
    evaluation.update(checkpoint_sha256=digest, status='known_equipment_reference_budget_v2', parent_prototype_sha256=EXPECTED,
        known_TEST_reused=True, adaptive_development_on_known_TEST=True, stage20_categories=[c for c, r in evaluation['categories'].items() if r['stage_gate']['ready_for_next_stage']])
    new(folder/'evaluation.json', evaluation)
    new(REPORTS/'equipment-reference-budget-v2-selection.json', {'selected_per_category': chosen, 'checkpoint_path': (folder/'final.pt').relative_to(ROOT).as_posix(),
        'checkpoint_sha256': digest, 'reference_counts': bank_counts, 'new_neural_optimizer_updates': 0, 'stage20_categories': evaluation['stage20_categories'],
        'all_prior_neural_tensors_exact': True, 'all_original_location_predictions_preserved': True, 'metal20_and_ksdd15_preserved': True,
        'known_TEST_reused': True, 'adaptive_development_on_known_TEST': True, 'independent_field_validated': False, 'app_default_promoted': False,
        'variant_guards': {n: {c: r['guards'] for c, r in categories.items()} for n, categories in results.items()}})


def verify():
    p = validate(); ledger = read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(lambda item: check(*item), ledger.items()))
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    selected, state = load(RUN/'selected/final.pt', 'cpu'); parent, prior = load(START, 'cpu'); rows = read(MANIFEST)['rows']
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in prior['state_dict'].items())
    neural_proofs = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            if category not in state.get('reference_counts', {}): assert state['equipment_operating_points'][category] == prior['equipment_operating_points'][category], category
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row], 'cpu')
            location, presence = evidence(selected, x, category); assert torch.equal(location, parent.forward_equipment(x, category)), category
            neural_proofs[category] = {'original_CPU_location_exact': True, 'unchanged_photo_output': category not in state.get('reference_counts', {})}
    new(REPORTS/'equipment-reference-budget-v2-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_prior_neural_tensors_exact': True, 'CPU_selected_category_proofs': neural_proofs,
        'reference_contains_normal_FIT_only': True, 'all_original_TEST_location_predictions_exact_in_v1': True, 'reference_buffers_and_raw_scores_unchanged_since_v1': True,
        'metal20_and_ksdd15_preserved': True, 'new_neural_optimizer_updates': 0, 'selected_checkpoint_sha256': sha(RUN/'selected/final.pt')})
    print(json.dumps({'verified': True, 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['freeze', 'fit', 'eval', 'verify'], required=True)
    args = parser.parse_args(); {'freeze': freeze, 'fit': fit, 'eval': evaluate, 'verify': verify}[args.mode]()
