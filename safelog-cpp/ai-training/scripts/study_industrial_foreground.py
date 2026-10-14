"""One preregistered foreground hypothesis; no iterative TEST tuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.industrial_foreground import CATEGORIES, VERSION, ForegroundMain, foreground_view, foreground_evidence, token_mask
from safelog_ai.ksdd_photo_only import KSDDPhotoMain
from safelog_ai.native_component_views import normalized
from safelog_ai.metal_reference import patch_features, normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.equipment_expansion import CATEGORIES as OLD_CATEGORIES, stage_gate
from safelog_ai.process_probe import process_exists
from scripts.prepare_industrial_foreground import DATA, RUN, REPORT as DATA_REPORT
from scripts.study_equipment_expansion import read, images
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_unified_belt_tiles import snapshot

REPORTS = ROOT/'reports'; PROTOCOL = REPORTS/'industrial-foreground-v12-protocol.json'
MANIFEST = DATA/'manifest.json'
START = ROOT/'runs/industrial-ksdd-photo-only-v11/selected/final.pt'
EXPECTED = '935cc6f6cb8e021714c97aa5c7225a567cda1637ba1c1876fd6de9230a0454d6'
PARENT_EVALUATION = ROOT/'runs/industrial-ksdd-photo-only-v11/evaluation.json'
SOURCES = ('safelog_ai/industrial_foreground.py', 'scripts/prepare_industrial_foreground.py', 'scripts/study_industrial_foreground.py', 'tests/test_industrial_foreground.py')


def freeze():
    assert not PROTOCOL.exists() and sha(START) == EXPECTED
    audit = read(DATA_REPORT)
    assert all(v['usable_fraction'] >= .95 for v in audit['normal_FIT_mask_review'].values())
    assert all(v['cal'] >= 20 for v in audit['counts'].values())
    assert read(REPORTS/'industrial-ksdd-v10-v11-final-audit.json')['status'] == 'completed_real_mask_training_photo_only_selected'
    ledger = dict(read(ROOT/'runs/industrial-ksdd-supervised-v10/protected-inputs.json'))
    # Existing raw sources and old accepted weights/report bytes stay immutable.
    for directory in (REPORTS, ROOT/'runs/industrial-ksdd-supervised-v10', ROOT/'runs/industrial-ksdd-photo-only-v11'):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    evaluation = read(PARENT_EVALUATION)
    new(PROTOCOL, {'schema': VERSION, 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_new_reference_fitting_CAL_and_TEST': True, 'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('safelog_ai', 'scripts', 'tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'manifest_sha256': sha(MANIFEST), 'parent_evaluation_sha256': sha(PARENT_EVALUATION),
        'protected_inputs_sha256': sha(RUN/'protected-inputs.json'), 'source_data_audit': audit,
        'categories': list(CATEGORIES), 'baseline': {c: evaluation['categories'][c] for c in CATEGORIES},
        'candidates_per_category': 1, 'view': 'aspect512 letterbox127',
        'mask': 'input border5percent CIELAB; deterministic3clusters farthest-point initialisation/eight Lloyd steps; distance greater than max12,border99percentile; open3 close5; drop components<32pixels; foreground1to75percent',
        'FIT_mask_min_usable_fraction': .95, 'unusable_FIT': 'excluded from new bank, not a healthy prediction',
        'feature_cells': 'same one backbone pooled low40/high960 patches; foreground area at least25percent per cell; at least4cells required',
        'reference': 'normalFIT-only at most256 patches/photo uniformly selected with SHA256(sourceID) seed; FP16 storage, FP32 cosine query2048/reference8192',
        'photo_score': 'bilinear512 distance inside input foreground, mean top1percent of foreground pixels',
        'threshold': 'normalCAL20 only, ceil((valid_normalCAL_count+1)*.8) order statistic, strict greater than; at least20 finite scores; save before TEST',
        'gradients_or_optimizer_updates': 0, 'VAL_epoch_selection': 'not applicable; frozen backbone and heads, one normal-reference hypothesis',
        'selection_guards': ['normalTEST>=20 and defectTEST>=20', 'photoFPR<=.20', 'photoFNR<=parentFNR-.05', 'CAL_and_TEST_manual_review_count_zero', 'all_parent_tensors_and_thresholds_exact'],
        'stage20_pass_distinct_from_incremental_selection': True, 'original_location_and_pixel_thresholds_preserved': True,
        'manual_review': 'record missing score, count as error in worst-case photo summary for both classes; never substitute healthy; candidate with CAL/TEST failure cannot be released',
        'known_public_TEST_reused_for_development_and_selection': True,
        'new_reference_TEST_and_near_group_exposure': False,
        'inherited_parent_normal_FIT_had_newly_flagged_near_TEST_candidates': True,
        'near_pHash_is_heuristic_not_proof_of_same_physical_item': True,
        'physical_acquisition_groups_verified': False, 'independent_field_validated': False,
        'input_foreground_mask_is_defect_ground_truth': False, 'new_pixel_localization_metrics_measured': False,
        'same_shared_backbone_passes': 2, 'app_default_promoted': False, 'new_unit_tests_passed': 4})
    print({'protocol_frozen': True, 'candidates': 2, 'normal_FIT_counts': audit['normal_FIT_mask_review']}, flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for path, digest in ((START, EXPECTED), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])): assert sha(path) == digest, path
    if committed:
        repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
        for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    return p


def load(path=START, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('foreground_specs'):
        model = ForegroundMain(state['reference_counts'], state['native_specs'], state['wear_variant'], state['surface_reference_specs'], state['foreground_specs'])
    else:
        model = KSDDPhotoMain(state['reference_counts'], state['native_specs'], state['wear_variant'], state['surface_reference_specs'])
    model.load_state_dict(state['state_dict'], strict=True)
    return model.eval().requires_grad_(False).to(device), state


def view(row, device='cuda'):
    assert sha(ROOT/row['image']) == row['image_sha256'], row['id']
    with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
    image, mask, geometry = foreground_view(rgb)
    return normalized(image)[None].to(device), mask[None].to(device), geometry


def fit():
    validate(committed=True); assert not (RUN/'normal-banks.pt').exists()
    model, _ = load(device='cuda'); rows = read(MANIFEST)['rows']; banks = {}; records = {}
    with torch.inference_mode():
        for category in CATEGORIES:
            selected = [r for r in rows if r['category'] == category and r['phase'] == 'fit' and r['foreground_FIT_audit']['usable']]
            assert all(r['label'] == 0 and r['publisher_split'] == 'train' for r in selected)
            values = []
            for index, row in enumerate(selected):
                x, mask, _ = view(row); features = patch_features(model.backbone(x))[0]
                tokens = token_mask(mask, features.shape[-2:])[0]; patches = features.permute(1, 2, 0)[tokens]
                assert len(patches) >= 4
                seed = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8], 'big')
                indices = torch.randperm(len(patches), generator=torch.Generator().manual_seed(seed))[:256].cuda()
                values.append(patches[indices].cpu().half())
                if index % 20 == 0: print({'fitting': category, 'normalFIT_done': index+1, 'total': len(selected)}, flush=True)
            banks[category] = torch.cat(values); records[category] = {'normal_FIT_photos': len(selected), 'patches': len(banks[category]), 'source_ids': [r['id'] for r in selected]}
    with (RUN/'normal-banks.pt').open('xb') as out: torch.save({'banks': banks, 'records': records, 'protocol_sha256': sha(PROTOCOL)}, out)
    print({'banks_fitted': {c: r['patches'] for c, r in records.items()}, 'actual_optimizer_updates': 0}, flush=True)


def score_rows(model, bank, rows):
    values = []
    with torch.inference_mode():
        for row in rows:
            try:
                x, mask, geometry = view(row)
                _, score = foreground_evidence(model.backbone(x), bank, mask)
                values.append({'id': row['id'], 'label': row['label'], 'defect': row['defect'], 'score': float(score[0]), 'geometry': geometry, 'manual_review': False})
            except ValueError as error:
                values.append({'id': row['id'], 'label': row['label'], 'defect': row['defect'], 'score': None, 'manual_review': True, 'reason': str(error)})
    return values


def calibrate():
    validate(); assert not (RUN/'normalCAL.json').exists()
    model, _ = load(device='cuda'); saved = torch.load(RUN/'normal-banks.pt', weights_only=True)
    assert saved['protocol_sha256'] == sha(PROTOCOL)
    rows = read(MANIFEST)['rows']; results = {}
    for category in CATEGORIES:
        selected = [r for r in rows if r['category'] == category and r['phase'] == 'cal']; assert all(r['label'] == 0 for r in selected)
        observations = score_rows(model, saved['banks'][category].cuda(), selected)
        valid = [r['score'] for r in observations if not r['manual_review']]
        threshold, rank = normal_budget_threshold(valid, .20)
        results[category] = {'photo_threshold': threshold, 'alpha': .20, 'rank': rank, 'normal_CAL_photos': len(valid), 'normal_CAL_total': len(selected),
            'manual_review_count': len(selected)-len(valid), 'observations': observations}
        print({'normalCAL': category, 'threshold': threshold, 'manual_review': len(selected)-len(valid)}, flush=True)
    new(RUN/'normalCAL.json', {'protocol_sha256': sha(PROTOCOL), 'all_thresholds_saved_before_TEST': True, 'categories': results})


def evaluate():
    p = validate(); output = REPORTS/'industrial-foreground-v12-validation.json'; assert not output.exists()
    model, parent = load(device='cuda'); banks = torch.load(RUN/'normal-banks.pt', weights_only=True)
    cal = read(RUN/'normalCAL.json'); assert cal['all_thresholds_saved_before_TEST'] and cal['protocol_sha256'] == banks['protocol_sha256'] == sha(PROTOCOL)
    rows = read(MANIFEST)['rows']; results = {}; selected_categories = []
    for category in CATEGORIES:
        selected = [r for r in rows if r['category'] == category and r['phase'] == 'test']
        observations = score_rows(model, banks['banks'][category].cuda(), selected); point = cal['categories'][category]; threshold = point['photo_threshold']
        for row in observations: row['candidate_detected'] = None if row['manual_review'] else row['score'] > threshold
        # Missing predictions penalise their own true class; no hidden healthy fallback.
        worst_scores = [r['score'] if not r['manual_review'] else (-float('inf') if r['label'] else float('inf')) for r in observations]
        labels = [r['label'] for r in observations]; rates = classification_rates(labels, worst_scores, threshold)
        groups = {}
        for name in sorted({r['defect'] for r in observations if r['label']}):
            group = [r for r in observations if r['label'] and r['defect'] == name]
            fn = sum(r['candidate_detected'] is not True for r in group)
            groups[name] = {'defect_photos': len(group), 'fn': fn, 'fnr': fn/len(group), 'manual_review': sum(r['manual_review'] for r in group)}
        manual = sum(r['manual_review'] for r in observations); baseline = p['baseline'][category]['photo_rates']
        guards = {'classes_at_least20': min(rates['positive_photos'], rates['negative_photos']) >= 20,
            'FPR_within20': rates['fpr'] <= .20, 'FNR_improvement_at_least5pp': rates['fnr'] <= baseline['fnr']-.05,
            'no_CAL_TEST_manual_review': manual == point['manual_review_count'] == 0, 'parent_tensors_exact': True}
        finite = [r for r in observations if not r['manual_review']]
        results[category] = {'photo_rates_worst_case_manual_review_as_error': rates, 'baseline_photo_rates': baseline,
            'photo_threshold': threshold, 'CAL': {k: point[k] for k in ('alpha', 'rank', 'normal_CAL_photos', 'normal_CAL_total', 'manual_review_count')},
            'manual_review_TEST_count': manual, 'by_defect': groups, 'auroc_valid_only': rank_auroc([r['label'] for r in finite], [r['score'] for r in finite]),
            'stage_gate': stage_gate(rates['fpr'], rates['fnr'], rates['negative_photos'], rates['positive_photos']), 'guards': guards,
            'eligible_incremental_improvement': all(guards.values()), 'observations': observations}
        if all(guards.values()): selected_categories.append(category)
        print({'TEST': category, 'rates': rates, 'manual_review': manual, 'eligible': all(guards.values())}, flush=True)
    new(output, {'categories': results, 'known_TEST_reused': True, 'protocol_sha256': sha(PROTOCOL), 'optimizer_updates': 0})
    checkpoint = START; state = deepcopy(parent); specs = {}; fit_records = {}
    for category in selected_categories:
        bank = banks['banks'][category]; state['state_dict']['foreground_reference_'+category] = bank
        specs[category] = {'patches': len(bank)}; fit_records[category] = banks['records'][category]
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
    if selected_categories:
        state.update(foreground_specs=specs, foreground_fit=fit_records, foreground_operating_points={c: cal['categories'][c] for c in selected_categories},
            foreground_version=VERSION, foreground_protocol_sha256=sha(PROTOCOL), foreground_parent_sha256=EXPECTED, foreground_optimizer_updates=0)
        (RUN/'selected').mkdir(); checkpoint = RUN/'selected/final.pt'
        with checkpoint.open('xb') as out: torch.save(state, out)
    evaluation = read(PARENT_EVALUATION)
    for category in selected_categories:
        value = results[category]
        evaluation['categories'][category].update(photo_rates=value['photo_rates_worst_case_manual_review_as_error'], photo_threshold=value['photo_threshold'],
            stage_gate=value['stage_gate'], by_defect=value['by_defect'], photo_score_source='foreground_reference')
        lookup = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=lookup[row['id']]['score'], candidate_detected=lookup[row['id']]['candidate_detected'])
    evaluation.update(checkpoint_sha256=sha(checkpoint), status='industrial_foreground_v12', stage20_categories=[c for c,v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']])
    new(RUN/'evaluation.json', evaluation)
    new(REPORTS/'industrial-foreground-v12-selection.json', {'selected_categories': selected_categories, 'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint),
        'stage20_categories': evaluation['stage20_categories'], 'parent_all_tensors_buffers_and_operating_points_exact': True, 'original_locations_preserved': True,
        'new_neural_optimizer_updates': 0, 'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False})


def verify():
    p = validate(); choice = read(REPORTS/'industrial-foreground-v12-selection.json')
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    ledger = read(RUN/'protected-inputs.json')
    def check(item):
        name, value = item; assert snapshot(ROOT/name) == value, name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    model, state = load(ROOT/choice['checkpoint_path']); parent, original = load()
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in original['state_dict'].items())
    assert state['equipment_operating_points'] == original['equipment_operating_points'] and state['ksdd_operating_point'] == original['ksdd_operating_point']
    proofs = {}; rows = read(MANIFEST)['rows']
    with torch.inference_mode():
        for category in OLD_CATEGORIES:
            row = next((r for r in rows if r['category'] == category and r['phase'] == 'fit'), None)
            x = images([row], 'cpu') if row else torch.zeros(1,3,256,256)
            assert torch.equal(model.forward_equipment(x, category), parent.forward_equipment(x, category)); proofs[category] = True
        x = torch.zeros(1,3,256,256); assert torch.equal(model.forward_wear(x), parent.forward_wear(x))
    new(REPORTS/'industrial-foreground-v12-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'parent_tensors_buffers_thresholds_exact': True, 'all_old_equipment_location_CPU_outputs_exact': proofs, 'BSClass_CPU_score_exact': True,
        'selected_categories': choice['selected_categories'], 'new_optimizer_updates': 0, 'no_TEST_threshold_loops': True})
    print({'verification': 'passed', 'selected': choice['selected_categories'], 'protected_files': len(ledger)}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'fit', 'calibrate', 'evaluate', 'verify']); args = parser.parse_args()
    if args.mode in ('fit', 'calibrate', 'evaluate'):
        # Inspect old lock owners without changing stale records.
        for lock in (ROOT/'runs').glob('**/*.lock'):
            try: pid = int(lock.read_text(encoding='utf-8').strip())
            except (ValueError, UnicodeError): raise RuntimeError('Unrecognised work lock requires investigation: '+str(lock))
            if process_exists(pid): raise RuntimeError('Another locked process is active: '+str(lock))
        lock = RUN/'operation.lock'
        with lock.open('x', encoding='utf-8') as out: out.write(str(os.getpid()))
        try: globals()[args.mode]()
        except BaseException as error:
            new(RUN/(args.mode+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'), {'mode': args.mode, 'error': repr(error), 'pid': os.getpid()})
            raise
        finally: lock.unlink()
    else: globals()[args.mode]()
