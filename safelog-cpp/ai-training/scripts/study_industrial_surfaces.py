"""Fixed normal-reference comparison across five remaining metal components."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.industrial_surface_reference import ARCH, SURFACES, METHODS, IndustrialSurfaceMain, surface_view, surface_evidence
from safelog_ai.native_component_views import normalized
from safelog_ai.metal_reference import patch_features, normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.equipment_expansion import CATEGORIES, stage_gate
from scripts import study_bsclass_supervised as previous
from scripts.study_equipment_expansion import read, images, MANIFEST
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_unified_belt_tiles import snapshot

RUN = ROOT/'runs/industrial-surfaces-v9'; REPORTS = ROOT/'reports'
START = previous.RUN/'selected/final.pt'
EXPECTED = 'eb083f9f4794a249998cf28b888da76df07b1ba75d139295240fddf2b7b003bd'
PARENT_EVALUATION = previous.RUN/'evaluation.json'
PROTOCOL = REPORTS/'industrial-surfaces-v9-protocol.json'
SOURCES = ('safelog_ai/industrial_surface_reference.py', 'scripts/study_industrial_surfaces.py', 'tests/test_industrial_surface_reference.py')
BUDGETS = {'normalCAL15': .15, 'normalCAL20': .20}


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START) == EXPECTED
    assert read(REPORTS/'industrial-bsclass-supervised-v8-verification.json')['status'] == 'passed'
    assert read(REPORTS/'industrial-bsclass-supervised-v8-selection.json')['selected'] == 'supervised_wear'
    RUN.mkdir(); ledger = dict(read(previous.RUN/'protected-inputs.json'))
    for directory in (REPORTS, previous.RUN):
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith(('.lock', '.log')): ledger[path.relative_to(ROOT).as_posix()] = snapshot(path)
    new(RUN/'protected-inputs.json', ledger)
    rows = read(MANIFEST)['rows']; evaluation = read(PARENT_EVALUATION)
    counts = {c: {phase: sum(r['category'] == c and r['phase'] == phase for r in rows) for phase in ('fit', 'cal', 'test')} for c in SURFACES}
    new(PROTOCOL, {'schema': 'safelog-industrial-surfaces-fixed-v9', 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'declared_before_new_banks_CAL_and_TEST_scores': True, 'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'tests', 'safelog_ai') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256': EXPECTED, 'manifest_sha256': sha(MANIFEST), 'parent_evaluation_sha256': sha(PARENT_EVALUATION), 'protected_inputs_sha256': sha(RUN/'protected-inputs.json'),
        'categories': list(SURFACES), 'counts': counts, 'baseline': {c: evaluation['categories'][c] for c in SURFACES},
        'methods': {'pooled256': 'same256 square input; ALL256 normalFIT pooled low40/high960 unit patches/photo',
            'aspect512': 'aspect-preserving512 letterbox127; same low40/high960 branch-normalized avgpool3 features; deterministic uniform256 normalFIT patches/photo seedSHA256(sourceID)'},
        'normal_FIT_only': True, 'no_CAL_or_TEST_in_reference_banks': True, 'neural_optimizer_updates': 0,
        'reference_storage': 'FP16, exact nearest cosine FP32 chunks2048/8192', 'photo_score': 'mean top1percent bilinear view evidence',
        'normalCAL_budgets': BUDGETS, 'thresholds': 'Only normalCAL order statistic ceil((n+1)*(1-alpha)), strict >; save all before TEST scoring',
        'guards': ['photoFPR<=.20', 'photoFNR<=.20', 'photoFNR<parent', 'normal_and_defect_TEST_each>=20', 'all_parent_tensors_and_buffers_exact'],
        'selection': 'Per-category eligible lower max(FPR,FNR), then FNR, then pooled256 before aspect512 then CAL15 before20; otherwise parent unchanged',
        'original256_location_branch_preserved': True, 'backbone_passes': 'pooled256 one; aspect512 two on the SAME shared encoder',
        'all_existing_accepted_five_and_new_BSClass_head_preserved': True, 'MPDD_connector_only14_defect_TEST_cannot_pass': True,
        'known_public_TEST_reused_for_development_and_selection': True, 'independent_field_validation': False, 'reference_heatmap_is_evaluated_segmentation': False,
        'app_default_promoted': False, 'new_unit_tests_passed': 3})
    print(json.dumps({'frozen': True, 'counts': counts, 'methods': list(METHODS), 'normal_FIT_photos': sum(v['fit'] for v in counts.values())}), flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, digest in ((START, EXPECTED), (MANIFEST, p['manifest_sha256']), (PARENT_EVALUATION, p['parent_evaluation_sha256']), (RUN/'protected-inputs.json', p['protected_inputs_sha256'])):
        assert sha(path) == digest, str(path)
    if committed:
        import subprocess
        repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
        for n in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+n])).hexdigest() == sha(ROOT/n), n
    return p


def batch(rows, method, device='cuda'):
    result = []
    for row in rows:
        assert sha(ROOT/row['image']) == row['image_sha256'], row['id']
        with Image.open(ROOT/row['image']) as im: rgb = im.convert('RGB')
        view, _ = surface_view(rgb, method); result.append(normalized(view))
    return torch.stack(result).to(device)


def fit():
    validate(committed=True); assert not (RUN/'normal-banks.pt').exists()
    model, _ = previous.load(START, 'cuda'); rows = read(MANIFEST)['rows']; banks = {}; records = {}
    with torch.inference_mode():
        for method in METHODS:
            banks[method] = {}; records[method] = {}
            for category in SURFACES:
                selected = [r for r in rows if r['category'] == category and r['phase'] == 'fit']; assert all(r['label'] == 0 for r in selected)
                values = []
                for start in range(0, len(selected), 8):
                    source = selected[start:start+8]; features = patch_features(model.backbone(batch(source, method)))
                    for row, value in zip(source, features):
                        patches = value.permute(1, 2, 0).reshape(-1, 1000)
                        if method == 'aspect512':
                            seed = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8], 'big')
                            indices = torch.randperm(len(patches), generator=torch.Generator().manual_seed(seed))[:256]
                            patches = patches[indices.cuda()]
                        assert len(patches) == 256; values.append(patches.cpu().half())
                banks[method][category] = torch.cat(values); records[method][category] = {'normal_FIT_photos': len(selected), 'patches': len(values)*256, 'source_ids': [r['id'] for r in selected]}
                print(json.dumps({'normal_reference_fit': category, 'method': method, 'photos': len(selected), 'patches': len(values)*256}), flush=True)
    with (RUN/'normal-banks.pt').open('xb') as out: torch.save({'banks': banks, 'records': records, 'protocol_sha256': sha(PROTOCOL)}, out)


def calibrate():
    validate(); assert not (RUN/'normalCAL.json').exists(); model, _ = previous.load(START, 'cuda')
    saved = torch.load(RUN/'normal-banks.pt', map_location='cpu', weights_only=True); assert saved['protocol_sha256'] == sha(PROTOCOL)
    rows = read(MANIFEST)['rows']; results = {}
    with torch.inference_mode():
        for category in SURFACES:
            selected = [r for r in rows if r['category'] == category and r['phase'] == 'cal']; assert all(r['label'] == 0 for r in selected); results[category] = {}
            for method in METHODS:
                bank = saved['banks'][method][category].cuda(); scores = []
                for start in range(0, len(selected), 4):
                    view = batch(selected[start:start+4], method); features = model.backbone(view)
                    _, score = surface_evidence(features, bank, view.shape[-2:]); scores.extend(score.cpu().tolist())
                points = {}
                for name, alpha in BUDGETS.items():
                    threshold, rank = normal_budget_threshold(scores, alpha); points[name] = {'photo_threshold': threshold, 'alpha': alpha, 'rank': rank, 'normal_CAL_photos': len(scores)}
                results[category][method] = {'scores': scores, 'points': points}
                print(json.dumps({'calibrated': category, 'method': method, 'normal_photos': len(scores)}), flush=True)
    new(RUN/'normalCAL.json', {'protocol_sha256': sha(PROTOCOL), 'all_thresholds_saved_before_TEST': True, 'categories': results})


def evaluate():
    p = validate(); assert not (REPORTS/'industrial-surfaces-v9-validation.json').exists(); model, parent = previous.load(START, 'cuda')
    saved = torch.load(RUN/'normal-banks.pt', map_location='cpu', weights_only=True); calibration = read(RUN/'normalCAL.json')
    assert saved['protocol_sha256'] == calibration['protocol_sha256'] == sha(PROTOCOL) and calibration['all_thresholds_saved_before_TEST']
    rows = read(MANIFEST)['rows']; results = {}; choices = {}
    with torch.inference_mode():
        for category in SURFACES:
            selected = [r for r in rows if r['category'] == category and r['phase'] == 'test']; labels = [r['label'] for r in selected]; results[category] = {}
            baseline = p['baseline'][category]['photo_rates']
            for method in METHODS:
                bank = saved['banks'][method][category].cuda(); scores = []
                for start in range(0, len(selected), 4):
                    view = batch(selected[start:start+4], method); _, score = surface_evidence(model.backbone(view), bank, view.shape[-2:]); scores.extend(score.cpu().tolist())
                for name, point in calibration['categories'][category][method]['points'].items():
                    threshold = point['photo_threshold']; rates = classification_rates(labels, scores, threshold)
                    guards = {'FPR_within20': rates['fpr'] <= .20, 'FNR_within20': rates['fnr'] <= .20, 'FNR_better_than_parent': rates['fnr'] < baseline['fnr'],
                        'each_class_at_least20': min(rates['negative_photos'], rates['positive_photos']) >= 20, 'parent_tensors_preserved': True}
                    groups = {}
                    for group in sorted({r['defect'] for r in selected if r['label']}):
                        indices = [i for i, r in enumerate(selected) if r['label'] and r['defect'] == group]; fn = sum(scores[i] <= threshold for i in indices)
                        groups[group] = {'defect_photos': len(indices), 'fn': fn, 'fnr': fn/len(indices)}
                    key = method+'/'+name
                    results[category][key] = {'method': method, 'calibration': name, **point, 'photo_rates': rates, 'auroc': rank_auroc(labels, scores), 'by_defect': groups,
                        'stage_gate': stage_gate(rates['fpr'], rates['fnr'], rates['negative_photos'], rates['positive_photos']), 'guards': guards, 'eligible': all(guards.values()),
                        'observations': [{'id': r['id'], 'label': r['label'], 'defect': r['defect'], 'score': s, 'candidate_detected': s > threshold} for r, s in zip(selected, scores)]}
                    print(json.dumps({'candidate': category+'/'+key, 'FPR': rates['fpr'], 'FNR': rates['fnr'], 'eligible': all(guards.values())}), flush=True)
            eligible = [key for key, value in results[category].items() if value['eligible']]
            if eligible:
                choices[category] = min(eligible, key=lambda key: (max(results[category][key]['photo_rates']['fpr'], results[category][key]['photo_rates']['fnr']),
                    results[category][key]['photo_rates']['fnr'], list(results[category]).index(key)))
    new(REPORTS/'industrial-surfaces-v9-validation.json', {'categories': results, 'known_TEST_reused': True})
    state = deepcopy(parent); specs = {}; fit_records = {}
    for category, key in choices.items():
        value = results[category][key]; method = value['method']; bank = saved['banks'][method][category]
        state['state_dict']['surface_reference_'+category] = bank
        specs[category] = {'method': method, 'patches': len(bank)}; fit_records[category] = saved['records'][method][category]
        state['equipment_operating_points'][category].update(photo_threshold=value['photo_threshold'], photo_score_method='industrial_surface_reference',
            photo_normalCAL_alpha=value['alpha'], photo_normalCAL_rank=value['rank'], surface_method=method)
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in parent['state_dict'].items())
    checkpoint = START
    if choices:
        state.update(architecture=ARCH, surface_reference_specs=specs, surface_reference_fit=fit_records, surface_protocol_sha256=sha(PROTOCOL),
            surface_parent_sha256=EXPECTED, surface_neural_updates=0, surface_known_TEST_reused=True)
        folder = RUN/'selected'; folder.mkdir(); checkpoint = folder/'final.pt'
        with checkpoint.open('xb') as out: torch.save(state, out)
    evaluation = read(PARENT_EVALUATION)
    for category, key in choices.items():
        value = results[category][key]; evaluation['categories'][category].update({k: value[k] for k in ('photo_rates', 'auroc', 'photo_threshold', 'stage_gate', 'by_defect')})
        evaluation['categories'][category].update(photo_score_source='industrial_surface_reference', method=value['method'])
        lookup = {r['id']: r for r in value['observations']}
        for row in evaluation['rows']:
            if row['category'] == category: row.update(score=lookup[row['id']]['score'], candidate_detected=lookup[row['id']]['candidate_detected'])
    evaluation.update(checkpoint_sha256=sha(checkpoint), stage20_categories=[c for c, v in evaluation['categories'].items() if v['stage_gate']['ready_for_next_stage']], status='industrial_surfaces_v9')
    new(RUN/'evaluation.json', evaluation)
    new(REPORTS/'industrial-surfaces-v9-selection.json', {'selected': choices, 'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': sha(checkpoint),
        'newly_selected_categories': list(choices), 'stage20_categories': evaluation['stage20_categories'], 'old_neural_tensors_and_buffers_exact': True,
        'new_normal_reference_FIT_photos_selected': sum(r['normal_FIT_photos'] for r in fit_records.values()), 'new_neural_optimizer_updates': 0,
        'BSClass_supervised_head_retained': True, 'known_TEST_reused_for_development': True, 'independent_field_validated': False, 'app_default_promoted': False})
    print(json.dumps({'selected': choices, 'stage20_categories': evaluation['stage20_categories'], 'checkpoint_sha256': sha(checkpoint)}), flush=True)


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['architecture'] != ARCH: return previous.load(path, device)
    model = IndustrialSurfaceMain(state['reference_counts'], state['native_specs'], state['wear_variant'], state['surface_reference_specs']).eval().requires_grad_(False)
    model.load_state_dict(state['state_dict'], strict=True); return model.to(device), state


def verify():
    p = validate(); choice = read(REPORTS/'industrial-surfaces-v9-selection.json')
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    ledger = read(RUN/'protected-inputs.json')
    def check(item):
        name, value = item; assert snapshot(ROOT/name) == value, name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    model, state = load(ROOT/choice['checkpoint_path']); parent, source = previous.load(START)
    assert all(torch.equal(v, state['state_dict'][n]) for n, v in source['state_dict'].items())
    proofs = {}; rows = read(MANIFEST)['rows']
    with torch.inference_mode():
        for category in CATEGORIES:
            row = next(r for r in rows if r['category'] == category and r['phase'] == 'fit'); x = images([row], 'cpu')
            if category in choice['selected']:
                method = state['surface_reference_specs'][category]['method']; view = batch([row], method, 'cpu')
                location, _, _ = model.forward_surface(x, view, category); assert torch.equal(location, parent.forward_equipment(x, category))
            else:
                a, b = model.forward_equipment_evidence(x, category), parent.forward_equipment_evidence(x, category)
                assert all(torch.equal(v, w) for v, w in zip(a, b)); assert state['equipment_operating_points'][category] == source['equipment_operating_points'][category]
            proofs[category] = True
        x = torch.zeros(1, 3, 256, 256); assert torch.equal(model.forward_wear(x), parent.forward_wear(x))
    new(REPORTS/'industrial-surfaces-v9-verification.json', {'status': 'passed', 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved': True, 'all_parent_neural_tensors_and_reference_buffers_exact': True, 'CPU_equipment_original_locations_or_unchanged_photo_evidence': proofs,
        'BSClass_wear_CPU_output_exact': True, 'newly_selected_categories': list(choice['selected']), 'new_optimizer_updates': 0})
    print(json.dumps({'verified': True, 'selected': choice['selected'], 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'fit', 'calibrate', 'evaluate', 'verify']); args = parser.parse_args(); globals()[args.mode]()
