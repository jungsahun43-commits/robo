"""TRAIN native uniform views and fixed one-model global/tile inference comparison."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.belt_tiles import uniform_boxes, raster_view, predict, policy, CONFIGS, select_tiles
from safelog_ai.belt_detail import configure_belt_only, non_belt_state, paired_flip
from safelog_ai.unified_main import ARCH, TASKS, UnifiedMain, freeze_bn_statistics, loss_for_task
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.belt_evaluation import summarize
from safelog_ai.belt_segmentation import pixel_counts
from scripts.unified_main_data import DATA, Cache, read, sha, normalize_batch
from scripts.prepare_beltcrack_starter import rasterize

BASE = ROOT/'runs/unified-belt-uniform-tiles-v1'
VIEWS = ROOT/'data/unified-belt-uniform-tiles-v1'
PROTOCOL = ROOT/'reports/unified-belt-tiles-protocol.json'
LEDGER = BASE/'protected-inputs.json'
START = ROOT/'runs/unified-belt-native-detail-v1/native_mix/final.pt'
START_SHA = '4b6d91c045cf6a33631ed5542c81d8e116166f541e97e6d745e1dfbe3f7fe5e5'
SOURCES = ('safelog_ai/belt_tiles.py', 'safelog_ai/unified_tiles_server.py', 'scripts/study_unified_belt_tiles.py', 'tests/test_belt_tiles.py')
EPOCHS, BATCH, SEED = 10, 8, 427


def new(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf-8', newline='\n') as out: json.dump(value, out, ensure_ascii=False, indent=2)


def snapshot(path):
    s = path.stat(); return {'sha256': sha(path), 'size_bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def native(row):
    assert row['split'] in ('train', 'val')
    path = ROOT/'data/equipment-beltcrack/expanded-cached-v2/original-annotations'/row['id']
    assert sha(path) == row['annotation_sha256']; annotation = read(path)
    raw = base64.b64decode(annotation['imageData'], validate=True)
    assert hashlib.sha256(raw).hexdigest() == row['embedded_image_sha256']
    with Image.open(io.BytesIO(raw)) as source:
        assert source.getexif().get(274) in (None, 1), 'Publisher encoded orientation needs explicit target audit'
        rgb = source.convert('RGB'); pixels = np.asarray(rgb)
        assert hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest() == row['decoded_rgb_sha256']
    assert rgb.size == (row['width'], row['height'])
    return rgb, annotation, path


def prepare():
    assert not BASE.exists() and not VIEWS.exists() and sha(START) == START_SHA
    source = read(DATA/'manifest.json')['fit']['belt_damage']; assert len(source) == 1397
    draws = np.load(ROOT/'runs/unified-belt-native-detail-v1/draws.npz')['belt_damage'].copy()
    assert draws.shape == (10, 1024)
    choices = np.random.default_rng(SEED+7).integers(0, 6, size=draws.shape)
    selected = sorted(set((int(draws[e, j]), int(choices[e, j])) for e in range(10) for j in range(0, 1024, 2)))
    allowed = sorted(set(draws.flatten().tolist())); by_source = {}
    for index, tile in selected: by_source.setdefault(index, []).append(tile)
    VIEWS.mkdir(); (VIEWS/'prepared').mkdir(); BASE.mkdir()
    def worker(index):
        row = source[index]; assert row['split'] == 'train'; rgb, annotation, path = native(row)
        for kind, actual in zip(('image', 'mask', 'valid'), rasterize(annotation, rgb)):
            with Image.open(ROOT/row[kind]) as previous: assert np.array_equal(np.asarray(actual), np.asarray(previous)), (row['id'], kind)
        boxes = uniform_boxes(*rgb.size); assert len(boxes) == 6
        results = []
        for tile in by_source.get(index, []):
            image, mask, valid = raster_view(rgb, annotation, boxes[tile])
            item = {'id': row['id']+f':uniform-view-{tile}', 'source_id': row['id'], 'source_index': index, 'view_index': tile,
                'split': 'train', 'task': 'belt_damage', 'sequence': row['sequence'], 'native_box': list(boxes[tile]),
                'annotation_path': path.relative_to(ROOT).as_posix(), 'annotation_sha256': row['annotation_sha256'],
                'source_embedded_image_sha256': row['embedded_image_sha256'], 'positive_pixels': int(np.count_nonzero(mask)),
                'valid_pixels': 640*640, 'whole_photo_normal_label': 'unknown'}
            for kind, value in [('image', image), ('mask', mask), ('valid', valid)]:
                dest = VIEWS/'prepared'/f'{index}-{tile}-{kind}.png'; value.save(dest, compress_level=1)
                item[kind] = dest.relative_to(ROOT).as_posix(); item[kind+'_sha256'] = sha(dest)
            results.append(item)
        return results
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for count, results in enumerate(pool.map(worker, allowed), 1):
            rows.extend(results)
            if count%100 == 0: print(json.dumps({'native_TRAIN_sources_verified': count, 'sources': len(allowed), 'scheduled_views_prepared': len(rows)}), flush=True)
    rows.sort(key=lambda r: (r['source_index'], r['view_index'])); lookup = {(r['source_index'], r['view_index']): i for i, r in enumerate(rows)}
    view_draws = np.full(draws.shape, -1, dtype=np.int64)
    for e in range(10):
        for j in range(0, 1024, 2): view_draws[e, j] = lookup[int(draws[e, j]), int(choices[e, j])]
    np.savez_compressed(BASE/'draws.npz', belt_damage=draws, uniform_views=view_draws)
    new(VIEWS/'manifest.json', {'status': 'frozen_TRAIN_only', 'rows': rows, 'source_TRAIN_photos_verified': len(allowed),
        'unique_source_photos_with_scheduled_views': len(by_source), 'derived_scheduled_views': len(rows),
        'empty_author_mask_views': sum(r['positive_pixels'] == 0 for r in rows), 'normal_whole_photo_labels_created': 0,
        'new_independent_photos': 0, 'new_labels': 0, 'source_VAL_or_TEST_used_for_training_views': False,
        'boxes_depend_on_annotations_or_predictions': False, 'draws_sha256': sha(BASE/'draws.npz'),
        'all_scheduled_source_RGB_mask_valid_match_previous_preparation': True,
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'license': 'BeltCrack CC BY-SA 4.0; native photos cropped/resized, original shapes rasterized'})
    print(json.dumps({'prepared_views': len(rows), 'empty_local_mask_views': sum(r['positive_pixels'] == 0 for r in rows), 'source_photos': len(allowed)}), flush=True)


def freeze():
    view_manifest = read(VIEWS/'manifest.json'); assert view_manifest['source_sha256'] == {n: sha(ROOT/n) for n in SOURCES}
    files = dict(read(ROOT/'runs/unified-belt-hard-background-v1/protected-inputs.json'))
    for folder in [ROOT/'reports', ROOT/'runs/unified-belt-hard-background-v1']:
        paths = list(folder.iterdir())
        if folder.name == 'unified-belt-hard-background-v1':
            for v in ('anchor_control', 'hard_background'): paths += list((folder/v).iterdir())
        for path in paths:
            if path.is_file() and not path.name.endswith(('.log', '.lock')): files[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for row in view_manifest['rows']:
        for kind in ('image', 'mask', 'valid'): files[row[kind]] = snapshot(ROOT/row[kind])
    for path in [DATA/'manifest.json', VIEWS/'manifest.json']: files[path.relative_to(ROOT).as_posix()] = snapshot(path)
    def check(pair):
        name, value = pair; assert snapshot(ROOT/name) == value, name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, files.items()))
    new(LEDGER, files)
    legacy = {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'safelog_ai', 'tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES}
    new(PROTOCOL, {'schema': 'unified-belt-uniform-tiles-v1', 'declared_before_fit_and_VAL': True, 'declared_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {n: sha(ROOT/n) for n in SOURCES}, 'legacy_source_sha256': legacy, 'input_ledger_sha256': sha(LEDGER),
        'start_sha256': START_SHA, 'unified_manifest_sha256': sha(DATA/'manifest.json'), 'view_manifest_sha256': sha(VIEWS/'manifest.json'),
        'draws_sha256': sha(BASE/'draws.npz'), 'epochs': 10, 'actual_neural_updates_planned': 1280, 'batch': 8, 'seed': SEED,
        'unique_source_TRAIN': view_manifest['source_TRAIN_photos_verified'], 'derived_views': view_manifest['derived_scheduled_views'],
        'empty_view_masks': view_manifest['empty_author_mask_views'], 'new_independent_photos': 0, 'new_labels': 0,
        'sampling': 'Exactly4uniform native views +4original fullframes per8; old TRAIN ids/order, view choice seed434 before labels/scores',
        'optimizer_loss': 'Belt-head only fresh AdamW LR3e-4 wd.0002 cosine10 eta_min5e-6; clip10; unchanged BCE posweight20+Dice; no hard-negative penalty or teacher',
        'non_belt_parameters_and_all_BN_stats_frozen': True, 'precision': 'FP32 TF32 disabled', 'checkpoint_selection': 'Final10 only; no per-epoch VAL',
        'configs': {'accepted_tiles': {'checkpoint': START.relative_to(ROOT).as_posix(), 'mode': 'global_tile_mean', 'new_neural_updates': 0},
                    'trained_global': {'checkpoint': (BASE/'train/final.pt').relative_to(ROOT).as_posix(), 'mode': 'global_only', 'new_neural_updates': 1280},
                    'trained_tiles': {'checkpoint': (BASE/'train/final.pt').relative_to(ROOT).as_posix(), 'mode': 'global_tile_mean', 'new_neural_updates': 1280}},
        'inference_policies': {m: policy(m) for m in ('global_only', 'global_tile_mean')}, 'same_weights_repeated_views_not_ensemble': True,
        'evaluation': 'Same known reused positive VAL353 at prior640 raster grid; threshold.5; decode native inputs, geometry-only views; reserved author TEST untouched',
        'selection': 'IoU>=accepted+.005, backgroundFPR<=accepted, photoFNR<=accepted, pixelFNR<=accepted, zeroGT-overlap<=accepted; rank IoU then FPR then trained_global; else prior_native_mix',
        'baseline': read(ROOT/'reports/unified-belt-detail-native_mix-validation.json')['summary'], 'normal_photo_FPR': None,
        'independent_factory_verified': False, 'app_default_promoted': False, 'tests_passed': 6, 'checkpoint_every_updates': 64})
    print(json.dumps({'protocol': 'frozen', 'protected_files': len(files), 'legacy_sources': len(legacy)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for n, h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path, h in [(START, START_SHA), (LEDGER, p['input_ledger_sha256']), (DATA/'manifest.json', p['unified_manifest_sha256']),
                    (VIEWS/'manifest.json', p['view_manifest_sha256']), (BASE/'draws.npz', p['draws_sha256'])]: assert sha(path) == h, path
    return p


def save_progress(path, model, opt, scheduler, epoch, cursor, updates, losses, elapsed):
    temp = path.with_suffix('.pending.pt'); assert path.resolve().is_relative_to(BASE.resolve())
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
        'epoch': epoch, 'cursor': cursor, 'updates': updates, 'losses': losses, 'elapsed_seconds': elapsed,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp); temp.replace(path)


def train():
    p = validate(); repo = ROOT.parents[1]; git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for n in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()): assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+n])).hexdigest() == sha(ROOT/n), n
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip(); run = BASE/'train'; run.mkdir(exist_ok=True)
    report = ROOT/'reports/unified-belt-tiles-training.json'
    if (run/'final.pt').exists():
        final = torch.load(run/'final.pt', map_location='cpu', weights_only=True)
        if not report.exists(): new(report, {**{k: v for k, v in final.items() if k != 'state_dict'}, 'weights_sha256': sha(run/'final.pt')})
        assert sha(run/'final.pt') == read(report)['weights_sha256']; print('Already complete; preserving neural weights', flush=True); return
    lock = run/'training.lock'
    if lock.exists():
        pid = int(lock.read_text())
        try: os.kill(pid, 0)
        except OSError: lock.unlink()
        else: raise RuntimeError('Owned study running '+str(pid))
    with lock.open('x') as out: out.write(str(os.getpid()))
    try:
        torch.manual_seed(SEED); start = torch.load(START, map_location='cpu', weights_only=True); model = UnifiedMain().cuda()
        model.load_state_dict(start['state_dict'], strict=True); fixed = state_sha256(non_belt_state(model)); initial_belt = state_sha256(model.belt_head)
        opt = torch.optim.AdamW(configure_belt_only(model), lr=3e-4, weight_decay=.0002); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=5e-6)
        manifest = read(DATA/'manifest.json'); draws = np.load(BASE/'draws.npz'); ids, views = draws['belt_damage'], draws['uniform_views']
        full = Cache(manifest['fit']['belt_damage'], set(ids.flatten().tolist()))
        detail = Cache(read(VIEWS/'manifest.json')['rows'], set(views[views >= 0].tolist()))
        progress = run/'progress.pt'; epoch, cursor, updates, losses, elapsed = 1, 0, 0, 0., 0.; torch.manual_seed(SEED)
        if progress.exists():
            s = torch.load(progress, map_location='cpu', weights_only=True); model.load_state_dict(s['state_dict'], strict=True)
            opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler']); epoch, cursor, updates, losses, elapsed = (s[k] for k in ('epoch', 'cursor', 'updates', 'losses', 'elapsed_seconds'))
            torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        started = time.monotonic()
        with (run/'update-journal.jsonl').open('a', encoding='utf-8', buffering=1) as journal:
            while epoch <= 10:
                model.eval(); model.belt_head.train(); freeze_bn_statistics(model)
                for index in range(cursor, 128):
                    offset = index*8; samples = [detail.row(int(views[epoch-1, offset+j])) if j%2 == 0 else full.row(int(ids[epoch-1, offset+j])) for j in range(8)]
                    images = torch.stack([v[0] for v in samples]); masks = torch.stack([v[1]['mask'] for v in samples]); valid = torch.stack([v[1]['valid'] for v in samples])
                    images, masks, valid = paired_flip(images, masks, valid, torch.rand(8) < .5)
                    truth = {'mask': masks.cuda().float(), 'valid': valid.cuda().float()}; images = normalize_batch(images)
                    opt.zero_grad(set_to_none=True); value = loss_for_task('belt_damage', model.forward_task(images, 'belt_damage'), truth)
                    assert torch.isfinite(value); value.backward(); torch.nn.utils.clip_grad_norm_(model.belt_head.parameters(), 10, error_if_nonfinite=True); opt.step()
                    updates += 1; cursor = index+1; losses += float(value.detach()); journal.write(json.dumps({'epoch': epoch, 'cursor': cursor, 'update': updates, 'uniform_view_draws': 4})+'\n')
                    if updates%64 == 0: save_progress(progress, model, opt, scheduler, epoch, cursor, updates, losses, elapsed+time.monotonic()-started)
                assert state_sha256(non_belt_state(model)) == fixed
                scheduler.step(); entry = {'epoch': epoch, 'updates': updates, 'mean_loss': losses/128, 'non_belt_unchanged': True, 'elapsed_minutes': (elapsed+time.monotonic()-started)/60}
                dest = run/f'epoch-{epoch:02d}.json'
                if dest.exists(): assert read(dest)['updates'] == updates
                else: new(dest, entry)
                print(json.dumps(entry), flush=True); epoch += 1; cursor = 0; losses = 0.; save_progress(progress, model, opt, scheduler, epoch, cursor, updates, losses, elapsed+time.monotonic()-started)
        assert updates == 1280 and state_sha256(model.belt_head) != initial_belt
        assert all(v.grad is None for n, v in model.named_parameters() if not n.startswith('belt_head.'))
        model.eval(); cpu = UnifiedMain().eval(); cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
        original = UnifiedMain().eval(); original.load_state_dict(start['state_dict'], strict=True); proofs = {}
        with torch.inference_mode():
            for task in TASKS:
                images, _ = Cache(manifest['fit'][task], {0, 1}).batch([0, 1]); a, b = model.forward_task(images, task), cpu.forward_task(images.cpu(), task)
                if task == 'facility_surface': a, b = a[0], b[0]
                assert torch.allclose(a.cpu(), b, atol=1e-4, rtol=.001)
                if task != 'belt_damage':
                    reference = original.forward_task(images.cpu(), task); reference = reference[0] if task == 'facility_surface' else reference; assert torch.equal(reference, b)
                proofs[task] = {'CPU_GPU_close': True, 'prior_CPU_exactly_equal': task != 'belt_damage'}
            rgb, _, _ = native(manifest['fit']['belt_damage'][0]); a, _, count = predict(model, rgb, 'global_tile_mean', 'cuda'); b, _, count_cpu = predict(cpu, rgb, 'global_tile_mean', 'cpu')
            assert count == count_cpu == 6 and torch.allclose(a.cpu(), b, atol=1e-4, rtol=.001)
        metadata = {'status': 'trained_fixed10', 'architecture': ARCH, 'tasks': list(TASKS), 'one_shared_backbone': True, 'input_size': 640,
            'thresholds': start['thresholds'], 'epochs': 10, 'optimizer_updates': 1280,
            'training_lineage': {'initial_unified_epochs': 20, 'accepted_native_belt_epochs': 10, 'additional_uniform_belt_epochs': 10},
            'initial_checkpoint_sha256': START_SHA, 'fit_protocol_commit': commit, 'protocol_sha256': sha(PROTOCOL),
            'non_belt_state_sha256': fixed, 'strict_CPU_GPU_three_tasks': proofs, 'native_tile_CPU_GPU_close': True,
            'elapsed_training_minutes': (elapsed+time.monotonic()-started)/60,
            'actual_journal_updates_including_replay': sum(1 for _ in (run/'update-journal.jsonl').open()), 'new_labels': 0, 'new_independent_photos': 0}
        with (run/'final.pt').open('xb') as out: torch.save({**metadata, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, out)
        new(report, {**metadata, 'weights_sha256': sha(run/'final.pt')}); print(json.dumps({'trained': 1280, 'sha256': sha(run/'final.pt')}), flush=True)
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()


def evaluate():
    p = validate(); rows = read(DATA/'manifest.json')['belt_known_validation']; truth_cache = Cache(rows, range(len(rows)))
    for config in CONFIGS:
        dest = ROOT/f'reports/unified-belt-tiles-{config}-validation.json'
        if dest.exists(): continue
        checkpoint = ROOT/p['configs'][config]['checkpoint']; mode = p['configs'][config]['mode']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True); model = UnifiedMain().cuda().eval(); model.load_state_dict(state['state_dict'], strict=True)
        observations = []; elapsed = time.monotonic()
        for index, row in enumerate(rows):
            rgb, _, _ = native(row)  # Source bytes only enter prediction; annotations are never passed.
            probability, box, views = predict(model, rgb, mode, 'cuda'); _, truth = truth_cache.batch([index]); valid, mask = truth['valid'], truth['mask']
            observations.append({'id': row['id'], 'sequence': row['sequence'], 'photo_positive': row['photo_positive'], 'detail_views': views,
                'score': float(probability[valid.bool()].max()), 'pixels': pixel_counts(probability, mask, valid)})
            if (index+1)%100 == 0: print(json.dumps({'evaluation': config, 'photos': index+1, 'total': len(rows)}), flush=True)
        summary = summarize(observations); new(dest, {'status': 'known_reused_native_VAL353', 'config': config, 'checkpoint_sha256': sha(checkpoint),
            'mode': mode, 'inference_policy': policy(mode), 'new_neural_updates': p['configs'][config]['new_neural_updates'],
            'threshold': .5, 'rows': observations, 'summary': summary, 'elapsed_seconds': time.monotonic()-elapsed,
            'normal_photo_FPR_available': False, 'reserved_TEST_used': False})
        print(json.dumps({'evaluation': config, 'summary': summary}), flush=True)


def select():
    p = validate(); results = {c: read(ROOT/f'reports/unified-belt-tiles-{c}-validation.json')['summary'] for c in CONFIGS}
    choice = select_tiles(results, p['baseline']); config = choice['selected']
    spec = {'checkpoint': START.relative_to(ROOT).as_posix(), 'mode': 'global_only', 'new_neural_updates': 0} if config == 'prior_native_mix' else p['configs'][config]
    choice.update(checkpoint_path=spec['checkpoint'], checkpoint_sha256=sha(ROOT/spec['checkpoint']), mode=spec['mode'],
        inference_policy=policy(spec['mode']), selected_new_neural_updates=spec['new_neural_updates'], protocol_sha256=sha(PROTOCOL), decided_utc=datetime.now(timezone.utc).isoformat())
    new(ROOT/'reports/unified-belt-tiles-selection.json', choice); print(json.dumps(choice), flush=True)


def verify():
    p = validate()
    for n, h in p['legacy_source_sha256'].items(): assert sha(ROOT/n) == h, n
    ledger = read(LEDGER)
    def check(pair):
        n, value = pair; assert snapshot(ROOT/n) == value, n
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    run = BASE/'train'; final = torch.load(run/'final.pt', map_location='cpu', weights_only=True); last = torch.load(run/'progress.pt', map_location='cpu', weights_only=True)
    original = torch.load(START, map_location='cpu', weights_only=True)['state_dict']
    assert last['epoch'] == 11 and last['cursor'] == 0 and last['updates'] == 1280
    assert all(torch.equal(v, last['state_dict'][k]) for k, v in final['state_dict'].items())
    assert all(torch.equal(v, original[k]) for k, v in final['state_dict'].items() if not k.startswith('belt_head.'))
    journal = [json.loads(s) for s in (run/'update-journal.jsonl').read_text().splitlines()]; committed = {r['update']: r for r in journal}
    assert set(committed) == set(range(1, 1281)) and sum(r['uniform_view_draws'] for r in committed.values()) == 5120
    history = [read(run/f'epoch-{e:02d}.json') for e in range(1, 11)]; assert all(r['updates'] == r['epoch']*128 and r['non_belt_unchanged'] for r in history)
    assert final['native_tile_CPU_GPU_close']; choice = read(ROOT/'reports/unified-belt-tiles-selection.json')
    assert sha(ROOT/choice['checkpoint_path']) == choice['checkpoint_sha256']
    new(ROOT/'reports/unified-belt-tiles-verification.json', {'status': 'passed', 'real_neural_updates': 1280, 'actual_journal_updates': len(journal),
        'uniform_view_draws': 5120, 'fullframe_draws': 5120, 'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']),
        'non_belt_tensors_exact_prior': True, 'final_equals_last_progress': True, 'strict_CPU_GPU_three_tasks': final['strict_CPU_GPU_three_tasks'],
        'native_tile_CPU_GPU_close': True, 'selected': choice['selected'], 'selected_new_neural_updates': choice['selected_new_neural_updates'],
        'new_labels': 0, 'new_independent_photos': 0, 'normal_photo_FPR_available': False, 'reserved_TEST_used': False, 'app_default_promoted': False})
    print(json.dumps({'verified': 'passed', 'selected': choice['selected'], 'real_neural_updates': 1280}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'freeze', 'train', 'evaluate', 'select', 'verify']); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    globals()[args.mode]()
