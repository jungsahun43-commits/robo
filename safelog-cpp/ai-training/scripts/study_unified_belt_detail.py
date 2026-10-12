"""Fixed belt-only control versus native TRAIN crop continuation of the SAME main."""
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
from safelog_ai.belt_detail import native_crop, paired_flip, configure_belt_only, non_belt_state, select_detail
from safelog_ai.unified_main import ARCH, TASKS, UnifiedMain, freeze_bn_statistics, loss_for_task
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.belt_evaluation import summarize
from safelog_ai.belt_segmentation import pixel_counts
from scripts.unified_main_data import DATA, Cache, read, sha, normalize_batch
from scripts.prepare_beltcrack_starter import rasterize

BASE = ROOT/'runs/unified-belt-native-detail-v1'
CROPS = ROOT/'data/unified-belt-native-detail-v1'
PROTOCOL = ROOT/'reports/unified-belt-detail-protocol.json'
LEDGER = BASE/'protected-inputs.json'
START = ROOT/'runs/unified-main-v1/final.pt'
START_SHA = '362f9cab4a137419e63118f73b6b622dccfacc37c94a8f52af62f217b1491598'
VARIANTS = ('full_control', 'native_mix')
EPOCHS, BATCH, SEED = 10, 8, 427
SOURCES = ('safelog_ai/belt_detail.py', 'scripts/study_unified_belt_detail.py', 'tests/test_belt_detail.py')


def new(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def prepare():
    assert not CROPS.exists() and not BASE.exists() and sha(START) == START_SHA
    rows = read(DATA/'manifest.json')['fit']['belt_damage']; assert len(rows) == 1397
    crops = []
    for i, row in enumerate(rows):
        assert row['split'] == 'train' and row['task'] == 'belt_damage' and row['photo_positive']
        path = ROOT/'data/equipment-beltcrack/expanded-cached-v2/original-annotations'/row['id']
        assert sha(path) == row['annotation_sha256']
        annotation = read(path); raw = base64.b64decode(annotation['imageData'], validate=True)
        assert hashlib.sha256(raw).hexdigest() == row['embedded_image_sha256']
        with Image.open(io.BytesIO(raw)) as original:
            rgb = original.convert('RGB'); pixels = np.asarray(rgb)
            assert hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest() == row['decoded_rgb_sha256']
            for kind, actual in zip(('image', 'mask', 'valid'), rasterize(annotation, rgb)):
                with Image.open(ROOT/row[kind]) as old: assert np.array_equal(np.asarray(actual), np.asarray(old)), (row['id'], kind)
            image, mask, valid, box = native_crop(rgb, annotation)
        item = {'id': row['id'], 'split': 'train', 'task': 'belt_damage', 'sequence': row['sequence'],
                'native_box': list(box), 'native_width': rgb.width, 'native_height': rgb.height,
                'annotation_sha256': row['annotation_sha256'], 'embedded_image_sha256': row['embedded_image_sha256'],
                'annotation_path': path.relative_to(ROOT).as_posix(), 'photo_positive': True,
                'positive_pixels': int(np.count_nonzero(mask)), 'valid_pixels': 640*640}
        for kind, value in [('image', image), ('mask', mask), ('valid', valid)]:
            dest = CROPS/'prepared'/f'{i}-{kind}.png'; dest.parent.mkdir(exist_ok=True, parents=True)
            value.save(dest); item[kind] = dest.relative_to(ROOT).as_posix(); item[kind+'_sha256'] = sha(dest)
        crops.append(item)
        if (i+1) % 100 == 0: print(json.dumps({'native_TRAIN_crops_verified': i+1, 'total': len(rows)}), flush=True)
    new(CROPS/'manifest.json', {'status': 'frozen_TRAIN_only', 'rows': crops, 'unique_source_photos': len(rows),
        'extra_independent_photos': 0, 'new_labels': 0, 'source_VAL_or_TEST_decoded_for_crops': False,
        'full_prepared_RGB_mask_valid_match_native_author_source': True,
        'crop_rule': 'First author shape bbox center; square side=max720,ceil1.5span capped by native shorter edge; clamp box to photo; rasterize all original shapes, clipping at crop boundary',
        'native_crops_not_upsampled_640_fullframe': True, 'normal_belt_photo_labels': 0,
        'license': 'BeltCrack CC BY-SA 4.0; publisher content cropped, resized, rasterized',
        'source_sha256': {name: sha(ROOT/name) for name in SOURCES}})
    old = np.load(DATA/'draws.npz', allow_pickle=False)['belt_damage'][:EPOCHS].copy()
    np.random.default_rng(SEED).shuffle(old, axis=1)
    BASE.mkdir(); np.savez_compressed(BASE/'draws.npz', belt_damage=old)
    print(json.dumps({'prepared': len(crops), 'draws': list(old.shape), 'unique_scheduled': len(set(old.flatten().tolist()))}), flush=True)


def freeze():
    manifest = read(CROPS/'manifest.json'); assert manifest['source_sha256'] == {n: sha(ROOT/n) for n in SOURCES}
    ledger = read(ROOT/'runs/unified-preservation-typed-v3/protected-inputs.json')
    files = {**ledger['current_inputs'], **ledger['previous_models_and_reports']}
    for folder in [ROOT/'reports', ROOT/'runs/unified-preservation-typed-v3', ROOT/'runs/unified-main-v1']:
        paths = folder.rglob('*') if folder != ROOT/'reports' else folder.iterdir()
        for path in paths:
            if path.is_file() and not path.name.endswith(('.lock', '.log')):
                stat = path.stat(); files[path.relative_to(ROOT).as_posix()] = {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    for row in manifest['rows']:
        for name in [row['annotation_path'], *(row[k] for k in ('image', 'mask', 'valid'))]:
            path = ROOT/name; stat = path.stat(); files[name] = {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    def check(pair):
        name, entry = pair; path = ROOT/name; stat = path.stat()
        assert sha(path) == entry['sha256'] and stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns'], name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, files.items()))
    new(LEDGER, files)
    legacy = {p.relative_to(ROOT).as_posix(): sha(p) for d in ('scripts', 'safelog_ai', 'tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES}
    baseline = read(ROOT/'reports/unified-main-results.json')['belt_val353']
    new(PROTOCOL, {'schema': 'unified-belt-native-detail-v1', 'declared_before_fit': True,
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': {n: sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256': legacy, 'input_ledger_sha256': sha(LEDGER), 'unified_manifest_sha256': sha(DATA/'manifest.json'),
        'crop_manifest_sha256': sha(CROPS/'manifest.json'), 'draws_sha256': sha(BASE/'draws.npz'), 'start_sha256': START_SHA,
        'variants': list(VARIANTS), 'epochs_each': EPOCHS, 'updates_each': 1280, 'total_updates': 2560,
        'batch': BATCH, 'seed': SEED, 'draws_per_epoch': 1024, 'unique_scheduled_TRAIN': len(set(np.load(BASE/'draws.npz')['belt_damage'].flatten().tolist())),
        'new_photos': 0, 'new_labels': 0, 'derived_TRAIN_crops': len(manifest['rows']),
        'single_shared_model_preserved': True, 'freeze': 'Entire backbone, facility heads and foreign head including every BN buffer; belt-head affine/conv only trainable; all BN running stats eval',
        'optimizer': 'Fresh AdamW belt parameters only; LR3e-4 weight_decay.0002 cosine10 epochs eta_min5e-6; gradient clip10',
        'loss': 'Unchanged v1 weighted BCE posweight20 plus per-image Dice; original mask/valid only',
        'training_difference': 'full_control uses original640 fullframes; native_mix replaces exactly4of8 draws per batch with deterministic native TRAIN crops; both same ids, steps, flips, loss, seed',
        'precision': 'FP32 TF32 disabled', 'checkpoint_selection': 'Final10 only; no per-epoch VAL or early stopping',
        'evaluation': 'Known reused author VAL353/4sequences, identical640 fullframe and .5 threshold; reserved author TEST never used',
        'selection': 'IoU>=v1+.005 AND photoFNR<=v1 AND pixelFNR<=v1 AND zero-GT-overlap<=v1; rank IoU then pixelFNR then control; else prior_v1',
        'baseline': baseline, 'other_tasks': 'Exact non-belt tensor retention and prediction probes; inherited v1 metrics explicitly not fresh evaluation; no new foreignCAL/TEST scoring',
        'normal_photo_FPR': None, 'independent_factory_verified': False, 'app_default_promoted': False,
        'tests_passed': 5, 'checkpoint_every_updates': 64, 'disposable_preflight_updates': 0})
    print(json.dumps({'protocol': 'frozen', 'protected_files': len(files), 'legacy_sources': len(legacy)}), flush=True)


def validate():
    p = read(PROTOCOL)
    for name, digest in p['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    for path, digest in [(START, START_SHA), (LEDGER, p['input_ledger_sha256']), (DATA/'manifest.json', p['unified_manifest_sha256']),
                         (CROPS/'manifest.json', p['crop_manifest_sha256']), (BASE/'draws.npz', p['draws_sha256'])]: assert sha(path) == digest, path
    return p


def batch_for(full, crops, indices, variant, device='cuda'):
    values = [crops.row(int(i)) if variant == 'native_mix' and j%2 == 0 else full.row(int(i)) for j, i in enumerate(indices)]
    images = torch.stack([v[0] for v in values]); mask = torch.stack([v[1]['mask'] for v in values]); valid = torch.stack([v[1]['valid'] for v in values])
    images, mask, valid = paired_flip(images, mask, valid, torch.rand(len(values)) < .5)
    return normalize_batch(images, device), {'mask': mask.to(device).float(), 'valid': valid.to(device).float()}


def save_progress(path, model, opt, scheduler, epoch, cursor, updates, losses, elapsed):
    assert path.resolve().is_relative_to(BASE.resolve()); temp = path.with_suffix('.pending.pt')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'cursor': cursor, 'updates': updates, 'losses': losses,
        'elapsed_seconds': elapsed, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
    temp.replace(path)


def train(variant):
    p = validate(); repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    for name in (*SOURCES, PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])).hexdigest() == sha(ROOT/name), name
    commit = subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()
    run = BASE/variant; run.mkdir(exist_ok=True)
    report = ROOT/f'reports/unified-belt-detail-{variant}-training.json'
    if (run/'final.pt').exists():
        if not report.exists():
            final = torch.load(run/'final.pt', map_location='cpu', weights_only=True)
            assert final['variant'] == variant and final['optimizer_updates'] == 1280 and final['epochs'] == EPOCHS
            new(report, {**{k: v for k, v in final.items() if k != 'state_dict'}, 'status': 'trained_fixed10',
                'weights_sha256': sha(run/'final.pt'), 'checkpoint_count': 1, 'backbone_count': 1})
        assert report.exists() and sha(run/'final.pt') == read(report)['weights_sha256']; print(variant+' already complete', flush=True); return
    lock = run/'training.lock'
    if lock.exists():
        old_pid = int(lock.read_text())
        try: os.kill(old_pid, 0)
        except OSError: lock.unlink()
        else: raise RuntimeError('Owned study already running '+str(old_pid))
    with lock.open('x') as out: out.write(str(os.getpid()))
    try:
        torch.manual_seed(SEED); start = torch.load(START, map_location='cpu', weights_only=True)
        model = UnifiedMain().cuda(); model.load_state_dict(start['state_dict'], strict=True)
        fixed = state_sha256(non_belt_state(model)); original_belt = state_sha256(model.belt_head)
        opt = torch.optim.AdamW(configure_belt_only(model), lr=3e-4, weight_decay=.0002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=5e-6)
        draws = np.load(BASE/'draws.npz')['belt_damage']; allowed = set(draws.flatten().tolist())
        full = Cache(read(DATA/'manifest.json')['fit']['belt_damage'], allowed)
        crops = Cache(read(CROPS/'manifest.json')['rows'], allowed)
        progress = run/'progress.pt'; epoch, cursor, updates, losses, elapsed = 1, 0, 0, 0., 0.
        torch.manual_seed(SEED)
        if progress.exists():
            saved = torch.load(progress, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['state_dict'], strict=True); opt.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            epoch, cursor, updates, losses, elapsed = (saved[k] for k in ('epoch', 'cursor', 'updates', 'losses', 'elapsed_seconds'))
            torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        started = time.monotonic()
        with (run/'update-journal.jsonl').open('a', encoding='utf-8', buffering=1) as journal:
            while epoch <= EPOCHS:
                model.eval(); model.belt_head.train(); freeze_bn_statistics(model)
                chunks = np.split(draws[epoch-1], 128)
                for offset in range(cursor, len(chunks)):
                    images, truth = batch_for(full, crops, chunks[offset], variant)
                    opt.zero_grad(set_to_none=True); value = loss_for_task('belt_damage', model.forward_task(images, 'belt_damage'), truth)
                    assert torch.isfinite(value); value.backward(); torch.nn.utils.clip_grad_norm_(model.belt_head.parameters(), 10, error_if_nonfinite=True)
                    opt.step(); updates += 1; cursor = offset+1; losses += float(value.detach())
                    journal.write(json.dumps({'epoch': epoch, 'cursor': cursor, 'update': updates, 'native_crop_draws': 4 if variant == 'native_mix' else 0})+'\n')
                    if updates%64 == 0: save_progress(progress, model, opt, scheduler, epoch, cursor, updates, losses, elapsed+time.monotonic()-started)
                assert state_sha256(non_belt_state(model)) == fixed
                scheduler.step(); record = {'variant': variant, 'epoch': epoch, 'updates': updates, 'mean_loss': losses/128,
                    'elapsed_minutes': (elapsed+time.monotonic()-started)/60, 'non_belt_state_unchanged': True}
                path = run/f'epoch-{epoch:02d}.json'
                if path.exists(): assert read(path)['updates'] == updates
                else: new(path, record)
                print(json.dumps(record), flush=True); epoch += 1; cursor = 0; losses = 0.
                save_progress(progress, model, opt, scheduler, epoch, cursor, updates, losses, elapsed+time.monotonic()-started)
        assert updates == 1280 and state_sha256(non_belt_state(model)) == fixed and state_sha256(model.belt_head) != original_belt
        assert all(param.grad is None for name, param in model.named_parameters() if not name.startswith('belt_head.'))
        model.eval(); cpu = UnifiedMain().eval(); cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
        original = UnifiedMain().eval(); original.load_state_dict(start['state_dict'], strict=True)
        probes = {}; manifest = read(DATA/'manifest.json')
        with torch.inference_mode():
            for task in TASKS:
                images, _ = Cache(manifest['fit'][task], {0, 1}).batch([0, 1])
                a, b = model.forward_task(images, task), cpu.forward_task(images.cpu(), task)
                if task == 'facility_surface': a, b = a[0], b[0]
                assert torch.allclose(a.cpu(), b, atol=1e-4, rtol=.001)
                if task != 'belt_damage':
                    ref = original.forward_task(images.cpu(), task); ref = ref[0] if task == 'facility_surface' else ref
                    assert torch.equal(ref, b)
                probes[task] = {'strict_CPU_GPU_close': True, 'v1_CPU_predictions_exactly_equal': task != 'belt_damage'}
        calls = sum(1 for _ in (run/'update-journal.jsonl').open())
        state = {'architecture': ARCH, 'tasks': list(TASKS), 'one_shared_backbone': True, 'input_size': 640,
            'thresholds': start['thresholds'], 'variant': variant, 'base_epochs': 20, 'epochs': EPOCHS, 'optimizer_updates': updates,
            'initial_checkpoint_sha256': START_SHA, 'protocol_sha256': sha(PROTOCOL), 'fit_protocol_commit': commit,
            'non_belt_state_sha256': fixed, 'other_task_predictions_exactly_equal': True, 'strict_CPU_GPU_three_tasks': probes,
            'elapsed_training_minutes': (elapsed+time.monotonic()-started)/60, 'actual_journal_updates_including_replay': calls,
            'field_validated': False, 'new_photos': 0, 'new_labels': 0}
        with (run/'final.pt').open('xb') as out: torch.save({**state, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, out)
        new(report, {**state, 'status': 'trained_fixed10', 'weights_sha256': sha(run/'final.pt'), 'checkpoint_count': 1, 'backbone_count': 1})
        print(json.dumps({'completed': variant, 'updates': updates, 'sha256': sha(run/'final.pt')}), flush=True)
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()): lock.unlink()


def evaluate():
    validate(); rows = read(DATA/'manifest.json')['belt_known_validation']; cache = Cache(rows, range(len(rows)))
    for variant in VARIANTS:
        path = ROOT/f'reports/unified-belt-detail-{variant}-validation.json'
        if path.exists(): continue
        trained = read(ROOT/f'reports/unified-belt-detail-{variant}-training.json')
        checkpoint = BASE/variant/'final.pt'; assert sha(checkpoint) == trained['weights_sha256']
        model = UnifiedMain().cuda().eval(); model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict'], strict=True)
        observations = []
        with torch.inference_mode():
            for offset in range(0, len(rows), BATCH):
                images, truth = cache.batch(range(offset, min(offset+BATCH, len(rows)))); scores = model.forward_task(images, 'belt_damage').sigmoid()
                for i, row in enumerate(rows[offset:offset+BATCH]):
                    score, mask, valid = scores[i:i+1], truth['mask'][i:i+1], truth['valid'][i:i+1]
                    observations.append({'id': row['id'], 'sequence': row['sequence'], 'photo_positive': row['photo_positive'],
                        'score': float(score[valid.bool()].max()), 'pixels': pixel_counts(score, mask, valid)})
        summary = summarize(observations)
        new(path, {'status': 'known_reused_VAL353_fullframe', 'variant': variant, 'model_sha256': trained['weights_sha256'],
            'rows': observations, 'summary': summary, 'threshold': .5, 'normal_photo_FPR_available': False, 'reserved_TEST_used': False})
        print(json.dumps({'validation': variant, 'summary': summary}), flush=True)


def select():
    p = validate(); rows = {v: read(ROOT/f'reports/unified-belt-detail-{v}-validation.json')['summary'] for v in VARIANTS}
    choice = select_detail(rows, p['baseline']); checkpoint = START if choice['selected'] == 'prior_v1' else BASE/choice['selected']/'final.pt'
    choice.update(checkpoint_path=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(checkpoint),
        protocol_sha256=sha(PROTOCOL), decided_utc=datetime.now(timezone.utc).isoformat(), known_VAL_reused=True, foreign_TEST_used=False)
    new(ROOT/'reports/unified-belt-detail-selection.json', choice); print(json.dumps(choice), flush=True)


def verify():
    p = validate()
    for name, digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name) == digest, name
    def check(pair):
        name, entry = pair; path = ROOT/name; stat = path.stat()
        assert sha(path) == entry['sha256'] and stat.st_size == entry['size_bytes'] and stat.st_mtime_ns == entry['mtime_ns'], name
    ledger = read(LEDGER)
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, ledger.items()))
    baseline = torch.load(START, map_location='cpu', weights_only=True)['state_dict']; proof = {}
    for variant in VARIANTS:
        run = BASE/variant; trained = read(ROOT/f'reports/unified-belt-detail-{variant}-training.json')
        final = torch.load(run/'final.pt', map_location='cpu', weights_only=True); progress = torch.load(run/'progress.pt', map_location='cpu', weights_only=True)
        assert progress['epoch'] == 11 and progress['cursor'] == 0 and progress['updates'] == 1280
        assert all(torch.equal(v, progress['state_dict'][k]) for k, v in final['state_dict'].items())
        assert all(torch.equal(v, baseline[k]) for k, v in final['state_dict'].items() if not k.startswith('belt_head.'))
        assert sha(run/'final.pt') == trained['weights_sha256']
        history = [read(run/f'epoch-{e:02d}.json') for e in range(1, EPOCHS+1)]
        assert all(r['updates'] == r['epoch']*128 and r['non_belt_state_unchanged'] for r in history)
        journal = [json.loads(s) for s in (run/'update-journal.jsonl').read_text().splitlines()]
        committed = {r['update']: r for r in journal}
        assert set(committed) == set(range(1, 1281))
        crop_draws = sum(r['native_crop_draws'] for r in committed.values())
        assert crop_draws == (5120 if variant == 'native_mix' else 0)
        proof[variant] = {'committed_updates': 1280, 'actual_journal_updates': len(journal), 'native_crop_draws': crop_draws,
            'fullframe_draws': 10240-crop_draws, 'model_sha256': trained['weights_sha256'], 'all_non_belt_tensors_exactly_v1': True,
            'final_matches_last_progress': True, 'strict_CPU_GPU_three_tasks': trained['strict_CPU_GPU_three_tasks']}
    choice = read(ROOT/'reports/unified-belt-detail-selection.json'); assert sha(ROOT/choice['checkpoint_path']) == choice['checkpoint_sha256']
    new(ROOT/'reports/unified-belt-detail-verification.json', {'status': 'passed', 'models': proof, 'total_real_committed_updates': 2560,
        'protected_files': len(ledger), 'legacy_sources': len(p['legacy_source_sha256']), 'selected': choice['selected'],
        'original_v1_preserved': True, 'new_photos': 0, 'new_labels': 0, 'reserved_TEST_used': False,
        'normal_photo_FPR_available': False, 'independent_factory_verified': False, 'app_default_promoted': False})
    print(json.dumps({'verified': 'passed', 'selected': choice['selected'], 'protected_files': len(ledger)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'freeze', 'train', 'evaluate', 'select', 'verify'])
    parser.add_argument('--variant', choices=VARIANTS); args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.mode == 'train':
        for variant in ([args.variant] if args.variant else VARIANTS): train(variant)
    else: globals()[args.mode]()
