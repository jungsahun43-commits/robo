"""One preregistered KSDD2 TRAIN/VAL development run; no CAL/TEST or release."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
WORK = REPO.parents[1]
OUT = WORK / 'outputs'
POLICY = ROOT / 'reports/industrial-ksdd2-stream-v173-protocol.json'
REPORT = ROOT / 'reports/KSDD2_STREAM_V173_PREREG_KO.md'
RUN = ROOT / 'runs/industrial-ksdd2-stream-v173'
GIT = 'C:/Program Files/Git/cmd/git.exe'
PACKET = (Path(__file__).relative_to(REPO).as_posix(),
          POLICY.relative_to(REPO).as_posix(), REPORT.relative_to(REPO).as_posix())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def inspect_packet():
    p = json.loads(POLICY.read_bytes())
    assert p['schema'] == 'safelog-ksdd2-stream-v173-development-only'
    assert digest(__file__) == p['script_sha256']
    for rel, sha in p['pinned_inputs'].items():
        assert digest(ROOT / rel) == sha, rel
    m = json.loads((ROOT / p['manifest']).read_bytes())
    counts = {phase: dict(Counter(str(r['label']) for r in m['rows']
                                 if r['phase'] == phase)) for phase in p['phase_counts']}
    assert counts == p['phase_counts']
    roles = {}
    for r in m['rows']:
        assert r['publisher_split'] == ('test' if r['phase'] == 'test' else 'train')
        roles.setdefault(r['group_id'], set()).add(r['phase'])
    assert all(len(v) == 1 for v in roles.values())
    rows = {phase: [r for r in m['rows'] if r['phase'] == phase]
            for phase in ('fit', 'validation')}
    for phase in rows:
        assert len({r['group_id'] for r in rows[phase]}) == p['development_group_counts'][phase]
        for r in rows[phase]:
            assert len(r['masks']) == len(r['mask_sha256']) == 1
            assert r['publisher_split'] == 'train'
            for rel in (r['image'], r['masks'][0]):
                path = (ROOT / rel).resolve()
                assert path.is_relative_to((ROOT / 'data').resolve()) and path.is_file()
    return p, rows


def publication_gate(receipt_path):
    p, rows = inspect_packet()
    r = json.loads(Path(receipt_path).read_bytes())
    assert r['human_approved_exact_payload'] is True
    assert r['remote_url'] == 'https://github.com/jungsahun43-commits/robo.git'
    assert r['remote_ref'] == 'refs/heads/feature/ai-engine'
    assert r['remote_exact_commit_verified'] is True
    assert r['paths'] == list(PACKET)
    commit = r['published_commit']
    for rel in PACKET:
        raw = subprocess.check_output([GIT, '--no-optional-locks', '-C', str(REPO),
                                       '-c', 'safe.directory=' + REPO.as_posix(),
                                       'show', commit + ':' + rel])
        assert hashlib.sha256(raw).hexdigest() == digest(REPO / rel), rel
    return p, rows, commit


def process_preflight():
    command = r"""$ErrorActionPreference='Stop';
    $all=@(Get-CimInstance Win32_Process);
    $other=@($all|Where-Object { $_.Name -match '(?i)^(python.*|safelog.*|.*mask.*|.*infer.*|robo.*|industrial.*)\.exe$' -and $_.ProcessId -ne SELF_PID }|Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine);
    $mem=Get-CimInstance Win32_OperatingSystem;
    @{utc=[DateTime]::UtcNow.ToString('o');other_related=$other;free_RAM_bytes=[int64]$mem.FreePhysicalMemory*1024}|ConvertTo-Json -Depth 5"""
    command = command.replace('SELF_PID', str(os.getpid()))
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    value = json.loads(result.stdout)
    assert not value['other_related'], value
    assert value['free_RAM_bytes'] >= 6 * 1024**3, value
    smi = shutil.which('nvidia-smi')
    assert smi, 'Read-only GPU process inventory unavailable'
    gpu = subprocess.check_output([smi, '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader,nounits'], text=True, encoding='utf-8')
    contexts = [line for line in gpu.splitlines() if line.strip()]
    known_jobs = [line for line in contexts if re.search(
        r'(?i)(python[^\\/]*|safelog[^\\/]*|[^\\/]*mask[^\\/]*|[^\\/]*infer[^\\/]*|robo[^\\/]*|industrial[^\\/]*|torchrun|ollama|llama[^\\/]*)\.exe$',
        line.split(',', 1)[-1].strip())]
    assert not known_jobs, 'Another recognized GPU model process exists'
    value['GPU_context_inventory'] = contexts
    value['GPU_inventory_scope'] = 'WDDM can include desktop graphics and inaccessible names; not proof all GPU contexts are idle or a whole C++ process audit'
    locks = [ROOT / 'operation.lock', OUT / 'operation.lock']
    locks += list((ROOT / 'runs').rglob('operation.lock'))
    assert not any(path.exists() for path in locks), 'Another operation lock exists'
    return value


def run(receipt):
    p, rows, commit = publication_gate(receipt)
    host = process_preflight()
    assert not RUN.exists(), 'Completed, interrupted or failed run must not be replayed'
    assert shutil.disk_usage(ROOT).free >= p['reserve_bytes'] + p['run_budget_bytes'] + p['temporary_budget_bytes']
    lock = OUT / 'operation.lock'
    token = uuid.uuid4().hex
    owned = {'pid': os.getpid(), 'token': token, 'operation': 'ksdd2-stream-v173'}
    write_json(lock, owned)
    actual = 0
    model = head = optimizer = scheduler = None
    started = time.monotonic()
    try:
        RUN.mkdir()
        write_json(RUN / 'start.json', {'host': host, 'published_commit': commit,
                   'protocol_sha256': digest(POLICY), 'receipt_sha256': digest(receipt),
                   'created_UTC': datetime.now(timezone.utc).isoformat()})
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        from safelog_ai.real_mask_server import RuntimeRealMaskMain
        from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
        from safelog_ai.magnetic_surface import RealMaskMain
        from safelog_ai.ksdd_supervised import SurfaceMaskHead, ksdd_view, HEIGHT, WIDTH
        from safelog_ai.native_component_views import normalized
        assert RuntimeRealMaskMain.forward_surface is IndustrialSurfaceMain.forward_surface
        assert RuntimeRealMaskMain.forward_real_mask is RealMaskMain.forward_surface
        torch.set_num_threads(4)
        assert torch.cuda.is_available()
        torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        def state_sha(state):
            h = hashlib.sha256()
            for name, tensor in sorted(state.items()):
                h.update(name.encode()); h.update(str(tensor.dtype).encode())
                h.update(json.dumps(list(tensor.shape)).encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        parent = torch.load(ROOT / p['parent_file'], map_location='cpu', weights_only=True)
        model = RuntimeRealMaskMain(parent['reference_counts'], parent['native_specs'],
                                   parent['wear_variant'], parent['surface_reference_specs'])
        model.load_state_dict(parent['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        parent_state_sha = state_sha(model.state_dict())
        del parent
        model.backbone.cuda().eval()
        head = SurfaceMaskHead().cuda()
        head.load_state_dict(model.ksdd_head.state_dict(), strict=True)
        head.high_projection.requires_grad_(False)
        params = list(head.fusion.parameters())
        optimizer = torch.optim.AdamW(params, lr=p['lr'], weight_decay=p['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, p['epochs'], eta_min=p['minimum_lr'])
        rng = np.random.default_rng(p['seed'])

        def loss_each(logit, target, valid):
            target = target.float(); valid = valid.bool()
            pos = valid & (target > 0); neg = valid & ~pos
            dims = (1, 2, 3)
            npix = pos.sum(dims); nnix = neg.sum(dims)
            a = (F.softplus(-logit) * pos).sum(dims) / npix.clamp_min(1)
            b = (F.softplus(logit) * neg).sum(dims) / nnix.clamp_min(1)
            pixel = torch.where(npix > 0, .5 * a + .5 * b, b)
            probability = logit.sigmoid()
            intersection = (probability * target * valid).sum(dims)
            denominator = ((probability + target) * valid).sum(dims)
            dice = 1 - (2 * intersection + 1) / (denominator + 1)
            score = probability.masked_fill(~valid, 0).flatten(1).topk(32, dim=1).values.mean(1)
            photo = F.binary_cross_entropy(score.clamp(1e-6, 1-1e-6), (npix > 0).float(), reduction='none')
            return pixel + dice + .5 * photo

        # Pure loss check after preregistration, before any dataset pixel or optimizer.
        z = torch.zeros(2, 1, 8, 8, requires_grad=True)
        y = torch.zeros_like(z); y[0, 0, 2, 2] = 1
        v = torch.ones_like(z, dtype=torch.bool); v[:, :, 0] = False
        values = loss_each(z, y, v); values.mean().backward()
        assert torch.isfinite(values).all() and torch.isfinite(z.grad).all()
        assert torch.count_nonzero(z.grad[:, :, 0]) == 0
        write_json(RUN / 'loss-check.json', {'finite_positive_normal': True,
                   'padding_gradient_zero': True, 'optimizer_updates': 0, 'dataset_pixels': 0})

        def resource_guard():
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            counters = MemoryCounters(); counters.cb = ctypes.sizeof(counters)
            current = ctypes.windll.kernel32.GetCurrentProcess
            current.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            assert get_memory(current(), ctypes.byref(counters), counters.cb)
            assert counters.PeakWorkingSetSize <= p['CPU_working_set_budget_bytes']
            assert time.monotonic() - started < p['maximum_seconds'], 'Time budget reached; no automatic replay'
            assert torch.cuda.max_memory_allocated() <= p['GPU_tensor_budget_bytes']
            assert sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) <= p['run_budget_bytes']
            assert shutil.disk_usage(ROOT).free > p['reserve_bytes']

        def batch(selected, augment=False):
            inputs = []; masks = []; valid = []
            for row in selected:
                assert digest(ROOT / row['image']) == row['image_sha256']
                assert digest(ROOT / row['masks'][0]) == row['mask_sha256'][0]
                with Image.open(ROOT / row['image']) as source:
                    assert list(source.size) == row['size'] and source.width * source.height <= p['maximum_source_pixels']
                    rgb = source.convert('RGB')
                with Image.open(ROOT / row['masks'][0]) as source:
                    assert source.size == rgb.size
                    truth = source.convert('L')
                if augment:
                    for method in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                        if rng.random() < .5:
                            rgb = rgb.transpose(method); truth = truth.transpose(method)
                view, mask, known, _ = ksdd_view(rgb, truth)
                assert bool(mask.any()) == bool(row['label'])
                inputs.append(normalized(view)); masks.append(mask); valid.append(known)
            image = torch.stack(inputs).cuda()
            with torch.no_grad():
                f = model.backbone(image)
                low = f['low'].half().float(); high = f['high'].half().float()
            return low, high, torch.stack(masks)[:, None].cuda(), torch.stack(valid)[:, None].cuda()

        def logits(low, high):
            return F.interpolate(head(low, high), size=(HEIGHT, WIDTH), mode='bilinear', align_corners=False)

        def validation():
            head.eval(); sums = [0., 0.]; counts = [0, 0]
            with torch.no_grad():
                for at in range(0, len(rows['validation']), p['microbatch']):
                    selected = rows['validation'][at:at+p['microbatch']]
                    low, high, mask, valid = batch(selected)
                    measured = loss_each(logits(low, high), mask, valid).cpu().tolist()
                    for row, value in zip(selected, measured):
                        label = int(row['label']); sums[label] += value; counts[label] += 1
                    resource_guard()
            assert counts == [389, 52]
            return .5 * (sums[0]/counts[0] + sums[1]/counts[1])

        pools = [[r for r in rows['fit'] if int(r['label']) == label] for label in (0, 1)]
        weights = []
        for pool in pools:
            groups = Counter(r['group_id'] for r in pool)
            w = np.asarray([1/groups[r['group_id']] for r in pool], dtype=float)
            weights.append(w/w.sum())
        initial = validation(); best = float('inf'); best_epoch = None; best_path = None
        write_json(RUN / 'initial-validation.json', {'loss': initial, 'epoch': 0, 'eligible': False})
        with (RUN / 'optimizer.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            for epoch in range(1, p['epochs']+1):
                head.train(); total = 0.
                for step in range(1, p['steps_per_epoch']+1):
                    resource_guard(); optimizer.zero_grad(set_to_none=True)
                    selected_ids = []
                    for _ in range(p['accumulation_microbatches']):
                        selected = [pools[label][int(i)] for label in (0, 1)
                                    for i in rng.choice(len(pools[label]), 2, replace=True, p=weights[label])]
                        selected_ids.extend(r['id'] for r in selected)
                        low, high, mask, valid = batch(selected, augment=True)
                        loss = loss_each(logits(low, high), mask, valid).mean()/p['accumulation_microbatches']
                        assert torch.isfinite(loss)
                        loss.backward(); total += float(loss.detach())
                    norm = torch.nn.utils.clip_grad_norm_(params, 5.)
                    assert torch.isfinite(norm) and norm > 0
                    optimizer.step(); actual += 1
                    journal.write(json.dumps({'actual_update': actual, 'epoch': epoch, 'step': step,
                                              'input_ids': selected_ids, 'gradient_norm_before_clip': float(norm)})+'\n')
                    journal.flush()
                value = validation(); scheduler.step()
                state = {'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'epoch': epoch, 'actual_updates': actual, 'validation_loss': value,
                         'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                         'numpy_rng_json': json.dumps(rng.bit_generator.state), 'protocol_sha256': digest(POLICY)}
                path = RUN / ('epoch-%02d.pt' % epoch)
                with path.open('xb') as f: torch.save(state, f)
                if value < best: best, best_epoch, best_path = value, epoch, path
                item = {'epoch': epoch, 'actual_updates': actual, 'mean_training_loss': total/p['steps_per_epoch'],
                        'validation_balanced_loss': value, 'selected_epoch': best_epoch}
                with (RUN / 'epochs.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(item)+'\n')
                print(json.dumps(item), flush=True)
        assert actual == p['epochs']*p['steps_per_epoch'] and best_epoch > 0
        assert state_sha(model.state_dict()) == parent_state_sha
        for rel, sha in p['pinned_inputs'].items(): assert digest(ROOT / rel) == sha, rel
        chosen = torch.load(best_path, map_location='cpu', weights_only=True)['head']
        added = {'ksdd_stream_v173.'+k: v for k, v in chosen.items()}
        combined = dict(model.state_dict()); assert not set(combined).intersection(added)
        combined.update(added)
        delta = {'schema': p['schema'], 'parent_file': p['parent_file'],
                 'parent_file_sha256': p['pinned_inputs'][p['parent_file']],
                 'parent_state_sha256': parent_state_sha, 'added_state_sha256': state_sha(added),
                 'combined_state_sha256': state_sha(combined), 'added_state_dict': added,
                 'preserved_sidecars': p['preserved_sidecars'], 'selected_epoch': best_epoch,
                 'actual_optimizer_updates': actual, 'protocol_sha256': digest(POLICY),
                 'standalone_app_model': False, 'CAL_TEST_used': False}
        path = RUN / 'development-delta.pt'
        with path.open('xb') as f: torch.save(delta, f)
        assert path.stat().st_size <= p['delta_budget_bytes']
        restored = torch.load(path, map_location='cpu', weights_only=True)
        assert state_sha(restored['added_state_dict']) == delta['added_state_sha256']
        combined = dict(model.state_dict()); combined.update(restored['added_state_dict'])
        assert state_sha(combined) == delta['combined_state_sha256']
        write_json(RUN / 'training-result.json', {'actual_optimizer_updates': actual,
                   'selected_epoch': best_epoch, 'initial_VAL_loss': initial, 'selected_VAL_loss': best,
                   'VAL_improved_over_initial': best < initial, 'parent_all_tensors_buffers_exact': True,
                   'pinned_files_exact': True, 'delta_sha256': digest(path), 'serialized_restore_exact': True,
                   'new_feature_cache_bytes': 0, 'CAL_TEST_predictions': 0, 'deployed': False,
                   'ready': False, 'independent_field_validated': False,
                   'next': 'Separate unchanged-gate CAL/one fixed TEST/retention/runtime preregistration before any adoption; no automatic publication or model replacement.'})
    except BaseException as error:
        if RUN.exists():
            write_json(RUN / 'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                       'actual_optimizer_updates_confirmed_in_process': actual,
                       'no_automatic_resume_or_replay': True, 'partial_artifacts_preserved': True})
            if head is not None and optimizer is not None:
                import torch
                with (RUN / 'failure-state.pt').open('xb') as f:
                    torch.save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
                                'actual_updates': actual, 'protocol_sha256': digest(POLICY),
                                'torch_rng': torch.get_rng_state(),
                                'cuda_rng': torch.cuda.get_rng_state_all()}, f)
        raise
    finally:
        if lock.exists() and json.loads(lock.read_bytes()) == owned:
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--publication-receipt')
    args = parser.parse_args()
    if args.inspect:
        p, rows = inspect_packet()
        print(json.dumps({'packet_metadata_verified': True,
                          'development_counts': {k: len(v) for k, v in rows.items()},
                          'photos_decoded': 0, 'torch_imported': False, 'optimizer_updates': 0}))
    else:
        if not args.publication_receipt: parser.error('Verified, human-approved exact publication receipt required')
        run(args.publication_receipt)
