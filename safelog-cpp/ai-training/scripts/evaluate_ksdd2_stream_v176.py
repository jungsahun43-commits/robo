"""One fixed KSDD2 CAL then TEST evaluation; no training, replay, release or app promotion."""
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
POLICY = ROOT / 'reports/industrial-ksdd2-stream-v176-evaluation-protocol.json'
REPORT = ROOT / 'reports/KSDD2_STREAM_V176_EVALUATION_PREREG_KO.md'
RUN = ROOT / 'runs/industrial-ksdd2-stream-v176-evaluation'
GIT = 'C:/Program Files/Git/cmd/git.exe'
PACKET = (Path(__file__).relative_to(REPO).as_posix(), POLICY.relative_to(REPO).as_posix(), REPORT.relative_to(REPO).as_posix())
# The following read-only process guards are exact function-source copies of the approved v175 script.

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def windows_argv(command):
    import ctypes
    from ctypes import wintypes
    if not command: return None
    count = ctypes.c_int()
    parse = ctypes.windll.shell32.CommandLineToArgvW
    parse.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    value = parse(command, ctypes.byref(count))
    if not value: return None
    try: return [value[i] for i in range(count.value)]
    finally:
        free = ctypes.windll.kernel32.LocalFree
        free.argtypes = [wintypes.HLOCAL]; free.restype = wintypes.HLOCAL
        free(ctypes.cast(value, wintypes.HLOCAL))

def same_owned_launcher(candidate, current, pid, parent_pid, executable, base_executable, arguments):
    try:
        if not candidate or not current or len(arguments) < 2: return False
        if int(current['ProcessId']) != pid: return False
        if int(candidate['ProcessId']) != parent_pid or int(current['ParentProcessId']) != parent_pid: return False
        expected = Path(executable).resolve(); base = Path(base_executable).resolve()
        if expected == base: return False
        if Path(candidate['ExecutablePath']).resolve() != expected: return False
        if Path(current['ExecutablePath']).resolve() != base: return False
        if candidate['Name'].lower() != expected.name.lower(): return False
        if not candidate.get('Argv') or not current.get('Argv'): return False
        if candidate['Argv'][1:] != current['Argv'][1:] or current['Argv'][1:] != list(arguments[1:]): return False
        elapsed = (datetime.fromisoformat(current['StartUTC']) - datetime.fromisoformat(candidate['StartUTC'])).total_seconds()
        return 0 <= elapsed <= 5
    except (KeyError, TypeError, ValueError, OSError):
        return False

def launcher_unit_checks():
    executable = ROOT / '.venv/Scripts/python.exe'
    base = ROOT / 'control-plane-test-base/python.exe'
    args = [str(base), '-u', str(Path(__file__)), '--inspect-process']
    parent = {'ProcessId': 10, 'Name': 'python.exe', 'ExecutablePath': str(executable),
              'Argv': [str(executable)] + args[1:], 'StartUTC': '2026-10-22T00:00:00+00:00'}
    current = {'ProcessId': 11, 'ParentProcessId': 10, 'ExecutablePath': str(base),
               'Argv': args, 'StartUTC': '2026-10-22T00:00:00.014+00:00'}
    check = lambda p, c: same_owned_launcher(p, c, 11, 10, executable, base, args)
    assert check(parent, current)
    assert not check(dict(parent, ProcessId=12), current)
    assert not check(dict(parent, Argv=[str(executable), '-u', 'another-job.py']), current)
    assert not check(dict(parent, ExecutablePath=str(base)), current)
    assert not check(dict(parent, StartUTC='2026-10-21T23:59:00+00:00'), current)
    assert not check(dict(parent, Argv=None), current)
    assert not check(parent, dict(current, ParentProcessId=12))
    return {'same_invocation_venv_parent_allowed': True, 'six_unrelated_or_unverified_cases_rejected': True,
            'scope': 'Administrative process records only; no dataset fixture, torch, feature or gradient'}

def process_preflight():
    command = r"""$ErrorActionPreference='Stop';
    $all=@(Get-CimInstance Win32_Process);
    $own=$all|Where-Object { $_.ProcessId -eq SELF_PID }|Select-Object -First 1 ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine,@{Name='StartUTC';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}};
    $other=@($all|Where-Object { $_.Name -match '(?i)^(python.*|safelog.*|.*mask.*|.*infer.*|robo.*|industrial.*)\.exe$' -and $_.ProcessId -ne SELF_PID }|Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine,@{Name='StartUTC';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}});
    $mem=Get-CimInstance Win32_OperatingSystem;
    @{utc=[DateTime]::UtcNow.ToString('o');self=$own;other_related=$other;free_RAM_bytes=[int64]$mem.FreePhysicalMemory*1024}|ConvertTo-Json -Depth 5"""
    command = command.replace('SELF_PID', str(os.getpid()))
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    value = json.loads(result.stdout)
    current = value['self']; assert current and int(current['ProcessId']) == os.getpid()
    current['Argv'] = windows_argv(current['CommandLine'])
    assert current['Argv'] and current['Argv'][1:] == list(sys.orig_argv[1:])
    assert Path(sys.argv[0]).resolve() == Path(__file__).resolve()
    owned = []; remaining = []
    for other in value['other_related']:
        other['Argv'] = windows_argv(other.get('CommandLine'))
        if same_owned_launcher(other, current, os.getpid(), os.getppid(), sys.executable,
                               sys._base_executable, sys.orig_argv):
            owned.append(other)
        else: remaining.append(other)
    assert len(owned) <= 1
    value['same_invocation_owned_launcher'] = owned
    value['other_related'] = remaining
    assert not remaining, value
    assert value['free_RAM_bytes'] >= 6 * 1024**3, value
    smi = shutil.which('nvidia-smi')
    assert smi, 'Read-only GPU process inventory unavailable'
    gpu = subprocess.check_output([smi, '--query-compute-apps=pid,process_name',
                                   '--format=csv,noheader,nounits'], text=True, encoding='utf-8')
    contexts = [line for line in gpu.splitlines() if line.strip()]
    known_jobs = [line for line in contexts if re.search(
        r'(?i)(python[^\\/]*|safelog[^\\/]*|[^\\/]*mask[^\\/]*|[^\\/]*infer[^\\/]*|robo[^\\/]*|industrial[^\\/]*|torchrun|ollama|llama[^\\/]*)\.exe$',
        line.split(',', 1)[-1].strip()) and not any(line.split(',', 1)[0].strip() == str(p['ProcessId']) for p in owned)]
    assert not known_jobs, 'Another recognized GPU model process exists'
    value['GPU_context_inventory'] = contexts
    value['GPU_inventory_scope'] = 'WDDM can include desktop graphics and inaccessible names; not proof all GPU contexts are idle or a whole C++ process audit'
    locks = [ROOT / 'operation.lock', OUT / 'operation.lock']
    locks += list((ROOT / 'runs').rglob('operation.lock'))
    assert not any(path.exists() for path in locks), 'Another operation lock exists'
    return value

def write_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    assert Path(path).read_bytes() == raw


def inspect_packet(protected=False):
    p = json.loads(POLICY.read_bytes())
    assert p['schema'] == 'safelog-ksdd2-stream-v176-fixed-evaluation-only'
    assert digest(__file__) == p['script_sha256']
    assert list(PACKET) == p['publication']['exact_paths']
    assert digest(REPORT) == p['report_sha256']
    for rel in p['metadata_pins']:
        assert digest(ROOT / rel) == p['pinned_inputs'][rel], rel
    if protected:
        for rel, sha in p['pinned_inputs'].items():
            assert digest(ROOT / rel) == sha, rel
    m = json.loads((ROOT / p['manifest']).read_bytes())
    rows = {phase: sorted([r for r in m['rows'] if r['phase'] == phase], key=lambda r: r['id'])
            for phase in p['phase_counts']}
    counts = {phase: dict(Counter(str(r['label']) for r in selected)) for phase, selected in rows.items()}
    assert counts == p['phase_counts']
    roles = {}
    for row in m['rows']:
        assert row['publisher_split'] == ('test' if row['phase'] == 'test' else 'train')
        assert row['category'] == 'ksdd2_surface' and row['source'] == 'kolektor_sdd2'
        assert row['whole_photo_scope'] == 'publisher_inspected_component_only'
        assert row['mechanical_health'] == 'unknown'
        roles.setdefault(row['group_id'], set()).add(row['phase'])
    assert all(len(v) == 1 for v in roles.values())
    assert len(roles) == 335
    for phase in ('calibration', 'test'):
        selected = rows[phase]
        summary = {
            'photos': len(selected), 'groups': len({r['group_id'] for r in selected}),
            'positive_pixels': sum(r['publisher_mask_pixels'] for r in selected),
            'source_pixels': sum(r['size'][0]*r['size'][1] for r in selected),
            'groups_per_label': {str(label): len({r['group_id'] for r in selected if r['label'] == label}) for label in (0, 1)}
        }
        assert summary == p['fixed_phase_metadata'][phase]
        assert min(counts[phase].values()) >= 20 and min(summary['groups_per_label'].values()) >= 3
        for row in selected:
            assert len(row['masks']) == len(row['mask_sha256']) == 1
            assert bool(row['publisher_mask_pixels']) == bool(row['label'])
            assert row['size'][0]*row['size'][1] <= p['maximum_source_pixels']
            for rel in (row['image'], row['masks'][0]):
                path = (ROOT / rel).resolve()
                assert path.is_relative_to((ROOT / 'data').resolve()) and path.is_file()
    t = json.loads((ROOT / p['training_result']).read_bytes())
    assert t['actual_optimizer_updates'] == 768 and t['selected_epoch'] == 6
    assert t['VAL_improved_over_initial'] and t['CAL_TEST_predictions'] == 0
    assert t['delta_sha256'] == p['pinned_inputs'][p['delta_file']]
    assert t['parent_all_tensors_buffers_exact'] and t['serialized_restore_exact']
    return p, rows


def publication_gate(receipt):
    p, rows = inspect_packet(protected=True)
    r = json.loads(Path(receipt).read_bytes())
    assert r['human_approved_exact_payload'] is True and r['remote_exact_commit_verified'] is True
    assert r['remote_url'] == 'https://github.com/jungsahun43-commits/robo.git'
    assert r['remote_ref'] == 'refs/heads/feature/ai-engine'
    assert r['paths'] == list(PACKET)
    commit = r['published_commit']
    for rel in PACKET:
        raw = subprocess.check_output([GIT, '--no-optional-locks', '-C', str(REPO),
                                       '-c', 'safe.directory=' + REPO.as_posix(),
                                       'show', commit + ':' + rel])
        assert hashlib.sha256(raw).hexdigest() == digest(REPO / rel), rel
    return p, rows, commit


def summarize(counts):
    tp, fp, tn, fn = (counts[k] for k in ('tp', 'fp', 'tn', 'fn'))
    assert tp+fn > 0 and tn+fp > 0
    return {'counts': counts, 'iou': tp/(tp+fp+fn), 'fnr': fn/(tp+fn), 'fpr': fp/(tn+fp)}


def photo_rates(labels, scores, cut):
    c = dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0)
    for label, score in zip(labels, scores):
        c['tp' if label and score > cut else 'fn' if label else 'fp' if score > cut else 'tn'] += 1
    value = summarize(c)
    value.update(positive_photos=c['tp']+c['fn'], negative_photos=c['tn']+c['fp'])
    return value


def add_counts(total, counts):
    for key in total:
        total[key] += counts[key]


def run(receipt):
    p, rows, commit = publication_gate(receipt)
    host = process_preflight()
    assert not RUN.exists(), 'No replay of completed, interrupted, failed or partial evaluation'
    assert shutil.disk_usage(ROOT).free >= p['reserve_bytes']+p['run_budget_bytes']+p['temporary_budget_bytes']
    lock = OUT / 'operation.lock'
    owned = {'pid': os.getpid(), 'token': uuid.uuid4().hex, 'operation': 'ksdd2-fixed-evaluation-v176'}
    write_json(lock, owned)
    completed = {'calibration': 0, 'test': 0}
    started = time.monotonic()
    try:
        RUN.mkdir()
        write_json(RUN / 'start.json', {'host': host, 'published_commit': commit,
                   'protocol_sha256': digest(POLICY), 'receipt_sha256': digest(receipt),
                   'created_UTC': datetime.now(timezone.utc).isoformat(), 'optimizer_updates': 0})
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        from safelog_ai.real_mask_server import RuntimeRealMaskMain
        from safelog_ai.industrial_surface_reference import IndustrialSurfaceMain
        from safelog_ai.magnetic_surface import RealMaskMain
        from safelog_ai.ksdd_supervised import SurfaceMaskHead, ksdd_view, ksdd_photo_score, restore_map, HEIGHT, WIDTH
        from safelog_ai.native_component_views import normalized
        from safelog_ai.belt_segmentation import pixel_counts
        assert RuntimeRealMaskMain.forward_surface is IndustrialSurfaceMain.forward_surface
        assert RuntimeRealMaskMain.forward_real_mask is RealMaskMain.forward_surface
        torch.set_num_threads(4); assert torch.cuda.is_available()
        torch.manual_seed(176176); torch.cuda.manual_seed_all(176176)
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
        old_photo_cut = float(parent['ksdd_operating_point']['photo_threshold'])
        old_pixel_cut = float(parent['equipment_operating_points']['ksdd2_surface']['pixel_threshold'])
        parent_sha = state_sha(model.state_dict())
        delta = torch.load(ROOT / p['delta_file'], map_location='cpu', weights_only=True)
        assert delta['schema'] == 'safelog-ksdd2-stream-v175-development-only'
        assert delta['parent_file'] == p['parent_file']
        assert delta['parent_file_sha256'] == p['pinned_inputs'][p['parent_file']]
        assert delta['protocol_sha256'] == p['pinned_inputs']['reports/industrial-ksdd2-stream-v175-protocol.json']
        assert delta['parent_state_sha256'] == parent_sha
        assert delta['selected_epoch'] == 6 and delta['actual_optimizer_updates'] == 768
        assert delta['CAL_TEST_used'] is False and delta['standalone_app_model'] is False
        assert delta['preserved_sidecars'] == p['preserved_sidecars']
        added = delta['added_state_dict']
        assert state_sha(added) == delta['added_state_sha256']
        assert not set(added).intersection(model.state_dict())
        combined = dict(model.state_dict()); combined.update(added)
        assert state_sha(combined) == delta['combined_state_sha256']
        head = SurfaceMaskHead()
        prefix = 'ksdd_stream_v175.'
        assert all(k.startswith(prefix) for k in added)
        head.load_state_dict({k[len(prefix):]: v for k, v in added.items()}, strict=True)
        head.eval().requires_grad_(False)
        assert state_sha({'ksdd_stream_v175.'+k: v for k, v in head.state_dict().items()}) == delta['added_state_sha256']
        del parent, combined
        model.cuda(); head.cuda()

        def resource_guard():
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            c = MemoryCounters(); c.cb = ctypes.sizeof(c)
            current = ctypes.windll.kernel32.GetCurrentProcess; current.restype = wintypes.HANDLE
            query = ctypes.windll.psapi.GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            assert query(current(), ctypes.byref(c), c.cb)
            assert c.PeakWorkingSetSize <= p['CPU_working_set_budget_bytes']
            assert torch.cuda.max_memory_allocated() <= p['GPU_tensor_budget_bytes']
            assert time.monotonic()-started < p['maximum_seconds']
            assert sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()) <= p['run_budget_bytes']
            assert shutil.disk_usage(ROOT).free > p['reserve_bytes']

        def measured_photo(row, need_parent):
            resource_guard()
            assert digest(ROOT / row['image']) == row['image_sha256']
            assert digest(ROOT / row['masks'][0]) == row['mask_sha256'][0]
            with Image.open(ROOT / row['image']) as source:
                assert list(source.size) == row['size']
                rgb = source.convert('RGB')
            with Image.open(ROOT / row['masks'][0]) as source:
                assert source.size == rgb.size
                truth = torch.from_numpy(np.array(source.convert('L')) > 0)
            assert int(truth.sum()) == row['publisher_mask_pixels']
            assert bool(truth.any()) == bool(row['label'])
            view, _, valid, geometry = ksdd_view(rgb)
            f = model.backbone(normalized(view)[None].cuda())
            low, high = f['low'].half().float(), f['high'].half().float()
            probability = F.interpolate(head(low, high), size=(HEIGHT, WIDTH),
                                        mode='bilinear', align_corners=False)[:, 0].sigmoid()
            score = float(ksdd_photo_score(probability, valid[None].cuda())[0])
            native = restore_map(probability.cpu()[0], geometry)
            assert torch.isfinite(native).all() and np.isfinite(score)
            parent_values = None
            if need_parent:
                previous = F.interpolate(model.ksdd_head(low, high), size=(HEIGHT, WIDTH),
                                         mode='bilinear', align_corners=False)[:, 0].sigmoid()
                old_score = float(ksdd_photo_score(previous, valid[None].cuda())[0])
                original = normalized(rgb.resize((256, 256), Image.Resampling.BICUBIC))[None].cuda()
                old_map = model.forward_equipment(original, 'ksdd2_surface')
                old_map = F.interpolate(old_map[:, None], size=(rgb.height, rgb.width),
                                        mode='bilinear', align_corners=False)[0, 0].cpu()
                assert torch.isfinite(old_map).all() and np.isfinite(old_score)
                parent_values = (old_score, pixel_counts(old_map, truth, torch.ones_like(truth), old_pixel_cut))
            return score, native, truth, parent_values

        cuts = p['pixel_threshold_candidates_CAL_only']
        cal_totals = {str(t): dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0) for t in cuts}
        cal_scores = []; cal_labels = []
        with torch.inference_mode(), (RUN / 'CAL-photos.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            for row in rows['calibration']:
                score, native, truth, _ = measured_photo(row, False)
                for cut in cuts:
                    add_counts(cal_totals[str(cut)], pixel_counts(native, truth, torch.ones_like(truth), cut))
                cal_scores.append(score); cal_labels.append(row['label'])
                completed['calibration'] += 1
                journal.write(json.dumps({'id': row['id'], 'group_id': row['group_id'], 'label': row['label'], 'score': score})+'\n')
                journal.flush()
            os.fsync(journal.fileno())
        assert completed['calibration'] == 490
        # Exactly the already registered v10 CAL minimax rule; no TEST score enters selection.
        labels = np.asarray(cal_labels, dtype=np.int64); scores = np.asarray(cal_scores, dtype=np.float64)
        values = np.unique(scores); candidates = np.r_[np.nextafter(values[0], -np.inf), values, .5]
        order = np.argsort(scores, kind='stable'); sorted_scores = scores[order]; sorted_labels = labels[order]
        prefix_sum = np.r_[0, np.cumsum(sorted_labels)]; k = np.searchsorted(sorted_scores, candidates, side='right')
        n, d = int((labels == 0).sum()), int(labels.sum())
        fn = prefix_sum[k]; fp = n-(k-fn); fpr, fnr = fp/n, fn/d
        chosen = min(range(len(candidates)), key=lambda i: (max(fpr[i], fnr[i]), fpr[i]+fnr[i], abs(candidates[i]-.5), float(candidates[i])))
        photo_cut = float(candidates[chosen])
        cal_pixels = {key: summarize(value) for key, value in cal_totals.items()}
        eligible = [t for t in cuts if cal_pixels[str(t)]['fnr'] <= .15]
        pixel_cut = min(eligible, key=lambda t: (-cal_pixels[str(t)]['iou'], cal_pixels[str(t)]['fnr'], -t)) if eligible else min(
            cuts, key=lambda t: (cal_pixels[str(t)]['fnr'], -cal_pixels[str(t)]['iou'], -t))
        for value in cal_pixels.values():
            c = value['counts']; expected = p['fixed_phase_metadata']['calibration']
            assert sum(c.values()) == expected['source_pixels'] and c['tp']+c['fn'] == expected['positive_pixels']
        calibration_path = RUN / 'calibration.json'
        write_json(calibration_path, {'protocol_sha256': digest(POLICY), 'candidate_delta_sha256': digest(ROOT / p['delta_file']),
                   'parent_state_sha256': parent_sha, 'CAL_photos': completed['calibration'], 'TEST_photos': 0,
                   'point': {'photo_threshold': photo_cut, 'pixel_threshold': pixel_cut}, 'photo_rates': photo_rates(cal_labels, cal_scores, photo_cut),
                   'pixel_candidates_CAL_only': cal_pixels, 'saved_before_any_TEST_prediction': True})
        calibration_sha = digest(calibration_path)
        frozen = json.loads(calibration_path.read_bytes())
        assert frozen['point'] == {'photo_threshold': photo_cut, 'pixel_threshold': pixel_cut}
        assert frozen['protocol_sha256'] == digest(POLICY)
        write_json(RUN / 'TEST-start.json', {'calibration_sha256': calibration_sha, 'candidate_delta_sha256': digest(ROOT / p['delta_file']),
                   'created_UTC': datetime.now(timezone.utc).isoformat(), 'one_fixed_pass_only': True})
        candidate_pixels = dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0)
        parent_pixels = dict.fromkeys(('tp', 'fp', 'tn', 'fn'), 0)
        test_scores = []; parent_scores = []; test_labels = []
        with torch.inference_mode(), (RUN / 'TEST-photos.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            for row in rows['test']:
                assert digest(calibration_path) == calibration_sha
                score, native, truth, previous = measured_photo(row, True)
                value = pixel_counts(native, truth, torch.ones_like(truth), pixel_cut)
                add_counts(candidate_pixels, value); add_counts(parent_pixels, previous[1])
                test_scores.append(score); parent_scores.append(previous[0]); test_labels.append(row['label'])
                completed['test'] += 1
                journal.write(json.dumps({'id': row['id'], 'group_id': row['group_id'], 'label': row['label'],
                                         'candidate_score': score, 'parent_photo_score': previous[0],
                                         'candidate_native_pixel_counts': value, 'parent_native_pixel_counts': previous[1]})+'\n')
                journal.flush()
            os.fsync(journal.fileno())
        assert completed['test'] == 1004
        for c in (candidate_pixels, parent_pixels):
            expected = p['fixed_phase_metadata']['test']
            assert sum(c.values()) == expected['source_pixels'] and c['tp']+c['fn'] == expected['positive_pixels']
        measured = photo_rates(test_labels, test_scores, photo_cut)
        previous = photo_rates(test_labels, parent_scores, old_photo_cut)
        pixels = summarize(candidate_pixels); old_pixels = summarize(parent_pixels)
        assert state_sha(model.state_dict()) == parent_sha
        assert state_sha({'ksdd_stream_v175.'+k: v for k, v in head.state_dict().items()}) == delta['added_state_sha256']
        for rel, sha in p['pinned_inputs'].items():
            assert digest(ROOT / rel) == sha, rel
        groups = p['fixed_phase_metadata']['test']['groups_per_label']
        guards = {
            'each_photo_label_at_least20': min(measured['positive_photos'], measured['negative_photos']) >= 20,
            'each_photo_label_at_least3_proxy_groups': min(groups.values()) >= 3,
            'photo_FPR_at_most20percent': measured['fpr'] <= .20,
            'photo_FNR_at_most20percent': measured['fnr'] <= .20,
            'photo_FPR_no_worse_than_current_parent': measured['fpr'] <= previous['fpr'],
            'photo_FNR_no_worse_than_current_parent': measured['fnr'] <= previous['fnr'],
            'native_pixel_IoU_strictly_better': pixels['iou'] > old_pixels['iou'],
            'native_pixel_FNR_no_worse': pixels['fnr'] <= old_pixels['fnr'],
            'native_pixel_FPR_no_worse': pixels['fpr'] <= old_pixels['fpr'],
            'parent_all_tensors_buffers_file_meta_exact': True,
            'candidate_added_state_exact': True,
            'CAL_saved_and_unchanged_before_one_fixed_TEST': digest(calibration_path) == calibration_sha
        }
        resource_guard()
        result = {'schema': p['schema'], 'protocol_sha256': digest(POLICY), 'candidate_delta_sha256': digest(ROOT / p['delta_file']),
                  'calibration_sha256': calibration_sha, 'candidate_TEST_photo_rates': measured, 'current_parent_TEST_photo_rates': previous,
                  'candidate_TEST_native_pixels': pixels, 'current_parent_TEST_native_pixels': old_pixels, 'guards': guards,
                  'metrics_eligible_for_separate_runtime_validation_only': all(guards.values()),
                  'actual_CAL_photos': completed['calibration'], 'actual_TEST_photos': completed['test'],
                  'candidate_TEST_passes': 1, 'optimizer_updates': 0, 'new_feature_cache_bytes': 0,
                  'other_old_tasks_full_retention_outputs_recomputed': False, 'CPU_GPU_equivalence_checked': False,
                  'actual_CPP_alloldalias_oldPNG_fixedZIP_checked': False, 'independent_field_validated': False,
                  'ready': False, 'deployed': False,
                  'next': 'On metric failure keep parent and close candidate. Only on all metric guards passing, prepare separate exact runtime/old-output/CPU-GPU/C++/alias/PNG/package policy and bounded artifacts; never reuse prior check counts or automatically promote.'}
        write_json(RUN / 'evaluation-result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if RUN.exists():
            write_json(RUN / 'failure.json', {'error_type': type(error).__name__, 'message': str(error),
                       'confirmed_completed_photos': completed, 'optimizer_updates': 0,
                       'no_automatic_resume_or_replay': True, 'partial_artifacts_preserved': True})
        raise
    finally:
        if lock.exists() and json.loads(lock.read_bytes()) == owned:
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--inspect-process', action='store_true')
    parser.add_argument('--inspect-publication', action='store_true')
    parser.add_argument('--publication-receipt')
    args = parser.parse_args()
    if args.inspect_publication:
        if not args.publication_receipt: parser.error('Publication receipt required')
        _, _, commit = publication_gate(args.publication_receipt)
        print(json.dumps({'publication_gate_verified': True, 'commit': commit, 'torch_imported': False, 'new_predictions_optimizer': 0}))
    elif args.inspect_process:
        print(json.dumps({'launcher_units': launcher_unit_checks(), 'host': process_preflight(), 'torch_imported': False, 'new_predictions_optimizer': 0}))
    elif args.inspect:
        launcher_unit_checks()
        p, rows = inspect_packet()
        print(json.dumps({'packet_metadata_verified': True, 'fixed_counts': p['fixed_phase_metadata'],
                          'protected_model_files_rehashed_by_this_inspection': False, 'torch_imported': False, 'new_predictions_optimizer': 0}))
    else:
        if not args.publication_receipt: parser.error('New human-approved exact publication receipt required')
        run(args.publication_receipt)
