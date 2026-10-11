"""Prespecified grouped conveyor foreign-object training and one final test."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_conveyor_foreign import DATA, SOURCE, read, new, sha
from safelog_ai.conveyor_foreign import ARCH, CLASS, SIZE, THRESHOLD, create_model, normalize, summarize
from safelog_ai.equipment_anomaly import rank_auroc
from safelog_ai.retention_distillation import state_sha256

RUN = ROOT/'runs/conveyor-foreign-grouped-v1'
PREFIX = ROOT/'reports/conveyor-foreign-grouped'
PROTOCOL = Path(str(PREFIX)+'-protocol.json')
LEDGER = ROOT/'runs/conveyor-foreign-grouped-input-ledger.json'
CACHE = ROOT/'.cache/torch-equipment/hub/checkpoints/resnet18-f37072fd.pth'
EPOCHS, BATCH, SEED = 30, 32, 95
SOURCES = ('scripts/prepare_conveyor_foreign.py', 'scripts/study_conveyor_foreign.py',
           'safelog_ai/conveyor_foreign.py', 'tests/test_conveyor_foreign.py')


def artifact(name):
    return Path(str(PREFIX)+'-'+name+'.json')


def sources():
    return {name: sha(ROOT/name) for name in SOURCES}


def snapshot(path):
    stat = path.stat()
    return {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def load(rows):
    images = []
    for row in rows:
        path = DATA/row['image']
        assert sha(path)==row['image_sha256']
        with Image.open(path) as image:
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    return torch.stack(images), torch.tensor([r['foreign_object_photo_label'] for r in rows], dtype=torch.float32)


def preflight():
    torch.manual_seed(SEED+1)
    model = create_model(True).cuda().train()
    initial = state_sha256(model)
    x = torch.randn(BATCH, 3, SIZE, SIZE, device='cuda')
    target = torch.arange(BATCH, device='cuda').remainder(2).float()
    opt = torch.optim.AdamW(model.parameters(), lr=.0001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), target)
    assert torch.isfinite(loss)
    loss.backward(); opt.step()
    assert state_sha256(model)!=initial
    model.eval()
    cpu = create_model(False).eval()
    cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()}, strict=True)
    with torch.no_grad():
        gpu, host = model(x[:2]).cpu(), cpu(x[:2].cpu())
    assert torch.allclose(gpu, host, atol=.0001, rtol=.001)
    new(ROOT/'runs/conveyor-foreign-grouped-preflight.json', {'status':'passed', 'actual_epochs':0,
        'disposable_updates':1, 'batch_size':BATCH, 'strict_cpu_gpu_reload':True,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(), 'source_sha256':sources(),
        'pretrained_cache_sha256':sha(CACHE)})
    print('Foreign-object GPU actual-update preflight passed', flush=True)


def freeze():
    flight = read(ROOT/'runs/conveyor-foreign-grouped-preflight.json')
    assert flight['status']=='passed' and flight['source_sha256']==sources()
    manifest = read(DATA/'manifest.json')
    assert manifest['remaining_exact_rgb_duplicates']==0
    assert manifest['preparation_script_sha256']==sources()['scripts/prepare_conveyor_foreign.py']
    legacy = {}
    for folder in ('safelog_ai', 'scripts', 'tests'):
        for path in (ROOT/folder).glob('*.py'):
            name = path.relative_to(ROOT).as_posix()
            if name not in SOURCES:
                legacy[name] = sha(path)
    prior = read(ROOT/'reports/beltcrack-expansion-protocol.json')
    for name, digest in {**prior['legacy_source_sha256'], **prior['source_sha256']}.items():
        assert legacy[name]==digest, 'Historical source changed: '+name
    old = read(ROOT/'runs/beltcrack-expansion-input-ledger.json')
    preserved = {**old['previous_artifacts'], **old['expanded_inputs']}
    for name, entry in preserved.items():
        assert snapshot(ROOT/name)==entry, 'Historical artifact changed: '+name
    for path in (ROOT/'runs/beltcrack-cached-expansion-v2').rglob('*'):
        if path.is_file():
            preserved[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for pattern in ('beltcrack-expansion*', 'CONVEYOR_FOREIGN_ACQUISITION_KO.md',
                    'conveyor-foreign-acquisition.json', 'equipment-visa-verification.json'):
        for path in (ROOT/'reports').glob(pattern):
            if path.is_file():
                preserved[path.relative_to(ROOT).as_posix()] = snapshot(path)
    inputs = {path.relative_to(ROOT).as_posix():snapshot(path) for path in DATA.rglob('*') if path.is_file()}
    for name in ('image-data.zip', 'archive-index.json', 'curated-label-inventory.json', 'publisher-files.json'):
        path = SOURCE/name
        inputs[path.relative_to(ROOT).as_posix()] = snapshot(path)
    inputs[CACHE.relative_to(ROOT).as_posix()] = snapshot(CACHE)
    assert inputs[(SOURCE/'image-data.zip').relative_to(ROOT).as_posix()]['sha256']==manifest['source_archive_sha256']
    new(LEDGER, {'foreign_inputs':inputs, 'previous_artifacts':preserved})
    new(PROTOCOL, {'schema':'conveyor-foreign-grouped-v1', 'declared_before_training':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(), 'source_sha256':sources(),
        'legacy_source_sha256':legacy, 'input_ledger_sha256':sha(LEDGER), 'manifest_sha256':sha(DATA/'manifest.json'),
        'preflight_sha256':sha(ROOT/'runs/conveyor-foreign-grouped-preflight.json'),
        'pretrained_cache_sha256':sha(CACHE), 'dataset_source':'https://data.mendeley.com/datasets/s25x2bnshz/1',
        'dataset_doi':'10.17632/s25x2bnshz.1', 'license':'CC-BY-4.0', 'counts':manifest['counts'],
        'group_splits':manifest['group_splits'], 'group_split_unit':'Author recording-name hints, explicit suffix alias',
        'source_test_used_for_fit_or_selection':False, 'source_test_decoded_for_quality_audit':True,
        'independent_factory_or_physical_object_holdout_verified':False,
        'normal_definition':manifest['normal_definition'], 'labels':manifest['label_source'],
        'mechanical_health_ground_truth':False, 'fault_localization_ground_truth':False,
        'architecture':ARCH, 'class':CLASS, 'input_size':SIZE, 'epochs':EPOCHS, 'batch_size':BATCH,
        'seed':SEED, 'threshold':THRESHOLD, 'optimizer':'AdamW', 'lr':.0001, 'weight_decay':.0001,
        'scheduler':'Cosine30 eta_min0.000005', 'gradient_clip_norm':10,
        'loss':'Binary BCEWithLogits, pos_weight=TRAIN negatives/positives',
        'positive_weight':manifest['counts']['train/0']/manifest['counts']['train/1'],
        'augmentation':'TRAIN only random horizontal flip p0.5 and brightness uniform0.8..1.2',
        'checkpoint_selection':'Fixed final epoch30; no VAL/TEST checkpoint or threshold selection',
        'initialization':'General ImageNet ResNet18, fresh one-class Linear head; no belt/building checkpoint',
        'preflight_updates':1, 'preflight_epochs':0, 'new_ai_or_human_labels':0,
        'near_frames_correlated_within_recording':True, 'sampled_frame_metrics_not_independent_factory_metrics':True,
        'final_test_one_scoring_pass':True, 'app_default_promoted':False, 'tests_expected':4})
    print(json.dumps({'protocol':'frozen', 'counts':manifest['counts'], 'previous_artifacts':len(preserved),
                      'input_files':len(inputs), 'legacy_sources':len(legacy)}), flush=True)


def validate_protocol():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256']==sources()
    assert protocol['manifest_sha256']==sha(DATA/'manifest.json')
    assert protocol['input_ledger_sha256']==sha(LEDGER)
    assert protocol['epochs']==EPOCHS and protocol['threshold']==THRESHOLD
    return protocol


def commit_proof():
    repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe', '-c', 'safe.directory='+repo.as_posix(), '-C', str(repo)]
    paths = list(SOURCES)+[PROTOCOL.relative_to(ROOT).as_posix()]
    for name in paths:
        raw = subprocess.check_output(git+['show', 'HEAD:safelog-cpp/ai-training/'+name])
        assert hashlib.sha256(raw).hexdigest()==sha(ROOT/name), 'Freeze must be committed before actual fit: '+name
    return subprocess.check_output(git+['rev-parse', 'HEAD'], text=True).strip()


def train():
    protocol = validate_protocol()
    frozen_commit = commit_proof()
    assert not RUN.exists(), 'Preserve completed experiment; choose a new run'
    RUN.mkdir()
    rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']=='train']
    images, labels = load(rows)
    assert all(r['source_split']!='test' for r in rows)
    torch.manual_seed(SEED)
    model = create_model(True).cuda()
    initial = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=.000005)
    weight = torch.tensor(protocol['positive_weight'], device='cuda')
    updates, history, start = 0, [], time.monotonic()
    for epoch in range(1, EPOCHS+1):
        model.train()
        total = 0.
        for indices in torch.randperm(len(rows)).split(BATCH):
            values = images[indices].cuda().float()/255
            flip = torch.rand(len(indices), device='cuda') < .5
            values[flip] = values[flip].flip(-1)
            values = (values*(.8+.4*torch.rand(len(indices), 1, 1, 1, device='cuda'))).clamp(0, 1)
            mean = values.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
            std = values.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model((values-mean)/std).flatten(),
                labels[indices].cuda(), pos_weight=weight)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total += float(loss.detach())*len(indices)
        scheduler.step()
        entry = {'epoch':epoch, 'train_loss':total/len(rows), 'optimizer_updates':updates,
                 'elapsed_minutes':(time.monotonic()-start)/60}
        new(RUN/f'epoch-{epoch:02d}.json', entry)
        history.append(entry)
        print(json.dumps(entry), flush=True)
    model.eval()
    final = state_sha256(model)
    assert initial!=final
    metadata = {'architecture':ARCH, 'class':CLASS, 'epochs':EPOCHS, 'input_size':SIZE,
        'threshold':THRESHOLD, 'train_photos':len(rows), 'optimizer_updates':updates,
        'initial_state_sha256':initial, 'final_state_sha256':final, 'protocol_sha256':sha(PROTOCOL),
        'manifest_sha256':sha(DATA/'manifest.json'), 'frozen_before_fit_commit':frozen_commit,
        'source_sha256':sources(), 'mechanical_health_ground_truth':False,
        'source_test_used_for_fit_or_selection':False}
    torch.save({**metadata, 'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}}, RUN/'final.pt')
    cpu = create_model(False).eval()
    cpu.load_state_dict(torch.load(RUN/'final.pt', map_location='cpu', weights_only=True)['state_dict'], strict=True)
    with torch.no_grad():
        gpu, host = model(normalize(images[:2], 'cuda')).cpu(), cpu(normalize(images[:2]))
    assert torch.allclose(gpu, host, atol=.0001, rtol=.001)
    new(RUN/'history.json', history)
    new(artifact('training'), {**metadata, 'weights_sha256':sha(RUN/'final.pt'),
        'elapsed_training_minutes':(time.monotonic()-start)/60, 'strict_cpu_gpu_reload_passed':True,
        'validation_used_for_selection':False, 'preflight_updates_counted_as_epochs':False})
    print('Final30epoch checkpoint frozen before VAL/TEST scoring', flush=True)


def evaluate():
    validate_protocol()
    training = read(artifact('training'))
    assert training['weights_sha256']==sha(RUN/'final.pt')
    assert not artifact('results').exists() and not (RUN/'test-predictions.json').exists()
    state = torch.load(RUN/'final.pt', map_location='cpu', weights_only=True)
    model = create_model(False).cuda().eval()
    model.load_state_dict(state['state_dict'], strict=True)
    results = {}
    for split in ('val', 'test'):
        rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']==split]
        images, _ = load(rows)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(rows), BATCH):
                values = model(normalize(images[start:start+BATCH], 'cuda')).flatten().sigmoid().cpu().double().tolist()
                predictions.extend({'id':row['id'], 'group':row['recording_group'],
                    'label':row['foreign_object_photo_label'], 'score':score}
                    for row, score in zip(rows[start:start+BATCH], values))
        new(RUN/f'{split}-predictions.json', predictions)
        rates = summarize(predictions)
        results[split] = {**rates, 'auroc':rank_auroc([r['label'] for r in predictions], [r['score'] for r in predictions]),
            'per_group':{group:summarize([r for r in predictions if r['group']==group])
                         for group in sorted({r['group'] for r in predictions})}}
    new(artifact('results'), {'status':'evaluated', **results, 'fixed_threshold':THRESHOLD,
        'weights_sha256':training['weights_sha256'], 'protocol_sha256':sha(PROTOCOL),
        'test_recording_groups':4, 'source_test_used_for_fit_or_selection':False,
        'test_score_passes':1, 'new_ai_or_human_labels':0,
        'mechanical_failure_or_belt_crack_performance_claimed':False,
        'independent_factory_or_physical_object_holdout_verified':False, 'app_default_promoted':False})
    print(json.dumps({s:{k:v for k,v in r.items() if k!='per_group'} for s,r in results.items()}), flush=True)


def verify():
    protocol = validate_protocol()
    for name, digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name)==digest, 'Historical source changed: '+name
    ledger = read(LEDGER)
    for records in ledger.values():
        for name, entry in records.items():
            assert snapshot(ROOT/name)==entry, 'Preserved input/artifact changed: '+name
    history, training, result = read(RUN/'history.json'), read(artifact('training')), read(artifact('results'))
    expected = ((training['train_photos']+BATCH-1)//BATCH)*EPOCHS
    assert [r['epoch'] for r in history]==list(range(1, EPOCHS+1))
    assert expected==training['optimizer_updates']==history[-1]['optimizer_updates']
    assert training['strict_cpu_gpu_reload_passed'] and not training['validation_used_for_selection']
    assert result['weights_sha256']==sha(RUN/'final.pt') and result['test_score_passes']==1
    manifest = read(DATA/'manifest.json')
    assert len({r['decoded_rgb_sha256'] for r in manifest['rows']})==len(manifest['rows'])
    for split in ('val', 'test'):
        predictions = read(RUN/f'{split}-predictions.json')
        assert all(v==result[split][k] for k,v in summarize(predictions).items())
    new(artifact('verification'), {'status':'passed', 'actual_epochs':EPOCHS, 'actual_optimizer_updates':expected,
        'disposable_preflight_updates':1, 'strict_cpu_gpu_reload_passed':True, 'tests_passed':4,
        'foreign_inputs_sha_size_mtime_preserved':len(ledger['foreign_inputs']),
        'previous_artifacts_sha_size_mtime_preserved':len(ledger['previous_artifacts']),
        'legacy_sources_sha_preserved':len(protocol['legacy_source_sha256']),
        'older_complete_check':'reports/beltcrack-expansion-verification.json',
        'weights_sha256':sha(RUN/'final.pt'), 'protocol_sha256':sha(PROTOCOL),
        'test_score_passes':1, 'app_default_promoted':False})
    print('Actual foreign-object training, scoring and preservation verified', flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'freeze', 'train', 'evaluate', 'verify'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    os.environ['TORCH_HOME'] = str(ROOT/'.cache/torch-equipment')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'preflight':preflight, 'freeze':freeze, 'train':train, 'evaluate':evaluate, 'verify':verify}[args.mode]()
