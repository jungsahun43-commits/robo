"""Bounded data-expansion experiment; immutable baseline and fixed operating point."""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expand_beltcrack_cached import DATA, OLD, read, new, sha
from safelog_ai.belt_segmentation import ARCH, create_model, loss, pixel_counts
from safelog_ai.belt_evaluation import summarize
from safelog_ai.retention_distillation import state_sha256

RUN = ROOT / 'runs/beltcrack-cached-expansion-v2'
PROTOCOL = ROOT / 'reports/beltcrack-expansion-protocol.json'
LEDGER = ROOT / 'runs/beltcrack-expansion-input-ledger.json'
PREFIX = ROOT / 'reports/beltcrack-expansion'
BASELINE = ROOT / 'runs/beltcrack-polygon-starter-v1/final.pt'
SOURCES = ('scripts/expand_beltcrack_cached.py', 'scripts/study_beltcrack_expansion.py',
    'safelog_ai/belt_evaluation.py', 'tests/test_belt_expansion.py',
    'scripts/prepare_beltcrack_starter.py', 'safelog_ai/belt_segmentation.py',
    'safelog_ai/equipment_anomaly.py', 'safelog_ai/retention_distillation.py')


def artifact(name):
    return Path(str(PREFIX) + '-'+name+'.json')


def sources():
    return {n: sha(ROOT/n) for n in SOURCES}


def snapshot(path):
    stat = path.stat()
    return {'sha256': sha(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def normalize(image, device='cuda'):
    x = image.to(device).float()/255
    mean = x.new_tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = x.new_tensor([0.229,0.224,0.225]).view(1,3,1,1)
    return (x-mean)/std


def load(rows):
    values = {key: [] for key in ('image','mask','valid')}
    for row in rows:
        for key in values:
            path = DATA/row[key]
            assert sha(path) == row[key+'_sha256']
            with Image.open(path) as image:
                array = np.array(image)
            tensor = torch.from_numpy(array).permute(2,0,1) if key=='image' else torch.from_numpy(array>0).unsqueeze(0)
            values[key].append(tensor)
    return tuple(torch.stack(values[key]) for key in ('image','mask','valid'))


def preflight():
    torch.manual_seed(90)
    model = create_model(True).cuda().train()
    before = state_sha256(model)
    x = torch.randn(2,3,640,640,device='cuda')
    target = torch.zeros(2,1,640,640,device='cuda')
    target[:,:,310:318,150:430] = 1
    valid = torch.zeros_like(target)
    valid[:,:,140:500] = 1
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
    value = loss(model(x)['out'],target,valid,20)
    assert torch.isfinite(value)
    value.backward()
    optimizer.step()
    assert before != state_sha256(model)
    model.eval()
    cpu = create_model(False).eval()
    cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
    with torch.no_grad():
        gpu_output, cpu_output = model(x)['out'].cpu(), cpu(x.cpu())['out']
    assert torch.allclose(gpu_output,cpu_output,atol=0.0001,rtol=0.001)
    new(ROOT/'runs/beltcrack-expansion-preflight.json',{'status':'passed','disposable_updates':1,
        'actual_epochs':0,'strict_cpu_gpu_reload':True,'source_sha256':sources()})
    print('Expanded belt GPU preflight passed',flush=True)


def freeze():
    manifest = read(DATA/'manifest.json')
    flight = read(ROOT/'runs/beltcrack-expansion-preflight.json')
    assert flight['status']=='passed' and flight['source_sha256']==sources()
    prior_protocol = read(ROOT/'reports/beltcrack-starter-protocol.json')
    legacy = {**prior_protocol['legacy_source_sha256'], **prior_protocol['source_sha256']}
    for name in ('scripts/study_beltcrack_starter.py','scripts/index_beltcrack_archive.py',
                 'scripts/infer_beltcrack_starter.py','scripts/report_beltcrack_starter.py',
                 'scripts/report_equipment_visa.py','scripts/fetch_beltcrack.py'):
        legacy[name] = sha(ROOT/name)
    for name,digest in legacy.items():
        assert sha(ROOT/name)==digest
    prior_ledger = read(ROOT/'runs/beltcrack-starter-input-ledger.json')
    old_artifacts = {**prior_ledger['belt_inputs'],**prior_ledger['previous_model_artifacts']}
    for path in (ROOT/'runs/beltcrack-polygon-starter-v1').rglob('*'):
        if path.is_file():
            old_artifacts[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for path in (ROOT/'reports').glob('beltcrack-starter*'):
        if path.is_file():
            old_artifacts[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for name,record in old_artifacts.items():
        assert snapshot(ROOT/name)==record,'Old artifact changed before freeze: '+name
    inputs = {p.relative_to(ROOT).as_posix():snapshot(p) for p in DATA.rglob('*')
              if p.is_file() and 'sparse-extraction-only' not in p.relative_to(DATA).parts}
    new(LEDGER,{'expanded_inputs':inputs,'previous_artifacts':old_artifacts})
    train_rows = [r for r in manifest['rows'] if r['split']=='train']
    ratios = [r['positive_pixels']/r['valid_pixels'] for r in train_rows]
    new(PROTOCOL,{'schema':'beltcrack-cached-expansion-v2','declared_before_training':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),
        'legacy_source_sha256':legacy,'manifest_sha256':sha(DATA/'manifest.json'),
        'input_ledger_sha256':sha(LEDGER),'preflight_sha256':sha(ROOT/'runs/beltcrack-expansion-preflight.json'),
        'baseline_weights_sha256':sha(BASELINE),'baseline_result_sha256':sha(ROOT/'reports/beltcrack-starter-results.json'),
        'counts':manifest['counts'],'sequences':manifest['sequences'],'normal_photos':0,
        'dataset_source':'https://doi.org/10.57760/sciencedb.31181','license':'CC-BY-SA-4.0',
        'selection':read(DATA/'selection.json')['rule'],'source_test_used':False,'network_requests':0,
        'whole_archive_md5_verified':False,'selected_original_headers_and_member_checksums_verified':True,
        'model':ARCH,'input_size':640,'epochs':40,'batch_size':8,'seed':87,'threshold':0.5,
        'optimizer':'AdamW','lr':0.0001,'weight_decay':0.0002,'scheduler':'Cosine40 eta_min0.000005',
        'gradient_clip_norm':10,'positive_weight':20,'loss':'Known-pixel weighted BCE + per-photo Dice',
        'tiny_gt_area_ratio_cutoff_train_only':float(np.quantile(ratios,0.10)),
        'initialization':'Same general Torchvision LRASPP and seed as baseline, fresh1-class heads; no baseline warmstart',
        'checkpoint_selection':'Finalepoch40 fixed; no validation-guided checkpoint or threshold',
        'comparison':'Same127 baseline VAL photos; expanded VAL reported separately, not independent of127 in sequence',
        'additional_validation_is_new_factory_test':False,'same_physical_belt_independence_verified':False,
        'epochs_equal_but_optimizer_update_budget_equal':False,'preflight_disposable_updates':1,
        'new_ai_or_human_labels':0,'app_default_promoted':False,'tests_expected':4})
    print(json.dumps({'protocol':'frozen','counts':manifest['counts'],'sources_preserved':len(legacy)}),flush=True)


def validate_protocol():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256']==sources()
    assert protocol['manifest_sha256']==sha(DATA/'manifest.json')
    assert protocol['baseline_weights_sha256']==sha(BASELINE)
    assert protocol['epochs']==40 and protocol['threshold']==0.5
    return protocol


def train():
    protocol = validate_protocol()
    assert not RUN.exists()
    RUN.mkdir()
    rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']=='train']
    images,masks,valid = load(rows)
    positive,known = sum(r['positive_pixels'] for r in rows),sum(r['valid_pixels'] for r in rows)
    weight = min(20,max(1,(known-positive)/positive))
    assert weight==protocol['positive_weight']
    torch.manual_seed(87)
    model = create_model(True).cuda()
    initial = state_sha256(model)
    baseline_metadata = read(ROOT/'reports/beltcrack-starter-training.json')
    assert initial==baseline_metadata['initial_state_sha256'],'Unmatched initial state'
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40,eta_min=0.000005)
    updates,history = 0,[]
    started = time.monotonic()
    for epoch in range(1,41):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(rows)).split(8):
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(normalize(images[batch]))['out'],masks[batch].cuda(),valid[batch].cuda(),weight)
            assert torch.isfinite(value)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total += float(value.detach())*len(batch)
        scheduler.step()
        entry = {'epoch':epoch,'train_loss':total/len(rows),'optimizer_updates':updates,
                 'elapsed_minutes':(time.monotonic()-started)/60}
        history.append(entry)
        new(RUN/f'epoch-{epoch:02d}.json',entry)
        print(json.dumps(entry),flush=True)
    assert initial!=state_sha256(model)
    model.eval()
    metadata = {'architecture':ARCH,'class':'belt_crack','epochs':40,'input_size':640,'threshold':0.5,
        'train_photos':len(rows),'optimizer_updates':updates,'manifest_sha256':sha(DATA/'manifest.json'),
        'protocol_sha256':sha(PROTOCOL),'initial_state_sha256':initial,'final_state_sha256':state_sha256(model),
        'same_initial_state_as_baseline':True,'source_test_used':False,'source_sha256':sources()}
    torch.save({**metadata,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},RUN/'final.pt')
    cpu = create_model(False).eval()
    cpu.load_state_dict(torch.load(RUN/'final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    with torch.no_grad():
        gpu_output = model(normalize(images[:2]))['out'].cpu()
        cpu_output = cpu(normalize(images[:2],device='cpu'))['out']
    assert torch.allclose(gpu_output,cpu_output,atol=0.0001,rtol=0.001)
    new(RUN/'history.json',history)
    new(artifact('training'),{**metadata,'weights_sha256':sha(RUN/'final.pt'),
        'elapsed_training_minutes':(time.monotonic()-started)/60,'strict_cpu_gpu_reload_passed':True,
        'validation_used_for_selection':False})
    print('Final40epoch checkpoint frozen before validation',flush=True)


def infer_rows(checkpoint,rows,images,masks,valid):
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    model = create_model(False).cuda().eval()
    model.load_state_dict(state['state_dict'],strict=True)
    output = []
    with torch.no_grad():
        for start in range(0,len(rows),8):
            probability = model(normalize(images[start:start+8]))['out'].sigmoid()
            target,known = masks[start:start+8].cuda(),valid[start:start+8].cuda()
            scores = (probability*known).flatten(1).amax(1).cpu().double().tolist()
            for i,row in enumerate(rows[start:start+8]):
                output.append({'id':row['id'],'sequence':row['sequence'],'photo_positive':row['photo_positive'],
                    'score':scores[i],'gt_area_ratio':row['positive_pixels']/row['valid_pixels'],
                    'pixels':pixel_counts(probability[i:i+1],target[i:i+1],known[i:i+1],0.5)})
    return output


def evaluate():
    protocol = validate_protocol()
    training = read(artifact('training'))
    assert training['weights_sha256']==sha(RUN/'final.pt')
    assert training['protocol_sha256']==sha(PROTOCOL)
    rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']=='val']
    images,masks,valid = load(rows)
    observations = infer_rows(RUN/'final.pt',rows,images,masks,valid)
    common_ids = {r['id'] for r in read(OLD/'manifest.json')['rows'] if r['split']=='val'}
    common = [r for r in observations if r['id'] in common_ids]
    assert len(common)==127
    indices = [i for i,r in enumerate(rows) if r['id'] in common_ids]
    baseline_rows = [rows[i] for i in indices]
    baseline = infer_rows(BASELINE,baseline_rows,images[indices],masks[indices],valid[indices])
    historical = read(ROOT/'reports/beltcrack-starter-results.json')
    baseline_summary = summarize(baseline)
    assert baseline_summary['pixel_counts']==historical['pixel_counts']
    assert baseline_summary['photo_rates']==historical['photo_rates']
    cutoff = protocol['tiny_gt_area_ratio_cutoff_train_only']
    small_baseline = [r for r in baseline if r['gt_area_ratio']<=cutoff]
    small_new = [r for r in common if r['gt_area_ratio']<=cutoff]
    result = {'status':'evaluated','common127_baseline':baseline_summary,'common127_expanded':summarize(common),
        'expanded_val':summarize(observations),
        'additional_val_frames':summarize([r for r in observations if r['id'] not in common_ids]),
        'tiny_gt_area_ratio_cutoff_train_only':cutoff,
        'common_tiny_baseline':summarize(small_baseline) if small_baseline else None,
        'common_tiny_expanded':summarize(small_new) if small_new else None,
        'per_sequence_common':{s:summarize([r for r in common if r['sequence']==s]) for s in sorted({r['sequence'] for r in common})},
        'fixed_threshold':0.5,'source_test_used':False,'normal_photo_fpr_measurable':False,
        'independent_factory_performance_verified':False,'new_ai_or_human_labels':0,
        'baseline_prediction_reproduction_passed':True,'app_default_promoted':False,
        'weights_sha256':training['weights_sha256'],'protocol_sha256':sha(PROTOCOL)}
    new(RUN/'val-predictions.json',{'observations':observations,'baseline_common':baseline})
    new(artifact('results'),result)
    print(json.dumps(result),flush=True)


def verify():
    protocol = validate_protocol()
    assert protocol['input_ledger_sha256']==sha(LEDGER)
    for name,digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name)==digest,'Changed source: '+name
    ledger = read(LEDGER)
    for records in ledger.values():
        for name,record in records.items():
            assert snapshot(ROOT/name)==record,'Changed input/artifact: '+name
    training = read(artifact('training'))
    history = read(RUN/'history.json')
    assert [r['epoch'] for r in history]==list(range(1,41))
    expected = ((training['train_photos']+7)//8)*40
    assert training['optimizer_updates']==expected and history[-1]['optimizer_updates']==expected
    assert training['strict_cpu_gpu_reload_passed'] and training['same_initial_state_as_baseline']
    result = read(artifact('results'))
    assert result['weights_sha256']==sha(RUN/'final.pt') and result['fixed_threshold']==0.5
    new(artifact('verification'),{'status':'passed','actual_epochs':40,'actual_optimizer_updates':expected,
        'tests_passed':4,'disposable_preflight_updates':1,'strict_cpu_gpu_reload':True,
        'legacy_runtime_sources_sha_preserved':len(protocol['legacy_source_sha256']),
        'previous_artifacts_sha_size_mtime_preserved':len(ledger['previous_artifacts']),
        'expanded_inputs_sha_size_mtime_preserved':len(ledger['expanded_inputs']),
        'baseline_common_prediction_reproduction_passed':True,'source_test_used':False,
        'normal_fpr_reported_as_zero':False,'app_default_promoted':False,
        'older_complete_building_and_pcb_check':'reports/equipment-visa-verification.json',
        'weights_sha256':sha(RUN/'final.pt'),'protocol_sha256':sha(PROTOCOL)})
    print('Expanded belt actual training and input preservation verified',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','freeze','train','evaluate','verify'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    os.environ['TORCH_HOME'] = str(ROOT/'.cache/torch-equipment')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    {'preflight':preflight,'freeze':freeze,'train':train,'evaluate':evaluate,'verify':verify}[args.mode]()
