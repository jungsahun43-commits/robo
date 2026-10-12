"""One fixed TRAIN expansion experiment; known benchmarks are openly reused."""
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
sys.path.insert(0,str(ROOT))
from prepare_conveyor_expansion import DATA, OLD, SOURCE, read, new, sha
from safelog_ai.conveyor_foreign import ARCH, CLASS, SIZE, THRESHOLD, create_model, normalize, summarize
from safelog_ai.equipment_anomaly import rank_auroc
from safelog_ai.retention_distillation import state_sha256

RUN = ROOT/'runs/conveyor-foreign-expansion-v2'
PREFIX = ROOT/'reports/conveyor-foreign-expansion'
PROTOCOL = Path(str(PREFIX)+'-protocol.json')
LEDGER = ROOT/'runs/conveyor-foreign-expansion-input-ledger.json'
SOURCES = ('scripts/prepare_conveyor_expansion.py','scripts/study_conveyor_expansion.py','tests/test_conveyor_expansion.py')
EPOCHS, BATCH, SEED = 30,32,95


def artifact(name):
    return Path(str(PREFIX)+'-'+name+'.json')


def sources():
    return {name:sha(ROOT/name) for name in SOURCES}


def snapshot(path):
    stat = path.stat()
    return {'sha256':sha(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def load(rows):
    tensors = []
    for row in rows:
        path = DATA/row['image']
        assert sha(path)==row['image_sha256']
        with Image.open(path) as image:
            tensors.append(torch.from_numpy(np.array(image)).permute(2,0,1))
    return torch.stack(tensors),torch.tensor([r['foreign_object_photo_label'] for r in rows],dtype=torch.float32)


def preflight():
    torch.manual_seed(SEED+1)
    model = create_model(True).cuda().train()
    initial = state_sha256(model)
    rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']=='train'][:BATCH]
    images,labels = load(rows)
    opt = torch.optim.AdamW(model.parameters(),lr=.0001)
    value = torch.nn.functional.binary_cross_entropy_with_logits(model(normalize(images,'cuda')).flatten(),labels.cuda())
    assert torch.isfinite(value)
    value.backward();opt.step()
    assert initial!=state_sha256(model)
    model.eval()
    cpu = create_model(False).eval()
    cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
    with torch.no_grad():
        gpu,host = model(normalize(images[:2],'cuda')).cpu(),cpu(normalize(images[:2]))
    assert torch.allclose(gpu,host,atol=.0001,rtol=.001)
    new(ROOT/'runs/conveyor-foreign-expansion-preflight.json',{'status':'passed','actual_epochs':0,
        'disposable_optimizer_updates':1,'strict_cpu_gpu_reload':True,'source_sha256':sources(),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'preflight_photos_split':'train'})
    print('TRAIN-only GPU update and CPU reload passed',flush=True)


def freeze():
    flight = read(ROOT/'runs/conveyor-foreign-expansion-preflight.json')
    assert flight['status']=='passed' and flight['source_sha256']==sources()
    manifest = read(DATA/'manifest.json')
    assert manifest['original_val_test_byte_identical'] and not manifest['source_test_used_for_fit_or_selection']
    old_protocol = read(ROOT/'reports/conveyor-foreign-grouped-protocol.json')
    legacy = {name:sha(path) for folder in ('scripts','safelog_ai','tests')
              for path in (ROOT/folder).glob('*.py') if (name:=path.relative_to(ROOT).as_posix()) not in SOURCES}
    for name,digest in {**old_protocol['legacy_source_sha256'],**old_protocol['source_sha256']}.items():
        assert legacy[name]==digest,'Old source changed: '+name
    old_ledger = read(ROOT/'runs/conveyor-foreign-grouped-input-ledger.json')
    preserved = {**old_ledger['previous_artifacts'],**old_ledger['foreign_inputs']}
    for i,(name,record) in enumerate(preserved.items()):
        assert snapshot(ROOT/name)==record,'Old input changed: '+name
        if i%4000==0:
            print(json.dumps({'prior_artifacts_checked':i,'total':len(preserved)}),flush=True)
    for path in (ROOT/'runs/conveyor-foreign-grouped-v1').rglob('*'):
        if path.is_file():
            preserved[path.relative_to(ROOT).as_posix()] = snapshot(path)
    for pattern in ('conveyor-foreign-grouped*','conveyor-dual-handoff*','CONVEYOR_FOREIGN_RESULTS_KO.md','CONVEYOR_FOREIGN_ATTRIBUTION_KO.md'):
        for path in (ROOT/'reports').glob(pattern):
            preserved[path.relative_to(ROOT).as_posix()] = snapshot(path)
    inputs = {p.relative_to(ROOT).as_posix():snapshot(p) for p in DATA.rglob('*') if p.is_file()}
    new(LEDGER,{'expanded_inputs':inputs,'previous_artifacts':preserved})
    baseline_training = read(ROOT/'reports/conveyor-foreign-grouped-training.json')
    new(PROTOCOL,{'schema':'conveyor-foreign-expansion-v2','declared_before_training':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),'legacy_source_sha256':legacy,
        'manifest_sha256':sha(DATA/'manifest.json'),'input_ledger_sha256':sha(LEDGER),
        'preflight_sha256':sha(ROOT/'runs/conveyor-foreign-expansion-preflight.json'),
        'counts':manifest['counts'],'group_splits':manifest['group_splits'],'architecture':ARCH,'class':CLASS,
        'input_size':SIZE,'epochs':EPOCHS,'batch_size':BATCH,'seed':SEED,'threshold':THRESHOLD,
        'optimizer':'AdamW','lr':.0001,'weight_decay':.0001,'scheduler':'Cosine30 eta_min0.000005',
        'gradient_clip_norm':10,'positive_weight':manifest['counts']['train/0']/manifest['counts']['train/1'],
        'loss':'Weighted BCE, TRAIN negative/positive ratio',
        'augmentation':'Same as v1: TRAIN horizontal flip p0.5 and brightness uniform0.8..1.2',
        'initialization':'Same general ImageNet ResNet18 and seed95; fresh head; no v1 warmstart',
        'expected_initial_state_sha256':baseline_training['initial_state_sha256'],
        'baseline_weights_sha256':baseline_training['weights_sha256'],
        'baseline_results_sha256':sha(ROOT/'reports/conveyor-foreign-grouped-results.json'),
        'checkpoint_selection':'Final30 fixed before new scores; no selection of epoch or threshold',
        'main_change':'More original frames from the same TRAIN recordings, with v1 TRAIN frames retained',
        'same_epochs_but_equal_optimizer_update_budget':False,
        'known_val_errors_motivated_this_experiment':True,'old_val_and_test_are_reused_known_benchmarks':True,
        'test_is_fresh_unseen_independent_evaluation':False,'source_test_used_for_fit':False,
        'additional_validation':'Previously unscored frames from the SAME3 VAL recording groups',
        'independent_factory_or_object_holdout_verified':False,'mechanical_health_ground_truth':False,
        'normal_definition':manifest['normal_definition'],'new_ai_or_human_labels':0,
        'acceptance_reference':'CommonVAL FPR no worse, FNR strictly lower; report all benchmarks and per-group regressions',
        'additional_validation_fpr_fnr_target':.05,'app_default_promoted':False,'tests_expected':4,
        'preflight_updates':1,'preflight_epochs':0})
    print(json.dumps({'protocol':'frozen','sources_preserved':len(legacy),'input_files':len(inputs),'counts':manifest['counts']}),flush=True)


def validate():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256']==sources() and protocol['manifest_sha256']==sha(DATA/'manifest.json')
    assert protocol['input_ledger_sha256']==sha(LEDGER) and protocol['threshold']==.5
    assert sha(ROOT/'reports/conveyor-foreign-grouped-results.json')==protocol['baseline_results_sha256']
    return protocol


def commit_proof():
    repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        raw = subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])
        assert hashlib.sha256(raw).hexdigest()==sha(ROOT/name), 'Commit freeze before fit: '+name
    return subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()


def train():
    protocol,commit = validate(),commit_proof()
    assert not RUN.exists(),'Preserve completed experiments'
    RUN.mkdir()
    rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']=='train']
    images,labels = load(rows)
    torch.manual_seed(SEED)
    model = create_model(True).cuda()
    initial = state_sha256(model)
    assert initial==protocol['expected_initial_state_sha256']
    opt = torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.000005)
    weight = torch.tensor(protocol['positive_weight'],device='cuda')
    start,updates,history = time.monotonic(),0,[]
    for epoch in range(1,EPOCHS+1):
        model.train(); total = 0.
        for indices in torch.randperm(len(rows)).split(BATCH):
            values = images[indices].cuda().float()/255
            flipped = torch.rand(len(indices),device='cuda')<.5
            values[flipped] = values[flipped].flip(-1)
            values = (values*(.8+.4*torch.rand(len(indices),1,1,1,device='cuda'))).clamp(0,1)
            mean = values.new_tensor([.485,.456,.406]).view(1,3,1,1)
            std = values.new_tensor([.229,.224,.225]).view(1,3,1,1)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model((values-mean)/std).flatten(),labels[indices].cuda(),pos_weight=weight)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
            updates += 1;total += float(loss.detach())*len(indices)
        scheduler.step()
        entry = {'epoch':epoch,'train_loss':total/len(rows),'optimizer_updates':updates,'elapsed_minutes':(time.monotonic()-start)/60}
        new(RUN/f'epoch-{epoch:02d}.json',entry);history.append(entry)
        print(json.dumps(entry),flush=True)
    model.eval()
    final = state_sha256(model)
    assert initial!=final
    metadata = {'architecture':ARCH,'class':CLASS,'epochs':EPOCHS,'input_size':SIZE,'threshold':THRESHOLD,
        'train_photos':len(rows),'optimizer_updates':updates,'initial_state_sha256':initial,'final_state_sha256':final,
        'manifest_sha256':sha(DATA/'manifest.json'),'protocol_sha256':sha(PROTOCOL),'source_sha256':sources(),
        'frozen_before_fit_commit':commit,'same_initial_state_as_v1':True,
        'source_test_used_for_fit':False,'mechanical_health_ground_truth':False}
    torch.save({**metadata,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},RUN/'final.pt')
    cpu = create_model(False).eval()
    cpu.load_state_dict(torch.load(RUN/'final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    with torch.no_grad():
        gpu,host = model(normalize(images[:2],'cuda')).cpu(),cpu(normalize(images[:2]))
    assert torch.allclose(gpu,host,atol=.0001,rtol=.001)
    new(RUN/'history.json',history)
    new(artifact('training'),{**metadata,'weights_sha256':sha(RUN/'final.pt'),
        'elapsed_training_minutes':(time.monotonic()-start)/60,'strict_cpu_gpu_reload_passed':True,
        'validation_used_for_checkpoint_or_threshold_selection':False})
    print('Final30 model saved before any new benchmark or audit scores',flush=True)


def evaluate():
    validate()
    training = read(artifact('training'))
    assert training['weights_sha256']==sha(RUN/'final.pt') and not artifact('results').exists()
    model = create_model(False).cuda().eval()
    model.load_state_dict(torch.load(RUN/'final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    result,changes = {},{}
    for phase in ('val','audit_val','test'):
        rows = [r for r in read(DATA/'manifest.json')['rows'] if r['split']==phase]
        images,_ = load(rows)
        observations = []
        with torch.inference_mode():
            for start in range(0,len(rows),BATCH):
                scores = model(normalize(images[start:start+BATCH],'cuda')).flatten().sigmoid().cpu().double().tolist()
                observations.extend({'id':r['id'],'group':r['recording_group'],'label':r['foreign_object_photo_label'],'score':score}
                                    for r,score in zip(rows[start:start+BATCH],scores))
        new(RUN/f'{phase}-predictions.json',observations)
        result[phase] = {**summarize(observations),'auroc':rank_auroc([r['label'] for r in observations],[r['score'] for r in observations]),
            'per_group':{g:summarize([r for r in observations if r['group']==g]) for g in sorted({r['group'] for r in observations})}}
        if phase in ('val','test'):
            previous = {r['id']:r for r in read(ROOT/f'runs/conveyor-foreign-grouped-v1/{phase}-predictions.json')}
            assert set(previous)=={r['id'] for r in observations}
            changes[phase] = [{'id':r['id'],'group':r['group'],'label':r['label'],'old_score':previous[r['id']]['score'],'new_score':r['score'],
                'change':'corrected' if (r['score']>.5)==bool(r['label']) else 'new_error'}
                for r in observations if (r['score']>.5)!=(previous[r['id']]['score']>.5)]
    baseline = read(ROOT/'reports/conveyor-foreign-grouped-results.json')
    acceptance = result['val']['fpr']<=baseline['val']['fpr'] and result['val']['fnr']<baseline['val']['fnr']
    new(artifact('results'),{'status':'evaluated',**result,'common_baseline':{s:baseline[s] for s in ('val','test')},
        'changed_classifications':changes,'common_val_reference_passed':acceptance,
        'extra_val_fpr_fnr_below_5_percent':result['audit_val']['photo_fpr_fnr_below_5_percent'],
        'fixed_threshold':THRESHOLD,'weights_sha256':training['weights_sha256'],'protocol_sha256':sha(PROTOCOL),
        'source_test_used_for_fit':False,'previous_test_reused_as_known_benchmark':True,
        'extra_val_is_independent_factory_or_recording':False,'new_ai_or_human_labels':0,'app_default_promoted':False})
    print(json.dumps({s:{k:v for k,v in r.items() if k!='per_group'} for s,r in result.items()}),flush=True)


def verify():
    protocol = validate()
    for name,digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name)==digest,'Changed legacy source: '+name
    ledger = read(LEDGER)
    for records in ledger.values():
        for i,(name,record) in enumerate(records.items()):
            assert snapshot(ROOT/name)==record,'Changed input or old artifact: '+name
            if i%5000==0:
                print(json.dumps({'preservation_checked':i,'total':len(records)}),flush=True)
    training,history,result = read(artifact('training')),read(RUN/'history.json'),read(artifact('results'))
    assert [r['epoch'] for r in history]==list(range(1,31))
    expected = ((training['train_photos']+BATCH-1)//BATCH)*EPOCHS
    assert training['optimizer_updates']==history[-1]['optimizer_updates']==expected
    assert training['same_initial_state_as_v1'] and training['strict_cpu_gpu_reload_passed']
    assert result['weights_sha256']==sha(RUN/'final.pt')
    for phase in ('val','test','audit_val'):
        observations = read(RUN/f'{phase}-predictions.json')
        assert all(result[phase][k]==v for k,v in summarize(observations).items())
    new(artifact('verification'),{'status':'passed','actual_epochs':EPOCHS,'actual_optimizer_updates':expected,
        'tests_passed':4,'disposable_preflight_updates':1,'strict_cpu_gpu_reload_passed':True,'same_initial_state_as_v1':True,
        'previous_artifacts_sha_size_mtime_preserved':len(ledger['previous_artifacts']),
        'new_inputs_sha_size_mtime_preserved':len(ledger['expanded_inputs']),
        'legacy_sources_sha_preserved':len(protocol['legacy_source_sha256']),'original_val_and_test_byte_identical':True,
        'baseline_results_unchanged':True,'old_test_known_and_reused':True,'app_default_promoted':False,
        'weights_sha256':sha(RUN/'final.pt'),'protocol_sha256':sha(PROTOCOL)})
    print('Actual expansion training, common benchmarks and source/input preservation verified',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','freeze','train','evaluate','verify'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    os.environ['TORCH_HOME'] = str(ROOT/'.cache/torch-equipment')
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    {'preflight':preflight,'freeze':freeze,'train':train,'evaluate':evaluate,'verify':verify}[args.mode]()
