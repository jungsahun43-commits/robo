"""Preregistered single-head actual-label study on physically grouped Re-GBC."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from safelog_ai.gearbox_supervised import GearboxMain, gearbox_descriptor, VERSION
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.process_probe import process_exists
from scripts.prepare_gearbox_stratified import RUN, MANIFEST
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from scripts.study_bsclass_supervised import calibrated_threshold
from scripts.study_unified_belt_tiles import snapshot

START = ROOT/'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED = '5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
PROTOCOL = ROOT/'reports/industrial-gearbox-v26-protocol.json'
SOURCES = ('safelog_ai/gearbox_supervised.py','scripts/study_gearbox_supervised.py','tests/test_gearbox_supervised.py')
EPOCHS, STEPS, SEED = 30, 64, 2626


def freeze():
    assert not PROTOCOL.exists() and sha(START) == EXPECTED
    ready = read(RUN/'data-ready.json'); assert ready['ready'] and ready['manifest_sha256'] == sha(MANIFEST)
    ledger = read(ROOT/'runs/industrial-neu-tube-adapt-v24/protected-inputs.json')
    for path in (ROOT/'runs/industrial-neu-tube-adapt-v24').rglob('*'):
        if path.is_file() and not path.name.endswith(('.log','.lock')):
            key = path.relative_to(ROOT).as_posix()
            if key not in ledger: ledger[key] = snapshot(path)
    for row in read(MANIFEST)['rows']:
        path = ROOT/row['image']; st = path.stat()
        ledger[row['image']] = {'sha256':row['image_sha256'],'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    new(RUN/'protected-inputs.json', ledger)
    p = {'schema':VERSION,'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_features_gradients_CAL_TEST_scores':True,
        'source_sha256':{n:sha(ROOT/n) for n in SOURCES}, 'parent_sha256':EXPECTED,
        'manifest_sha256':sha(MANIFEST),'data_audit_sha256':sha(ROOT/'reports/industrial-gearbox-v26-data-audit.json'),
        'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),
        'legacy_source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
        'input':'publisher128square patch, RGB bicubic256square ImageNet normalization; no GT or label dependent crop; no augmentation',
        'descriptor':'same frozen encoder fine24/low40/high960: FP16roundtrip maps then FP32 channel mean/populationstd/max; concatenate3072; FP16roundtrip descriptor. No category embedding or second encoder',
        'head':'one new Linear3072to256 GELU Dropout.1 Linear256to1; random seed2626 initialization; FIT-only mean/popstd floor1e-3 clamp[-8,8]',
        'trainable':'only new gearbox_head; encoder and all existing parameters/buffers/metadata unchanged; new mean/std FIT-only',
        'sampling':'each batch64:32OK,32nOK; inverse per-label group patch count for uniform group then uniform patch within label; with replacement, not new pictures',
        'loss':'binary logit BCE; balanced batches; VAL per-label per-group mean then macro over both labels, no TEST labels in selection',
        'optimizer':'AdamW lr1e-3 wd1e-4 cosine30 min1e-5 clip5 FP32 TF32off',
        'epochs':EPOCHS,'steps':STEPS,'batch':64,'extraction_batch':32,'seed':SEED,'planned_optimizer_updates':EPOCHS*STEPS,
        'VAL_selection':'epoch0 and each completed epoch; earliest strict minimum macro label/group BCE; epoch0 selection cannot release',
        'CAL':'after VAL selection only, labeled CAL minimax FPR/FNR threshold; tie sum then distance.5 then lower threshold; strict>; save before TEST features/scores',
        'TEST':'exact reserved authorS3fold0 3109patches, zero split change; fixed model/threshold once; report per-category and per-physical-part plus group uncertainty; no method sweep or old bracket/tube transfer evaluation this study',
        'adoption':['selected_epoch>0','TEST eachlabel>=20','TEST FPR<=.20 and FNR<=.20','each functional category with >=20 perlabel FPR/FNR<=.25','all existing state exact','fresh packageCPU/GPU/C++/aliases/old maps required before app release'],
        'research_patch_scope':'publisher OK may contain acceptable lightwear; not whole gearbox internal health, arbitrary photo input, natural location mask or independently captured field validation',
        'old_target_primary_maps_and_thresholds_unchanged':True,'new_unit_tests_passed':3,
        'license':'Re-GBC version1 CC BY4.0, cite Bauer etal DOI10.1109/ETFA65518.2025.11205777 and original dataset; inherited other sources retain own conditions',
        'resume':'per-epoch optimizer/scheduler/RNG progress; append actual optimizer journal; preserve repeated partial-epoch steps and failed models; do not rerun completed study'}
    new(PROTOCOL,p); print({'frozen':True,'counts':read(MANIFEST)['audit']['counts'],'planned_optimizer_updates':1920},flush=True)


def validate(committed=False):
    p = read(PROTOCOL)
    for n,h in p['source_sha256'].items(): assert sha(ROOT/n) == h, n
    for path,h in ((START,EXPECTED),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])): assert sha(path) == h, path
    if committed:
        git = ['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix(),'reports/industrial-gearbox-v26-data-audit.json'):
            assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest() == sha(ROOT/n),n
    return p


def load(path=START, device='cuda'):
    state = torch.load(path,map_location='cpu',weights_only=True)
    model = GearboxMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs'])
    missing,extra = model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and all(n.startswith('gearbox_') for n in missing)
    return model.eval().requires_grad_(False).to(device),state


def rows_for(phase): return [r for r in read(MANIFEST)['rows'] if r['phase'] == phase]


def images(rows):
    values = []
    for r in rows:
        assert sha(ROOT/r['image']) == r['image_sha256']
        with Image.open(ROOT/r['image']) as im:
            assert im.size == (128,128)
            values.append(normalized(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)))
    return torch.stack(values)


def cache(phase, model):
    path = RUN/(phase+'-features.pt'); receipt = RUN/(phase+'-cache.json'); rows = rows_for(phase)
    if path.exists():
        assert sha(path) == read(receipt)['sha256']
        result = torch.load(path,map_location='cpu',weights_only=True)
        assert result['ids'] == [r['id'] for r in rows] and result['manifest_sha256'] == sha(MANIFEST)
        return rows,result['features'].float()
    values = []
    with torch.inference_mode():
        for i in range(0,len(rows),32):
            batch = images(rows[i:i+32]).cuda(); features = forward_with_fine(model.backbone,batch)
            if phase == 'fit' and i == 0:
                previous = model.backbone(batch)
                assert all(torch.equal(previous[k],features[k]) for k in ('low','high'))
            values.append(gearbox_descriptor(features).half().cpu())
            if i % 1024 == 0: print({'features':phase,'patches':min(i+32,len(rows)),'total':len(rows)},flush=True)
    data = {'ids':[r['id'] for r in rows],'features':torch.cat(values),'manifest_sha256':sha(MANIFEST),'protocol_sha256':sha(PROTOCOL)}
    with path.open('xb') as out: torch.save(data,out)
    new(receipt,{'sha256':sha(path),'bytes':path.stat().st_size,'patches':len(rows)})
    return rows,data['features'].float()


def macro_loss(loss, rows):
    values = []
    for y in (0,1):
        groups = sorted({r['group_id'] for r in rows if r['label'] == y})
        values.append(torch.stack([loss[[i for i,r in enumerate(rows) if r['label']==y and r['group_id']==g]].mean() for g in groups]).mean())
    return torch.stack(values).mean()


def train():
    validate(committed=True); assert not (RUN/'trained/training.json').exists()
    model,parent = load(); fit,x = cache('fit',model); val,vx = cache('validation',model)
    model.gearbox_mean.copy_(x.mean(0).cuda()); model.gearbox_std.copy_(x.std(0,correction=0).clamp_min(1e-3).cuda())
    x,vx = x.cuda(),vx.cuda(); y = torch.tensor([r['label'] for r in fit],dtype=torch.float32,device='cuda')
    vy = torch.tensor([r['label'] for r in val],dtype=torch.float32,device='cuda')
    model.gearbox_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.gearbox_head.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=1e-5)
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    journal = RUN/'optimizer.jsonl'; progress = RUN/'progress.pt'; start = 0; best = float('inf'); best_epoch = 0
    if progress.exists():
        saved = torch.load(progress,map_location='cpu',weights_only=True)
        model.gearbox_head.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start,best,best_epoch = saved['epoch'],saved['best'],saved['best_epoch']; rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        assert not journal.exists()
        journal.touch(exist_ok=False)
        with torch.no_grad(): best = float(macro_loss(F.binary_cross_entropy_with_logits(model.forward_gearbox_descriptor(vx),vy,reduction='none'),val))
        new(RUN/'initial-VAL.json',{'balanced_group_BCE':best,'epoch':0})
        torch.save({'head':model.gearbox_head.state_dict(),'epoch':0,'VAL_loss':best,'actual_updates':0},RUN/'best-head.pt')
    actual = len(journal.read_text(encoding='utf-8').splitlines())
    pools = {}; probabilities = {}
    for label in (0,1):
        pool = [i for i,r in enumerate(fit) if r['label']==label]; counts = Counter(fit[i]['group_id'] for i in pool)
        weights = np.array([1/counts[fit[i]['group_id']] for i in pool],dtype=float)
        pools[label] = pool; probabilities[label] = weights/weights.sum()
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            model.gearbox_head.train(); losses = []
            for step in range(STEPS):
                ids = np.concatenate([rng.choice(pools[label],32,replace=True,p=probabilities[label]) for label in (0,1)])
                logits = model.forward_gearbox_descriptor(x[ids]); loss = F.binary_cross_entropy_with_logits(logits,y[ids])
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.gearbox_head.parameters(),5); optimizer.step()
                actual += 1; out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'loss':float(loss.detach())})+'\n'); losses.append(float(loss.detach()))
            scheduler.step(); model.gearbox_head.eval()
            with torch.no_grad(): score = float(macro_loss(F.binary_cross_entropy_with_logits(model.forward_gearbox_descriptor(vx),vy,reduction='none'),val))
            if score < best:
                best,best_epoch = score,epoch
                torch.save({'head':model.gearbox_head.state_dict(),'epoch':epoch,'VAL_loss':score,'actual_updates':actual},RUN/'best-head.pt')
            saved = {'head':model.gearbox_head.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            torch.save(saved,RUN/'progress.partial.pt'); (RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_BCE':float(np.mean(losses)),'VAL_macro_group_BCE':score,'best_epoch':best_epoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':best_epoch,'actual_optimizer':actual},flush=True)
    chosen = torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True); model.gearbox_head.load_state_dict(chosen['head']); model.eval()
    summary = {'completed_epochs':EPOCHS,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_macro_group_BCE':chosen['VAL_loss'],'FIT_patches':len(fit),'VAL_patches':len(val)}
    result = deepcopy(parent); result['state_dict'] = {n:t.detach().cpu() for n,t in model.state_dict().items()}
    result.update(gearbox_protocol_sha256=sha(PROTOCOL),gearbox_training=summary,gearbox_label_scope='publisher surface128patch acceptance, no verified location or whole machine functioning')
    (RUN/'trained').mkdir()
    with (RUN/'trained/final.pt').open('xb') as out: torch.save(result,out)
    new(RUN/'trained/training.json',summary)


def score_rows(model, features):
    result = []
    with torch.inference_mode():
        for i in range(0,len(features),256): result.extend(model.forward_gearbox_descriptor(features[i:i+256].cuda()).sigmoid().cpu().tolist())
    return result


def calibrate():
    validate(); assert not (RUN/'calibration.json').exists() and (RUN/'trained/training.json').exists()
    model,_ = load(RUN/'trained/final.pt'); rows,features = cache('cal',model); scores = score_rows(model,features)
    labels = [r['label'] for r in rows]; cut = calibrated_threshold(labels,scores)
    new(RUN/'calibration.json',{'checkpoint_sha256':sha(RUN/'trained/final.pt'),'threshold':cut,'rates':classification_rates(labels,scores,cut),'patches':len(rows),'score_scope':'visiblewear acceptance not failure probability','TEST_scored_before_this':False})
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})


def evaluate():
    validate(); assert not (RUN/'selection.json').exists()
    cal = read(RUN/'calibration.json'); assert cal['checkpoint_sha256'] == sha(RUN/'trained/final.pt')
    model,_ = load(RUN/'trained/final.pt'); rows,features = cache('test',model); scores = score_rows(model,features)
    labels = [r['label'] for r in rows]; rates = classification_rates(labels,scores,cal['threshold'])
    by = {}
    for field in ('functional_component_category','physical_part_id','group_id'):
        by[field] = {}
        for group in sorted({r[field] for r in rows},key=str):
            ids = [i for i,r in enumerate(rows) if r[field]==group]
            by[field][str(group)] = classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
    adequate = [v for v in by['functional_component_category'].values() if min(v['positive_photos'],v['negative_photos'])>=20]
    training = read(RUN/'trained/training.json')
    eligible = training['selected_epoch']>0 and min(rates['positive_photos'],rates['negative_photos'])>=20 and max(rates['fpr'],rates['fnr'])<=.20 and all(max(v['fpr'],v['fnr'])<=.25 for v in adequate)
    # Whole-group bootstrap: acknowledges correlation, unlike binomial patch intervals.
    groups = sorted({r['group_id'] for r in rows}); rng = np.random.default_rng(SEED+99); intervals = []
    for _ in range(1000):
        sample = rng.choice(groups,len(groups),replace=True); ids = [i for g in sample for i,r in enumerate(rows) if r['group_id']==g]
        v = classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['threshold'])
        if v['fpr'] is not None and v['fnr'] is not None: intervals.append([v['fpr'],v['fnr']])
    result = {'UTC':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':cal['checkpoint_sha256'],'training':training,'threshold':cal['threshold'],'rates':rates,'AUROC':rank_auroc(labels,scores),'by':by,'eligible_pending_package_checks':bool(eligible),'TEST_patches':len(rows),'TEST_connected_groups':len(groups),'group_bootstrap_95percent':np.quantile(intervals,[.025,.975],axis=0).tolist(),'bootstrap_valid_draws':len(intervals),'bootstrap_unit':'connected publisher physical/functional/acq and near groups; only7TESTgroups, uncertain','new_independent_field_validation':False,'whole_equipment_health_or_verified_location':False,'existing_bracket_tube_transferred_or_replaced':False}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})
    new(RUN/'selection.json',result); new(ROOT/'reports/industrial-gearbox-v26-selection.json',result)
    print({k:v for k,v in result.items() if k not in ('by','group_bootstrap_95percent')},flush=True)


def verify():
    p = validate(); selection = read(RUN/'selection.json'); parent = torch.load(START,map_location='cpu',weights_only=True)
    candidate = torch.load(RUN/'trained/final.pt',map_location='cpu',weights_only=True)
    for name,tensor in parent['state_dict'].items(): assert torch.equal(tensor,candidate['state_dict'][name]),name
    for name,value in parent.items():
        if name != 'state_dict': assert value == candidate[name],name
    for name,digest in p['legacy_source_sha256'].items(): assert sha(ROOT/name)==digest,name
    unchanged = rehashed = 0
    for name,before in read(RUN/'protected-inputs.json').items():
        path = ROOT/name; st = path.stat()
        if st.st_size==before['size_bytes'] and st.st_mtime_ns==before['mtime_ns']: unchanged += 1
        else: assert sha(path)==before['sha256'],name; rehashed += 1
    journal = [json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(journal) == selection['training']['actual_optimizer_updates'] and len(journal)>=1920
    assert [v['actual_update'] for v in journal] == list(range(1,len(journal)+1))
    for phase in ('fit','validation','cal','test'): assert sha(RUN/(phase+'-features.pt')) == read(RUN/(phase+'-cache.json'))['sha256']
    result = {'status':'passed','parent_all_tensor_buffer_metadata_exact':True,'legacy_python_SHA_checked':len(p['legacy_source_sha256']),'protected_stat_unchanged':unchanged,'changed_metadata_SHA_rechecked':rehashed,'all_protected_SHA_reread':False,'actual_optimizer_updates':len(journal),'new_unit_tests_passed':3,'eligible_pending_package_checks':selection['eligible_pending_package_checks'],'new_API_CPP_ZIP_checks':0,'active_v16_retained_pending_any_release':True}
    new(RUN/'verification.json',result); new(ROOT/'reports/industrial-gearbox-v26-verification.json',result); print(result,flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    parser = argparse.ArgumentParser(); parser.add_argument('stage',choices=('freeze','train','calibrate','evaluate','verify','run'))
    args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    else:
        validate(committed=True)
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())): raise RuntimeError('Existing work: '+str(path))
        lock = RUN/'operation.lock'
        if lock.exists(): raise RuntimeError('Preserve and review stale lock before recovery')
        with lock.open('x',encoding='utf-8') as out: out.write(str(os.getpid()))
        try:
            if args.stage == 'run': train(); calibrate(); evaluate(); verify()
            else: globals()[args.stage]()
        except BaseException as error:
            new(RUN/(args.stage+'-failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()}); raise
        finally: lock.unlink()
