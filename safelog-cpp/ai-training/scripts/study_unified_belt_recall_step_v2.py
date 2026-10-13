"""Matched normal-ROI study with a positive-recall projected Adam displacement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from safelog_ai.belt_normal_learning import normal_roi_loss
from safelog_ai.belt_recall_step_v2 import positive_recall_loss,project_step
from safelog_ai.belt_positive_retention import pixel_retention
from safelog_ai.belt_detail import configure_belt_only,non_belt_state,paired_flip
from safelog_ai.belt_tiles import predict,policy
from safelog_ai.belt_fine_tiles import guards,BASELINE
from safelog_ai.belt_evaluation import summarize
from safelog_ai.belt_segmentation import pixel_counts
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.unified_main import ARCH,TASKS,UnifiedMain,freeze_bn_statistics,loss_for_task
from safelog_ai.process_probe import process_exists
from scripts.unified_main_data import DATA,Cache,read,sha,normalize_batch
from scripts.study_unified_belt_tiles import native,snapshot,START,START_SHA
from scripts.study_unified_belt_fine import slices

BASE=ROOT/'runs/unified-belt-recall-step-v2'; PROTOCOL=ROOT/'reports/unified-belt-recall-step-v2-protocol.json'
NORMAL=ROOT/'data/belt-normal-rois-v1/manifest.json'; NATIVE=ROOT/'data/unified-belt-native-detail-v1/manifest.json'
SOURCES=('safelog_ai/belt_recall_step_v2.py','scripts/study_unified_belt_recall_step_v2.py','tests/test_belt_recall_step_v2.py')
SEED=443


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def freeze():
    assert not BASE.exists() and sha(START)==START_SHA; BASE.mkdir()
    assert sha(ROOT/'runs/unified-belt-normal-roi-pilot-v1/train/final.pt')=='c4e70e28cd596f1bc4726fd8be617ce5fd7c598ef442076116af942ae2be103a'
    normal=read(NORMAL); fit=[r for r in normal['rows'] if r['split']=='fit']; cal=[r for r in normal['rows'] if r['split']=='cal']
    assert len(fit)==14 and len(cal)==3
    for n,h in normal['preparation_source_sha256'].items(): assert sha(ROOT/n)==h,n
    ids=np.load(ROOT/'runs/unified-belt-native-detail-v1/draws.npz')['belt_damage'][:4].copy()
    negatives=np.random.default_rng(SEED).integers(0,14,size=(4,128,2)); np.savez_compressed(BASE/'draws.npz',belt_damage=ids,normal_roi=negatives)
    files=dict(read(ROOT/'runs/unified-belt-normal-roi-pilot-v1/protected-inputs.json'))
    for folder in [ROOT/'reports',ROOT/'runs/unified-belt-normal-roi-pilot-v1',ROOT/'runs/unified-belt-recall-step-v1',ROOT/'data/belt-normal-rois-v1',ROOT/'data/belt-normal-roboflow-v1',ROOT/'runs/belt-normal-rois-v1']:
        for path in folder.rglob('*') if folder.name!='reports' else folder.iterdir():
            if path.is_file() and not path.name.endswith(('.lock','.log')): files[path.relative_to(ROOT).as_posix()]=snapshot(path)
    files[NATIVE.relative_to(ROOT).as_posix()]=snapshot(NATIVE)
    def check(item):
        n,v=item; assert snapshot(ROOT/n)==v,n
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check,files.items()))
    new(BASE/'protected-inputs.json',files)
    old_draws=np.load(ROOT/'runs/unified-belt-normal-roi-pilot-v1/draws.npz')
    assert np.array_equal(ids,old_draws['belt_damage']) and np.array_equal(negatives,old_draws['normal_roi'])
    prior=read(ROOT/'reports/unified-belt-fine-global_tile_mean-validation.json')
    positive_ids=[int(ids[e,i*8+j]) for e in range(4) for i in range(128) for j in range(6)]
    new(PROTOCOL,{'schema':'unified-belt-recall-step-v2','declared_before_fit_or_model_CAL':True,'declared_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'preparation_source_sha256':normal['preparation_source_sha256'],
        'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for d in ('scripts','safelog_ai','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'input_ledger_sha256':sha(BASE/'protected-inputs.json'),'normal_manifest_sha256':sha(NORMAL),'native_manifest_sha256':sha(NATIVE),
        'unified_manifest_sha256':sha(DATA/'manifest.json'),'draws_sha256':sha(BASE/'draws.npz'),'start_sha256':START_SHA,
        'matched_control_checkpoint_sha256':'c4e70e28cd596f1bc4726fd8be617ce5fd7c598ef442076116af942ae2be103a','matched_control_arrays_exact':True,
        'gradient_projection':'Actual Adam parameter delta projected if it raises first-order mean TRAIN GT-positive-pixel softplus(-logit) loss; numerical signed-dot check each update. Raw Adam moments retained. Not standard PCGrad and not a validation recall guarantee.',
        'method_reference':'https://arxiv.org/abs/2001.06782','numerical_repair':'FP32-write dot error bound margin; exact parameter revert if numerical guard still fails, raw optimizer state retained',
        'excluded_attempt':{'run':'unified-belt-recall-step-v1','status':'first_optimizer_call_failed_numeric_guard','optimizer_calls':1,'journal_committed_updates':0,'saved_checkpoint':False},
        'epochs':4,'optimizer_updates_planned':512,'batch':8,'seed':SEED,'positive_unique_TRAIN_photos':len(set(positive_ids)),
        'publisher_normal_TRAIN_ROIs':14,'CAL_normal_ROIs':3,'unscored_reserved_normal_ROIs':7,
        'positive_draws':3072,'negative_ROI_draws':1024,'new_normal_whole_photo_labels':0,'new_normal_pixel_masks':0,
        'sampling':'Per8:4 original positive TRAIN frames +2 contextual native positive crops +2 publisher-normal ROI classification examples; fixed first4 draw schedule, seed443 normal choices',
        'loss':'Unchanged positive BCE posweight20+Dice +10*teacher KL over positive-image valid pixels +.5*normalROI softplus(mean top64 logits); no invented negative segmentation masks',
        'optimizer':'Belt-head only fresh AdamW LR5e-5 wd.0002 cosine4 eta_min5e-6; clip10; FP32 TF32 disabled; all BN and nonbelt tensors frozen',
        'checkpoint':'Final4 only; checkpoint64 with RNG/optimizer/cursor; no per-epoch CAL or knownVAL',
        'policy':policy(BASELINE),'baseline':prior['summary'],'baseline_slices':prior['slices'],
        'quality_gate':'Research-only selection if all8 current knownVAL guards and CAL3 false alarms strictly improve; original groups unknown and CAL3 insufficient for field claims; app default never auto-promoted; reserved7 unscored.',
        'comparison_guards':'Original8 knownVAL guards against CURRENT accepted tile baseline, plus CAL ROI false alarms descriptive only',
        'source':'https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/2','publisher':'Gofaones Workspace','license':'CC BY 4.0',
        'known_belt_VAL_reused':True,'old_BeltCrack_TEST_used':False,'original_sequence_groups_verified':False,
        'new_independent_factory_photos_verified':0,'app_default_promoted':False,'unit_tests_passed':4})
    print(json.dumps({'frozen':True,'pilot_updates_planned':512,'normal_fit':14,'normal_cal':3,'app_default_adoption_allowed':False,'protected_files':len(files)}),flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['source_sha256'],**p['preparation_source_sha256']}.items(): assert sha(ROOT/n)==h,n
    for path,h in [(START,START_SHA),(NORMAL,p['normal_manifest_sha256']),(NATIVE,p['native_manifest_sha256']),(DATA/'manifest.json',p['unified_manifest_sha256']),
        (BASE/'draws.npz',p['draws_sha256']),(BASE/'protected-inputs.json',p['input_ledger_sha256'])]: assert sha(path)==h,path
    return p


def progress(path,model,opt,scheduler,epoch,cursor,updates,losses,elapsed):
    assert path.resolve().is_relative_to(BASE.resolve()); tmp=path.with_suffix('.pending.pt')
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
        'epoch':epoch,'cursor':cursor,'updates':updates,'losses':losses,'elapsed_seconds':elapsed,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},tmp); tmp.replace(path)


def train():
    p=validate(); repo=ROOT.parents[1]; git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip(); run=BASE/'train'; run.mkdir(exist_ok=True)
    if (run/'final.pt').exists():
        s=torch.load(run/'final.pt',map_location='cpu',weights_only=True); report=ROOT/'reports/unified-belt-recall-step-v2-training.json'
        if not report.exists(): new(report,{**{k:v for k,v in s.items() if k!='state_dict'},'weights_sha256':sha(run/'final.pt')})
        assert sha(run/'final.pt')==read(report)['weights_sha256']; print('Completed pilot preserved',flush=True); return
    lock=run/'training.lock'
    if lock.exists():
        if process_exists(int(lock.read_text())): raise RuntimeError('Owned pilot is already running')
        lock.unlink()
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        torch.manual_seed(SEED); initial=torch.load(START,map_location='cpu',weights_only=True)
        model=UnifiedMain().cuda(); model.load_state_dict(initial['state_dict'],strict=True)
        teacher=UnifiedMain().cuda().eval(); teacher.load_state_dict(initial['state_dict'],strict=True); teacher.requires_grad_(False)
        fixed=state_sha256(non_belt_state(model)); initial_head=state_sha256(model.belt_head); teacher_state=state_sha256(teacher)
        opt=torch.optim.AdamW(configure_belt_only(model),lr=5e-5,weight_decay=.0002); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4,eta_min=5e-6)
        manifest=read(DATA/'manifest.json'); draws=np.load(BASE/'draws.npz'); ids,negative_ids=draws['belt_damage'],draws['normal_roi']; allowed=set(ids.flatten().tolist())
        full=Cache(manifest['fit']['belt_damage'],allowed); detail=Cache(read(NATIVE)['rows'],allowed)
        normal=[r for r in read(NORMAL)['rows'] if r['split']=='fit']; negatives=[]
        for row in normal:
            assert sha(ROOT/row['image'])==row['image_sha256'] and row['label_scope']=='publisher_normal_bbox_interior_ROI_only'
            with Image.open(ROOT/row['image']) as im: negatives.append(torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1))
        epoch,cursor,updates,losses,elapsed=1,0,0,0.,0.; checkpoint=run/'progress.pt'; torch.manual_seed(SEED)
        if checkpoint.exists():
            s=torch.load(checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(s['state_dict'],strict=True)
            opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler']); epoch,cursor,updates,losses,elapsed=(s[k] for k in ('epoch','cursor','updates','losses','elapsed_seconds'))
            torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng'])
        started=time.monotonic()
        with (run/'update-journal.jsonl').open('a',encoding='utf-8',buffering=1) as journal:
            while epoch<=4:
                model.eval(); model.belt_head.train(); freeze_bn_statistics(model)
                for i in range(cursor,128):
                    source=[int(ids[epoch-1,i*8+j]) for j in range(6)]
                    samples=[full.row(index) if j<4 else detail.row(index) for j,index in enumerate(source)]
                    images=torch.stack([x[0] for x in samples]); masks=torch.stack([x[1]['mask'] for x in samples]); valid=torch.stack([x[1]['valid'] for x in samples])
                    images,masks,valid=paired_flip(images,masks,valid,torch.rand(6)<.5)
                    ni=torch.stack([negatives[int(x)] for x in negative_ids[epoch-1,i]]).clone(); flips=torch.rand(2)<.5; ni[flips]=ni[flips].flip(-1)
                    all_images=normalize_batch(torch.cat([images,ni])); truth={'mask':masks.cuda().float(),'valid':valid.cuda().float()}
                    with torch.no_grad(): reference=teacher.forward_task(all_images[:6],'belt_damage')
                    opt.zero_grad(set_to_none=True); logits=model.forward_task(all_images,'belt_damage')
                    retention=pixel_retention(logits[:6],reference,truth['valid']); negative_loss=normal_roi_loss(logits[6:])
                    value=loss_for_task('belt_damage',logits[:6],truth)+10*retention+.5*negative_loss
                    assert torch.isfinite(value)
                    params=list(model.belt_head.parameters()); recall=positive_recall_loss(logits[:6],truth['mask'],truth['valid'])
                    recall_grads=torch.autograd.grad(recall,params,retain_graph=True,allow_unused=True)
                    before=[parameter.detach().clone() for parameter in params]
                    value.backward(); torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True); opt.step()
                    projection=project_step(params,before,recall_grads)
                    updates+=1; cursor=i+1; losses+=float(value.detach()); journal.write(json.dumps({'epoch':epoch,'cursor':cursor,'update':updates,'normal_ROI_draws':2,'negative_loss':float(negative_loss.detach()),'recall_loss':float(recall.detach()),'projection':projection})+'\n')
                    if updates%64==0: progress(checkpoint,model,opt,scheduler,epoch,cursor,updates,losses,elapsed+time.monotonic()-started)
                assert state_sha256(non_belt_state(model))==fixed; scheduler.step()
                entry={'epoch':epoch,'updates':updates,'mean_loss':losses/128,'elapsed_minutes':(elapsed+time.monotonic()-started)/60,'non_belt_unchanged':True}
                dest=run/f'epoch-{epoch:02d}.json'
                if not dest.exists(): new(dest,entry)
                else: assert read(dest)['updates']==updates
                print(json.dumps(entry),flush=True); epoch+=1; cursor=0; losses=0.
                progress(checkpoint,model,opt,scheduler,epoch,cursor,updates,losses,elapsed+time.monotonic()-started)
        assert updates==512 and state_sha256(model.belt_head)!=initial_head and state_sha256(teacher)==teacher_state
        assert all(v.grad is None for n,v in model.named_parameters() if not n.startswith('belt_head.'))
        model.eval(); cpu=UnifiedMain().eval(); cpu.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()},strict=True)
        old=UnifiedMain().eval(); old.load_state_dict(initial['state_dict'],strict=True); proofs={}
        with torch.inference_mode():
            for task in TASKS:
                x,_=Cache(manifest['fit'][task],{0,1}).batch([0,1]); a,b=model.forward_task(x,task),cpu.forward_task(x.cpu(),task)
                if task=='facility_surface': a,b=a[0],b[0]
                assert torch.allclose(a.cpu(),b,atol=1e-4,rtol=.001)
                if task!='belt_damage':
                    expected=old.forward_task(x.cpu(),task); expected=expected[0] if task=='facility_surface' else expected; assert torch.equal(expected,b)
                proofs[task]={'CPU_GPU_close':True,'prior_CPU_exact':task!='belt_damage'}
            rgb,_,_=native(manifest['fit']['belt_damage'][0]); a,box,count=predict(model,rgb,BASELINE,'cuda'); b,b_cpu,c_cpu=predict(cpu,rgb,BASELINE,'cpu')
            assert box==b_cpu and count==c_cpu==6 and torch.allclose(a.cpu(),b,atol=1e-4,rtol=.001)
        metadata={'status':'recall_projected_trained_fixed4','architecture':ARCH,'tasks':list(TASKS),'one_shared_backbone':True,'input_size':640,
            'thresholds':initial['thresholds'],'epochs':4,'optimizer_updates':512,'initial_checkpoint_sha256':START_SHA,'fit_protocol_commit':commit,'protocol_sha256':sha(PROTOCOL),
            'strict_CPU_GPU_three_tasks':proofs,'native_tile_CPU_GPU_close':True,'non_belt_state_sha256':fixed,'teacher_unchanged_not_deployed':True,
            'normal_fit_ROIs':14,'negative_ROI_draws':1024,'normal_whole_photo_labels':0,'normal_dense_pixel_labels':0,
            'elapsed_training_minutes':(elapsed+time.monotonic()-started)/60,'actual_journal_updates_including_replay':sum(1 for _ in (run/'update-journal.jsonl').open()),
            'field_adoption_allowed':False,'quality_gate':'Research-only; CAL3 and recording IDs not independently verified'}
        with (run/'final.pt').open('xb') as f: torch.save({**metadata,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},f)
        new(ROOT/'reports/unified-belt-recall-step-v2-training.json',{**metadata,'weights_sha256':sha(run/'final.pt')})
        print(json.dumps({'actual_updates':512,'weights_sha256':sha(run/'final.pt')}),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()): lock.unlink()


def evaluate():
    p=validate(); model=UnifiedMain().cuda().eval(); model.load_state_dict(torch.load(BASE/'train/final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    rows=read(DATA/'manifest.json')['belt_known_validation']; truth=Cache(rows,range(len(rows))); observations=[]
    for i,row in enumerate(rows):
        rgb,_,_=native(row); probability,box,count=predict(model,rgb,BASELINE,'cuda'); _,t=truth.batch([i])
        observations.append({'id':row['id'],'sequence':row['sequence'],'photo_positive':row['photo_positive'],'detail_views':count,
            'score':float(probability[t['valid'].bool()].max()),'pixels':pixel_counts(probability,t['mask'],t['valid'])})
        if (i+1)%100==0: print(json.dumps({'known_VAL_evaluated':i+1,'total':353}),flush=True)
    summary=summarize(observations); summary['photo_score_definition']='Maximum fused global/tile probabilities over valid pixels, not calibrated'
    normal=[r for r in read(NORMAL)['rows'] if r['split']=='cal']; assert len(normal)==3
    old=UnifiedMain().cuda().eval(); old.load_state_dict(torch.load(START,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    roi_results={}
    for name,m in [('accepted',old),('recall_projected',model)]:
        scores=[]
        for row in normal:
            with Image.open(ROOT/row['image']) as im: rgb=im.convert('RGB')
            prob,box,count=predict(m,rgb,BASELINE,'cuda'); assert count==0
            scores.append({'id':row['id'],'source_parent_hint':row['parent'],'normal_ROI_score':float(prob.max()),'false_alarm':float(prob.max())>.5})
        alarms=sum(x['false_alarm'] for x in scores); roi_results[name]={'ROIs':3,'false_alarms':alarms,'ROI_FPR':alarms/3,'rows':scores}
    new(ROOT/'reports/unified-belt-recall-step-v2-validation.json',{'status':'known_positive_VAL353_and_provisional_normal_CAL3','known_belt_rows':observations,
        'known_summary':summary,'known_slices':slices(observations),'normal_ROI_CAL':roi_results,'normal_whole_photo_FPR':None,
        'normal_ROI_FPR_is_whole_photo_FPR':False,'reserved_normal7_predictions_performed':False,'old_BeltCrack_TEST_used':False,'checkpoint_sha256':sha(BASE/'train/final.pt')})
    print(json.dumps({'known_summary':summary,'normal_CAL':{k:{n:v for n,v in d.items() if n!='rows'} for k,d in roi_results.items()}}),flush=True)


def verify():
    p=validate(); r=read(ROOT/'reports/unified-belt-recall-step-v2-validation.json'); checks=guards(r['known_summary'],p['baseline'],r['known_slices'],p['baseline_slices'])
    ledger=read(BASE/'protected-inputs.json')
    def check(item):
        n,v=item; assert snapshot(ROOT/n)==v,n
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check,ledger.items()))
    for n,h in p['legacy_source_sha256'].items(): assert sha(ROOT/n)==h,n
    final=torch.load(BASE/'train/final.pt',map_location='cpu',weights_only=True); saved=torch.load(BASE/'train/progress.pt',map_location='cpu',weights_only=True); old=torch.load(START,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,saved['state_dict'][n]) for n,v in final['state_dict'].items())
    assert all(torch.equal(v,old['state_dict'][n]) for n,v in final['state_dict'].items() if not n.startswith('belt_head.'))
    journal=[json.loads(line) for line in (BASE/'train/update-journal.jsonl').read_text().splitlines()]
    assert set(r['update'] for r in journal)==set(range(1,513))
    assert all(j['projection']['dot_after']<=j['projection']['tolerance'] for j in journal)
    projection_count=sum(j['projection']['projected'] for j in journal)
    roi_gain=r['normal_ROI_CAL']['recall_projected']['false_alarms']<r['normal_ROI_CAL']['accepted']['false_alarms']
    accepted=all(checks.values()) and roi_gain
    selected='recall_projected_research' if accepted else 'prior_accepted_tiles'
    chosen=BASE/'train/final.pt' if accepted else START
    new(ROOT/'reports/unified-belt-recall-step-v2-selection.json',{'selected':selected,'known_metric_guards':checks,'normal_ROI_CAL_alarm_reduction':roi_gain,
        'field_quality_gate_passed':False,'research_comparison_guards_passed':accepted,
        'reason':'Research selection requires knownVAL preservation plus descriptive CAL3 alarm reduction. No field5percent or normal whole-photo FPR claim.',
        'checkpoint_path':chosen.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(chosen),'policy':policy(BASELINE),'new_neural_updates_performed':512,
        'selected_new_neural_updates':512 if accepted else 0,'app_default_promoted':False})
    new(ROOT/'reports/unified-belt-recall-step-v2-verification.json',{'status':'passed','protected_files':len(ledger),'legacy_sources':len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved':True,'final_equals_progress':True,'non_belt_tensors_exact_prior':True,'actual_journal_updates':len(journal),'unique_committed_updates':512,'projected_updates':projection_count,'reverted_parameter_steps':sum(j['projection']['reverted'] for j in journal),'all_effective_step_protection_checks_passed':True,
        'normal_ROI_source_images_and_sha_preserved':True,'reserved_normal7_model_predictions_performed':False,'old_BeltCrack_TEST_used':False})
    print(json.dumps({'verified':True,'old_files':len(ledger),'actual_updates':len(journal),'projected_updates':projection_count,'selected':selected}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['freeze','train','eval','verify'],required=True)
    args=parser.parse_args(); {'freeze':freeze,'train':train,'eval':evaluate,'verify':verify}[args.mode]()
