"""One preregistered normal-only projected-density study; all prior tasks fixed."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.insplad_lower_flow import InspladLowerFlowMain
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.gearbox_supervised import gearbox_descriptor
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates, rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.research_delta import pack_delta, restore_delta
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

START=ROOT/'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED='5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
PRIOR_DELTA=ROOT/'runs/industrial-insplad-suspension-v51/trained/final-delta.pt'
PRIOR_SHA='b91b6578eccd1886c5be18805a9475875ff319f77bda019cd0b3f518d84f46dc'
PRIOR_COMBINED='bb388200af167ffde35f6281a5e4626be904f9026411c401151e9bc3b4d7c45a'
MANIFEST=ROOT/'data/industrial-insplad-lower-shackle-v57/manifest.json'
MANIFEST_SHA='5ae6838ea6acbdbd76bd2030a58feb3ca6792b63b9967f69a76861935e740cbb'
AUDIT=ROOT/'reports/industrial-insplad-lower-shackle-v57-data-audit.json'
RUN=ROOT/'runs/industrial-insplad-lower-flow-v58'
PROTOCOL=ROOT/'reports/industrial-insplad-lower-flow-v58-protocol.json'
SOURCES=('safelog_ai/insplad_lower_flow.py','scripts/study_insplad_lower_flow.py','tests/test_insplad_lower_flow.py')
EPOCHS,STEPS,BATCH,SEED=30,64,32,5858
PREFIXES=('insplad_classifier.','insplad_glass_classifier.','insplad_suspension_classifier.','insplad_lower_flow.')


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]
def group(r):return r['pixel_group_id']


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists()
    assert sha(START)==EXPECTED and sha(PRIOR_DELTA)==PRIOR_SHA and sha(MANIFEST)==MANIFEST_SHA
    audit=read(AUDIT);assert audit['ready'] and audit['manifest_sha256']==MANIFEST_SHA
    assert shutil.disk_usage(ROOT).free>1024**3+100*1024**2
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                st=path.stat();protected[path.relative_to(ROOT).as_posix()]=[st.st_size,st.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    fit=rows_for('fit');assert len(fit)==87 and all(r['label']==0 for r in fit)
    counts=Counter(group(r) for r in fit);w=np.array([1/counts[group(r)] for r in fit]);w/=w.sum()
    draws=np.random.default_rng(SEED).choice(len(fit),(EPOCHS*STEPS,BATCH),replace=True,p=w)
    np.save(RUN/'draws.npy',draws.astype(np.int32),allow_pickle=False)
    p={'UTC':utc(),'schema':'safelog-insplad-lower-flow-v58','fixed_before_new_features_PCA_gradient_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':EXPECTED,
       'approved_three_head_delta_sha256':PRIOR_SHA,'approved_three_head_combined_sha256':PRIOR_COMBINED,
       'manifest_sha256':MANIFEST_SHA,'data_audit_sha256':sha(AUDIT),
       'legacy_Python_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
       'protected_stat_sha256':sha(RUN/'protected-stat.json'),'draws_sha256':sha(RUN/'draws.npy'),
       'prior_calibration_sha256':{n:sha(ROOT/n) for n in ('runs/industrial-insplad-corrosion-v45/calibration.json','runs/industrial-insplad-glass-v48/calibration.json','runs/industrial-insplad-suspension-v51/calibration.json')},
       'split':audit['counts'],'groups_each_label':audit['groups_each_label'],
       'scope':'Only cropped InsPLAD polymer-insulator-lower-shackle good-vs-rust appearance; normal-only density score, not rust mask supervision, internal/electrical/machine health or independent field accuracy.',
       'input':'original RGB bicubic256square ImageNet; no augmentation/mask/crop selected with TEST/source/date/category embedding',
       'features':'same frozen encoder once perRGB fine24/low40/high960 FP16roundtrip maps then FP32 channel mean/popstd/max 3072 FP16roundtrip; no separateencoder',
       'normalization':'FIT87 only channel mean/popstd floor.001; standardize/clamp[-8,8], FIT standardizedcenter; CPUfloat64 SVD full_matricesFalse fixed32PCA columns sign largestabs entry positive, storedFP32; latentFITmean/popstd floor.001 from storedtransform fixed',
       'architecture':'one insplad_lower_flow with6 affinecoupling blocks latent32 half16 conditioner16to64 GELU64to64 GELU64to32 scale16+shift16; finalLinearzero, scale1.5*tanh(raw), reversechannels aftereachblock absdet1;43968trainableparams',
       'density':'NLL=.5sum(z^2+log2pi)-sum(logscale). Invertible only within PCA32 latent, not density on3072/fullRGB. Flowblocks only learned; PCA/normalization frozen.',
       'loss':'mean NLL on original normalFIT87 only. No new realrustGT, masks, synthetic defects, source/TESTlabels in optimizer',
       'sampling':'1920fixed draws x32 uniform connectedgroup thenphoto withreplacement; repeated draws not newphotos',
       'optimizer':'AdamW lr.0003 wd.01 cosine30 min.000001 clip5 FP32 TF32off threads4',
       'epochs':30,'steps_per_epoch':64,'batch':32,'seed':SEED,'planned_optimizer':1920,
       'VAL_selection':'normal-only groupmean NLL on26photos/25proxygroups epoch0andeachcompletedepoch earlieststrictminimum; epoch0 rejects before CAL_TEST. No positiveVAL may limit anomaly sensitivity.',
       'bounded_score':'selectedmodel FIT87 NLLmean/popstd floor.001 fixed, sigmoid((NLL-FITmean)/FITstd). Monotone compatibility score, not calibrated failure probability. No CAL/TEST statistics in transformation.',
       'CAL':'normal28 only alpha.15 rankceil((28+1)*.85)=25 strictscore>cut stored before TESTfeatures/scores; one checkpoint and cut',
       'TEST':'original327good315rust=642, single fixed evaluation; bydate report. No locationGT or prediction-driven relabel; no threshold/architecture rescue',
       'adoption_guards':['selectedepoch>0','TESTeachlabel>=20 andeachconnectedgroups>=3','bothphotoFPRandFNR<=.20','allv16ANDapprovedthreeheadtensor_buffer_metadata_exact','fresh ZIP_CPU_newGPU_alloldalias_oldPNG_actualCPP beforehandoff'],
       'cache':'pooled<8MiB, sourceID/rawSHA/order/geometry/dtype/encoderSHA/priorDeltaSHA/protocolSHA;FIT_VALfirst,CALaftermodelselection,TESTafterCALsaved; firstFIT low/high standardforward bitwiseexact',
       'FP16_device_policy':'new package ifeligible freshRGB CPU/newGPU/cachedscore atol5e-4 rtol1e-3 and decisionexact; failed compatibility rejects package; no cutadjustment',
       'delta':'originalv16 parent-bound, old3headprefixes + insplad_lower_flow. only; assert approvedv51 state/meta beforepack and diskrestore, allparent/file/state/added/combinedSHA verified; notstandalone appmodel',
       'location':'no new validated localization, all old basicmaps/pixelthresholds exact unchanged',
       'source_QA':'fixedfirst3normalFIT only; lowerhardware/bolts visible,2lowresolution adjacentframes resemble sameassembly. No all-data humanreview or originalGTedit.',
       'limits':'frame/pHashproxy not physicalpart/flight independence; adjacentframe similarity and priorparent near exposure unresolved; normal-onlyVAL; no oldbracket/tubeimprovement claimed',
       'license':'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1 CC BY-NC3.0 noncommercialacademic author/source/version/changes, paper10.1080/01431161.2023.2283900; otherlicensesunchanged',
       'references':{'RealNVP':'https://arxiv.org/abs/1605.08803','DifferNet':'https://openaccess.thecvf.com/content/WACV2021/html/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.html'},
       'reference_scope':'density/changeofvariables inspiration only; not paperarchitecture/result reproduction',
       'storage':'newpooledcache/progress/delta/protectedstat<100MiB,>=1GiBreserve; nofullparentduplicate/densecache/originalexpansion/deletions',
       'resume':'completedepoch progress optimizer/scheduler/RNG/newbranch; journal counts allactualupdates including repeats; preservefailures',
       'new_units_passed':3,'actual_new_FIT_rust_photos':0,'independent_field_validation':False,'auto_app_adoption':False}
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'protected_files':len(protected),'planned_optimizer':1920},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(PRIOR_DELTA,PRIOR_SHA),(MANIFEST,MANIFEST_SHA),(AUDIT,p['data_audit_sha256']),(RUN/'draws.npy',p['draws_sha256']),(RUN/'protected-stat.json',p['protected_stat_sha256'])):assert sha(path)==h,path
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix(),AUDIT.relative_to(ROOT).as_posix()):
            assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def research_parent():
    original=torch.load(START,map_location='cpu',weights_only=True)
    prior=torch.load(PRIOR_DELTA,map_location='cpu',weights_only=True)
    assert prior['combined_state_sha256']==PRIOR_COMBINED
    return original,restore_delta(original,prior,EXPECTED)


def load(device='cuda',delta=False):
    original,parent=research_parent();state=parent
    if delta:
        state=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),EXPECTED);assert_preserved(parent,state)
    torch.manual_seed(SEED)
    model=InspladLowerFlowMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and all(n.startswith('insplad_lower_flow.') for n in missing)
    if delta:assert not missing
    return model.eval().requires_grad_(False).to(device),parent,original


def images(rows):
    batch=[]
    for r in rows:
        assert sha(ROOT/r['image'])==r['image_sha256']
        with Image.open(ROOT/r['image']) as im:
            assert list(im.size)==r['size'] and im.getexif().get(274) in (None,1)
            batch.append(normalized(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)))
    return torch.stack(batch)


def cache(phase,model):
    rows=rows_for(phase);path=RUN/(phase+'-features.pt')
    if phase=='test':assert (RUN/'calibration.json').exists()
    if phase=='cal':assert read(RUN/'trained/training.json')['selected_epoch']>0
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache.json'))['sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['ids']==[r['id'] for r in rows] and value['protocol_sha256']==sha(PROTOCOL)
        return rows,value['features'].float().cuda()
    values=[]
    with torch.inference_mode():
        for i in range(0,len(rows),16):
            x=images(rows[i:i+16]).cuda();f=forward_with_fine(model.backbone,x)
            if phase=='fit' and i==0:
                old=model.backbone(x);assert all(torch.equal(old[k],f[k]) for k in ('low','high'))
            values.append(gearbox_descriptor(f).half().cpu())
    value={'ids':[r['id'] for r in rows],'raw_sha256':[r['image_sha256'] for r in rows],'features':torch.cat(values),
           'manifest_sha256':MANIFEST_SHA,'protocol_sha256':sha(PROTOCOL),'encoder_parent_sha256':EXPECTED,
           'approved_three_head_delta_sha256':PRIOR_SHA,'input':'RGBbicubic256square_ImageNet','dtype':'FP16descriptor'}
    with path.open('xb') as out:torch.save(value,out)
    new(RUN/(phase+'-cache.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'photos':len(rows)})
    print({'cached':phase,'photos':len(rows)},flush=True)
    return rows,value['features'].float().cuda()


def group_mean(values,rows):
    return torch.stack([values[[i for i,r in enumerate(rows) if group(r)==g]].mean() for g in sorted({group(r) for r in rows})]).mean()


def train():
    validate(True);assert not (RUN/'trained/training.json').exists()
    model,parent,original=load();fit,x=cache('fit',model);val,vx=cache('validation',model);head=model.insplad_lower_flow
    head.fit_projection(x);head.blocks.requires_grad_(True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==43968
    opt=torch.optim.AdamW(head.blocks.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    draws=np.load(RUN/'draws.npy',allow_pickle=False);journal=RUN/'optimizer.jsonl';progress=RUN/'progress.pt';start=0;bestepoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        assert not journal.exists();journal.touch(exist_ok=False)
        with torch.no_grad():best=float(group_mean(head(vx),val))
        new(RUN/'initial-VAL.json',{'normal_group_NLL':best,'epoch':0})
        torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines())
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            losses=[]
            for step in range(STEPS):
                ids=draws[(epoch-1)*STEPS+step];loss=head(x[ids]).mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.blocks.parameters(),5);opt.step();actual+=1
                out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'normal_NLL':float(loss.detach())})+'\n');losses.append(float(loss.detach()))
            scheduler.step()
            with torch.no_grad():score=float(group_mean(head(vx),val))
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'normal_NLL':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head'])
    with torch.no_grad():
        nll=head(x);head.score_mean.copy_(nll.mean());head.score_std.copy_(nll.std(correction=0).clamp_min(.001))
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],
             'VAL_normal_group_NLL':chosen['VAL'],'FIT_normal_photos':len(fit),'actual_FIT_defect_photos':0,'VAL_normal_photos':len(val),
             'score_FIT_NLL_mean':float(head.score_mean),'score_FIT_NLL_std':float(head.score_std)}
    result=deepcopy(parent);result['state_dict']={n:t.detach().cpu() for n,t in model.state_dict().items()}
    result.update(insplad_lower_protocol_sha256=sha(PROTOCOL),insplad_lower_training=summary,insplad_lower_scope='normal_only_cropped_polymer_insulator_lower_shackle_appearance')
    assert_preserved(parent,result)
    payload=pack_delta(original,result,EXPECTED,START.relative_to(ROOT).as_posix(),PREFIXES);(RUN/'trained').mkdir()
    with (RUN/'trained/final-delta.pt').open('xb') as out:torch.save(payload,out)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),EXPECTED);assert_preserved(parent,restored)
    assert all(torch.equal(t,restored['state_dict'][n]) for n,t in result['state_dict'].items())
    assert all(v==restored[n] for n,v in result.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary)


def calibrate():
    assert not (RUN/'calibration.json').exists();model,_,_=load(delta=True);rows,x=cache('cal',model)
    with torch.no_grad():scores=model.forward_insplad_lower_descriptor(x).tolist()
    cut,rank=normal_budget_threshold(scores,alpha=.15)
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'scores':scores})
    new(RUN/'calibration.json',{'delta_sha256':sha(RUN/'trained/final-delta.pt'),'cut':cut,'rank':rank,'normal_photos':len(rows),'alpha':.15,'TEST_features_or_scores_before_this':0})


def evaluate():
    assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['delta_sha256']==sha(RUN/'trained/final-delta.pt')
    model,_,_=load(delta=True);rows,x=cache('test',model)
    with torch.no_grad():scores=model.forward_insplad_lower_descriptor(x).tolist()
    labels=[r['label'] for r in rows];rates=classification_rates(labels,scores,cal['cut'])
    groups={str(label):len({group(r) for r in rows if r['label']==label}) for label in (0,1)}
    eligible=max(rates['fpr'],rates['fnr'])<=.2 and min(rates['positive_photos'],rates['negative_photos'])>=20 and min(groups.values())>=3
    by={}
    for date in sorted({r['frame_proxy'].split('_DJI_')[0] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['frame_proxy'].startswith(date+'_DJI_')]
        by[date]=classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['cut'])
    result={'UTC':utc(),'rates':rates,'AUROC':rank_auroc(labels,scores),'groups_each_label':groups,'threshold':cal['cut'],
            'training':read(RUN/'trained/training.json'),'delta_sha256':cal['delta_sha256'],
            'source_candidate_eligible_pending_package_checks':bool(eligible),'by_capture_date':by,
            'independent_field_validation':False,'whole_machine_health_or_verified_pixel_localization':False,
            'old_bracket_tube_improvement_tested':False,'new_API_CPP_ZIP_checks':0,'approved_v16_three_heads_unchanged':True}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'scores':scores});new(RUN/'selection.json',result)
    new(ROOT/'reports/industrial-insplad-lower-flow-v58-selection.json',result);print(result,flush=True)


def verify():
    p=validate();train=read(RUN/'trained/training.json');original,parent=research_parent()
    payload=torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True);restored=restore_delta(original,payload,EXPECTED);assert_preserved(parent,restored)
    for n,h in p['legacy_Python_sha256'].items():assert sha(ROOT/n)==h,n
    checked=0
    for n,before in read(RUN/'protected-stat.json').items():
        st=(ROOT/n).stat();assert [st.st_size,st.st_mtime_ns]==before,n;checked+=1
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(journal)==train['actual_optimizer_updates'] and len(journal)>=1920
    assert [r['actual_update'] for r in journal]==list(range(1,len(journal)+1))
    for phase in ('fit','validation','cal','test'):
        path=RUN/(phase+'-cache.json')
        if path.exists():assert sha(RUN/(phase+'-features.pt'))==read(path)['sha256']
    result={'status':'passed','parent_all_tensor_buffer_metadata_exact':True,'approved_three_heads_all_tensor_buffer_metadata_exact':True,
            'disk_delta_restore_exact':True,'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':checked,
            'whole_protected_source_SHA_recalculated':False,'actual_optimizer_updates':len(journal),'new_units_passed':3,
            'new_API_CPP_ZIP_checks':0,'default_model_unchanged':True,'delta_not_standalone_app_model':True}
    new(RUN/'verification.json',result);new(ROOT/'reports/industrial-insplad-lower-flow-v58-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        validate(True)
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job: '+str(path))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if not (RUN/'trained/training.json').exists():train()
            if read(RUN/'trained/training.json')['selected_epoch']>0:
                if not (RUN/'calibration.json').exists():calibrate()
                if not (RUN/'selection.json').exists():evaluate()
            elif not (RUN/'selection.json').exists():
                new(RUN/'selection.json',{'rejected':'epoch0 selected','new_CAL_TEST_scores':0,'source_candidate_eligible_pending_package_checks':False})
            if not (RUN/'verification.json').exists():verify()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
