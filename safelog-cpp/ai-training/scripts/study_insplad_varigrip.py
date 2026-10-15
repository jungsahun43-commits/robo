"""One three-condition vari-grip study preserving all three approved heads and base."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.insplad_varigrip import InspladVariGripMain,CONDITIONS,condition_decisions
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.gearbox_supervised import gearbox_descriptor
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

START=ROOT/'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED='5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
MANIFEST=ROOT/'data/industrial-insplad-varigrip-v53/manifest.json'
RUN=ROOT/'runs/industrial-insplad-varigrip-v54'
PROTOCOL=ROOT/'reports/industrial-insplad-varigrip-v54-protocol.json'
SOURCES=('safelog_ai/insplad_varigrip.py','scripts/study_insplad_varigrip.py','tests/test_insplad_varigrip.py')
EPOCHS,STEPS,BATCH,SEED=30,64,36,5454


PRIOR_DELTA=ROOT/'runs/industrial-insplad-suspension-v51/trained/final-delta.pt'
PRIOR_SHA='b91b6578eccd1886c5be18805a9475875ff319f77bda019cd0b3f518d84f46dc'
PRIOR_COMBINED='bb388200af167ffde35f6281a5e4626be904f9026411c401151e9bc3b4d7c45a'


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(START)==EXPECTED and sha(PRIOR_DELTA)==PRIOR_SHA
    assert sha(MANIFEST)=='fc2fe5560a94a787dc6d0866d10d3bec8ec673e625a57534b64d5d8dce1ea59b'
    audit=read(ROOT/'reports/industrial-insplad-varigrip-v53-data-audit.json');assert audit['ready'] and audit['manifest_sha256']==sha(MANIFEST)
    review=read(ROOT/'reports/industrial-insplad-varigrip-v53-FIT-review.json');assert not review['GT_modified']
    assert read(ROOT/'runs/industrial-insplad-suspension-release-v52/release.json')['status']=='passed'
    assert shutil.disk_usage(ROOT).free>1024**3+100*1024**2
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                st=path.stat();protected[path.relative_to(ROOT).as_posix()]=[st.st_size,st.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    fit=rows_for('fit');rng=np.random.default_rng(SEED);draws=[];pools={};weights={}
    for condition in CONDITIONS:
        ids=[i for i,r in enumerate(fit) if r['condition']==condition];counts=Counter(fit[i]['group_id'] for i in ids)
        w=np.array([1/counts[fit[i]['group_id']] for i in ids]);pools[condition]=ids;weights[condition]=w/w.sum()
    for _ in range(EPOCHS*STEPS):
        draws.append([i for condition in CONDITIONS for i in rng.choice(pools[condition],BATCH//3,replace=True,p=weights[condition]).tolist()])
    np.save(RUN/'draws.npy',np.array(draws,dtype=np.int32),allow_pickle=False)
    p={'UTC':utc(),'schema':'safelog-insplad-varigrip-v54','fixed_before_features_gradients_or_model_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':EXPECTED,'manifest_sha256':sha(MANIFEST),
       'approved_three_head_delta_sha256':PRIOR_SHA,'approved_three_head_combined_state_sha256':PRIOR_COMBINED,
       'data_audit_sha256':sha(ROOT/'reports/industrial-insplad-varigrip-v53-data-audit.json'),
       'FIT_review_sha256':sha(ROOT/'reports/industrial-insplad-varigrip-v53-FIT-review.json'),
       'prior_calibration_sha256':{n:sha(ROOT/n) for n in ('runs/industrial-insplad-corrosion-v45/calibration.json','runs/industrial-insplad-glass-v48/calibration.json','runs/industrial-insplad-suspension-v51/calibration.json')},
       'legacy_Python_sha256':{f.relative_to(ROOT).as_posix():sha(f) for d in ('safelog_ai','scripts','tests') for f in (ROOT/d).glob('*.py') if f.relative_to(ROOT).as_posix() not in SOURCES},
       'protected_stat_sha256':sha(RUN/'protected-stat.json'),'draws_sha256':sha(RUN/'draws.npy'),'split':audit['counts'],
       'source_scope':'ONLY original cropped InsPLAD vari-grip good/bird-nest/rust photo annotations. Exterior visual condition, not electrical/internal/wholemachinehealth or pixelGT.',
       'condition_order':list(CONDITIONS),'label_contract':'headindex0good/1bird-nest/2rust from literal condition names. Existing sourcebinary label is preserved; originalauthornumeric1good is not a defect.',
       'license':'InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1 CC BY-NC3.0 noncommercialacademic author/source/version/change attribution paper10.1080/01431161.2023.2283900, inherited licenses unchanged',
       'FIT_QA':'fixed first3FIT percondition ninephotos only: brightnormal metal against pale sky; twigs/nest near attachment; brownrust on longarms. Two rust examples sameframe/similarpart. OriginalGT unchanged; no VAL_CAL_TEST content view or all-data humanreview.',
       'input':'original RGB bicubic256square ImageNet normalization. No augmentation/GTcrop/extra source orcapture input.',
       'features':'same frozen sharedencoder once perRGB fine24/low40/high960 FP16roundtrip maps, mean/popstd/max3072 FP16roundtrip. No secondencoder or source/category embedding.',
       'architecture':'fresh insplad_varigrip_classifier3072to128/GELU/Dropout.2/to3. PyTorchdefaultfirstLinear seed5454, lastLinearzero, no oldlearnedheadcopy.',
       'normalization':'FIT142 descriptors only perchannelmean/popstd floor.001 clamp[-8,8]; fixed buffers thereafter',
       'trainable':'new varigripclassifier.head393731params only. v16 and v45corrosion/v48glass/v51suspension allstate/meta exact fixed.',
       'sampling':'fixed1920x36 draws12percondition, inverse percondition/group photosize;69120draws not69120newnaturalphotos',
       'loss':'mean3class crossentropy on actual named original conditions only. No binarycollapse for headtraining, pseudo/synthetic/pixelGT.',
       'optimizer':'AdamW lr.0003 wd.01 cosine30 min.000001 clip5 FP32 TF32off threads4',
       'epochs':30,'steps_per_epoch':64,'batch':36,'seed':SEED,'planned_optimizer':1920,
       'VAL_selection':'macro condition/group crossentropy epoch0 andeachcompleteepoch, earlieststrictminimum; epoch0selection rejects beforeCAL_TEST',
       'CAL':'only normalCAL26, score=1-softmaxgood, alpha.15 rankceil((26+1)*.85)=23 strictscore>cut savedbeforeTESTfeatures/scores. Scores not calibrated failureprobabilities.',
       'decision':'if overallnongoodscore<=CALcut predictgood0; otherwise argmax overbird-nest/rust softmaxcomponents+1. No condition-specificcuts.',
       'TEST':'all281 originalpublisherVAL good238/bird-nest23/rust20, fixedcheckpoint/cut once; report binaryscreening and typed FN separately; wrongfaulttype is a typedmiss even if overallcandidateflagtrue.',
       'adoption_guards':['selectedepoch>0','TESTeachcondition>=20 andeachconnectedgroups>=3','normalphotoFPR<=.20','overallfaultscreeningFNR<=.20','bird-nest typedFNR<=.20 andrust typedFNR<=.20, nofaulttypecollapse torescue','v16ANDallthreeapprovedheadstate/metaexact','fresh ZIP_CPU_GPU_alloldalias_oldPNG_actualCPP beforehandoff'],
       'cache':'pooled only<8MiB allphases, IDs/rawSHA/order/condition/RGB256/dtype/parentSHA/priorDeltaSHA/protocolSHA bound;FIT_VALfirst,CALafterselection,TESTafterCALsaved',
       'delta':'pack againstoriginalv16 allowedprefixes insplad_classifier.+insplad_glass_classifier.+insplad_suspension_classifier.+insplad_varigrip_classifier. only; assert prior3heads exact beforepack/afterdiskrestore, parent/state/added/combinedSHA and serializedrestoreexact, notstandalone appmodel',
       'location':'no new verifiedpixel map, alloldlocation/pixelthresholds preserved',
       'previous_tasks':'no oldtaskcut/weights changed, no bracket/tubeimprovement claimed',
       'data_limits':'FIT12bird/15rust photos not27verifiedparts; VAL4bird/3rust only andnormal26CAL small. VALno datehasgoodandfault, TEST5dateshaveboth but physicalpart/flight/siteindependence notverified.',
       'storage':'cache/progress/delta<100MiB,>=1GiBreserve, no fullparent435MBduplicates/densecache/deletions',
       'independent_field_validation':False,'actual_new_FIT_fault_photographs':27,'new_units_passed':3,'auto_app_adoption':False,
       'resume':'percompleteepoch optimizer/scheduler/RNG/newhead, actualstepjournal preserves partialrepeatedsteps/failures'}
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'protected_files':len(protected),'planned_optimizer':1920},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(PRIOR_DELTA,PRIOR_SHA),(MANIFEST,p['manifest_sha256']),(RUN/'draws.npy',p['draws_sha256']),(RUN/'protected-stat.json',p['protected_stat_sha256'])):assert sha(path)==h,path
    for name,h in p['prior_calibration_sha256'].items():assert sha(ROOT/name)==h,name
    assert sha(ROOT/'reports/industrial-insplad-varigrip-v53-FIT-review.json')==p['FIT_review_sha256']
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix(),'reports/industrial-insplad-varigrip-v53-data-audit.json'):
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
        state=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),EXPECTED)
        assert_preserved(parent,state)
    torch.manual_seed(SEED)
    model=InspladVariGripMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and all(n.startswith('insplad_varigrip_classifier.') for n in missing)
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
    if phase=='cal':assert (RUN/'trained/training.json').exists()
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache.json'))['sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['ids']==[r['id'] for r in rows] and value['protocol_sha256']==sha(PROTOCOL)
        assert value['conditions']==[r['condition'] for r in rows]
        return rows,value['features'].float().cuda()
    values=[]
    with torch.inference_mode():
        for i in range(0,len(rows),16):
            x=images(rows[i:i+16]).cuda();f=forward_with_fine(model.backbone,x)
            if phase=='fit' and i==0:
                old=model.backbone(x);assert all(torch.equal(old[k],f[k]) for k in ('low','high'))
            values.append(gearbox_descriptor(f).half().cpu())
    value={'ids':[r['id'] for r in rows],'raw_sha256':[r['image_sha256'] for r in rows],
           'conditions':[r['condition'] for r in rows],'features':torch.cat(values),'manifest_sha256':sha(MANIFEST),'protocol_sha256':sha(PROTOCOL),
           'encoder_parent_sha256':EXPECTED,'approved_three_head_delta_sha256':PRIOR_SHA,'input':'RGBbicubic256square_ImageNet','dtype':'FP16descriptor'}
    with path.open('xb') as out:torch.save(value,out)
    new(RUN/(phase+'-cache.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'photos':len(rows)})
    print({'cached':phase,'photos':len(rows)},flush=True)
    return rows,value['features'].float().cuda()


def macro_loss(values,rows):
    return torch.stack([torch.stack([values[[i for i,r in enumerate(rows) if r['condition']==condition and r['group_id']==group]].mean()
        for group in sorted({r['group_id'] for r in rows if r['condition']==condition})]).mean() for condition in CONDITIONS]).mean()


def train():
    validate(True);assert not (RUN/'trained/training.json').exists()
    model,parent,original=load();fit,x=cache('fit',model);val,vx=cache('validation',model);head=model.insplad_varigrip_classifier
    head.mean.copy_(x.mean(0));head.std.copy_(x.std(0,correction=0).clamp_min(1e-3));head.head.requires_grad_(True)
    y=torch.tensor([CONDITIONS.index(r['condition']) for r in fit],device='cuda',dtype=torch.long);vy=torch.tensor([CONDITIONS.index(r['condition']) for r in val],device='cuda',dtype=torch.long)
    assert sum(t.numel() for t in model.parameters() if t.requires_grad)==393731
    assert all(n.startswith('insplad_varigrip_classifier.head.') for n,t in model.named_parameters() if t.requires_grad)
    opt=torch.optim.AdamW(head.head.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    draws=np.load(RUN/'draws.npy',allow_pickle=False);journal=RUN/'optimizer.jsonl';progress=RUN/'progress.pt'
    start=0;bestepoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        assert not journal.exists();journal.touch(exist_ok=False)
        with torch.no_grad():best=float(macro_loss(F.cross_entropy(head(vx),vy,reduction='none'),val))
        new(RUN/'initial-VAL.json',{'cross_entropy':best,'epoch':0})
        torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines())
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            losses=[];head.head.train()
            for step in range(STEPS):
                ids=draws[(epoch-1)*STEPS+step];loss=F.cross_entropy(head(x[ids]),y[ids])
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.head.parameters(),5);opt.step()
                actual+=1;out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'cross_entropy':float(loss.detach())})+'\n');losses.append(float(loss.detach()))
            scheduler.step();head.head.eval()
            with torch.no_grad():score=float(macro_loss(F.cross_entropy(head(vx),vy,reduction='none'),val))
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'cross_entropy':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head'])
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_macro_condition_group_CE':chosen['VAL'],'FIT_photos':len(fit),'actual_FIT_defect_photos':sum(r['condition']!='good' for r in fit),'VAL_photos':len(val)}
    result=deepcopy(parent);result['state_dict']={n:t.detach().cpu() for n,t in model.state_dict().items()}
    result.update(insplad_varigrip_protocol_sha256=sha(PROTOCOL),insplad_varigrip_training=summary,insplad_varigrip_scope='cropped_vari_grip_good_bird_nest_rust')
    assert_preserved(parent,result)
    payload=pack_delta(original,result,EXPECTED,START.relative_to(ROOT).as_posix(),('insplad_classifier.','insplad_glass_classifier.','insplad_suspension_classifier.','insplad_varigrip_classifier.'))
    (RUN/'trained').mkdir()
    with (RUN/'trained/final-delta.pt').open('xb') as out:torch.save(payload,out)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),EXPECTED)
    assert_preserved(parent,restored)
    assert all(torch.equal(t,restored['state_dict'][n]) for n,t in result['state_dict'].items())
    assert all(v==restored[n] for n,v in result.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary)


def calibrate():
    training=read(RUN/'trained/training.json');assert training['selected_epoch']>0 and not (RUN/'calibration.json').exists()
    model,_,_=load(delta=True);rows,x=cache('cal',model)
    with torch.no_grad():probabilities=model.forward_insplad_varigrip_descriptor(x).softmax(1)
    scores=(1-probabilities[:,0]).tolist()
    cut,rank=normal_budget_threshold([s for r,s in zip(rows,scores) if r['condition']=='good'],alpha=.15)
    new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in rows],'condition_order':list(CONDITIONS),'probabilities':probabilities.tolist(),'scores':scores})
    new(RUN/'calibration.json',{'delta_sha256':sha(RUN/'trained/final-delta.pt'),'cut':cut,'rank':rank,'normal_photos':sum(r['condition']=='good' for r in rows),'alpha':.15,'TEST_features_or_scores_before_this':0})


def evaluate():
    assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['delta_sha256']==sha(RUN/'trained/final-delta.pt')
    model,_,_=load(delta=True);rows,x=cache('test',model)
    with torch.no_grad():probabilities=model.forward_insplad_varigrip_descriptor(x).softmax(1)
    score_tensor,prediction=condition_decisions(probabilities,cal['cut']);scores=score_tensor.tolist();predictions=prediction.tolist()
    labels=[int(r['condition']!='good') for r in rows];rates=classification_rates(labels,scores,cal['cut'])
    typed={}
    for index,condition in enumerate(CONDITIONS):
        ids=[i for i,r in enumerate(rows) if r['condition']==condition]
        typed[condition]={'photos':len(ids),'groups':len({rows[i]['group_id'] for i in ids}),
             'correct':sum(predictions[i]==index for i in ids),'misses':sum(predictions[i]!=index for i in ids),
             'error_rate':sum(predictions[i]!=index for i in ids)/len(ids),
             'predicted_good':sum(predictions[i]==0 for i in ids),
             'predicted_bird_nest':sum(predictions[i]==1 for i in ids),'predicted_rust':sum(predictions[i]==2 for i in ids)}
    eligible=rates['fpr']<=.2 and rates['fnr']<=.2 and all(v['photos']>=20 and v['groups']>=3 for v in typed.values())
    eligible=eligible and all(typed[c]['error_rate']<=.2 for c in CONDITIONS[1:]) and read(RUN/'trained/training.json')['selected_epoch']>0
    by={}
    for date in sorted({r['frame_proxy'].split('_DJI_')[0] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['frame_proxy'].startswith(date+'_DJI_')]
        by[date]={'binary_screening':classification_rates([labels[i] for i in ids],[scores[i] for i in ids],cal['cut']),
                  'typed':{c:{'photos':sum(rows[i]['condition']==c for i in ids),'misses':sum(rows[i]['condition']==c and predictions[i]!=j for i in ids)} for j,c in enumerate(CONDITIONS)}}
    result={'UTC':utc(),'binary_screening_rates':rates,'typed_condition_results':typed,'binary_AUROC':rank_auroc(labels,scores),'threshold':cal['cut'],
        'training':read(RUN/'trained/training.json'),'delta_sha256':cal['delta_sha256'],
        'source_candidate_eligible_pending_package_checks':bool(eligible),'by_capture_date':by,
        'independent_field_validation':False,'whole_machine_health_or_verified_pixel_localization':False,
        'old_bracket_tube_improvement_tested':False,'new_API_CPP_ZIP_checks':0,'accepted_v16_and_three_head_overlay_unchanged':True}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in rows],'condition_order':list(CONDITIONS),'probabilities':probabilities.tolist(),'scores':scores,'typed_predictions':predictions})
    new(RUN/'selection.json',result);new(ROOT/'reports/industrial-insplad-varigrip-v54-selection.json',result)
    print({k:v for k,v in result.items() if k!='by_capture_date'},flush=True)


def verify():
    p=validate();train=read(RUN/'trained/training.json');original,parent=research_parent()
    payload=torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True);restored=restore_delta(original,payload,EXPECTED);assert_preserved(parent,restored)
    for n,t in parent['state_dict'].items():assert torch.equal(t,restored['state_dict'][n]),n
    for n,v in parent.items():
        if n!='state_dict':assert v==restored[n],n
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
    result={'status':'passed','parent_all_tensor_buffer_metadata_exact':True,'approved_corrosion_glass_suspension_all_tensor_buffer_metadata_exact':True,'disk_delta_restore_exact':True,
            'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':checked,
            'whole_protected_source_SHA_recalculated':False,'actual_optimizer_updates':len(journal),'new_units_passed':3,
            'new_API_CPP_ZIP_checks':0,'accepted_v16_unchanged':True,'delta_not_standalone_app_model':True}
    new(RUN/'verification.json',result);new(ROOT/'reports/industrial-insplad-varigrip-v54-verification.json',result);print(result,flush=True)


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
            train()
            if read(RUN/'trained/training.json')['selected_epoch']>0:calibrate();evaluate()
            else:new(RUN/'selection.json',{'rejected':'epoch0 selected','new_CAL_TEST_scores':0,'source_candidate_eligible_pending_package_checks':False})
            verify()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
