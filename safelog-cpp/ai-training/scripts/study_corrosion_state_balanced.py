"""One controlled FIT-budget loss correction; v69 initial state and draws reused."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_state_mask import CorrosionStateMain,state_view,native_probabilities,confusion,rates
from safelog_ai.corrosion_json_pairs import embedded_image,polygon_labels
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_ostrava_rust_mask as base
from safelog_ai.corrosion_balanced_state_loss import weighted_state_loss
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

MANIFEST=ROOT/'data/industrial-corrosion-json-pairs-v68/manifest.json'
MANIFEST_SHA='af40f9f6a0cc91709c0006ec5c9e21c9d7dd94a93f079fe2401dc1dab56888ce'
PARENT=base.RUN/'trained/final-delta.pt'
PARENT_SHA='26c26bc5739930f153d1553b9b22840f4e83ba925cddeca5f46eee4f9f25a01b'
PARENT_STATE='61677d5f1d038006cbcfc83894af03a90b3105055ca94d5eef3c40567dc7332f'
RUN=ROOT/'runs/industrial-corrosion-state-balanced-v71'
PROTOCOL=ROOT/'reports/industrial-corrosion-state-balanced-v71-protocol.json'
SOURCES=('scripts/study_corrosion_state_balanced.py','safelog_ai/corrosion_balanced_state_loss.py','tests/test_corrosion_balanced_state_loss.py')
PREFIXES=base.PREFIXES+('corrosion_state_head.',)
EPOCHS,STEPS,BATCH,SEED=30,64,4,6969


INIT_RUN=ROOT/'runs/industrial-corrosion-state-v69'
DIAGNOSIS=ROOT/'runs/industrial-corrosion-state-balance-v70/diagnosis.json'
CLASS_WEIGHTS=(0.6987341772151905,0.7156439066551432,1.0739299610894926,4.160804020100503)


def state_loss(logits,target):return weighted_state_loss(logits,target,CLASS_WEIGHTS)


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(MANIFEST)==MANIFEST_SHA
    diagnostic=read(DIAGNOSIS);assert np.allclose(diagnostic['FIT']['candidate_inverse_expected_coefficient_weights_not_trained'],CLASS_WEIGHTS,atol=1e-14,rtol=0)
    assert diagnostic['selected6_model_state_before_after_exact'] and diagnostic['new_CAL_TEST_scores']==0
    assert diagnostic['VAL_head_views']['6']['metrics']['per_class_recall'][3]==0
    assert read(MANIFEST)['audit']['ready'] and sha(PARENT)==PARENT_SHA
    assert read(ROOT/'reports/industrial-corrosion-json-pair-v68-FIT-QA-review.json')['AI_visual_contact_sheet_reviewed']
    assert sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA
    assert shutil.disk_usage(ROOT).free>1024**3+256*1024**2
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                s=path.stat();protected[path.relative_to(ROOT).as_posix()]=[s.st_size,s.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    fit=rows_for('fit');groups=Counter(r['group_id'] for r in fit)
    weights=np.array([1/groups[r['group_id']] for r in fit]);weights/=weights.sum()
    np.save(RUN/'draws.npy',np.random.default_rng(SEED).choice(len(fit),(EPOCHS*STEPS,BATCH),replace=True,p=weights).astype(np.int32),allow_pickle=False)
    files=[PARENT,MANIFEST,ROOT/'reports/industrial-corrosion-json-pair-v68-data-audit.json',ROOT/'reports/industrial-corrosion-json-pair-v68-FIT-QA.json',ROOT/'reports/industrial-corrosion-json-pair-v68-FIT-QA-review.json',ROOT/'safelog_ai/corrosion_json_pairs.py',ROOT/'scripts/study_ostrava_rust_mask.py',ROOT/'safelog_ai/ostrava_rust_mask.py']
    files += [ROOT/rows_for('fit')[0]['archive'],DIAGNOSIS,INIT_RUN/'head-epoch-00.pt',INIT_RUN/'fit-input-binding.json',INIT_RUN/'validation-input-binding.json',INIT_RUN/'draws.npy',ROOT/'safelog_ai/corrosion_state_mask.py']
    assert sha(RUN/'draws.npy')==sha(INIT_RUN/'draws.npy')
    p={'UTC':utc(),'fixed_before_real_features_gradient_or_model_scores':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
       'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},'parent_file_sha256':PARENT_SHA,'parent_combined_state_sha256':PARENT_STATE,
       'original_v16_sha256':base.EXPECTED,'approved_v51_delta_sha256':base.PRIOR_SHA,'manifest_sha256':MANIFEST_SHA,
       'protected_stat_sha256':sha(RUN/'protected-stat.json'),'draws_sha256':sha(RUN/'draws.npy'),
       'legacy_Python_sha256':{v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES},
       'target':'Author polygon appearance0Good/background(no OR minimalcorrosion),1Fair,2Poor,3Severe,255ignore. These are not measureddepth/structuralgrade/allrustabsence/wholemachinehealth. v66JPEG/nativePNG contract staysFAILED; use only v68embeddedRGB+originalpolygons.',
       'data':read(MANIFEST)['audit']['counts'],'input':'Full embeddedRGB, aspectpreservingletterbox512: scale=min(512/w,512/h), round resizedsizes, centeredfloorpadding RGB(124,116,104), targetnearest with255padding. ImageNetnormalization. No GTchosen crop/augmentation/fakecategory/source/dateembedding.',
       'model':'Exactsamev69architecture/seed/draws andhead100692params, initializefromfixedv69epoch0head includingFITonlynormalization, notselected6/last30. Onlysourceobjective changes; allv16/approved3heads/researchv62 frozeneval exact; no oldmethodchanges.',
       'normalization':'Reusev69epoch0 FITonlymean/popstd/clampbuffers exactly, no new normalization fitting/features. TheywerecomputedfromFITletterbox cells includingpadding, FP16boundary/float64moments/std floor.001/clamp[-8,8]. Alloldbuffers untouched.',
       'loss':'Onlychange: multiplyoriginalper-presentclass categoricalCE+softDice byfixedFITweights1/(4*a_k), wherea_k=FITgroup-uniform expectation of1(classpresent)/n_present512. Do notrenormalizeweights perphoto; bothFITandVAL use sameweightedobjective,255excluded. Notgradientimportance/causalproof orbinaryzero-rust.', 'class_weights':list(CLASS_WEIGHTS), 'initializer_epoch0_sha256':sha(INIT_RUN/'head-epoch-00.pt'),
       'optimizer':'AdamW lr.0003 wd.01 cosine30 min.000001 gradclip5 FP32, no AMP, TF32off threads4; seed6969',
       'epochs':30,'steps_per_epoch':64,'batch':4,'planned_optimizer_updates':1920,
       'sampling':'Fixedgroup-uniformthenphoto1920x4 replacementdraws; repeats are not newnaturalphotos',
       'VAL_selection':'InternalVAL60 perphoto state_loss thenconnectedgroupmean, epoch0 andeachcompletedepoch, earlieststrictminimum. Epoch0 rejects before newCAL_TESTscores. No last-epoch retrospective promotion.',
       'CAL':'Afterselectedepoch>0 onlyCAL60. Fixed categorical softmax temperature1.0/argmax, no quantile/cut search or fittedtemperature. Save CALmetrics and fixed classificationpolicy BEFORE TESTfeatures/scores; wholehealthy normalCAL doesnotexist so85percentphotoquantile notapplicable.',
       'native_coordinates':'softmax512, crop centeredpadding, bilinear4probabilities tooriginalembeddedheight/width align_cornersFalse thenargmax. Compare authornativepolygonraster with255ignored, never bilinearGT. Record4x4confusion/perclassIoU-Dice-recall/native>=FairvsGoodFPR_FNR; these are not wholephotoFP_FN or allrustvszero-rust.',
       'TEST':'Once original44TEST afterCALjson; preserveall44. Sourcepointgate selectedepoch>0, eachstate>=5photos/groups>=3, macroIoU>=.40, annotated>=FairpixelFPRandFNR<=.20, eachofFair/Poor/Severe recall>=.25, parentexact. Fixed1000proxygroupconfusionbootstrap seed6970 is uncertaintyonly, not newmodeltrials or physicalgroupadjustment.',
       'scope':'Source4statepixelresearch only, wholehealthyphotoGT0; no newapp20percentphoto pass/oldaliasclearance/bracket-tube accuracy claim or automaticZIP. MissingnativePNGexactmatch cause remainsunresolved; thisdifferentprovidedJSONcontract doesnotrepairoldfiles.',
       'features_storage':'No extra densefeature diskcache. Keep onlyFIT276+VAL60 preprocesseduint8RGB512/uint8target in bounded512MiBRAM; nativebuffers sequential. Encoder onlineeachbatch,no gradient. CAL/TEST decodeonlineonlyatallowedstage. TEST512probability references<=256MiBRAM forconditionaldevicecheck, no fullnative map files/fullparentcopy. Newdelta/checkpoints/statmetadata<=256MiB disk,>=1GiBreserve.',
       'conditional_device_check':'Onlysourcegatepasses: all44originalembeddedRGB freshCPU andfreshGPU vs firstTESTGPU FP32prob512refs keptinRAM. atol.002/rtol.002 and512argmax disagreementfraction<=.001 perphoto; recordmax/nonexact differences, notnativepixelperfectidentity or Androidspeed/packageAPI/C++test.',
       'preservation':'Assertv62parentalltensor/buffer/meta exact; parent-bound originalv16delta allowed3approvedheadprefixes+ostrava_rust_head+corrosion_state_head only; diskrestoreallstate/meta exact, file/parent/added/combinedhashes. Preserveall30epochheads/optimizers, failurecheckpoint/log/actualstepjournal. No oldraw/model/report deletions.',
       'limits':'Publicsourcephotos withburned-indates, unknownphysicalpart/report/sitegroups; pHash/exif notidentityproof. No independentfieldperformance/0missguarantee. SevereTEST12photos islimitedsupport. No multiplearchitectures/losscoefficients/TESTthreshold tuning.',
       'license':'VT Bianchi/HebdonV2 dataCC0 cite10.7294/16624663.v2 and10.1061/(ASCE)CP.1943-5487.0001045 plusauthor/source/version/changes; oldInsPLADCC BY-NC3.0/OstravaCC BY4.0/otherparenttermsunchanged',
       'new_units_passed':3,'independent_field_validation':False}
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'FIT':len(fit),'planned_optimizer':1920,'protected_files':len(protected)},flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['source_sha256'],**p['files_sha256'],**p['legacy_Python_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(RUN/'protected-stat.json')==p['protected_stat_sha256'] and sha(RUN/'draws.npy')==p['draws_sha256']
    assert sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load():
    original,approved=base.research_parent();parent=restore_delta(original,torch.load(PARENT,map_location='cpu',weights_only=True),base.EXPECTED)
    assert_preserved(approved,parent);assert state_sha256(parent['state_dict'])==PARENT_STATE
    torch.manual_seed(SEED);model=CorrosionStateMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(parent['state_dict'],strict=False);assert not extra and all(n.startswith('corrosion_state_head.') for n in missing)
    initial=torch.load(INIT_RUN/'head-epoch-00.pt',map_location='cpu',weights_only=True);assert initial['epoch']==0 and initial['actual_updates']==0
    model.corrosion_state_head.load_state_dict(initial['head'])
    return model.eval().requires_grad_(False).cuda(),parent,original


def pair(archive,r,stage):
    raw=archive.read(r['source_json_member']);assert hashlib.sha256(raw).hexdigest()==r['source_json_encoded_sha256']
    a=json.loads(raw);rgb,encoded,meta=embedded_image(a);truth,_=polygon_labels(a,rgb.size)
    assert hashlib.sha256(encoded).hexdigest()==r['embedded_encoded_sha256'] and hashlib.sha256(truth.tobytes()).hexdigest()==r['class_raster_sha256']
    view,target,geometry=state_view(rgb,truth)
    with (RUN/'input-events.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps({'UTC':utc(),'id':r['id'],'stage':stage,'embedded_decode_polygon_raster':True})+'\n')
    return view,target,geometry,truth


def prepared(phase):
    rows=rows_for(phase);xs=[];ys=[]
    with zipfile.ZipFile(ROOT/rows[0]['archive']) as archive:
        for r in rows:
            x,y,_,_=pair(archive,r,'prepare_'+phase);xs.append(torch.from_numpy(x).permute(2,0,1));ys.append(torch.from_numpy(y))
    x=torch.stack(xs);y=torch.stack(ys);assert x.numel()+y.numel()<=512*1024**2
    new(RUN/(phase+'-input-binding.json'),{'ids':[r['id'] for r in rows],'manifest_sha256':MANIFEST_SHA,'protocol_sha256':sha(PROTOCOL),
        'uint8_RGB512_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'uint8_target512_sha256':hashlib.sha256(y.numpy().tobytes()).hexdigest(),'bytes_in_RAM':x.numel()+y.numel(),'feature_disk_cache':False})
    binding=read(INIT_RUN/(phase+'-input-binding.json'));current=read(RUN/(phase+'-input-binding.json'))
    assert binding['ids']==current['ids'] and binding['uint8_RGB512_sha256']==current['uint8_RGB512_sha256'] and binding['uint8_target512_sha256']==current['uint8_target512_sha256']
    return rows,x,y


def features(model,x,stage,ids):
    device=next(model.parameters()).device;x=x.to(device).float()/255
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    with torch.no_grad():
        result=forward_with_fine(model.backbone,(x-mean)/std)
        if stage=='fit_normalization' and not (RUN/'first-feature-contract.json').exists():
            old=model.backbone((x-mean)/std);assert all(torch.equal(old[n],result[n]) for n in ('low','high'))
            new(RUN/'first-feature-contract.json',{'ordinary_low_high_bitwise_exact':True,'comparison_inputs':len(ids),'not_extra_photos':True,'geometry':'letterbox512'})
    with (RUN/'feature-events.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps({'stage':stage,'ids':ids,'encoder_inputs':len(ids)})+'\n')
    return {n:result[n].half() for n in ('fine','low','high')}


def loss_on(model,x,y,rows,stage):
    model.corrosion_state_head.eval();values=[]
    with torch.no_grad():
        for i in range(0,len(rows),BATCH):
            f=features(model,x[i:i+BATCH],stage,[r['id'] for r in rows[i:i+BATCH]])
            values.extend(state_loss(model.forward_corrosion_state_features(**f),y[i:i+BATCH].cuda()).cpu().tolist())
    bygroup=defaultdict(list)
    for r,v in zip(rows,values):bygroup[r['group_id']].append(v)
    return float(np.mean([np.mean(v) for v in bygroup.values()]))


def train(model,parent,original):
    fit,x,y=prepared('fit');val,vx,vy=prepared('validation');assert x.numel()+y.numel()+vx.numel()+vy.numel()<=512*1024**2
    head=model.corrosion_state_head;head.requires_grad_(True)
    new(RUN/'normalization-reuse.json',{'v69_epoch0_head_sha256':sha(INIT_RUN/'head-epoch-00.pt'),'class_weights':list(CLASS_WEIGHTS),'no_new_FIT_normalization_forward':True})
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==100692
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    draws=np.load(RUN/'draws.npy',allow_pickle=False);actual=0;best=loss_on(model,vx,vy,val,'initial_VAL');bestepoch=0
    torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'head-epoch-00.pt')
    new(RUN/'initial-VAL.json',{'loss':best,'epoch':0});journal=(RUN/'optimizer.jsonl').open('x',encoding='utf-8',buffering=1)
    try:
        for epoch in range(1,EPOCHS+1):
            losses=[];head.train()
            for step in range(STEPS):
                ids=draws[(epoch-1)*STEPS+step];f=features(model,x[ids],'fit_optimizer',[fit[i]['id'] for i in ids])
                loss=state_loss(model.forward_corrosion_state_features(**f),y[ids].cuda()).mean();assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();actual+=1
                losses.append(float(loss.detach()));journal.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'loss':losses[-1]})+'\n')
            scheduler.step();score=loss_on(model,vx,vy,val,'VAL_epoch_'+str(epoch))
            if score<best:best,bestepoch=score,epoch
            torch.save({'head':head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/f'head-epoch-{epoch:02d}.pt')
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/f'optimizer-epoch-{epoch:02d}.pt')
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_loss':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    except BaseException:
        torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/('failure-checkpoint-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.pt'));raise
    finally:journal.close()
    chosen=torch.load(RUN/f'head-epoch-{bestepoch:02d}.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval().requires_grad_(False)
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':bestepoch,'selected_actual_updates':chosen['actual_updates'],'selected_VAL':best,'FIT_photos':len(fit),'VAL_photos':len(val),'actual_FIT_state_polygons':True,'objective':'FITclassbudgetweighted CE+Dice','class_weights':list(CLASS_WEIGHTS)}
    result=deepcopy(parent);result['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()}
    result.update(corrosion_state_protocol_sha256=sha(PROTOCOL),corrosion_state_training=summary,corrosion_state_scope='VT suppliedpolygon fourappearance states; research only, notwholehealth')
    assert_preserved(parent,result);payload=pack_delta(original,result,base.EXPECTED,base.START.relative_to(ROOT).as_posix(),PREFIXES)
    (RUN/'trained').mkdir();torch.save(payload,RUN/'trained/final-delta.pt')
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED);assert_preserved(parent,restored)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in result['state_dict'].items()) and all(v==restored[n] for n,v in result.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary)
    return summary


def evaluate(model,phase):
    if phase=='test':assert (RUN/'calibration.json').exists()
    rows=rows_for(phase);matrices=[];references=[];photos=[];model.eval()
    with zipfile.ZipFile(ROOT/rows[0]['archive']) as archive,torch.inference_mode():
        for r in rows:
            view,target,g,truth=pair(archive,r,'native_'+phase);x=torch.from_numpy(view).permute(2,0,1)[None]
            f=features(model,x,phase,[r['id']]);prob=model.forward_corrosion_state_features(**f).softmax(1)[0]
            native=native_probabilities(prob,g);matrix=confusion(native,torch.from_numpy(truth).to(native.device)).cpu().numpy();matrices.append(matrix)
            photos.append({'id':r['id'],'group_id':r['group_id'],**rates(matrix),'probability512_sha256':hashlib.sha256(prob.cpu().numpy().tobytes()).hexdigest()})
            if phase=='test':references.append(prob.cpu())
    result={'phase':phase,'metrics':rates(np.sum(matrices,axis=0)),'photos':photos,'pixel_metrics_not_photo_error_rates':True}
    if phase=='test':
        bygroup=defaultdict(lambda:np.zeros((4,4),dtype=np.int64))
        for r,c in zip(rows,matrices):bygroup[r['group_id']]+=c
        values=np.stack([bygroup[k] for k in sorted(bygroup)]);rng=np.random.default_rng(6970);samples=[]
        for _ in range(1000):
            rr=rates(values[rng.integers(len(values),size=len(values))].sum(0));samples.append([rr['macroIoU'],rr['at_least_Fair_pixel_FPR'],rr['at_least_Fair_pixel_FNR']])
        result['proxygroup_bootstrap95']={'macroIoU_FPR_FNR':np.quantile(samples,[.025,.975],axis=0).tolist(),'not_physical_group_independence_adjustment':True}
    return result,references


def devices(model,references):
    rows=rows_for('test');results=[]
    for device in ('cpu','cuda'):
        model.to(device);maximum=0.;disagreements=0;pixels=0
        with zipfile.ZipFile(ROOT/rows[0]['archive']) as archive,torch.inference_mode():
            for r,ref in zip(rows,references):
                view,_,_,_=pair(archive,r,'fresh_device_'+device);x=torch.from_numpy(view).permute(2,0,1)[None]
                f=features(model,x,'fresh_'+device,[r['id']]);p=model.forward_corrosion_state_features(**f).softmax(1)[0].cpu()
                maximum=max(maximum,float((p-ref).abs().max()));assert torch.allclose(p,ref,atol=.002,rtol=.002)
                changed=int((p.argmax(0)!=ref.argmax(0)).sum());assert changed/(512*512)<=.001
                disagreements+=changed;pixels+=512*512
        results.append({'device':device,'fresh_embedded_inputs':len(rows),'max_probability512_abs_difference':maximum,'argmax512_changed_pixels':disagreements,'argmax512_pixels':pixels})
    new(RUN/'device-compatibility.json',{'status':'passed','checks':results,'not_native_pixel_exact_or_APK_package_check':True})


def verify(model,parent,original,p):
    current=deepcopy(parent);current['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};assert_preserved(parent,current)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in current['state_dict'].items())
    for n,h in p['legacy_Python_sha256'].items():assert sha(ROOT/n)==h,n
    protected=read(RUN/'protected-stat.json')
    for n,(size,mtime) in protected.items():
        st=(ROOT/n).stat();assert (st.st_size,st.st_mtime_ns)==(size,mtime),n
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['actual_update'] for r in journal]==list(range(1,1921))
    events=[json.loads(v) for v in (RUN/'feature-events.jsonl').read_text(encoding='utf-8').splitlines()]
    new(RUN/'verification.json',{'status':'passed','parent_v62_all_tensor_buffer_metadata_exact':True,'delta_disk_restore_exact':True,'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':len(protected),'all_original_raw_SHA_recalculated':False,'actual_optimizer_updates':len(journal),'encoder_input_counts_by_stage':dict(Counter({s:sum(r['encoder_inputs'] for r in events if r['stage']==s) for s in {r['stage'] for r in events}})),'extra_ordinary_forward_inputs':0,'delta_sha256':sha(RUN/'trained/final-delta.pt'),'combined_state_sha256':state_sha256(restored['state_dict']),'new_API_CPP_ZIP_APK_checks':0})


def run():
    p=validate();assert not (RUN/'trained/training.json').exists();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,parent,original=load();summary=train(model,parent,original)
    if summary['selected_epoch']>0:
        cal,_=evaluate(model,'cal');new(RUN/'calibration.json',{'classification_policy':'softmax temperature1.0, fixedargmax, no photoquantile or threshold fit','CAL':cal,'saved_before_TEST_features_scores':True})
        test,refs=evaluate(model,'test');new(RUN/'test-metrics.json',test);m=test['metrics']
        support={str(k):{'photos':sum(r['class_pixels'].get(str(k),0)>0 for r in rows_for('test')),
                         'groups':len({r['group_id'] for r in rows_for('test') if r['class_pixels'].get(str(k),0)>0})} for k in range(4)}
        accepted=all(v['photos']>=5 and v['groups']>=3 for v in support.values()) and m['macroIoU']>=.4 and m['at_least_Fair_pixel_FPR']<=.2 and m['at_least_Fair_pixel_FNR']<=.2 and min(m['per_class_recall'][1:])>=.25
        new(RUN/'selection.json',{'source_four_state_research_gate_passed':accepted,'automatic_app_adoption':False,'wholephoto_gate_passed':False,'TEST_state_support':support,'metrics':m})
        if accepted:devices(model,refs)
    else:new(RUN/'selection.json',{'source_four_state_research_gate_passed':False,'reason':'epoch0 minimum, no newCAL_TEST scores','automatic_app_adoption':False})
    verify(model,parent,original,p);print(read(RUN/'selection.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:run()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()});raise
        finally:(RUN/'operation.lock').unlink()
