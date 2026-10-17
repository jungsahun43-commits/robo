"""One paired cross-view teacher-retention candidate, preregistered before gradients."""
import argparse
from contextlib import ExitStack
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
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_teacher_retention import CorrosionRetainedMain,retained_weights,teacher_tile,teacher_kl
from safelog_ai.corrosion_background_roi import target512
from safelog_ai.corrosion_partial_mask import native_probability
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.process_probe import process_exists
from safelog_ai.ostrava_rust_mask import pixel_rates
from scripts import study_corrosion_native as native
from scripts import study_ostrava_rust_mask as base
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-corrosion-teacher-retention-v83'
PROTOCOL=ROOT/'reports/industrial-corrosion-teacher-retention-v83-protocol.json'
INIT=ROOT/'runs/industrial-corrosion-native-v81/trained/final-delta.pt'
INIT_SHA='f5938468e7acb4b5dfbb7025b0c986c8e5a2cfc90bfe3c0f5ec6431fa983cf10'
SOURCES=('scripts/study_corrosion_teacher_retention.py','safelog_ai/corrosion_teacher_retention.py','tests/test_corrosion_teacher_retention.py')
PREFIXES=base.PREFIXES+('corrosion_retained_head.',)
SEED=8383
native.RUN=RUN


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(INIT)==INIT_SHA
    diag=read(ROOT/'reports/industrial-corrosion-views-v82-diagnosis.json')
    assert diag['model_state_before_after_exact'] and diag['new_optimizer_gradient_VAL_CAL_TEST_visual_API_CPP_ZIP']==0
    free=shutil.disk_usage(ROOT).free;assert free>=1024**3+256*1024**2
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                s=path.stat();protected[path.relative_to(ROOT).as_posix()]=[s.st_size,s.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    rios=np.stack([np.random.default_rng(SEED+e).permutation(15) for e in range(30)]).astype(np.int32)
    ost=np.concatenate([np.random.default_rng(SEED+100+b).permutation(60) for b in range(8)])[:450].reshape(30,15).astype(np.int32)
    np.savez(RUN/'paired-orders.npz',rios=rios,ostrava=ost)
    files=[INIT,native.PARENT,base.START,base.PRIOR_DELTA,native.MANIFEST,base.MANIFEST,ROOT/'reports/industrial-corrosion-views-v82-diagnosis.json']
    files += [base.RUN/(phase+suffix) for phase in ('fit','validation') for suffix in ('-features.pt','-cache.json')]
    files += [base.PROTOCOL,ROOT/'reports/industrial-corrosion-native-v81-protocol.json',ROOT/'scripts/study_corrosion_native.py',ROOT/'scripts/study_corrosion_partial.py',ROOT/'scripts/prepare_corrosion_partial.py',ROOT/'safelog_ai/corrosion_native_mask.py',ROOT/'safelog_ai/native_corrosion_grid.py']
    p={'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_source_gradient_optimizer':True,'sources':{n:sha(ROOT/n) for n in SOURCES},'inputs':{v.relative_to(ROOT).as_posix():sha(v) for v in files},'paired_orders_SHA256':sha(RUN/'paired-orders.npz'),'protected_stat_SHA256':sha(RUN/'protected-stat.json'),
       'legacy_Python_sha256':{v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES},
       'FIT_evidence':diag['FIT_aggregates'],
       'hypothesis':'One cross-view response-retention regularizer addresses FIT evidence of both input and head drift; no repeat of positive-only mixedmask or loss coefficient/cut sweep. Oldv81 remains excluded, used only as explicitly failed initializer. Teacher is originalv62 whole512 on its own locked FIT/VAL, not TEST and not new natural annotation.',
       'model':'One unchanged common encoder; newcorrosion_retained_head100593params initialized exactly from selectedfailedv81corrosion_native_head inclnorm. All v16/approved3heads/originalv62 tensors/buffers/meta fixed; no failedv81branch is added as frozen deployed state. Newforward_corrosion_retained_features/tiles; internalcorrosion_native compatibilityproperty/method only lets unchangedv81gridhelpers call newhead without duplicate state. Existingforward_surface/RuntimeRealMaskMain.forward_real_mask untouched.',
       'data':'Locked RiosFIT15/VAL6/CAL6/TEST7 plus originalOstravaFIT60/VAL15 as regularization only; v61/v78 roles, graphs, rawGT unchanged. No newphotos/splits/audits or GT changes. Sparse ROI unknown255 preserved. ActualOstravamask0/169 usesL>0. Models responses are regularization, not actualnewGT.',
       'input':'Same single native512stride384 full input-size grid/edge/rgbpad124116104/unknown255/overlapuniformprobability asv81. CAL_TEST alltiles regardlessGT; no newstride/size/pooling. Teacher originalOstravaWHOLE512 features from unchanged completedv62 cache, IDs/raw/mask/order/encoder/protocol/FP16 SHA checked. Teacherheadoutputs75views inRAM only; no new teacherencoder extraction.',
       'normalization':'Copy selectedv81buffers exactly, equaloriginalv62FITnorm; no newfitting. FP32encoder->FP16maps->FP32head/std.001/clamp[-8,8], noAMP/TF32off/threads4.',
       'loss':'Each update L=.5*RiosNativeBalancedKnownBCE + .5*OstravaNativeClassBalancedKL(originalv62WHOLE512teacher || newnativestudent). Eachsourcephoto objective sums over alltiles with1/coverage divided by UNIQUE native GTclasspixelcount and numberofpresentclasses. Rios bothknownclasseseach.5; Ostrava fullnegative usespresentbackgroundonly. Original fullGT only defines class weights; teacherBernoulliprobability is softretentiontarget, not GT. KL= BCEwithteacher -teacherentropy, teacherclamp1e-6/1-1e-6, clampnumericalnegativeKLto0. Unknown/paddingzero. NoDice ornew supervisedOstravaBCE, no VT data.',
       'sampling':'30epochs, eachsavedpermutation of15Riosonce, pair eachwithoneOstravaFIT. Ostrava preseeded consecutive60permutationblocks slicedinto30x15: eachFITphoto7or8exposures. OneupdateperPAIR after sequentialmicrobatch4backward forRios andOstrava, clip5once. 450updates,450Rios+450Ostrava=900photoexposures, not new900photos. FullOstrava grid25tiles/photo; Rios trainingemptyGTtiles skip onlyzeroobjective. No alltilegraph stack.',
       'optimizer':'AdamW1e-4 wd.01 cosine30min1e-6 clip5,seed8383,FP32. No coefficient/structure/initializer search.',
       'planned_optimizer_updates':450,'planned_photo_pairs':450,'planned_Rios_exposures':450,'planned_Ostrava_exposures':450,
       'VAL_selection':'Initialepoch0andeach30epoch, SAME .5*meanRios6balancedBCE + .5*meanOstrava15teacherclassbalancedKL, groups equally within source. Fullnativeinputgrid alltiles. Earlieststrictminimum; epoch0rejectwithoutnewCAL_TEST. Never select on TEST or finalepoch afterseeingTEST.',
       'CAL_source_TEST':'Onlyselectedepoch>0; lockedRiosCAL6fullgridfixed19cuts.05:.05:.95 minimizepooledannotatednativebalancederror,smallesttie; savebeforeonceTEST7. Compareoriginalv62samegridfixedcut.20. Samev81sourcegates:eachclass5photos/3groups/375pixels,pooledFPFN<=.20,macro<=.25,worst<=.50,BAgain>=.05. Unknowncoverage notFPR/wholephoto7notappgate. No teacher CAL/TEST used in training.',
       'retention':'Onlysourcegatepass:originalOstrava15TEST once,newnativegrid atnewRioscut vsv62whole512/cut.20. FP_FN<=.20,IoU>=.50,FP_FNincrease<=.05each/IoUdrop<=.05. 25+1encoderinputs/photo/TWOviews. Wholecandidateexcludedifanygatefails. No guardrelaxation/sourceonlyadoption.',
       'conditional_device':'Onlybothgatespass,freshall7Rios+15OstravaoriginalCPU/newGPU nativeprobvsfirstTESTrefs atol/rtol.002/decisiondisagreement<=.001perphoto. No automaticAPI_CPP_ZIP_APK/aliaswarningmapchanges; separatehandoffprotocol required.',
       'preservation':'Parentoriginalv62+approved3heads/v16 exacttensor-buffer-meta, unchangednorm, no oldoutputs replaced. Originalv16-boundallowedprefixesapproved3+ostrava_rust_head+corrosion_retained_head only; parent/add/combined/fileSHA andserializedrestoreexact. Everyepochhead/optimizer/RNG andpartialfailure/actualjournal retained.',
       'storage':{'reserve_bytes':1024**3,'new_disk_budget_bytes':256*1024**2,'free_at_freeze':free,'RAM_plan':'RiosFIT_VALnativeRGB_GT_coverage<=512MiB,teacher75x512FP32<=75MiB,streamONEOstravaoriginal/nativeGT/teacherresponse atatime, conditionalrefs<=512MiB','new_dense_feature_files':False,'fullparentcopy':False,'archiveexpansion':False,'deletions':False},
       'scope':'Publicrepeateddevelopment, notindependentfield/wholephoto20pass/depth/health/zeroFN. TeacherFIT75responsesnotnewnaturalGT. Riosmissing83/5x5descriptiondiscrepancy/checkerplatephysicalcaptureuncertainty remain. Allpreviousfailuresheldoutpreserved.',
       'licenses':'RiosNettoRoehlMendeleyV1CC BY4.0 DOI10.17632/kcyn4nhv2c.1;OstravaFricZenodoV2CC BY4.0 DOI10.5281/zenodo.11235637;VTCC0/InsPLADCC BY-NC3.0/allparentterms/attribution/source/version/changes retained.',
       'synthetic_units_passed':3,'unit_prototype_uint16_overflow_fixed_before_source_gradient':True,'independent_field_validation':False}
    new(PROTOCOL,p);print({'protocol_SHA256':sha(PROTOCOL),'planned_updates':450,'protected_files':len(protected)},flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['sources'],**p['inputs'],**p['legacy_Python_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(RUN/'paired-orders.npz')==p['paired_orders_SHA256'] and sha(RUN/'protected-stat.json')==p['protected_stat_SHA256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert shutil.disk_usage(ROOT).free>=1024**3+128*1024**2
    return p


def load():
    original,approved=base.research_parent();parent=restore_delta(original,torch.load(native.PARENT,map_location='cpu',weights_only=True),base.EXPECTED)
    assert_preserved(approved,parent);assert state_sha256(parent['state_dict'])==native.PARENT_STATE
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=CorrosionRetainedMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(parent['state_dict'],strict=False);assert not extra and all(n.startswith('corrosion_retained_head.') for n in missing)
    init=torch.load(INIT,map_location='cpu',weights_only=True)
    head={n.removeprefix('corrosion_native_head.'):v for n,v in init['added_state_dict'].items() if n.startswith('corrosion_native_head.')}
    model.corrosion_retained_head.load_state_dict(head)
    assert all(torch.equal(v,model.corrosion_retained_head.state_dict()[n]) for n,v in head.items())
    new(RUN/'initializer.json',{'failed_v81_file_SHA256':sha(INIT),'head_state_SHA256':state_sha256(head),'new_head_initialization_exact':True,'old_failed_model_not_promoted':True})
    return model.eval().requires_grad_(False).cuda(),parent,original


def teachers(model,phase):
    cache=base.RUN/(phase+'-features.pt');assert sha(cache)==read(base.RUN/(phase+'-cache.json'))['sha256']
    data=torch.load(cache,map_location='cpu',weights_only=True);rows=base.rows_for(phase)
    assert data['ids']==[r['id'] for r in rows] and data['raw_sha256']==[r['encoded_sha256'] for r in rows] and data['mask_sha256']==[r['mask_encoded_sha256'] for r in rows]
    assert data['protocol_sha256']==sha(base.PROTOCOL) and data['manifest_sha256']==base.MANIFEST_SHA and data['encoder_parent_sha256']==base.EXPECTED and data['approved_three_head_delta_sha256']==base.PRIOR_SHA
    values=[]
    with torch.no_grad():
        for i in range(0,len(rows),4):
            f={n:v[i:i+4].cuda() for n,v in data['features'].items()};values.extend(model.forward_ostrava_features(**f).sigmoid().cpu().numpy())
    assert len(values)==len(rows)
    new(RUN/(phase+'-teacher-binding.json'),{'ids':data['ids'],'cache_SHA256':sha(cache),'original_teacher_delta_SHA256':native.PARENT_SHA,'teacher_probability512_SHA256':hashlib.sha256(np.stack(values).tobytes()).hexdigest(),'RAM_bytes':sum(v.nbytes for v in values),'head_forward_views':len(values),'new_encoder_inputs':0,'responses_not_new_GT':True})
    return rows,values


def retention_objective(model,item,teacher512,stage,training=False):
    _,g=target512(item['truth']);teacher=native_probability(torch.from_numpy(teacher512),g).numpy()
    total=0.;micro=0;tiles=0
    for start in range(0,len(item['boxes']),4):
        boxes=item['boxes'][start:start+4]
        x=native.batch_for(item,boxes,False)
        w=torch.from_numpy(np.stack([retained_weights(item['truth'],item['coverage'],b)[1] for b in boxes])).cuda()
        t=torch.from_numpy(np.stack([teacher_tile(teacher,b) for b in boxes])).cuda()
        f=native.features(model,x,stage,[item['row']['id']]*len(boxes))
        value=teacher_kl(model.forward_corrosion_retained_features(**f),t,w);assert torch.isfinite(value)
        if training:(.5*value).backward()
        total+=float(value.detach());micro+=1;tiles+=len(boxes)
        native.event('teacher-microbatches.jsonl',{'stage':stage,'id':item['row']['id'],'tile_count':len(boxes),'KL_contribution':float(value.detach()),'backward':training})
    return total,micro,tiles


def validation(model,rios,ost,teachers512,archives,stage):
    model.corrosion_retained_head.eval();newloss=native.val_loss(model,rios,stage+'_Rios');values=[]
    with torch.no_grad():
        for row,t in zip(ost,teachers512):
            item=native.native_pair(archives,row,stage+'_Ostrava');value,_,_=retention_objective(model,item,t,stage+'_Ostrava');values.append(value)
    oldloss=float(np.mean(values));return .5*(newloss+oldloss),newloss,oldloss


def train(model,parent,original):
    rios,used=native.prepared('fit');vrios,vused=native.prepared('validation');assert used+vused<=512*1024**2
    ost,teacher=teachers(model,'fit');vost,vteacher=teachers(model,'validation');head=model.corrosion_retained_head;head.requires_grad_(True)
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==100593
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=1e-6)
    orders=np.load(RUN/'paired-orders.npz',allow_pickle=False);actual=0;tiles_r=tiles_o=micros=0
    with ExitStack() as stack:
        archives=native.archives_for(ost+vost,stack)
        best,vr,vo=validation(model,vrios,vost,vteacher,archives,'initial_VAL');bestepoch=0
        new(RUN/'initial-VAL.json',{'objective':best,'Rios_BCE':vr,'Ostrava_teacher_KL':vo})
        torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'head-epoch-00.pt')
        print({'initial_VAL':best,'Rios_BCE':vr,'Ostrava_KL':vo},flush=True)
        with (RUN/'optimizer.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
            try:
                for e in range(30):
                    assert sorted(orders['rios'][e].tolist())==list(range(15));assert shutil.disk_usage(ROOT).free>1024**3+16*1024**2
                    head.train();losses=[]
                    for ir,io in zip(orders['rios'][e],orders['ostrava'][e]):
                        opt.zero_grad(set_to_none=True)
                        # Original helper backward is unscaled; scale its accumulated gradient once.
                        a,ma,ta=native.photo_objective(model,rios[ir],'FIT_Rios',True)
                        for value in head.parameters():
                            if value.grad is not None:value.grad.mul_(.5)
                        item=native.native_pair(archives,ost[io],'FIT_Ostrava')
                        b,mb,tb=retention_objective(model,item,teacher[io],'FIT_Ostrava',True)
                        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5);assert torch.isfinite(norm)
                        opt.step();actual+=1;tiles_r+=ta;tiles_o+=tb;micros+=ma+mb;losses.append(.5*(a+b))
                        journal.write(json.dumps({'actual_update':actual,'epoch':e+1,'photo_id':rios[ir]['row']['id'],'Ostrava_id':ost[io]['id'],'Rios_BCE':a,'Ostrava_teacher_KL':b,'paired_objective':losses[-1],'Rios_encoder_tiles':ta,'Ostrava_encoder_tiles':tb,'microbatches':ma+mb,'gradient_norm_before_clip':float(norm)})+'\n')
                    scheduler.step();score,vr,vo=validation(model,vrios,vost,vteacher,archives,'VAL_epoch_'+str(e+1))
                    if score<best:best,bestepoch=score,e+1
                    torch.save({'head':head.state_dict(),'epoch':e+1,'VAL':score,'actual_updates':actual},RUN/f'head-epoch-{e+1:02d}.pt')
                    torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':e+1,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/f'optimizer-epoch-{e+1:02d}.pt')
                    new(RUN/f'epoch-{e+1:02d}.json',{'epoch':e+1,'VAL':score,'Rios_BCE':vr,'Ostrava_teacher_KL':vo,'train_objective':float(np.mean(losses)),'best_epoch':bestepoch,'actual_updates':actual})
                    print({'epoch':e+1,'VAL':score,'Rios_BCE':vr,'Ostrava_KL':vo,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
            except BaseException:
                torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'epoch':e+1,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/'partial-failure.pt');raise
    chosen=torch.load(RUN/f'head-epoch-{bestepoch:02d}.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval().requires_grad_(False)
    info={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':bestepoch,'selected_actual_updates':chosen['actual_updates'],'selected_VAL':best,'initial_VAL':read(RUN/'initial-VAL.json')['objective'],'Rios_original_FIT':15,'Ostrava_original_FIT':60,'Rios_photo_exposures':450,'Ostrava_photo_exposures':450,'Rios_encoder_tiles':tiles_r,'Ostrava_encoder_tiles':tiles_o,'microbatches':micros,'new_natural_photos':0,'teacher_outputs_are_not_new_GT':True}
    candidate=deepcopy(parent);candidate['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};candidate.update(corrosion_retained_protocol_SHA256=sha(PROTOCOL),corrosion_retained_training=info,corrosion_retained_scope='paired_native_teacher_retention_research_not_app_health')
    assert_preserved(parent,candidate);payload=pack_delta(original,candidate,base.EXPECTED,base.START.relative_to(ROOT).as_posix(),PREFIXES)
    (RUN/'trained').mkdir();torch.save(payload,RUN/'trained/final-delta.pt')
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED);assert_preserved(parent,restored)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in candidate['state_dict'].items()) and all(v==restored[n] for n,v in candidate.items() if n!='state_dict')
    info['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',info);return info


def checks(model,parent,original,p):
    now=deepcopy(parent);now['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};assert_preserved(parent,now)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in now['state_dict'].items())
    for n,v in model.ostrava_rust_head.state_dict().items():
        if n.endswith(('_mean','_std')):assert torch.equal(v,model.corrosion_retained_head.state_dict()[n])
    for n,h in {**p['legacy_Python_sha256'],**p['inputs']}.items():assert sha(ROOT/n)==h,n
    protected=read(RUN/'protected-stat.json')
    for n,(size,mtime) in protected.items():
        s=(ROOT/n).stat();assert (s.st_size,s.st_mtime_ns)==(size,mtime),n
    journal=[json.loads(s) for s in (RUN/'optimizer.jsonl').read_text().splitlines()];assert [r['actual_update'] for r in journal]==list(range(1,451))
    assert all(sum(row['photo_id']==r['id'] for row in journal)==30 for r in native.rows_for('fit'))
    counts={r['id']:sum(row['Ostrava_id']==r['id'] for row in journal) for r in base.rows_for('fit')};assert set(counts.values())<={7,8}
    events=[json.loads(s) for s in (RUN/'feature-events.jsonl').read_text().splitlines()];stages={s:sum(r['encoder_inputs'] for r in events if r['stage']==s) for s in {r['stage'] for r in events}}
    assert stages['FIT_Rios']==sum(r['Rios_encoder_tiles'] for r in journal) and stages['FIT_Ostrava']==sum(r['Ostrava_encoder_tiles'] for r in journal)
    size=sum(v.stat().st_size for v in RUN.rglob('*') if v.is_file());assert size<=p['storage']['new_disk_budget_bytes'] and shutil.disk_usage(ROOT).free>=1024**3
    new(RUN/'verification.json',{'status':'passed','parent_all_tensor_buffer_metadata_exact':True,'norm_unchanged':True,'initializer_selected_failed_v81_exact':True,'disk_delta_restore_exact':True,'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':len(protected),'all_original_raw_SHA_recalculated':False,'actual_optimizer_updates':450,'Rios_exposures':450,'Ostrava_exposures':450,'Ostrava_per_photo_exposure_range':[min(counts.values()),max(counts.values())],'encoder_inputs_by_stage':stages,'teacher_cached_head_views':75,'new_teacher_encoder_inputs':0,'new_run_disk_bytes':size,'delta_SHA256':sha(RUN/'trained/final-delta.pt'),'combined_state_SHA256':state_sha256(restored['state_dict']),'new_API_CPP_ZIP_APK_checks':0})


def run():
    p=validate();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,parent,original=load();info=train(model,parent,original)
    if info['selected_epoch']>0:
        cut=native.calibrate(model);rows=native.rows_for('test');result,refs=native.evaluate(model,rows,cut,'source_test')
        a,b=result['metrics']['candidate'],result['metrics']['parent'];photos=result['photos'];gain=.5*(b['fpr']+b['fnr']-a['fpr']-a['fnr'])
        passed=len(rows)>=5 and len({r['group_id'] for r in rows})>=3 and min(a['tp']+a['fn'],a['tn']+a['fp'])>=375 and a['fpr']<=.20 and a['fnr']<=.20 and all(np.mean([r['candidate'][k] for r in photos])<=.25 and max(r['candidate'][k] for r in photos)<=.50 for k in ('fpr','fnr')) and gain>=.05
        result.update(source_sparse_ROI_point_gate_passed=bool(passed),balanced_accuracy_gain_vs_samegrid_parent=gain,cut=cut,parent_cut=.20,macrophoto_FPR=float(np.mean([r['candidate']['fpr'] for r in photos])),macrophoto_FNR=float(np.mean([r['candidate']['fnr'] for r in photos])))
        new(RUN/'source-test.json',result);retained=False
        if passed:
            ost=base.rows_for('test');retention,orefs=native.evaluate(model,ost,cut,'Ostrava_retention_test',True);a,b=retention['metrics']['candidate'],retention['metrics']['parent']
            retained=a['fpr']<=.20 and a['fnr']<=.20 and a['IoU']>=.50 and a['fpr']-b['fpr']<=.05 and a['fnr']-b['fnr']<=.05 and b['IoU']-a['IoU']<=.05
            retention['retention_gate_passed']=bool(retained);new(RUN/'retention.json',retention)
            if retained:native.devices(model,rows+ost,refs+orefs,cut)
        new(RUN/'selection.json',{'source_sparse_ROI_point_gate_passed':bool(passed),'Ostrava_retention_attempted':bool(passed),'Ostrava_retention_gate_passed':bool(retained),'combined_source_research_gate_passed':bool(passed and retained),'automatic_app_adoption':False,'wholephoto_app_gate_passed':False,'new_API_CPP_ZIP_APK_checks':0})
    else:new(RUN/'selection.json',{'combined_source_research_gate_passed':False,'reason':'epoch0 selected; newCAL_TEST0; failedinitializer not promoted','automatic_app_adoption':False,'wholephoto_app_gate_passed':False})
    checks(model,parent,original,p);print(read(RUN/'selection.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text().strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
        try:run()
        except BaseException as error:new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':datetime.now(timezone.utc).isoformat(),'error':repr(error)});raise
        finally:(RUN/'operation.lock').unlink()
