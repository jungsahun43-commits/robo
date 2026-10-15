"""One real-corrosion-mask experiment; immutable shared encoder and prior tasks."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.ostrava_rust_mask import OstravaRustMain,source_view,native_probability,pixel_loss,confusion_at,pixel_rates
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

START=ROOT/'runs/industrial-magnetic-tiles-v16/released/final.pt'
EXPECTED='5285353bfd6f0d11684e1684f5a0128049a923804035081ed7fa67d0632f3f3e'
PRIOR_DELTA=ROOT/'runs/industrial-insplad-suspension-v51/trained/final-delta.pt'
PRIOR_SHA='b91b6578eccd1886c5be18805a9475875ff319f77bda019cd0b3f518d84f46dc'
PRIOR_COMBINED='bb388200af167ffde35f6281a5e4626be904f9026411c401151e9bc3b4d7c45a'
MANIFEST=ROOT/'data/industrial-ostrava-masks-v61/manifest.json'
MANIFEST_SHA='90ab369b28d8ad6955ca0f3114e04c1fd87f7fc464c66a28fbccc763407e4f8c'
AUDIT=ROOT/'reports/industrial-ostrava-v61-data-audit.json'
ARCHIVE=ROOT/'data/industrial-new-rust-belt-v60-source/publisher-ostrava-v2.zip'
RUN=ROOT/'runs/industrial-ostrava-rust-mask-v62'
PROTOCOL=ROOT/'reports/industrial-ostrava-v62-protocol.json'
SOURCES=('safelog_ai/ostrava_rust_mask.py','scripts/study_ostrava_rust_mask.py','tests/test_ostrava_rust_mask.py')
PREFIXES=('insplad_classifier.','insplad_glass_classifier.','insplad_suspension_classifier.','ostrava_rust_head.')
EPOCHS,STEPS,BATCH,SEED=30,64,4,6262
CUTS=[round(i*.05,6) for i in range(1,20)]


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists()
    assert sha(START)==EXPECTED and sha(PRIOR_DELTA)==PRIOR_SHA and sha(MANIFEST)==MANIFEST_SHA
    assert read(AUDIT)['ready'] and read(AUDIT)['manifest_sha256']==MANIFEST_SHA
    assert shutil.disk_usage(ROOT).free>1024**3+450*1024**2
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                st=path.stat();protected[path.relative_to(ROOT).as_posix()]=[st.st_size,st.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    fit=rows_for('fit');assert len(fit)==60 and sum(r['label'] for r in fit)==58
    counts=Counter(r['group_id'] for r in fit);weights=np.array([1/counts[r['group_id']] for r in fit]);weights/=weights.sum()
    draws=np.random.default_rng(SEED).choice(len(fit),(EPOCHS*STEPS,BATCH),replace=True,p=weights)
    np.save(RUN/'draws.npy',draws.astype(np.int32),allow_pickle=False)
    p={'UTC':utc(),'schema':'safelog-ostrava-real-corrosion-mask-v62','fixed_before_encoder_features_gradients_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':EXPECTED,'approved_three_head_delta_sha256':PRIOR_SHA,
       'approved_three_head_combined_sha256':PRIOR_COMBINED,'manifest_sha256':MANIFEST_SHA,'data_audit_sha256':sha(AUDIT),
       'source_archive_sha256':sha(ARCHIVE),'QA_sha256':sha(ROOT/'reports/industrial-ostrava-v61-FIT-QA.json'),
       'legacy_Python_sha256':{v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES},
       'protected_stat_sha256':sha(RUN/'protected-stat.json'),'draws_sha256':sha(RUN/'draws.npy'),
       'prior_calibration_sha256':{n:sha(ROOT/n) for n in ('runs/industrial-insplad-corrosion-v45/calibration.json','runs/industrial-insplad-glass-v48/calibration.json','runs/industrial-insplad-suspension-v51/calibration.json')},
       'split':read(AUDIT)['counts'],'actual_FIT_corrosion_photos':58,'actual_FIT_annotated_fullnegative_photos':2,
       'input':'full original2048square RGB bicubic512square ImageNet; source actual Lmask169positive viaL>0 and nearest512 target. No GTchosen crop/source/date/category input or augmentation.',
       'features':'unchanged sharedencoder forward_with_fine fine24@128x128,low40@64x64,high960@32x32; FP16 stored maps and FP32head. FirstFITbatch4 ordinarylow/highbitwiseexact, these4comparisonpasses are not newphotos.',
       'architecture':'fresh ostrava_rust_head high960to32 1x1 GELU, fine24to16 1x1 GELU, bilinearup low40/high32 to128, concatfine16+low40+high32=88; Conv88to64 3x3 GN8 GELU dropout.1,Conv64to32 3x3 GN4 GELU,Conv32to1 zerooutput; bilinearlogit512;100593trainableparams',
       'normalization':'eachnewfeaturechannel mean/popstd overonlyFIT60 cachedspatialcells CPUfloat64, floor.001 FP32buffers, clampstandardized[-8,8]; allbuffersfixedafterfit',
       'trainable':'new entireRustMaskHead only; v16andall3approvedheads/normalizations/tensor/buffer/meta exact, backbone frozeneval',
       'loss':'perphoto mean ofpresent foreground/background BCE classmeans +softDice(smooth1); allnegativephoto backgroundBCEonly plusDice, no false fullmachinehealth labels',
       'sampling':'fixed1920x4 uniformFITproxygroup thenphoto withreplacement; no forced labeloversampling; repeats are not newnaturalphotos',
       'optimizer':'AdamW lr.0003 wd.01 cosine30 min.000001 clip5 FP32 TF32off threads4',
       'epochs':30,'steps_per_epoch':64,'batch':4,'seed':SEED,'planned_optimizer':1920,
       'VAL_selection':'15originalVAL, full512 loss perphoto thenproxygroupmean, epoch0andcompletedepochs earlieststrictminimum;epoch0rejectbeforeCAL_TEST',
       'native_geometry':'sigmoid512 thenbilinear probability restore2048 align_cornersFalse; compare withoriginal L>0 actualmask, never resizeGT withbilinear',
       'CAL':'afterselectedepoch>0 originalCAL15 masks, globalnativepixel .5FNR+.5FPR overfixedcuts .05:.05:.95; earliestlowestcut tie, save19countsandchosen threshold beforeTESTfeatures/scores',
       'TEST':'once original15positivephotos, fulloriginalcoordinatepixels andmacrophotoIoU; no wholephotoFPR since0normalTEST; no 20photonewappgate claim',
       'source_research_gate':['selectedepoch>0','nativeTESTpixelFPR<=.20 andFNR<=.20','nativeIoU>=.50','positiveTESTphotos>=5 andproxygroups>=3','baseandapprovedthreeheadexact'],
       'source_gate_limit':'Researchcorrosionsegmentation only. Even ifpassed, no automaticappalias/ZIP due15positivesand0normalTEST; targettransfer requires ownfuturepreregisteredCAL_TESTprotocol. Oldbracket/tube/InsPLADphotoaccuracy not re-evaluated here.',
       'conditional_device_check':'onlyifsourcegatepasses, original15RGB freshCPU andfreshGPU vsstoredcachedGPU512maps, atol.002 rtol.002 andnativepixeldecisiondisagreement<=.001 atfixedCALcut; preserve mismatch, no retune; notAndroidspeed or appintegrationcheck',
       'cache':'FIT_VAL beforetraining, CAL afterselection,TEST onlyafterCALjson. ThreeFP16maps+uint8mask512 perphoto<400MiB all105; bindIDs/raw/maskSHA/order/geometry/dtype/encoderParentSHA/priorDeltaSHA/protocolSHA. SaveCAL_TEST512probmaps<32MiB not2048maps.',
       'delta':'parent-bound originalv16, allowed3oldheadprefixes+ostrava_rust_head. only; assertpriorv51beforepack anddiskrestore allstate/meta exact, parent/file/state/added/combinedSHA; small researchdelta notstandalone appmodel',
       'source_limits':'singleindustrialarea; EXIFsecond/pHash no physicalidentity/siteindependenceproof; publishermaskboundaries not fullyhumanreviewed; sourcepositive15 insufficientfor20photogate; pixelFP notfullphotoFP; notcorrosiondepth/severity/machine/electricalhealth',
       'location':'source actualmask researchonly, no existingprimarymap/pixelthreshold or alias changed; transferlocationnotvalidated bysourceGT',
       'QA':'fixedtrain01/02/03 only: rustnuts,pipe/bar,handle/panel majorareas broadlyaligned. No full105humanreview/GT edits/VAL_CAL_TESTcontentchoice.',
       'license':'MatejFric Ostrava ZenodoV2 DOI10.5281/zenodo.11235637 dataCC BY4.0 author/source/version/changeattribution. InsPLADCC BY-NC3.0 andotherlicensesunchanged.',
       'storage':'newcache+probmaps+delta+progress+stat<450MiB,>=1GiBreserve; no fullparentduplicate or originalexpansion/deletions',
       'resume':'percompletedepoch optimizer/scheduler/RNG/newheadstate; appendactualstepjournal includingpartialrepeats/failures; no redo completedstudy',
       'new_units_passed':3,'independent_field_validation':False,'automatic_app_adoption':False}
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'protected_files':len(protected),'planned_optimizer':1920},flush=True)


def validate(committed=False):
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((START,EXPECTED),(PRIOR_DELTA,PRIOR_SHA),(MANIFEST,MANIFEST_SHA),(ARCHIVE,p['source_archive_sha256']),(AUDIT,p['data_audit_sha256']),
            (RUN/'draws.npy',p['draws_sha256']),(RUN/'protected-stat.json',p['protected_stat_sha256'])):assert sha(path)==h,path
    if committed:
        git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
        for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix(),AUDIT.relative_to(ROOT).as_posix(),'reports/industrial-ostrava-v61-FIT-QA.json'):
            assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def research_parent():
    original=torch.load(START,map_location='cpu',weights_only=True);prior=torch.load(PRIOR_DELTA,map_location='cpu',weights_only=True)
    assert prior['combined_state_sha256']==PRIOR_COMBINED
    return original,restore_delta(original,prior,EXPECTED)


def load(device='cuda',delta=False):
    original,parent=research_parent();state=parent
    if delta:
        state=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),EXPECTED);assert_preserved(parent,state)
    torch.manual_seed(SEED);model=OstravaRustMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(state['state_dict'],strict=False)
    assert not extra and all(n.startswith('ostrava_rust_head.') for n in missing)
    if delta:assert not missing
    return model.eval().requires_grad_(False).to(device),parent,original


def images(rows,stage):
    x=[];y=[]
    with zipfile.ZipFile(ARCHIVE) as archive,(RUN/'input-events.jsonl').open('a',encoding='utf-8',buffering=1) as events:
        for r in rows:
            raw=archive.read(r['member']);maskraw=archive.read(r['mask_member'])
            assert hashlib.sha256(raw).hexdigest()==r['encoded_sha256'] and hashlib.sha256(maskraw).hexdigest()==r['mask_encoded_sha256']
            with Image.open(io.BytesIO(raw)) as rgb,Image.open(io.BytesIO(maskraw)) as mask:
                assert list(rgb.size)==r['size'] and mask.size==rgb.size
                view,target=source_view(rgb.convert('RGB'),mask);x.append(normalized(view));y.append(target.to(torch.uint8))
            events.write(json.dumps({'UTC':utc(),'id':r['id'],'stage':stage,'RGB_and_mask_decode':True})+'\n')
    return torch.stack(x),torch.stack(y)


def native_truth(r,device='cuda'):
    with zipfile.ZipFile(ARCHIVE) as archive:
        raw=archive.read(r['mask_member']);assert hashlib.sha256(raw).hexdigest()==r['mask_encoded_sha256']
        with Image.open(io.BytesIO(raw)) as mask:value=np.array(mask.convert('L'))>0
    return torch.from_numpy(value).to(device)


def cache(phase,model):
    if phase=='test':assert (RUN/'calibration.json').exists()
    if phase=='cal':assert read(RUN/'trained/training.json')['selected_epoch']>0
    rows=rows_for(phase);path=RUN/(phase+'-features.pt')
    if path.exists():
        assert sha(path)==read(RUN/(phase+'-cache.json'))['sha256'];value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['ids']==[r['id'] for r in rows] and value['protocol_sha256']==sha(PROTOCOL)
    else:
        values={n:[] for n in ('fine','low','high')};targets=[]
        with torch.inference_mode():
            for i in range(0,len(rows),4):
                x,y=images(rows[i:i+4],'cache_'+phase);x=x.cuda();features=forward_with_fine(model.backbone,x)
                if phase=='fit' and i==0:
                    old=model.backbone(x);assert all(torch.equal(old[n],features[n]) for n in ('low','high'))
                for n in values:values[n].append(features[n].half().cpu())
                targets.append(y)
        value={'ids':[r['id'] for r in rows],'raw_sha256':[r['encoded_sha256'] for r in rows],'mask_sha256':[r['mask_encoded_sha256'] for r in rows],
               'features':{n:torch.cat(v) for n,v in values.items()},'target512':torch.cat(targets),'manifest_sha256':MANIFEST_SHA,'protocol_sha256':sha(PROTOCOL),
               'encoder_parent_sha256':EXPECTED,'approved_three_head_delta_sha256':PRIOR_SHA,'input':'RGBbicubic512square_ImageNet_maskLgt0Nearest512',
               'dtype':'FP16maps_uint8targets','native_size':[2048,2048]}
        with path.open('xb') as out:torch.save(value,out)
        new(RUN/(phase+'-cache.json'),{'sha256':sha(path),'bytes':path.stat().st_size,'photos':len(rows)});print({'cached':phase,'photos':len(rows)},flush=True)
    return rows,{n:v.cuda() for n,v in value['features'].items()},value['target512'].cuda()


def predict(model,features,ids=None):
    values=[features[n] if ids is None else features[n][ids] for n in ('fine','low','high')]
    return model.forward_ostrava_features(*values)


def group_mean(values,rows):
    return torch.stack([values[[i for i,r in enumerate(rows) if r['group_id']==g]].mean() for g in sorted({r['group_id'] for r in rows})]).mean()


def validation_loss(model,features,target,rows):
    model.ostrava_rust_head.eval();values=[]
    with torch.no_grad():
        for i in range(0,len(rows),4):values.append(pixel_loss(predict(model,features,slice(i,i+4)),target[i:i+4]))
    return float(group_mean(torch.cat(values),rows))


def train():
    validate(True);assert not (RUN/'trained/training.json').exists()
    model,parent,original=load();fit,x,y=cache('fit',model);val,vx,vy=cache('validation',model);head=model.ostrava_rust_head
    head.fit_normalization(x);head.requires_grad_(True);assert sum(v.numel() for v in model.parameters() if v.requires_grad)==100593
    opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    draws=np.load(RUN/'draws.npy',allow_pickle=False);journal=RUN/'optimizer.jsonl';progress=RUN/'progress.pt';start=0;bestepoch=0
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=True);head.load_state_dict(saved['head']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start,best,bestepoch=saved['epoch'],saved['best'],saved['bestepoch'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    else:
        assert not journal.exists();journal.touch(exist_ok=False);best=validation_loss(model,vx,vy,val)
        new(RUN/'initial-VAL.json',{'group_mean_mask_loss':best,'epoch':0})
        torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'best-head.pt')
    actual=len(journal.read_text(encoding='utf-8').splitlines())
    with journal.open('a',encoding='utf-8',buffering=1) as out:
        for epoch in range(start+1,EPOCHS+1):
            losses=[];head.train()
            for step in range(STEPS):
                ids=draws[(epoch-1)*STEPS+step];loss=pixel_loss(predict(model,x,ids),y[ids]).mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();actual+=1
                out.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'mask_loss':float(loss.detach())})+'\n');losses.append(float(loss.detach()))
            scheduler.step();score=validation_loss(model,vx,vy,val)
            if score<best:
                best,bestepoch=score,epoch;torch.save({'head':head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/'best-head.pt')
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bestepoch':bestepoch,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/'progress.partial.pt');(RUN/'progress.partial.pt').replace(progress)
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'mask_loss':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
            print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    chosen=torch.load(RUN/'best-head.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval()
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':chosen['epoch'],'selected_actual_updates':chosen['actual_updates'],'VAL_group_mean_mask_loss':chosen['VAL'],
             'FIT_photos':60,'actual_FIT_corrosion_photos':58,'actual_FIT_annotated_fullnegative_photos':2,'VAL_photos':15}
    result=deepcopy(parent);result['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()}
    result.update(ostrava_mask_protocol_sha256=sha(PROTOCOL),ostrava_mask_training=summary,ostrava_mask_scope='source_industrial_corrosion_actual_masks_research_no_app_photo_gate')
    assert_preserved(parent,result);payload=pack_delta(original,result,EXPECTED,START.relative_to(ROOT).as_posix(),PREFIXES);(RUN/'trained').mkdir()
    with (RUN/'trained/final-delta.pt').open('xb') as out:torch.save(payload,out)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),EXPECTED);assert_preserved(parent,restored)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in result['state_dict'].items())
    assert all(v==restored[n] for n,v in result.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary)


def probabilities(phase,model,features,rows):
    with torch.inference_mode():
        values=torch.cat([predict(model,features,slice(i,i+4)).sigmoid().cpu() for i in range(0,len(rows),4)])
    path=RUN/(phase+'-probability512.pt')
    with path.open('xb') as out:torch.save({'ids':[r['id'] for r in rows],'probability512':values,'delta_sha256':sha(RUN/'trained/final-delta.pt'),'protocol_sha256':sha(PROTOCOL)},out)
    return values


def calibrate():
    assert not (RUN/'calibration.json').exists();model,_,_=load(delta=True);rows,x,_=cache('cal',model);values=probabilities('cal',model,x,rows)
    counts=torch.zeros(len(CUTS),4,dtype=torch.long,device='cuda')
    for r,prob in zip(rows,values):counts+=confusion_at(native_probability(prob.cuda()),native_truth(r),CUTS)
    rates=[pixel_rates(c) for c in counts.cpu().tolist()];errors=[.5*(r['fpr']+r['fnr']) for r in rates];best=min(range(len(CUTS)),key=lambda i:(errors[i],i))
    new(RUN/'calibration.json',{'delta_sha256':sha(RUN/'trained/final-delta.pt'),'source_photos':len(rows),'cuts':CUTS,'native_rates':rates,'balanced_errors':errors,
        'selected_index':best,'cut':CUTS[best],'selected_error':errors[best],'TEST_features_or_scores_before_this':0,'probability512_sha256':sha(RUN/'cal-probability512.pt')})


def evaluate():
    assert not (RUN/'selection.json').exists();cal=read(RUN/'calibration.json');assert cal['delta_sha256']==sha(RUN/'trained/final-delta.pt')
    model,_,_=load(delta=True);rows,x,_=cache('test',model);values=probabilities('test',model,x,rows);total=torch.zeros(4,dtype=torch.long,device='cuda');perphoto=[]
    for r,prob in zip(rows,values):
        counts=confusion_at(native_probability(prob.cuda()),native_truth(r),[cal['cut']])[0];total+=counts
        perphoto.append({'id':r['id'],'native_pixel_rates':pixel_rates(counts.cpu().tolist())})
    rates=pixel_rates(total.cpu().tolist());train=read(RUN/'trained/training.json')
    passed=train['selected_epoch']>0 and rates['fpr']<=.2 and rates['fnr']<=.2 and rates['IoU']>=.5 and sum(r['label'] for r in rows)>=5 and len({r['group_id'] for r in rows})>=3
    result={'UTC':utc(),'native_pixel_rates':rates,'macro_photo_IoU':float(np.mean([r['native_pixel_rates']['IoU'] for r in perphoto])),
        'per_photo':perphoto,'source_research_gate_passed':bool(passed),'app_photo_gate_passed':False,'reason_no_app_gate':'TESTnormal0/positive15<20',
        'original_TEST_photos':len(rows),'whole_photo_FPR':None,'whole_photo_FNR':None,'threshold':cal['cut'],'training':train,
        'delta_sha256':cal['delta_sha256'],'probability512_sha256':sha(RUN/'test-probability512.pt'),'old_tasks_unchanged':True,
        'new_API_CPP_ZIP_checks':0,'new_device_compatibility_photos':0,'independent_field_validation':False,'target_transfer_or_target_localization_tested':False}
    new(RUN/'selection.json',result);new(ROOT/'reports/industrial-ostrava-v62-selection.json',result);print(result,flush=True)


def device_check():
    result=read(RUN/'selection.json');assert result['source_research_gate_passed']
    reference=torch.load(RUN/'test-probability512.pt',map_location='cpu',weights_only=True);rows=rows_for('test');assert reference['ids']==[r['id'] for r in rows]
    report={};cut=result['threshold']
    for device in ('cpu','cuda'):
        model,parent,_=load(device,delta=True);maximum=0.;changed=0;pixels=0;allclose=True
        with torch.inference_mode():
            for i,r in enumerate(rows):
                x,_=images([r],'fresh_'+device+'_compatibility');actual=model.forward_ostrava_mask(x.to(device))[0].sigmoid().cpu();expected=reference['probability512'][i]
                maximum=max(maximum,float((actual-expected).abs().max()));allclose=allclose and torch.allclose(actual,expected,atol=.002,rtol=.002)
                a=native_probability(actual)>cut;b=native_probability(expected)>cut;changed+=int((a!=b).sum());pixels+=a.numel()
        assert_preserved(parent,{**parent,'state_dict':{n:v.cpu() for n,v in model.state_dict().items()}})
        report[device]={'fresh_original_RGB_photos':len(rows),'maximum_probability512_abs_difference':maximum,'map_allclose_atol002_rtol002':bool(allclose),
            'native_pixel_decision_disagreement':changed/pixels,'disagreeing_pixels':changed,'assessed_pixels':pixels}
        del model
    passed=all(v['map_allclose_atol002_rtol002'] and v['native_pixel_decision_disagreement']<=.001 for v in report.values())
    value={'status':'passed' if passed else 'failed','devices':report,'cut_unchanged':cut,'new_APP_CPP_ZIP_checks':0,'not_Android_speed_or_wholephoto_accuracy':True}
    new(RUN/'device-compatibility.json',value);new(ROOT/'reports/industrial-ostrava-v62-device-compatibility.json',value);print(value,flush=True)


def verify():
    p=validate();train=read(RUN/'trained/training.json');original,parent=research_parent()
    payload=torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True);restored=restore_delta(original,payload,EXPECTED);assert_preserved(parent,restored)
    for n,h in p['legacy_Python_sha256'].items():assert sha(ROOT/n)==h,n
    checked=0
    for n,before in read(RUN/'protected-stat.json').items():
        st=(ROOT/n).stat();assert [st.st_size,st.st_mtime_ns]==before,n;checked+=1
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(journal)==train['actual_optimizer_updates'] and len(journal)>=1920 and [r['actual_update'] for r in journal]==list(range(1,len(journal)+1))
    cachebytes=0
    for phase in ('fit','validation','cal','test'):
        path=RUN/(phase+'-cache.json')
        if path.exists():assert sha(RUN/(phase+'-features.pt'))==read(path)['sha256'];cachebytes+=(RUN/(phase+'-features.pt')).stat().st_size
    assert cachebytes<400*1024**2 and shutil.disk_usage(ROOT).free>=1024**3
    result={'status':'passed','base_and_approved_three_head_tensor_buffer_metadata_exact':True,'serialized_delta_restore_exact':True,
        'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':checked,'all_protected_raw_SHA_recalculated':False,
        'actual_optimizer_updates':len(journal),'cached_original_photos':sum(read(v)['photos'] for v in RUN.glob('*-cache.json')),
        'new_cache_bytes':cachebytes,'ordinary_encoder_comparison_inputs':4,'new_units_passed':3,'new_API_CPP_ZIP_checks':0,'default_and_approved_overlay_unchanged':True}
    new(RUN/'verification.json',result);new(ROOT/'reports/industrial-ostrava-v62-verification.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        validate(True)
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:
            if not (RUN/'trained/training.json').exists():train()
            if read(RUN/'trained/training.json')['selected_epoch']>0:
                if not (RUN/'calibration.json').exists():calibrate()
                if not (RUN/'selection.json').exists():evaluate()
                if read(RUN/'selection.json')['source_research_gate_passed'] and not (RUN/'device-compatibility.json').exists():device_check()
            elif not (RUN/'selection.json').exists():new(RUN/'selection.json',{'source_research_gate_passed':False,'app_photo_gate_passed':False,'rejected':'epoch0selected','new_CAL_TEST_scores':0})
            if not (RUN/'verification.json').exists():verify()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
