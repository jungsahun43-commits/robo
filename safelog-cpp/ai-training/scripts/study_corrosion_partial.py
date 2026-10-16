"""One mixed-source partial-positive experiment on the frozen common encoder."""
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
from io import BytesIO
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
from safelog_ai.corrosion_partial_mask import CorrosionPartialMain,partial_loss,loss_components,native_probability,known_confusion
from safelog_ai.corrosion_partial_data import partial_target,mask_view
from safelog_ai.corrosion_json_pairs import embedded_image,polygon_labels
from safelog_ai.ostrava_rust_mask import pixel_rates
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_ostrava_rust_mask as base
from scripts import prepare_corrosion_partial as data
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

MANIFEST=data.OUT/'manifest.json'
PARENT=base.RUN/'trained/final-delta.pt'
PARENT_SHA='26c26bc5739930f153d1553b9b22840f4e83ba925cddeca5f46eee4f9f25a01b'
PARENT_STATE='61677d5f1d038006cbcfc83894af03a90b3105055ca94d5eef3c40567dc7332f'
RUN=ROOT/'runs/industrial-corrosion-partial-v73'
PROTOCOL=ROOT/'reports/industrial-corrosion-partial-v73-protocol.json'
SOURCES=('scripts/study_corrosion_partial.py','safelog_ai/corrosion_partial_mask.py','tests/test_corrosion_partial_mask.py')
PREFIXES=base.PREFIXES+('corrosion_partial_head.',)
EPOCHS,STEPS,BATCH,SEED=30,64,4,7373
CUTS=[round(i*.05,6) for i in range(1,20)]


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and read(MANIFEST)['audit']['ready']
    assert sha(PARENT)==PARENT_SHA and sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA
    assert shutil.disk_usage(ROOT).free>1024**3+256*1024**2
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                s=path.stat();protected[path.relative_to(ROOT).as_posix()]=[s.st_size,s.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    fit=rows_for('fit');rng=np.random.default_rng(SEED);draws=np.empty((EPOCHS*STEPS,4),dtype=np.int32)
    for source,columns in (('ostrava',slice(0,2)),('vt',slice(2,4))):
        indices=[i for i,r in enumerate(fit) if r['source']==source];groups=Counter(fit[i]['group_id'] for i in indices)
        weights=np.array([1/groups[fit[i]['group_id']] for i in indices]);weights/=weights.sum()
        draws[:,columns]=rng.choice(indices,(EPOCHS*STEPS,2),replace=True,p=weights)
    np.save(RUN/'draws.npy',draws,allow_pickle=False)
    files=[MANIFEST,PARENT,data.POLICY,ROOT/'reports/industrial-corrosion-partial-v72-data-audit.json',ROOT/'safelog_ai/corrosion_partial_data.py',ROOT/'scripts/prepare_corrosion_partial.py',ROOT/'safelog_ai/corrosion_json_pairs.py',ROOT/'safelog_ai/ostrava_rust_mask.py',ROOT/'scripts/study_ostrava_rust_mask.py',base.RUN/'selection.json']
    files += [ROOT/n for n in sorted({r['archive'] for r in fit})]
    p={'UTC':utc(),'fixed_before_real_features_gradient_or_model_scores':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
       'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},'manifest_sha256':sha(MANIFEST),'parent_file_sha256':PARENT_SHA,'parent_combined_state_sha256':PARENT_STATE,
       'protected_stat_sha256':sha(RUN/'protected-stat.json'),'draws_sha256':sha(RUN/'draws.npy'),
       'legacy_Python_sha256':{v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES},
       'task':'Partialbinary rust: VT suppliedFair/Poor/Severe positive1, Good/minimal/background andrawignore unknown255. OstravaactualLgt0fullmask knownpositive1/knownnegative0. Not severity/depth/wholemachinehealth or newnaturalphotos; VTunknown FP unmeasurable.',
       'data':read(MANIFEST)['audit']['counts'],'input':'OriginalauthorRGB, fullaspectletterbox512, bicubicRGB/pad124,116,104; targetnearest/padding255 usingv72 exactgeometry/SHA. ImageNetnormalization. No augmentation/GTcrop/newsource/date/categoryembedding.',
       'model':'One unchangedencoder fine24@128/low40@64/high960@32. Newcorrosion_partial_head exactclone selectedv62binaryhead inclall100593trainparameters andnormbuffers. Trainallnewheadparameters; freezeevalv16/approved3heads/oldv62; no failedv69/v71head inherited. Distinctforward_corrosion_partial methods.',
       'normalization':'Reuseclonedv62FIT-onlymeans/popstd/floor.001/clamp[-8,8] exactly, do notrefit any buffers. Newfeatures FP16boundary ->FP32head, originalv62 sameboundary. First4FIT low/high mustbitwiseequalordinaryforward.',
       'loss':'For eachphoto compute knownpositiveBCEmean P_i andknownnegativeBCEmean N_i, unknown255/padding excluded fromboth. L=.25*mean(P_i overpresentpositiveOstrava)+.25*mean(P_i overpresentpositiveVT)+.5*mean(N_i overpresentnegativeOstrava)+mean(fullmasksoftDiceOstrava,smooth1). Missingpositivepool contributes0, fixedweightsnotrenormalized. NoVTDice/noVTnegative. ThusBCEpositivebudget.5 splitby source,knownbackgroundbudget.5. VAL uses source/groupmean ofsamecomponents. This is not measuredgradient equality.',
       'optimizer':'AdamW lr.0001 wd.01 cosine30 min.000001 gradclip5 FP32/noAMP/TF32off/threads4 seed7373',
       'epochs':30,'steps_per_epoch':64,'batch':4,'planned_optimizer_updates':1920,
       'sampling':'Fixed1920x4draws:2Ostrava+2VT perbatch, groupuniformthenphoto per source replacement. 7680encoderFITdraws notnewphotos. InitialVALand30epochVAL with fixedobjective; alloriginal336FIT+75VAL retained.',
       'VAL_selection':'Earlieststrictminimum of.25Ostpositivegroupmean+.25VTpositivegroupmean+.5Ostnegativegroupmean+OstDicegroupmean; epoch0included. Epoch0rejects with no newCAL_TESTfeatures/scores. No lasthead promotion.',
       'CAL':'Afterselectedepoch>0 only. Original15OstravafullmaskCAL, fixed19cuts.05:.05:.95 onnativeprobabilities, minimizepooled.5(FPR+FNR), tieearliest/smallest. VT60CAL positive-recalldiagnosticatselectedcut only, notFP/thresholdfit. Savecut/policy beforeTESTfeatures/scores.',
       'native':'sigmoid512cropcenterpaddingthenbilinearnativeprobabilityalignFalse; compareoriginalnativeGT unchanged. Thresholdstrict>. VT positivepixelrecall/perrawFair-Poor-Severe recall only; unknownexcludedfromnegativecounts. OstravaFP/FN/IoUandperphoto outliers reported separately.',
       'TEST':'Original59 once afterCAL:15Ostrava+44VT, shareoneencoderfeatures betweenfrozenv62 andnewhead. Parentbaselinecut.20fixed. Sourcegate selectedepoch>0; Ost>=10positivephotos/groups>=3,FPandFN<=.20,IoU>=.50 andFP/FNincrease<=.05each/IoUdrop<=.05vsfreshparent; VT>=20positivephotos/groups>=3,pooledpositive recall>=.80 andgain>=.05vsparent,eachrawstateFair/Poor/Severerecall>=.60. Allconditionsrequired; unknownFP unavailable. No automaticappadoption/aliaswarningclearance/ZIPevenifsourcegatepasses.',
       'uncertainty':'1000fixedproxygroupconfusionbootstrap seed7374 eachsource. Notphysicalpart/siteindependenceadjustment; VTpositive-onlyIoU/FP mustnotbereported.',
       'storage':'OnlyFIT336+VAL75uint8RGB512/target512 =411MiBRAM, bounded<=512MiB; native sequentialbuffers; encoderonline/nograd. No newdenseencoder diskcache/fullparentclone/archiveexpansion/deletions. TEST512refs59MiBRAM conditionalcheck. Newhead+optimizercheckpoints/delta/statmetadata<=256MiB disk; >=1GiBreserve.',
       'preservation':'v62parentalltensor-buffer-metadata/policies/maps exact. Newnormmustremaininitializerexact. packoriginalv16-bounddelta withapproved3prefixes+oldostrava_rust_head+newcorrosion_partial_head, diskrestoreallstate/metaexact. Keepall30epochheads/optimizers/failurecheckpoints/journal. FileSHA andcombinedtensorSHA differentcontracts, notstandaloneappmodel.',
       'conditional_device':'Onlysourcegatepasses,59originalRGBfreshCPU+freshGPU vsfirstTESTnewheadprob512 refs. atol.002/rtol.002,strictcut512pixeldecisiondifffraction<=.001perphoto. NotAPK/Androidspeed/newAPI/C++/ZIPcheck.',
       'limits':'WholehealthyTEST0Ostrava/unknownVT so no appphotoFP_FN20percentpass. BroadVTpolygons/burnedindates/sourceappearance notexpertallrustboundary/fieldhealth. No weights/mixtures/structure/TESTcut retries, oldv63ANDnotrepeated.',
       'licenses':'VTBianchi/HebdonV2 dataCC0 DOI10.7294/16624663.v2 and10.1061/(ASCE)CP.1943-5487.0001045; OstravaFricZenodoV2CC BY4.0 DOI10.5281/zenodo.11235637; preserveauthors/source/version/derivedprojection changes, InsPLADCC BY-NC3.0 andallparentterms.',
       'new_units_passed':3,'independent_field_validation':False}
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'manifest_sha256':sha(MANIFEST),'protected_files':len(protected)},flush=True)


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
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=CorrosionPartialMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(parent['state_dict'],strict=False);assert not extra and all(n.startswith('corrosion_partial_head.') for n in missing)
    model.corrosion_partial_head.load_state_dict(model.ostrava_rust_head.state_dict())
    assert all(torch.equal(v,model.ostrava_rust_head.state_dict()[n]) for n,v in model.corrosion_partial_head.state_dict().items())
    return model.eval().requires_grad_(False).cuda(),parent,original


def archives_for(rows,stack):return {n:stack.enter_context(zipfile.ZipFile(ROOT/n)) for n in sorted({r['archive'] for r in rows})}


def pair(archives,r,stage):
    archive=archives[r['archive']];classes=None
    if r['source']=='vt':
        raw=archive.read(r['source_json_member']);assert hashlib.sha256(raw).hexdigest()==r['source_json_encoded_sha256']
        a=json.loads(raw);rgb,encoded,_=embedded_image(a);classes,_=polygon_labels(a,tuple(r['size']))
        assert hashlib.sha256(encoded).hexdigest()==r['embedded_encoded_sha256'] and hashlib.sha256(classes.tobytes()).hexdigest()==r['class_raster_sha256']
        truth=partial_target('vt',classes)
    else:
        raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['encoded_sha256']
        with Image.open(BytesIO(raw)) as im:rgb=im.convert('RGB')
        truth=data.native_target(archive,r,'ostrava')
    assert list(rgb.size)==r['size'] and hashlib.sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256']
    assert hashlib.sha256(truth.tobytes()).hexdigest()==r['derived_native_sha256']
    target,_,geometry=mask_view(truth);assert geometry==r['input_geometry'] and hashlib.sha256(target.tobytes()).hexdigest()==r['derived_512_sha256']
    rw,rh=geometry['resized_size'];view=Image.new('RGB',(512,512),(124,116,104));view.paste(rgb.resize((rw,rh),Image.Resampling.BICUBIC),(geometry['left'],geometry['top']))
    with (RUN/'input-events.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps({'UTC':utc(),'id':r['id'],'stage':stage,'source_RGB_redecode':True,'not_new_natural_photo':True})+'\n')
    return np.array(view),target,geometry,truth,classes


def prepared(phase):
    rows=rows_for(phase);x=torch.empty((len(rows),3,512,512),dtype=torch.uint8);y=torch.empty((len(rows),512,512),dtype=torch.uint8)
    xhash=hashlib.sha256();yhash=hashlib.sha256()
    with ExitStack() as stack:
        archives=archives_for(rows,stack)
        for i,r in enumerate(rows):
            xx,yy,_,_,_=pair(archives,r,'prepare_'+phase);x[i].copy_(torch.from_numpy(xx).permute(2,0,1));y[i].copy_(torch.from_numpy(yy))
            xhash.update(x[i].numpy().tobytes());yhash.update(y[i].numpy().tobytes())
    assert x.numel()+y.numel()<=512*1024**2
    new(RUN/(phase+'-input-binding.json'),{'ids':[r['id'] for r in rows],'manifest_sha256':sha(MANIFEST),'protocol_sha256':sha(PROTOCOL),'uint8_RGB512_sha256':xhash.hexdigest(),'uint8_target512_sha256':yhash.hexdigest(),'bytes_in_RAM':x.numel()+y.numel(),'feature_disk_cache':False})
    return rows,x,y


def features(model,x,stage,ids):
    device=next(model.parameters()).device;x=x.to(device).float()/255
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    with torch.no_grad():
        result=forward_with_fine(model.backbone,(x-mean)/std)
    with (RUN/'feature-events.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps({'stage':stage,'ids':ids,'encoder_inputs':len(ids)})+'\n')
    return {n:result[n].half() for n in ('fine','low','high')}


def loss_on(model,x,y,rows,stage):
    model.corrosion_partial_head.eval();components=[]
    with torch.no_grad():
        for i in range(0,len(rows),BATCH):
            f=features(model,x[i:i+BATCH],stage,[r['id'] for r in rows[i:i+BATCH]])
            values=loss_components(model.forward_corrosion_partial_features(**f),y[i:i+BATCH].cuda())
            components.extend(zip(*(v.cpu().tolist() for v in values)))
    def grouped(component,source,present=None):
        groups=defaultdict(list)
        for r,c in zip(rows,components):
            if r['source']==source and (present is None or c[present]):groups[r['group_id']].append(c[component])
        return float(np.mean([np.mean(v) for v in groups.values()])) if groups else 0.
    return .25*grouped(0,'ostrava',3)+.25*grouped(0,'vt',3)+.5*grouped(1,'ostrava',4)+grouped(2,'ostrava')


def train(model,parent,original):
    fit,x,y=prepared('fit');val,vx,vy=prepared('validation');assert x.numel()+y.numel()+vx.numel()+vy.numel()<=512*1024**2
    f=features(model,x[:4],'FIT_contract',[r['id'] for r in fit[:4]])
    xx=x[:4].cuda().float()/255;xx=(xx-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():ordinary=model.backbone(xx);fresh=forward_with_fine(model.backbone,xx)
    assert all(torch.equal(ordinary[n],fresh[n]) and torch.equal(f[n],fresh[n].half()) for n in ('low','high'))
    new(RUN/'first-feature-contract.json',{'FIT_comparison_inputs':4,'ordinary_low_high_bitwise_exact':True,'extra_encoder_comparison_inputs':8,'FP16_boundary_exact':True})
    head=model.corrosion_partial_head;head.requires_grad_(True)
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==100593
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    draws=np.load(RUN/'draws.npy',allow_pickle=False);actual=0;best=loss_on(model,vx,vy,val,'initial_VAL');bestepoch=0
    torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'head-epoch-00.pt')
    new(RUN/'initial-VAL.json',{'loss':best,'epoch':0});journal=(RUN/'optimizer.jsonl').open('x',encoding='utf-8',buffering=1)
    try:
        for epoch in range(1,EPOCHS+1):
            losses=[];head.train()
            for step in range(STEPS):
                ids=draws[(epoch-1)*STEPS+step];assert [fit[i]['source'] for i in ids]==['ostrava','ostrava','vt','vt']
                f=features(model,x[ids],'fit_optimizer',[fit[i]['id'] for i in ids])
                loss=partial_loss(model.forward_corrosion_partial_features(**f),y[ids].cuda(),[True,True,False,False]);assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();actual+=1
                losses.append(float(loss.detach()));journal.write(json.dumps({'actual_update':actual,'epoch':epoch,'step':step+1,'loss':losses[-1]})+'\n')
            scheduler.step();score=loss_on(model,vx,vy,val,'VAL_epoch_'+str(epoch))
            if score<best:best,bestepoch=score,epoch
            torch.save({'head':head.state_dict(),'epoch':epoch,'VAL':score,'actual_updates':actual},RUN/f'head-epoch-{epoch:02d}.pt')
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/f'optimizer-epoch-{epoch:02d}.pt')
            new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_loss':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual});print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
    except BaseException:
        torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/('failure-checkpoint-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.pt'));raise
    finally:journal.close()
    chosen=torch.load(RUN/f'head-epoch-{bestepoch:02d}.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval().requires_grad_(False)
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':bestepoch,'selected_actual_updates':chosen['actual_updates'],'selected_VAL':best,'FIT_existing_photos':len(fit),'VAL_existing_photos':len(val),'new_natural_photos':0,'partial_positive_supervision_not_full_VT_binarymask':True}
    result=deepcopy(parent);result['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};result.update(corrosion_partial_protocol_sha256=sha(PROTOCOL),corrosion_partial_training=summary,corrosion_partial_scope='sourcepartialpositivepixelresearch_notappwholephotohealth')
    assert_preserved(parent,result);payload=pack_delta(original,result,base.EXPECTED,base.START.relative_to(ROOT).as_posix(),PREFIXES)
    (RUN/'trained').mkdir();torch.save(payload,RUN/'trained/final-delta.pt')
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED);assert_preserved(parent,restored)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in result['state_dict'].items()) and all(v==restored[n] for n,v in result.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary);return summary


def calibrate(model):
    rows=rows_for('cal');totals=np.zeros((19,4),dtype=np.int64);vt=[]
    with ExitStack() as stack,torch.inference_mode():
        archives=archives_for(rows,stack)
        for r in rows:
            view,_,g,truth,_=pair(archives,r,'cal');f=features(model,torch.from_numpy(view).permute(2,0,1)[None],'cal',[r['id']])
            prob=native_probability(model.forward_corrosion_partial_features(**f)[0].sigmoid(),g);t=torch.from_numpy(truth).to(prob.device)
            values=np.stack([known_confusion(prob,t,cut).cpu().numpy() for cut in CUTS])
            if r['source']=='ostrava':totals+=values
            else:vt.append(values)
    metrics=[pixel_rates(c) for c in totals];best=min(range(len(CUTS)),key=lambda i:(.5*(metrics[i]['fpr']+metrics[i]['fnr']),i))
    vtc=np.sum(vt,axis=0)[best];value={'cut':CUTS[best],'selected_index':best,'fixed_cuts':CUTS,'Ostrava_native_metrics':metrics,'VT_CAL_positive_recall_diagnostic_only':int(vtc[0])/int(vtc[0]+vtc[1]),'VT_unknown_FPR':None,'saved_before_TEST_features_scores':True,'delta_sha256':sha(RUN/'trained/final-delta.pt')}
    new(RUN/'calibration.json',value);return CUTS[best]


def evaluate(model,cut):
    assert (RUN/'calibration.json').exists();rows=rows_for('test');photos=[];refs=[];by_source={s:{'candidate':np.zeros(4,dtype=np.int64),'parent':np.zeros(4,dtype=np.int64)} for s in ('ostrava','vt')};severity={str(k):[0,0] for k in (1,2,3)}
    with ExitStack() as stack,torch.inference_mode():
        archives=archives_for(rows,stack)
        for r in rows:
            view,_,g,truth,classes=pair(archives,r,'test');f=features(model,torch.from_numpy(view).permute(2,0,1)[None],'test',[r['id']])
            p512=model.forward_corrosion_partial_features(**f)[0].sigmoid();old512=model.forward_ostrava_features(**f)[0].sigmoid()
            prob=native_probability(p512,g);old=native_probability(old512,g);t=torch.from_numpy(truth).to(prob.device)
            c=known_confusion(prob,t,cut).cpu().numpy();b=known_confusion(old,t,.20).cpu().numpy()
            for n,v in (('candidate',c),('parent',b)):by_source[r['source']][n]+=v
            value={'id':r['id'],'source':r['source'],'group_id':r['group_id'],'candidate_known_counts':c.tolist(),'parent_known_counts':b.tolist()}
            if r['source']=='ostrava':value.update(candidate_native_pixel_rates=pixel_rates(c),parent_native_pixel_rates=pixel_rates(b))
            else:
                value['candidate_explicit_positive_recall']=int(c[0])/int(c[0]+c[1]);value['unknown_FPR']=None
                cl=torch.from_numpy(classes).to(prob.device)
                for k in (1,2,3):
                    actual=cl==k;severity[str(k)][0]+=int(((prob>cut)&actual).sum());severity[str(k)][1]+=int(actual.sum())
            photos.append(value);refs.append(p512.cpu())
    ost={n:pixel_rates(c) for n,c in by_source['ostrava'].items()};vt={n:{'positive_recall':int(c[0])/int(c[0]+c[1]),'tp':int(c[0]),'fn':int(c[1]),'unknown_FPR':None} for n,c in by_source['vt'].items()}
    sr={k:{'recalled':v[0],'annotated_positive':v[1],'recall':v[0]/v[1]} for k,v in severity.items()};a,b=ost['candidate'],ost['parent']
    support={s:{'positive_photos':sum(r['source']==s and r['derived_native_pixels']['positive']>0 for r in rows),'groups':len({r['group_id'] for r in rows if r['source']==s and r['derived_native_pixels']['positive']>0})} for s in ('ostrava','vt')}
    passed=support['ostrava']['positive_photos']>=10 and support['vt']['positive_photos']>=20 and all(v['groups']>=3 for v in support.values()) and a['fpr']<=.20 and a['fnr']<=.20 and a['IoU']>=.50 and a['fpr']-b['fpr']<=.05 and a['fnr']-b['fnr']<=.05 and b['IoU']-a['IoU']<=.05 and vt['candidate']['positive_recall']>=.80 and vt['candidate']['positive_recall']-vt['parent']['positive_recall']>=.05 and min(v['recall'] for v in sr.values())>=.60
    intervals={};rng=np.random.default_rng(7374)
    for source in ('ostrava','vt'):
        grouped=defaultdict(lambda:np.zeros(4,dtype=np.int64))
        for p in photos:
            if p['source']==source:grouped[p['group_id']]+=np.array(p['candidate_known_counts'])
        values=np.stack([grouped[k] for k in sorted(grouped)]);samples=[]
        for _ in range(1000):
            c=values[rng.integers(len(values),size=len(values))].sum(0)
            samples.append([pixel_rates(c)[k] for k in ('fpr','fnr','IoU')] if source=='ostrava' else [int(c[0])/int(c[0]+c[1])])
        intervals[source]={'metrics':'FPR_FNR_IoU' if source=='ostrava' else 'explicit_positive_recall','95':np.quantile(samples,[.025,.975],axis=0).tolist(),'proxygroup_not_physical_independence_adjustment':True}
    result={'UTC':utc(),'source_partial_research_gate_passed':bool(passed),'automatic_app_adoption':False,'wholephoto_gate_passed':False,'Ostrava_native_pixel_metrics':ost,'VT_explicit_positive_metrics':vt,'VT_per_raw_state_recall':sr,'TEST_support':support,'cut':cut,'parent_cut':.20,'photos':photos,'bootstrap':intervals,'independent_field_validation':False,'new_API_CPP_ZIP_APK_checks':0}
    new(RUN/'selection.json',result);return result,refs


def devices(model,references,cut):
    rows=rows_for('test');reports=[]
    for device in ('cpu','cuda'):
        model.to(device);maximum=0.;changed=0
        with ExitStack() as stack,torch.inference_mode():
            archives=archives_for(rows,stack)
            for r,ref in zip(rows,references):
                view,_,_,_,_=pair(archives,r,'fresh_'+device);f=features(model,torch.from_numpy(view).permute(2,0,1)[None],'fresh_'+device,[r['id']]);p=model.forward_corrosion_partial_features(**f)[0].sigmoid().cpu()
                maximum=max(maximum,float((p-ref).abs().max()));assert torch.allclose(p,ref,atol=.002,rtol=.002)
                count=int(((p>cut)!=(ref>cut)).sum());assert count/(512*512)<=.001;changed+=count
        reports.append({'device':device,'fresh_original_RGB':len(rows),'max_probability512_difference':maximum,'changed_cut_pixels_512':changed,'pixels':len(rows)*512*512})
    new(RUN/'device-compatibility.json',{'status':'passed','checks':reports,'not_pixel_exact_or_API_CPP_APK_check':True})


def verify(model,parent,original,p):
    current=deepcopy(parent);current['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};assert_preserved(parent,current)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in current['state_dict'].items())
    for n,v in model.ostrava_rust_head.state_dict().items():
        if n.endswith(('_mean','_std')):assert torch.equal(v,model.corrosion_partial_head.state_dict()[n])
    for n,h in p['legacy_Python_sha256'].items():assert sha(ROOT/n)==h,n
    protected=read(RUN/'protected-stat.json')
    for n,(size,mtime) in protected.items():
        st=(ROOT/n).stat();assert (st.st_size,st.st_mtime_ns)==(size,mtime),n
    journal=[json.loads(v) for v in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()];assert [r['actual_update'] for r in journal]==list(range(1,1921))
    events=[json.loads(v) for v in (RUN/'feature-events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert shutil.disk_usage(ROOT).free>=1024**3
    new(RUN/'verification.json',{'status':'passed','parent_v62_and_approved_heads_all_tensor_buffer_metadata_exact':True,'new_normalization_clone_unchanged':True,'delta_disk_restore_exact':True,'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':len(protected),'all_original_raw_SHA_recalculated':False,'actual_optimizer_updates':len(journal),'encoder_input_counts_by_stage':dict(Counter({s:sum(r['encoder_inputs'] for r in events if r['stage']==s) for s in {r['stage'] for r in events}})),'extra_ordinary_and_probe_forward_inputs':8,'delta_sha256':sha(RUN/'trained/final-delta.pt'),'combined_state_sha256':state_sha256(restored['state_dict']),'new_API_CPP_ZIP_APK_checks':0})


def run():
    p=validate();assert not (RUN/'trained/training.json').exists();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,parent,original=load();summary=train(model,parent,original)
    if summary['selected_epoch']>0:
        cut=calibrate(model);result,refs=evaluate(model,cut)
        if result['source_partial_research_gate_passed']:devices(model,refs,cut)
    else:new(RUN/'selection.json',{'source_partial_research_gate_passed':False,'reason':'epoch0 minimum, newCAL_TEST scores0','automatic_app_adoption':False})
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
