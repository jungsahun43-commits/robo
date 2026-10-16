"""One sparse same-photo contrast study; thirty passes are 120 optimizer updates."""
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
from safelog_ai.corrosion_background_mask import CorrosionBackgroundMain,background_loss
from safelog_ai.corrosion_subpixel_roi import provided_target
from safelog_ai.corrosion_background_roi import target512
from safelog_ai.corrosion_partial_mask import native_probability,known_confusion
from safelog_ai.ostrava_rust_mask import pixel_rates
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_ostrava_rust_mask as base
from scripts import study_corrosion_partial as feature_helper
from scripts import prepare_corrosion_partial as fullmask_helper
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

MANIFEST=ROOT/'data/industrial-corrosion-subpixel-v78/manifest.json'
RUN=ROOT/'runs/industrial-corrosion-background-v79'
PROTOCOL=ROOT/'reports/industrial-corrosion-background-v79-protocol.json'
PARENT=base.RUN/'trained/final-delta.pt'
PARENT_SHA='26c26bc5739930f153d1553b9b22840f4e83ba925cddeca5f46eee4f9f25a01b'
PARENT_STATE='61677d5f1d038006cbcfc83894af03a90b3105055ca94d5eef3c40567dc7332f'
PREFIXES=base.PREFIXES+('corrosion_background_head.',)
SOURCES=('scripts/study_corrosion_background.py','safelog_ai/corrosion_background_mask.py','tests/test_corrosion_background_mask.py')
EPOCHS,BATCH,SEED=30,4,7979
CUTS=[round(i*.05,6) for i in range(1,20)]


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and read(MANIFEST)['audit']['ready'] and sha(PARENT)==PARENT_SHA
    assert read(ROOT/'reports/industrial-corrosion-subpixel-v78-FIT-QA-review.json')['AI_visual_contact_sheet_reviewed']
    assert sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA and shutil.disk_usage(ROOT).free>1024**3+256*1024**2
    fit=rows_for('fit');assert len(fit)==15 and len({r['group_id'] for r in fit})==15
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                st=path.stat();protected[path.relative_to(ROOT).as_posix()]=[st.st_size,st.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    rng=np.random.default_rng(SEED);orders=np.stack([rng.permutation(15) for _ in range(30)]).astype(np.int32);np.save(RUN/'epoch-orders.npy',orders,allow_pickle=False)
    files=[MANIFEST,PARENT,base.MANIFEST,base.RUN/'selection.json',ROOT/'reports/industrial-corrosion-subpixel-v78-data-audit.json',ROOT/'reports/industrial-corrosion-subpixel-v78-FIT-QA.json',ROOT/'reports/industrial-corrosion-subpixel-v78-FIT-QA-review.json']
    files += [ROOT/n for n in ('scripts/study_corrosion_partial.py','scripts/prepare_corrosion_partial.py','scripts/study_ostrava_rust_mask.py','safelog_ai/ostrava_rust_mask.py','safelog_ai/corrosion_partial_mask.py','safelog_ai/corrosion_subpixel_roi.py','safelog_ai/corrosion_background_roi.py')]
    files += [ROOT/fit[0]['archive'],base.ARCHIVE]
    p={'UTC':utc(),'fixed_before_real_features_scores_gradient_optimizer':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},'manifest_sha256':sha(MANIFEST),'parent_file_sha256':PARENT_SHA,'parent_combined_state_sha256':PARENT_STATE,'protected_stat_sha256':sha(RUN/'protected-stat.json'),'epoch_orders_sha256':sha(RUN/'epoch-orders.npy'),
       'legacy_Python_sha256':{v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES},
       'task':'ActualprovidedROI positives+knownnegativefromsameRiosphotos; outsidebothunknown255. Thisaddresseslackofsame-domainknownbackground, notrepeatv73mixedsourcepositive-onlyloss. No newOstravaTRAIN/VT data, no wholehealthyphotolabel/machinehealth/depthGT. v76descriptioncontractstillFAILED; v78directsubpixelcontract/5x5webdiscrepancy retained.',
       'data':read(MANIFEST)['audit']['counts'],'input':'FullnativeRGBaspect512letterbox,roundresize/centerfloorpadding RGB124,116,104, bicubicRGB/nearestknownGT with255padding; ImageNetnormalization. No GTchosencrop/augmentation. No inferencecropchosenfromTESTannotations.',
       'model':'Oneunchangedcommonencoder fine24@128/low40@64/high960@32. Newcorrosion_background_head100593params exactclone ofselectedv62allweights+FITnormalization, allnewheadparameters train; allv16/approved3head/oldv62 frozen/eval. No failedv69/v71/v73weights. Distinctforward_corrosion_background[_features]. No secondencoder/source/date/categoryinputs.',
       'normalization_features':'Reusev62FITonlybuffersexact, no newnormfit. FP16featureboundary->FP32head/std floor.001/clamp[-8,8] unchanged. Usev73genericfeaturehelper only, no failedweights/decisions. First4FIT fullprecisionlow/high ordinaryforwardbitwiseexact;8extraordinary/probeinputs separatelycounted.',
       'loss':'Perphoto L=.5*mean(BCE overprovidedpositivepixels)+.5*mean(BCE overprovidedknownnegativepixels). Requirebothclassespresentin512view; unknown/padding gradientsanddenominatorszero. NoDice/GTfill/backgroundguessing. VALsamephotoobjective,groupmean.',
       'optimizer':'AdamW lr.0001 wd.01 cosine30 min.000001 gradclip5 FP32/noAMP/TF32off/threads4/seed7979',
       'epochs':30,'steps_per_epoch':4,'planned_optimizer_updates':120,'planned_FIT_encoder_draws':450,'sampling':'Everyepoch onefixedseedshuffleofall15FITonce, batches4/4/4/3;30passes/450draws are not450newnaturalphotos. No replacement/paddedduplicates or increasedstepcount.',
       'VAL_selection':'SixlockedVALphotos, initialepoch0andeach30epoch, meanperphoto/group, earlieststrictminimum. Epoch0rejects beforeCAL_TESTnewfeatures/scores. No finalepochretrospectivepromotion.',
       'CAL':'Onlyafterselectedepoch>0,locked6CALnativeprovidedpatches,19fixedcuts.05:.05:.95 minimizepooled.5(FPR+FNR),tieearliest/smallest. Savecut/policybeforeTEST. Nohealthyphotocalibration/85percentphotoquantile becauseonlysparsepixelGT.',
       'native_metrics':'Sigmoid512cropcenterpadding/bilinearnativeprobability alignFalse, GTunchangednativeprovidedROI raster. Strictpred>cut. CountFPonlywithinprovidednegativeROIs/FNonlywithinpositiveROIs, outsideunknownnotFP. Reportperphoto/macro/maxerrorandunknownpredictedcoverage(notFPR).',
       'source_TEST_gate':'Once7reservedsourceTESTafterCAL, frozenv62cut.20 andnewCALcutsharefeatures. Researchpointgateonly:selectedepoch>0,eachclass>=5photos/groups>=3,eachclass>=375nativepixels,pooledFPR/FNR<=.20,macrophotoFPR/FNR<=.25,worstphotoFPR/FNR<=.50,balancedaccuracygain>=.05vsv62. Allrequired. TEST7<20wholephotogatealwaysfalse; sparsepointgateisnotfullmaskaccuracy.',
       'retention':'OnlyifsourceTESTgatepasses, originalOstrava15heldphotos once atnewCALcut vsfrozenv62cut.20,sharedoneencoder. ActualfullmaskpixelFP/FN<=.20/IoU>=.50 andFP,FNincrease<=.05each/IoUdrop<=.05. No newVT/InsPLADTESTscore/AND/pooling/hardcasecutchanges. Ifeitherstagefailswholecandidateexcluded; source success alone notadoption.',
       'uncertainty':'1000fixedproxygroupbootstrap seed7980 forsourceROIpooledFPR/FNR;7sourcegroups notverifiedphysicalparts/sites,107/115similarplateappearance maybecorrelated. Noindependentfieldclaimor0missguarantee.',
       'storage':'KeepFIT15+VAL6 uint8RGB+target21MiB RAM, encoderonline/no newdensefeaturefiles/fullparentcopy/archiveexpansion/deletion. Nativebufferssequential, conditionaldeviceprob512refs22MiB RAM. Newcheckpoints/delta/protectedstat<=256MiBdisk,>=1GiBreserve.',
       'preservation':'Allv62parenttensor/buffer/meta/policies/maps exact,newnormbuffersinitializerexact. Originalv16-bounddelta allowedapproved3prefixes+oldostrava_rust_head+newcorrosion_background_head only; file/parent/added/combinedSHAanddiskrestoreexact. Saveall30epochheads/optimizers, actualstepjournal/failurecheckpoint. Notstandaloneappmodel.',
       'conditional_device':'Onlybothgatespass:all7source+15OstravaoriginalfreshCPU+freshGPU vsfirstTESTprob512refs,atol.002/rtol.002 andcutdecisiondisagreementfraction<=.001perphoto. ThisisnotAPI/CPP/Android/APK/ZIPvalidation.',
       'deployment':'Evenbothsource/retentionpass, no autoapp/alias/warningclearance/newZIP. A separate boundedresearchhandoff protocol withactualCPU_GPU_CPP_alloldalias_primaryPNG checks isneeded beforedelivery. OldappZIPand3approvedheadsoverlayunchanged.',
       'licenses':'Rios/Netto/Roehl,MendeleyV1dataCC BY4.0 DOI10.17632/kcyn4nhv2c.1, author/source/version/actualROIrepresentationchanges. OstravaFricZenodoV2CC BY4.0 DOI10.5281/zenodo.11235637. AllVTCC0/InsPLADCC BY-NC3.0/parenttermsunchanged. Noexternalcodeexecuted.',
       'new_units_passed':3,'independent_field_validation':False}
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'manifest_sha256':sha(MANIFEST),'planned_optimizer':120,'protected_files':len(protected)},flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['source_sha256'],**p['files_sha256'],**p['legacy_Python_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(RUN/'protected-stat.json')==p['protected_stat_sha256'] and sha(RUN/'epoch-orders.npy')==p['epoch_orders_sha256'] and sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def load():
    original,approved=base.research_parent();parent=restore_delta(original,torch.load(PARENT,map_location='cpu',weights_only=True),base.EXPECTED);assert_preserved(approved,parent);assert state_sha256(parent['state_dict'])==PARENT_STATE
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);model=CorrosionBackgroundMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(parent['state_dict'],strict=False);assert not extra and all(n.startswith('corrosion_background_head.') for n in missing)
    model.corrosion_background_head.load_state_dict(model.ostrava_rust_head.state_dict());return model.eval().requires_grad_(False).cuda(),parent,original


def archives_for(rows,stack):return {n:stack.enter_context(zipfile.ZipFile(ROOT/n)) for n in sorted({r['archive'] for r in rows})}


def pair(archives,r,stage):
    archive=archives[r['archive']]
    if 'original' in r:
        item=r['original'];raw=archive.read(item['member']);assert hashlib.sha256(raw).hexdigest()==item['encoded_SHA256']
        with Image.open(BytesIO(raw)) as im:rgb=im.convert('RGB')
        truth,_=provided_target(tuple(r['size']),r['positive_ROIs'],r['negative_ROIs']);assert hashlib.sha256(truth.tobytes()).hexdigest()==r['native_sparse_SHA256']
    else:
        raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['encoded_sha256']
        with Image.open(BytesIO(raw)) as im:rgb=im.convert('RGB')
        truth=fullmask_helper.native_target(archive,r,'ostrava')
    assert list(rgb.size)==r['size'] and hashlib.sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256']
    target,g=target512(truth)
    if 'original' in r:assert hashlib.sha256(target.tobytes()).hexdigest()==r['target512_SHA256'] and g==r['geometry']
    rw,rh=g['resized_size'];view=Image.new('RGB',(512,512),(124,116,104));view.paste(rgb.resize((rw,rh),Image.Resampling.BICUBIC),(g['left'],g['top']))
    with (RUN/'input-events.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps({'stage':stage,'id':r['id'],'original_RGB_redecode':1,'not_new_natural_photo':True})+'\n')
    return np.array(view),target,g,truth


def features(model,x,stage,ids):
    feature_helper.RUN=RUN
    return feature_helper.features(model,x,stage,ids)


def prepared(phase):
    rows=rows_for(phase);x=torch.empty(len(rows),3,512,512,dtype=torch.uint8);y=torch.empty(len(rows),512,512,dtype=torch.uint8);xh=hashlib.sha256();yh=hashlib.sha256()
    with ExitStack() as stack:
        archives=archives_for(rows,stack)
        for i,r in enumerate(rows):
            a,b,_,_=pair(archives,r,'prepare_'+phase);x[i].copy_(torch.from_numpy(a).permute(2,0,1));y[i].copy_(torch.from_numpy(b));xh.update(x[i].numpy().tobytes());yh.update(b.tobytes())
    new(RUN/(phase+'-input-binding.json'),{'ids':[r['id'] for r in rows],'RGB512_SHA256':xh.hexdigest(),'target512_SHA256':yh.hexdigest(),'RAM_bytes':x.numel()+y.numel(),'dense_feature_disk_cache':False})
    return rows,x,y


def val_loss(model,x,y,rows,stage):
    head=model.corrosion_background_head;head.eval();values=[]
    with torch.no_grad():
        for i in range(0,len(rows),4):
            f=features(model,x[i:i+4],stage,[r['id'] for r in rows[i:i+4]]);values+=background_loss(model.forward_corrosion_background_features(**f),y[i:i+4].cuda()).cpu().tolist()
    grouped=defaultdict(list)
    for r,v in zip(rows,values):grouped[r['group_id']].append(v)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def train(model,parent,original):
    fit,x,y=prepared('fit');val,vx,vy=prepared('validation');head=model.corrosion_background_head;head.requires_grad_(True)
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==100593
    f=features(model,x[:4],'FIT_contract',[r['id'] for r in fit[:4]]);xx=x[:4].cuda().float()/255;xx=(xx-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():ordinary=model.backbone(xx);fresh=forward_with_fine(model.backbone,xx)
    assert all(torch.equal(ordinary[n],fresh[n]) and torch.equal(f[n],fresh[n].half()) for n in ('low','high'));new(RUN/'first-feature-contract.json',{'FIT_inputs':4,'ordinary_low_high_bitwise_exact':True,'extra_comparison_encoder_inputs':8})
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=1e-6);orders=np.load(RUN/'epoch-orders.npy',allow_pickle=False)
    best=val_loss(model,vx,vy,val,'initial_VAL');bestepoch=0;actual=0;torch.save({'head':head.state_dict(),'VAL':best,'epoch':0,'actual_updates':0},RUN/'head-epoch-00.pt');new(RUN/'initial-VAL.json',{'loss':best})
    with (RUN/'optimizer.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
        try:
            for epoch,order in enumerate(orders,1):
                assert sorted(order.tolist())==list(range(15));head.train();losses=[]
                for ids in np.array_split(order,4):
                    f=features(model,x[ids],'FIT_optimizer',[fit[i]['id'] for i in ids]);loss=background_loss(model.forward_corrosion_background_features(**f),y[ids].cuda()).mean();assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();actual+=1;losses.append(float(loss.detach()));journal.write(json.dumps({'actual_update':actual,'epoch':epoch,'loss':losses[-1],'photo_ids':[fit[i]['id'] for i in ids]})+'\n')
                scheduler.step();score=val_loss(model,vx,vy,val,'VAL_epoch_'+str(epoch))
                if score<best:best,bestepoch=score,epoch
                torch.save({'head':head.state_dict(),'VAL':score,'epoch':epoch,'actual_updates':actual},RUN/f'head-epoch-{epoch:02d}.pt');torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/f'optimizer-epoch-{epoch:02d}.pt')
                new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_loss':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual});print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
        except BaseException:
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'actual_updates':actual},RUN/('failure-checkpoint-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.pt'));raise
    chosen=torch.load(RUN/f'head-epoch-{bestepoch:02d}.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval().requires_grad_(False)
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':bestepoch,'selected_actual_updates':chosen['actual_updates'],'selected_VAL':best,'FIT_photos':15,'VAL_photos':6,'FIT_encoder_draws':450,'new_natural_photos':0,'source_sparse_ROI_not_fullmask':True}
    candidate=deepcopy(parent);candidate['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};candidate.update(corrosion_background_protocol_SHA256=sha(PROTOCOL),corrosion_background_training=summary,corrosion_background_scope='sparseprovidedROIresearch_notwholephoto_or_fieldhealth')
    assert_preserved(parent,candidate);payload=pack_delta(original,candidate,base.EXPECTED,base.START.relative_to(ROOT).as_posix(),PREFIXES);(RUN/'trained').mkdir();torch.save(payload,RUN/'trained/final-delta.pt')
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED);assert_preserved(parent,restored);assert all(torch.equal(v,restored['state_dict'][n]) for n,v in candidate['state_dict'].items()) and all(v==restored[n] for n,v in candidate.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary);return summary


def calibrate(model):
    rows=rows_for('cal');totals=np.zeros((19,4),dtype=np.int64)
    with ExitStack() as stack,torch.inference_mode():
        archives=archives_for(rows,stack)
        for r in rows:
            x,_,g,t=pair(archives,r,'cal');f=features(model,torch.from_numpy(x).permute(2,0,1)[None],'cal',[r['id']]);p=native_probability(model.forward_corrosion_background_features(**f)[0].sigmoid(),g);truth=torch.from_numpy(t).cuda()
            totals+=np.stack([known_confusion(p,truth,c).cpu().numpy() for c in CUTS])
    values=[pixel_rates(c) for c in totals];best=min(range(19),key=lambda i:(.5*(values[i]['fpr']+values[i]['fnr']),i))
    new(RUN/'calibration.json',{'cut':CUTS[best],'cuts':CUTS,'native_provided_ROI_metrics':values,'TEST_features_scores_before_this':0,'wholephoto_quantile_not_applicable':True,'delta_sha256':sha(RUN/'trained/final-delta.pt')});return CUTS[best]


def evaluate(model,rows,cut,stage):
    assert (RUN/'calibration.json').exists();photos=[];refs=[];totals={'candidate':np.zeros(4,dtype=np.int64),'parent':np.zeros(4,dtype=np.int64)}
    with ExitStack() as stack,torch.inference_mode():
        archives=archives_for(rows,stack)
        for r in rows:
            x,_,g,t=pair(archives,r,stage);f=features(model,torch.from_numpy(x).permute(2,0,1)[None],stage,[r['id']]);p512=model.forward_corrosion_background_features(**f)[0].sigmoid();b512=model.forward_ostrava_features(**f)[0].sigmoid();p=native_probability(p512,g);b=native_probability(b512,g);truth=torch.from_numpy(t).cuda()
            c=known_confusion(p,truth,cut).cpu().numpy();old=known_confusion(b,truth,.20).cpu().numpy();totals['candidate']+=c;totals['parent']+=old;unknown=truth==255
            photos.append({'id':r['id'],'group_id':r['group_id'],'candidate':pixel_rates(c),'parent':pixel_rates(old),'candidate_unknown_predicted_coverage_not_FPR':float(((p>cut)&unknown).sum()/unknown.sum()) if unknown.any() else None});refs.append(p512.cpu())
    return {'metrics':{k:pixel_rates(v) for k,v in totals.items()},'photos':photos},refs


def devices(model,rows,refs,cut):
    results=[]
    for device in ('cpu','cuda'):
        model.to(device);maximum=0.;changed=0
        with ExitStack() as stack,torch.inference_mode():
            archives=archives_for(rows,stack)
            for r,ref in zip(rows,refs):
                x,_,_,_=pair(archives,r,'fresh_'+device);f=features(model,torch.from_numpy(x).permute(2,0,1)[None],'fresh_'+device,[r['id']]);p=model.forward_corrosion_background_features(**f)[0].sigmoid().cpu()
                maximum=max(maximum,float((p-ref).abs().max()));assert torch.allclose(p,ref,atol=.002,rtol=.002);count=int(((p>cut)!=(ref>cut)).sum());assert count/(512*512)<=.001;changed+=count
        results.append({'device':device,'fresh_original_RGB':len(rows),'max_probability512_difference':maximum,'cut_changed_pixels512':changed,'pixels':len(rows)*512*512})
    new(RUN/'device-compatibility.json',{'status':'passed','checks':results,'not_pixel_exact_API_CPP_APK_or_Android_speed':True})


def verify(model,parent,original,p):
    current=deepcopy(parent);current['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};assert_preserved(parent,current);restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in current['state_dict'].items())
    for n,v in model.ostrava_rust_head.state_dict().items():
        if n.endswith(('_mean','_std')):assert torch.equal(v,model.corrosion_background_head.state_dict()[n])
    for n,h in p['legacy_Python_sha256'].items():assert sha(ROOT/n)==h,n
    protected=read(RUN/'protected-stat.json')
    for n,(size,mtime) in protected.items():
        s=(ROOT/n).stat();assert (s.st_size,s.st_mtime_ns)==(size,mtime),n
    journal=[json.loads(s) for s in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()];assert [r['actual_update'] for r in journal]==list(range(1,121)) and sum(len(r['photo_ids']) for r in journal)==450
    events=[json.loads(s) for s in (RUN/'feature-events.jsonl').read_text(encoding='utf-8').splitlines()];assert shutil.disk_usage(ROOT).free>=1024**3
    new(RUN/'verification.json',{'status':'passed','parent_v62_approvedheads_all_tensor_buffer_metadata_exact':True,'normalization_clone_unchanged':True,'delta_disk_restore_exact':True,'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':len(protected),'all_original_raw_SHA_recalculated':False,'actual_optimizer_updates':120,'FIT_photo_exposures':450,'encoder_inputs_by_stage':dict(Counter({s:sum(r['encoder_inputs'] for r in events if r['stage']==s) for s in {r['stage'] for r in events}})),'extra_comparison_inputs':8,'delta_sha256':sha(RUN/'trained/final-delta.pt'),'combined_state_SHA256':state_sha256(restored['state_dict']),'new_API_CPP_ZIP_APK_checks':0})


def run():
    p=validate();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model,parent,original=load();summary=train(model,parent,original)
    if summary['selected_epoch']>0:
        cut=calibrate(model);rows=rows_for('test');result,refs=evaluate(model,rows,cut,'source_test');a,b=result['metrics']['candidate'],result['metrics']['parent'];photos=result['photos'];gain=.5*(b['fpr']+b['fnr']-a['fpr']-a['fnr'])
        passed=len(rows)>=5 and len({r['group_id'] for r in rows})>=3 and min(a['tp']+a['fn'],a['tn']+a['fp'])>=375 and a['fpr']<=.20 and a['fnr']<=.20 and all(np.mean([r['candidate'][k] for r in photos])<=.25 and max(r['candidate'][k] for r in photos)<=.50 for k in ('fpr','fnr')) and gain>=.05
        rng=np.random.default_rng(7980);values=np.array([[r['candidate'][k] for k in ('tp','fn','fp','tn')] for r in photos]);samples=[]
        for _ in range(1000):
            c=pixel_rates(values[rng.integers(len(values),size=len(values))].sum(0));samples.append([c['fpr'],c['fnr']])
        result.update(source_sparse_ROI_point_gate_passed=bool(passed),balanced_accuracy_gain=gain,cut=cut,parent_cut=.20,macrophoto_FPR=float(np.mean([r['candidate']['fpr'] for r in photos])),macrophoto_FNR=float(np.mean([r['candidate']['fnr'] for r in photos])),proxygroup_bootstrap95_FPR_FNR=np.quantile(samples,[.025,.975],axis=0).tolist(),not_physical_group_independence_adjustment=True)
        new(RUN/'source-test.json',result);retained=False
        if passed:
            ost=base.rows_for('test');retention,orefs=evaluate(model,ost,cut,'Ostrava_retention_test');a,b=retention['metrics']['candidate'],retention['metrics']['parent'];retained=a['fpr']<=.20 and a['fnr']<=.20 and a['IoU']>=.50 and a['fpr']-b['fpr']<=.05 and a['fnr']-b['fnr']<=.05 and b['IoU']-a['IoU']<=.05;retention['retention_gate_passed']=bool(retained);new(RUN/'retention.json',retention)
            if retained:devices(model,rows+ost,refs+orefs,cut)
        new(RUN/'selection.json',{'source_sparse_ROI_point_gate_passed':bool(passed),'Ostrava_retention_gate_passed':bool(retained),'combined_source_research_gate_passed':bool(passed and retained),'automatic_app_adoption':False,'wholephoto_app_gate_passed':False,'new_API_CPP_ZIP_APK_checks':0})
    else:new(RUN/'selection.json',{'combined_source_research_gate_passed':False,'reason':'epoch0 minimum, newCAL_TESTfeatures/scores0','automatic_app_adoption':False,'wholephoto_app_gate_passed':False})
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
