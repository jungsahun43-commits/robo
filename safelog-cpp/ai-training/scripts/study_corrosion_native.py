"""One preregistered native512/stride384 study; 15 photos x 30 passes = 450 updates."""
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
from safelog_ai.corrosion_native_mask import CorrosionNativeMain,coverage_map,rgb_tile,supervised_weights,native_loss,add_probability
from safelog_ai.native_corrosion_grid import grid
from safelog_ai.corrosion_subpixel_roi import provided_target
from safelog_ai.corrosion_background_roi import target512
from safelog_ai.corrosion_partial_mask import known_confusion,native_probability
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
MANIFEST_SHA='4b2a9a8f1f37bf4515558b1cbcfccb56aceb7fde228986545402c7fa711df6e4'
RUN=ROOT/'runs/industrial-corrosion-native-v81'
PROTOCOL=ROOT/'reports/industrial-corrosion-native-v81-protocol.json'
PARENT=base.RUN/'trained/final-delta.pt'
PARENT_SHA='26c26bc5739930f153d1553b9b22840f4e83ba925cddeca5f46eee4f9f25a01b'
PARENT_STATE='61677d5f1d038006cbcfc83894af03a90b3105055ca94d5eef3c40567dc7332f'
PREFIXES=base.PREFIXES+('corrosion_native_head.',)
SOURCES=('scripts/study_corrosion_native.py','safelog_ai/corrosion_native_mask.py','tests/test_corrosion_native_mask.py')
EPOCHS,MICROBATCH,SEED=30,4,8181
CUTS=[round(i*.05,6) for i in range(1,20)]
RESERVE=1024**3


def utc():return datetime.now(timezone.utc).isoformat()
def rows_for(phase):return [r for r in read(MANIFEST)['rows'] if r['phase']==phase]
def event(name,value):
    with (RUN/name).open('a',encoding='utf-8') as out:out.write(json.dumps(value)+'\n')


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(MANIFEST)==MANIFEST_SHA
    assert read(MANIFEST)['audit']['ready'] and sha(PARENT)==PARENT_SHA
    assert sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA
    assert read(ROOT/'reports/industrial-corrosion-subpixel-v78-FIT-QA-review.json')['AI_visual_contact_sheet_reviewed']
    free=shutil.disk_usage(ROOT).free;assert free>=RESERVE+256*1024**2
    fit=rows_for('fit');assert len(fit)==15 and len({r['group_id'] for r in fit})==15
    RUN.mkdir();protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                st=path.stat();protected[path.relative_to(ROOT).as_posix()]=[st.st_size,st.st_mtime_ns]
    new(RUN/'protected-stat.json',protected)
    orders=np.stack([np.random.default_rng(SEED+epoch).permutation(15) for epoch in range(30)]).astype(np.int32)
    np.save(RUN/'epoch-orders.npy',orders,allow_pickle=False)
    files=[MANIFEST,PARENT,base.MANIFEST,base.RUN/'selection.json',ROOT/'reports/industrial-corrosion-subpixel-v78-data-audit.json',ROOT/'reports/industrial-corrosion-subpixel-v78-FIT-QA-review.json',ROOT/'reports/industrial-corrosion-native-geometry-v80-probe.json']
    files += [ROOT/n for n in ('scripts/study_corrosion_partial.py','scripts/prepare_corrosion_partial.py','scripts/study_ostrava_rust_mask.py','safelog_ai/ostrava_rust_mask.py','safelog_ai/corrosion_partial_mask.py','safelog_ai/corrosion_subpixel_roi.py','safelog_ai/corrosion_background_roi.py','safelog_ai/native_corrosion_grid.py')]
    files += [ROOT/fit[0]['archive'],base.ARCHIVE]
    p={'UTC':utc(),'fixed_before_real_features_scores_gradient_optimizer':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},
       'manifest_sha256':MANIFEST_SHA,'parent_file_sha256':PARENT_SHA,'parent_combined_state_sha256':PARENT_STATE,
       'protected_stat_sha256':sha(RUN/'protected-stat.json'),'epoch_orders_sha256':sha(RUN/'epoch-orders.npy'),
       'legacy_Python_sha256':{v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES},
       'task':'One native-input sparse ROI hypothesis. Rios original 34 photos/roles unchanged, FIT15/VAL6/CAL6/TEST7. Provided Corrosion1/Background0, outside255 unknown. No new photos/GT, wholemask/wholehealthy/machinehealth/depth labels. Missing83 and publisher5x5-vs-actualsubpixel discrepancy retained; v76 remains failed.',
       'data':read(MANIFEST)['audit']['counts'],
       'input':'Native RGB without rescale; input-size-only full512 grid stride384, regular starts0,384,... then append axis_length-512 if necessary; axis<=512 starts0. Right/bottom RGB padding124,116,104; GT255. Row-major grid. No augmentation/TEST-GT-selected crop. At inference/CAL/TEST execute ALL tiles, uniform native probability mean in overlaps.',
       'model':'One unchanged common encoder fine24@128/low40@64/high960@32; new corrosion_native_head100593params exact clone of selected ORIGINAL v62 head+normalization, all its parameters trained. All v16/approved3heads/oldv62 tensors/buffers/metadata fixed/eval. No failedv79 or other failed head initialization. Distinct forward_corrosion_native_tiles/features; no source/date/category input or second encoder.',
       'normalization':'Originalv62 FIT normalization buffers exact, no new fitting. FP32 encoder, explicit FP16 fine/low/high boundary, then FP32 head/std floor.001/clamp[-8,8]. First4FIT tiles fullprecision low/high bitwise exact with ordinary backbone; 8 extra comparison encoder inputs counted separately.',
       'loss':'For each photo unique native N0,N1 and input-grid coverage C(x): sum over tiles/known native pixels of .5*BCE(logit,0)/(N0*C(x)) for class0 plus .5*BCE(logit,1)/(N1*C(x)) for class1. Both classes required per PHOTO, not per tile. Unknown/padding exact zero weight/gradient and excluded from both denominators. No Dice/unknown background fill. Native GT is neither resized nor inferred.',
       'training':'Each epoch one saved permutation of15photos, each exactly once; sequential tile microbatches<=4 within a photo, sum gradients across its supported tiles, clip5 then ONE optimizer update per photo. FIT empty-GT tiles zero objective and skipped; VAL/CAL/TEST full grid never skipped by GT. 30epochs*15=450actualupdates/450photoexposures, not new450photos. Tile/encoder/microbatch counts journaled separately.',
       'optimizer':'AdamW lr1e-4 wd.01, cosine30epochs min1e-6, FP32/noAMP/TF32off/threads4, torch/numpy seed8181; order epoch e uses numpy default_rng(8181+e_zero_based). No hyperparameter/stride/size/structure search.',
       'epochs':30,'photo_updates_per_epoch':15,'planned_optimizer_updates':450,'planned_FIT_photo_exposures':450,'microbatch':4,
       'VAL_selection':'Same coverage-weighted per-photo objective on all6lockedVAL, groupmean, initialepoch0 and each30epoch. Earliest strict minimum. Epoch0 rejected without newCAL_TEST features/scores. No retrospective last-epoch promotion.',
       'CAL':'Afterselectedepoch>0 only: locked6CAL full input grid; 19 fixed cuts .05:.05:.95 minimize pooled annotated-native .5(FPR+FNR), tie smallestcut. Save calibration before TEST. No wholephoto85percentile because sparse ROI GT only.',
       'source_TEST':'Once7reserved photos after CAL. New head/newcut versus originalv62 head/FIXEDcut.20 on SAME native input grids/shared encoder features. Report native per-photo, pooled, macro, worst, and unknown predicted coverage (notFPR). This comparison includes fixed threshold policy; not a causal proof solely of learned features. Priorv79 whole512 v62 metrics are historical, not this same-grid baseline.',
       'source_gate':{'selected_epoch_gt0':True,'each_class_photos_min':5,'each_class_groups_min':3,'each_class_native_pixels_min':375,'pooled_FPR_FNR_max':.20,'macrophoto_FPR_FNR_max':.25,'worstphoto_FPR_FNR_max':.50,'balanced_accuracy_gain_vs_samegrid_v62_min':.05},
       'retention':'Only if source gate passes: originalOstrava15TEST once, new native512stride384 at newCALcut versus originalv62 WHOLE512/cut.20. Native maps/fullL>0 GT. 2048square=>25native tile encoder inputs +1whole512 comparison input/photo; TWO views, not one forward. Require pooledFP/FN<=.20,IoU>=.50,FP/FNincrease<=.05each,IoUdrop<=.05. Otherwise full candidate excluded. If source fails new retentionTEST0.',
       'conditional_device':'Only both gates pass: fresh originals all7source+15Ostrava, fullnative grid CPU then GPU vs firstTEST nativeprobability references. atol.002/rtol.002 and cut disagreement<=.001nativepixels per photo. Not pixel-exact identity or API/CPP/APK/Android test.',
       'uncertainty':'1000proxygroup bootstrap seed8182 on source annotated pooledFPR/FNR. Groups not verified physical parts; similar checkerplates107/115 possible shared capture. 7TEST is not wholephoto20percent app gate/independent field validation. No zero-miss guarantee.',
       'storage':{'free_at_freeze_bytes':free,'reserve_bytes':RESERVE,'new_disk_budget_bytes':256*1024**2,'native_FIT_VAL_RGB_GT_coverage_RAM_limit_bytes':512*1024**2,'conditional_native_refs_RAM_limit_bytes':512*1024**2,'dense_feature_files':False,'full_parent_copy':False,'archive_expansion':False,'deletions':False},
       'preservation':'Originalv16-bound delta includes approved3prefixes+oldostrava_rust_head+newcorrosion_native_head only; parent/file/state/added/combined SHA and serialized restore exact. All parent tensors/buffers/meta/policies/primarymaps unchanged; clone normalization unchanged. Save every epoch head/optimizer/RNG, actualstep/microbatch journal, partial checkpoint on failure. Delta not standalone app model.',
       'deployment':'Source/retention pass is source research only, no automatic app/alias/warning/map/ZIP replacement. Any later research handoff requires separate preregistered actual CPU/GPU/C++/alloldalias/oldPNG verification; this study API_CPP_ZIP_APK0.',
       'licenses':'Rios/Netto/Roehl2023 MendeleyV1 DOI10.17632/kcyn4nhv2c.1 dataCC BY4.0 attribution/source/version/changes; unresolved stated third-party rights retained. OstravaFricZenodoV2 DOI10.5281/zenodo.11235637 CC BY4.0. VTCC0/InsPLADCC BY-NC3.0/allparentconditions unchanged. No external code executed.',
       'new_synthetic_units_passed':3,'new_source_gradient_before_preregister':0,'independent_field_validation':False}
    new(PROTOCOL,p)
    print({'protocol_SHA256':sha(PROTOCOL),'planned_updates':450,'protected_files':len(protected),'free_bytes':free},flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['source_sha256'],**p['files_sha256'],**p['legacy_Python_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(RUN/'protected-stat.json')==p['protected_stat_sha256'] and sha(RUN/'epoch-orders.npy')==p['epoch_orders_sha256']
    assert sha(base.START)==base.EXPECTED and sha(base.PRIOR_DELTA)==base.PRIOR_SHA
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    assert shutil.disk_usage(ROOT).free>=RESERVE+128*1024**2
    return p


def load():
    original,approved=base.research_parent()
    parent=restore_delta(original,torch.load(PARENT,map_location='cpu',weights_only=True),base.EXPECTED)
    assert_preserved(approved,parent);assert state_sha256(parent['state_dict'])==PARENT_STATE
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=CorrosionNativeMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(parent['state_dict'],strict=False)
    assert not extra and missing and all(n.startswith('corrosion_native_head.') for n in missing)
    model.corrosion_native_head.load_state_dict(model.ostrava_rust_head.state_dict())
    return model.eval().requires_grad_(False).cuda(),parent,original


def archives_for(rows,stack):return {n:stack.enter_context(zipfile.ZipFile(ROOT/n)) for n in sorted({r['archive'] for r in rows})}


def native_pair(archives,r,stage):
    archive=archives[r['archive']]
    if 'original' in r:
        item=r['original'];raw=archive.read(item['member']);assert hashlib.sha256(raw).hexdigest()==item['encoded_SHA256']
        truth,_=provided_target(tuple(r['size']),r['positive_ROIs'],r['negative_ROIs'])
        assert hashlib.sha256(truth.tobytes()).hexdigest()==r['native_sparse_SHA256']
    else:
        raw=archive.read(r['member']);assert hashlib.sha256(raw).hexdigest()==r['encoded_sha256']
        truth=fullmask_helper.native_target(archive,r,'ostrava')
    with Image.open(BytesIO(raw)) as image:
        assert getattr(image,'n_frames',1)==1 and list(image.size)==r['size']
        rgb=np.array(image.convert('RGB'))
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256']
    event('input-events.jsonl',{'stage':stage,'id':r['id'],'original_RGB_redecode':1,'new_natural_photos':0})
    boxes=grid(tuple(r['size']));cov=coverage_map(tuple(r['size']))
    counts=[int((truth==k).sum()) for k in (0,1)]
    return {'row':r,'rgb':rgb,'truth':truth,'coverage':cov,'boxes':boxes,'counts':counts}


def prepared(phase):
    rows=rows_for(phase);data=[]
    with ExitStack() as stack:
        archives=archives_for(rows,stack)
        for r in rows:
            item=native_pair(archives,r,'prepare_'+phase);assert min(item['counts'])>0;data.append(item)
    used=sum(sum(d[n].nbytes for n in ('rgb','truth','coverage')) for d in data)
    new(RUN/(phase+'-input-binding.json'),{'ids':[r['id'] for r in rows],'native_RGB_GT_and_coverage_RAM_bytes':used,'original_hashes':{r['id']:[r['rgb_sha256'],r['native_sparse_SHA256']] for r in rows},'full_input_tiles':sum(len(d['boxes']) for d in data),'dense_feature_files':False})
    return data,used


def features(model,x,stage,ids):
    feature_helper.RUN=RUN
    return feature_helper.features(model,x,stage,ids)


def batch_for(item,boxes,with_weights):
    x=torch.from_numpy(np.stack([rgb_tile(item['rgb'],b) for b in boxes])).permute(0,3,1,2)
    if not with_weights:return x
    pairs=[supervised_weights(item['truth'],item['coverage'],b,item['counts']) for b in boxes]
    return x,torch.from_numpy(np.stack([p[0] for p in pairs])),torch.from_numpy(np.stack([p[1] for p in pairs]))


def photo_objective(model,item,stage,training=False):
    boxes=item['boxes']
    if training:
        boxes=[b for b in boxes if np.any(item['truth'][b['top']:b['top']+b['height'],b['left']:b['left']+b['width']]!=255)]
    total=0.;microbatches=0;tile_count=0
    for offset in range(0,len(boxes),MICROBATCH):
        chunk=boxes[offset:offset+MICROBATCH];x,y,w=batch_for(item,chunk,True)
        f=features(model,x,stage,[item['row']['id']]*len(chunk))
        loss=native_loss(model.forward_corrosion_native_features(**f),y.cuda(),w.cuda())
        assert torch.isfinite(loss)
        if training:loss.backward()
        total+=float(loss.detach());microbatches+=1;tile_count+=len(chunk)
        event('microbatches.jsonl',{'stage':stage,'id':item['row']['id'],'offset':offset,'tile_count':len(chunk),'loss_contribution':float(loss.detach()),'backward':training})
    return total,microbatches,tile_count


def val_loss(model,data,stage):
    model.corrosion_native_head.eval();grouped=defaultdict(list)
    with torch.no_grad():
        for item in data:
            value,_,_=photo_objective(model,item,stage)
            grouped[item['row']['group_id']].append(value)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def train(model,parent,original):
    fit,used=prepared('fit');val,vused=prepared('validation');assert used+vused<=512*1024**2
    head=model.corrosion_native_head;head.requires_grad_(True)
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==100593
    x=batch_for(fit[0],fit[0]['boxes'][:4],False);f=features(model,x,'FIT_contract',[fit[0]['row']['id']]*len(x))
    xx=x.cuda().float()/255;xx=(xx-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():ordinary=model.backbone(xx);fresh=forward_with_fine(model.backbone,xx)
    assert all(torch.equal(ordinary[n],fresh[n]) and torch.equal(f[n],fresh[n].half()) for n in ('low','high'))
    new(RUN/'first-feature-contract.json',{'FIT_tiles':len(x),'ordinary_low_high_bitwise_exact':True,'extra_comparison_encoder_inputs':2*len(x),'new_source_gradient':0})
    opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-6)
    orders=np.load(RUN/'epoch-orders.npy',allow_pickle=False)
    best=val_loss(model,val,'initial_VAL');bestepoch=0;actual=0;exposures=0;total_tiles=0;total_micro=0
    torch.save({'head':head.state_dict(),'VAL':best,'epoch':0,'actual_updates':0},RUN/'head-epoch-00.pt')
    new(RUN/'initial-VAL.json',{'loss':best});print({'initial_VAL':best},flush=True)
    with (RUN/'optimizer.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
        try:
            for epoch,order in enumerate(orders,1):
                assert sorted(order.tolist())==list(range(15));assert shutil.disk_usage(ROOT).free>=RESERVE+16*1024**2
                head.train();losses=[]
                for index in order:
                    opt.zero_grad(set_to_none=True)
                    value,micro,tiles=photo_objective(model,fit[index],'FIT_optimizer',True)
                    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5);assert torch.isfinite(norm)
                    opt.step();actual+=1;exposures+=1;total_tiles+=tiles;total_micro+=micro;losses.append(value)
                    journal.write(json.dumps({'actual_update':actual,'epoch':epoch,'loss':value,'photo_id':fit[index]['row']['id'],'microbatches':micro,'encoder_tile_inputs':tiles,'gradient_norm_before_clip':float(norm)})+'\n')
                scheduler.step();score=val_loss(model,val,'VAL_epoch_'+str(epoch))
                if score<best:best,bestepoch=score,epoch
                torch.save({'head':head.state_dict(),'VAL':score,'epoch':epoch,'actual_updates':actual},RUN/f'head-epoch-{epoch:02d}.pt')
                torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/f'optimizer-epoch-{epoch:02d}.pt')
                new(RUN/f'epoch-{epoch:02d}.json',{'epoch':epoch,'train_loss':float(np.mean(losses)),'VAL':score,'best_epoch':bestepoch,'actual_updates':actual})
                print({'epoch':epoch,'VAL':score,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
        except BaseException:
            torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'actual_updates':actual,'epoch':epoch,'FIT_photo_exposures':exposures,'FIT_encoder_tiles':total_tiles,'FIT_microbatches':total_micro,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/('failure-checkpoint-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.pt'))
            raise
    chosen=torch.load(RUN/f'head-epoch-{bestepoch:02d}.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval().requires_grad_(False)
    summary={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':bestepoch,'selected_actual_updates':chosen['actual_updates'],'initial_VAL':read(RUN/'initial-VAL.json')['loss'],'selected_VAL':best,'FIT_photos':15,'VAL_photos':6,'FIT_photo_exposures':exposures,'FIT_encoder_tile_inputs':total_tiles,'FIT_microbatches':total_micro,'new_natural_photos':0,'source_sparse_ROI_not_fullmask':True}
    candidate=deepcopy(parent);candidate['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()}
    candidate.update(corrosion_native_protocol_SHA256=sha(PROTOCOL),corrosion_native_training=summary,corrosion_native_scope='native512_stride384_sparseROI_research_not_wholephoto_or_fieldhealth')
    assert_preserved(parent,candidate)
    payload=pack_delta(original,candidate,base.EXPECTED,base.START.relative_to(ROOT).as_posix(),PREFIXES)
    (RUN/'trained').mkdir();torch.save(payload,RUN/'trained/final-delta.pt')
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED)
    assert_preserved(parent,restored)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in candidate['state_dict'].items()) and all(v==restored[n] for n,v in candidate.items() if n!='state_dict')
    summary['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',summary)
    return summary


def predict(model,item,stage,reference=False):
    totals=[np.zeros(item['truth'].shape,np.float32) for _ in range(2 if reference else 1)]
    with torch.no_grad():
        for offset in range(0,len(item['boxes']),MICROBATCH):
            boxes=item['boxes'][offset:offset+MICROBATCH]
            f=features(model,batch_for(item,boxes,False),stage,[item['row']['id']]*len(boxes))
            probabilities=[model.forward_corrosion_native_features(**f).sigmoid().cpu().numpy()]
            if reference:probabilities.append(model.forward_ostrava_features(**f).sigmoid().cpu().numpy())
            for total,values in zip(totals,probabilities):
                for box,p in zip(boxes,values):add_probability(total,p,box)
    return [total/item['coverage'] for total in totals]


def calibrate(model):
    rows=rows_for('cal');totals=np.zeros((19,4),np.int64);tiles=0
    with ExitStack() as stack:
        archives=archives_for(rows,stack)
        for r in rows:
            item=native_pair(archives,r,'cal');prob=predict(model,item,'cal')[0];truth=torch.from_numpy(item['truth'])
            totals+=np.stack([known_confusion(torch.from_numpy(prob),truth,c).numpy() for c in CUTS]);tiles+=len(item['boxes'])
    values=[pixel_rates(c) for c in totals];best=min(range(19),key=lambda i:(.5*(values[i]['fpr']+values[i]['fnr']),i))
    new(RUN/'calibration.json',{'cut':CUTS[best],'cuts':CUTS,'annotated_native_ROI_metrics':values,'CAL_photos':6,'encoder_tile_inputs':tiles,'TEST_features_scores_before_this':0,'wholephoto_calibration':False,'delta_sha256':sha(RUN/'trained/final-delta.pt')})
    return CUTS[best]


def whole512_reference(model,item,stage):
    _,g=target512(item['truth']);rw,rh=g['resized_size']
    view=Image.new('RGB',(512,512),(124,116,104));view.paste(Image.fromarray(item['rgb']).resize((rw,rh),Image.Resampling.BICUBIC),(g['left'],g['top']))
    x=torch.from_numpy(np.array(view)).permute(2,0,1)[None]
    with torch.no_grad():
        f=features(model,x,stage,[item['row']['id']])
        return native_probability(model.forward_ostrava_features(**f)[0].sigmoid(),g).cpu().numpy()


def evaluate(model,rows,cut,stage,retention=False):
    assert (RUN/'calibration.json').exists();photos=[];refs=[]
    totals={'candidate':np.zeros(4,np.int64),'parent':np.zeros(4,np.int64)}
    with ExitStack() as stack:
        archives=archives_for(rows,stack)
        for r in rows:
            item=native_pair(archives,r,stage);values=predict(model,item,stage,not retention)
            prob=values[0];old=whole512_reference(model,item,stage+'_parent_whole512') if retention else values[1]
            truth=torch.from_numpy(item['truth'])
            c=known_confusion(torch.from_numpy(prob),truth,cut).numpy();b=known_confusion(torch.from_numpy(old),truth,.20).numpy()
            totals['candidate']+=c;totals['parent']+=b;unknown=item['truth']==255
            photos.append({'id':r['id'],'group_id':r['group_id'],'candidate':pixel_rates(c),'parent':pixel_rates(b),'native_grid_encoder_inputs':len(item['boxes']),'extra_parent_whole512_encoder_inputs':int(retention),'unknown_positive_coverage_not_FPR':float((prob[unknown]>cut).mean()) if unknown.any() else None})
            refs.append(prob)
    assert sum(p.nbytes for p in refs)<=512*1024**2
    return {'metrics':{k:pixel_rates(v) for k,v in totals.items()},'photos':photos,'parent_input':'whole512' if retention else 'same_native_grid','candidate_input':'native512_stride384'},refs


def devices(model,rows,refs,cut):
    assert sum(p.nbytes for p in refs)<=512*1024**2;results=[]
    for device in ('cpu','cuda'):
        model.to(device);maximum=0.;changed=0;pixels=0
        with ExitStack() as stack:
            archives=archives_for(rows,stack)
            for r,ref in zip(rows,refs):
                item=native_pair(archives,r,'fresh_'+device);p=predict(model,item,'fresh_'+device)[0]
                maximum=max(maximum,float(np.max(np.abs(p-ref))))
                np.testing.assert_allclose(p,ref,atol=.002,rtol=.002)
                count=int(((p>cut)!=(ref>cut)).sum());assert count/p.size<=.001
                changed+=count;pixels+=p.size
        results.append({'device':device,'fresh_original_RGB':len(rows),'max_native_probability_difference':maximum,'native_cut_disagreement_pixels':changed,'native_pixels':pixels})
    new(RUN/'device-compatibility.json',{'status':'passed','checks':results,'not_pixel_exact_API_CPP_APK_or_Android_speed':True})


def verify(model,parent,original,p):
    current=deepcopy(parent);current['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};assert_preserved(parent,current)
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in current['state_dict'].items())
    for n,v in model.ostrava_rust_head.state_dict().items():
        if n.endswith(('_mean','_std')):assert torch.equal(v,model.corrosion_native_head.state_dict()[n])
    for n,h in p['legacy_Python_sha256'].items():assert sha(ROOT/n)==h,n
    protected=read(RUN/'protected-stat.json')
    for n,(size,mtime) in protected.items():
        s=(ROOT/n).stat();assert (s.st_size,s.st_mtime_ns)==(size,mtime),n
    journal=[json.loads(s) for s in (RUN/'optimizer.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['actual_update'] for r in journal]==list(range(1,451)) and len(journal)==450
    for epoch in range(1,31):assert sorted(r['photo_id'] for r in journal if r['epoch']==epoch)==sorted(r['id'] for r in rows_for('fit'))
    events=[json.loads(s) for s in (RUN/'feature-events.jsonl').read_text(encoding='utf-8').splitlines()]
    inputs=dict(Counter({s:sum(r['encoder_inputs'] for r in events if r['stage']==s) for s in {r['stage'] for r in events}}))
    assert inputs['FIT_optimizer']==sum(r['encoder_tile_inputs'] for r in journal)
    assert shutil.disk_usage(ROOT).free>=RESERVE
    size=sum(v.stat().st_size for v in RUN.rglob('*') if v.is_file());assert size<=p['storage']['new_disk_budget_bytes']
    new(RUN/'verification.json',{'status':'passed','parent_v62_approvedheads_all_tensor_buffer_metadata_exact':True,'normalization_clone_unchanged':True,'delta_disk_restore_exact':True,'legacy_Python_SHA_files':len(p['legacy_Python_sha256']),'protected_stat_files':len(protected),'all_original_raw_SHA_recalculated':False,'actual_optimizer_updates':450,'FIT_photo_exposures':450,'FIT_encoder_tiles':inputs['FIT_optimizer'],'FIT_microbatches':sum(r['microbatches'] for r in journal),'encoder_inputs_by_stage':inputs,'extra_comparison_encoder_inputs':8,'new_run_disk_bytes':size,'delta_sha256':sha(RUN/'trained/final-delta.pt'),'combined_state_SHA256':state_sha256(restored['state_dict']),'new_API_CPP_ZIP_APK_checks':0})


def run():
    p=validate();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,parent,original=load();summary=train(model,parent,original)
    if summary['selected_epoch']>0:
        cut=calibrate(model);rows=rows_for('test');result,refs=evaluate(model,rows,cut,'source_test')
        a,b=result['metrics']['candidate'],result['metrics']['parent'];photos=result['photos'];gain=.5*(b['fpr']+b['fnr']-a['fpr']-a['fnr'])
        support=all(sum((r['candidate']['tp']+r['candidate']['fn'] if k==1 else r['candidate']['tn']+r['candidate']['fp'])>0 for r in photos)>=5 and len({r['group_id'] for r in photos if (r['candidate']['tp']+r['candidate']['fn'] if k==1 else r['candidate']['tn']+r['candidate']['fp'])>0})>=3 for k in (0,1))
        passed=support and min(a['tp']+a['fn'],a['tn']+a['fp'])>=375 and a['fpr']<=.20 and a['fnr']<=.20 and all(np.mean([r['candidate'][k] for r in photos])<=.25 and max(r['candidate'][k] for r in photos)<=.50 for k in ('fpr','fnr')) and gain>=.05
        values=np.array([[r['candidate'][k] for k in ('tp','fn','fp','tn')] for r in photos]);rng=np.random.default_rng(8182);samples=[]
        for _ in range(1000):
            rates=pixel_rates(values[rng.integers(len(values),size=len(values))].sum(0));samples.append([rates['fpr'],rates['fnr']])
        result.update(source_sparse_ROI_point_gate_passed=bool(passed),balanced_accuracy_gain_vs_samegrid_parent=gain,cut=cut,parent_cut=.20,macrophoto_FPR=float(np.mean([r['candidate']['fpr'] for r in photos])),macrophoto_FNR=float(np.mean([r['candidate']['fnr'] for r in photos])),proxygroup_bootstrap95_FPR_FNR=np.quantile(samples,[.025,.975],axis=0).tolist(),not_physical_group_independence_adjustment=True)
        new(RUN/'source-test.json',result);retained=False
        if passed:
            ost=base.rows_for('test');retention,orefs=evaluate(model,ost,cut,'Ostrava_retention_test',True)
            a,b=retention['metrics']['candidate'],retention['metrics']['parent']
            retained=a['fpr']<=.20 and a['fnr']<=.20 and a['IoU']>=.50 and a['fpr']-b['fpr']<=.05 and a['fnr']-b['fnr']<=.05 and b['IoU']-a['IoU']<=.05
            retention['retention_gate_passed']=bool(retained);new(RUN/'retention.json',retention)
            if retained:devices(model,rows+ost,refs+orefs,cut)
        new(RUN/'selection.json',{'source_sparse_ROI_point_gate_passed':bool(passed),'Ostrava_retention_attempted':bool(passed),'Ostrava_retention_gate_passed':bool(retained),'combined_source_research_gate_passed':bool(passed and retained),'automatic_app_adoption':False,'wholephoto_app_gate_passed':False,'new_API_CPP_ZIP_APK_checks':0})
    else:new(RUN/'selection.json',{'combined_source_research_gate_passed':False,'reason':'epoch0 minimum; new CAL_TEST features/scores0','automatic_app_adoption':False,'wholephoto_app_gate_passed':False})
    verify(model,parent,original,p);print(read(RUN/'selection.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x',encoding='utf-8') as lock:lock.write(str(os.getpid()))
        try:run()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()});raise
        finally:(RUN/'operation.lock').unlink()
