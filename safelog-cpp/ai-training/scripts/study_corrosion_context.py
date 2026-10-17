"""One context-input change to the fixed v83 schedule/objectives; no new GT."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_context_head import CorrosionContextMain
from safelog_ai.corrosion_context_input import whole_geometry,native_context,context_tile128,fixed_probe_tiles
from safelog_ai.corrosion_native_mask import native_loss,add_probability
from safelog_ai.corrosion_teacher_retention import retained_weights,teacher_tile,teacher_kl
from safelog_ai.ostrava_rust_mask import RustMaskHead,pixel_rates
from safelog_ai.corrosion_partial_mask import known_confusion,native_probability
from safelog_ai.insplad_glass import assert_preserved
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.process_probe import process_exists
from scripts import study_corrosion_native as native
from scripts import study_corrosion_teacher_retention as previous
from scripts import study_ostrava_rust_mask as base
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

OLD_RUN=previous.RUN;OLD_PROTOCOL=previous.PROTOCOL
RUN=ROOT/'runs/industrial-corrosion-context-v85'
PROTOCOL=ROOT/'reports/industrial-corrosion-context-v85-protocol.json'
SOURCES=('scripts/study_corrosion_context.py','safelog_ai/corrosion_context_head.py','tests/test_corrosion_context_head.py')
PREFIXES=base.PREFIXES+('corrosion_context_head.',)
ORIGINAL_PAIR=native.native_pair;ORIGINAL_PREPARED=native.prepared;ORIGINAL_TEACHERS=previous.teachers
MODEL=None;WHOLE={}


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists()
    probe=read(ROOT/'reports/industrial-corrosion-context-v84-probe.json');assert probe['coordinate_checks_passed'] and probe['new_head_gradient_optimizer_VAL_CAL_TEST_visual_accuracy_scores']==0
    assert shutil.disk_usage(ROOT).free>=1024**3+256*1024**2;RUN.mkdir()
    protected={}
    for directory in ('data','runs','reports'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_relative_to(RUN) and not path.name.endswith(('.lock','.log')):
                s=path.stat();protected[path.relative_to(ROOT).as_posix()]=[s.st_size,s.st_mtime_ns]
    new(RUN/'protected-stat.json',protected);shutil.copyfile(OLD_RUN/'paired-orders.npz',RUN/'paired-orders.npz')
    p=read(OLD_PROTOCOL);old_sources=dict(p['sources']);old_inputs=dict(p['inputs'])
    p.update(UTC=datetime.now(timezone.utc).isoformat(),sources={n:sha(ROOT/n) for n in SOURCES},inputs={**old_inputs,**old_sources,OLD_PROTOCOL.relative_to(ROOT).as_posix():sha(OLD_PROTOCOL)},paired_orders_SHA256=sha(RUN/'paired-orders.npz'),protected_stat_SHA256=sha(RUN/'protected-stat.json'))
    for name in ('safelog_ai/corrosion_context_input.py','reports/industrial-corrosion-context-v84-policy.json','reports/industrial-corrosion-context-v84-probe.json','reports/industrial-corrosion-teacher-retention-v83-results.json'):
        p['inputs'][name]=sha(ROOT/name)
    p['legacy_Python_sha256']={v.relative_to(ROOT).as_posix():sha(v) for d in ('safelog_ai','scripts','tests') for v in (ROOT/d).glob('*.py') if v.relative_to(ROOT).as_posix() not in SOURCES}
    p['hypothesis']='One actualwholeRGB response feature addresses FITv82inputdrift/v84coordinatefeasibility. Fixedv83loss/coefficients/sampling/seed/initializer oldweights/VAL/CAL/gates unchanged. Not a TESTcut/losscoefficient/stride sweep. Previousv83remains excluded.'
    p['model']='Newcorrosion_context_head101169params =selectedfailedv81RustMaskHead100593exact+zeroConvcontext1to64x3x3(noBias)576. Old88channelconvolution computedunchanged thenaddcontextprojection; equivalentextra input withseparatezeropath preserving original kernel reduction. No extraencoder/fakecategory. Normbuffers unchanged. Newforward_corrosion_context_features/tiles requires explicitcontext. Internalcorrosion_retained_head property for unchangedsharedv83preservationchecks, no duplicate state. Allv16/approved3heads/originalv62 exactfrozen.'
    p['context']='Originalv62whole512sigmoid response, letterboxpaddingremoved/bilinearprobabilityrestoretonative alignFalse; clampP[1e-6,1-1e-6]/logit/8/clamp[-1,1], nativecrop512/neutral0padding/bilinear128alignFalse. Feature notnewnaturalGT. Contextproducerfrozen/detached. ForOstravaFIT_VAL share same75cachedteacherheadresponses/context withoutnewencoder. RiosFIT_VAL wholeviewencoder1/photo precomputedRAM; CAL_TEST/freshdevices wholeview1/photo atallowedstage; allnativegridtiles executed. Retentionparentwholecomparison shares samewholecontext response, noadditionalduplicateencoder. Nnative+1whole perinferencephoto, notoneforward.'
    p['initialization_check']='Before realgradient, fixedfirst3FITperRios/Ostrava x first/middle/lastsizetiles18. SameFP16features pass newcontexthead and unboundoriginalRustMaskHead.forward usingidenticaloldweights. Requiretorch.equal logits, maxdiff0, addedweightsall0; noaccuracy/labelselection. Synthetic3units alreadyprove CPUidentity/RNGpreserved/newprojectiongradient andcontextdetach/inputvalidation.'
    p['controlled_v83_conditions']='Original pairedorders.npz byteSHAidentical; same450pairupdates/900photoexposures, seed8383/AdamW/cosine30/loss.5BCE+.5classbalancedKL/earliestVAL/CAL/source+retentionlimits. ContextconvconstructorusesCPUfork_rng to preserve originalRNGstream. Initfailedv81, notv83selected/last. No newnormfit. Newinputonlyhypothesis.'
    p['preservation']='Originalv16-bounddelta approved3prefixes+ostrava_rust_head+corrosion_context_head ONLY. Existingalltensor-buffer-meta/norm/maps exact; parent/add/combined/fileSHA andserializedrestoreexact. Saveallepoch/optimizer/RNG/actualjournal/partialfailure. Noapp/alias/warning/map/ZIPautomaticreplacement.'
    p['storage']['free_at_freeze']=shutil.disk_usage(ROOT).free
    p['storage']['RAM_plan']='RiosnativeRGB_GT_coverage_context<=512MiB,teacher75x512FP32<=75MiB, oneOstravaRGB_GT_context_teacher atatime, deviceprobrefs<=512MiB; no newdensefiles/fullparentcopy/deletion'
    p['synthetic_units_passed']=3;p['v84_probe_summary']={k:v for k,v in probe.items() if k!='rows'}
    new(PROTOCOL,p);print({'protocol_SHA256':sha(PROTOCOL),'old_paired_orders_byte_exact':sha(RUN/'paired-orders.npz')==sha(OLD_RUN/'paired-orders.npz'),'planned_updates':450,'protected_files':len(protected)},flush=True)


def load():
    global MODEL
    original,approved=base.research_parent();parent=restore_delta(original,torch.load(native.PARENT,map_location='cpu',weights_only=True),base.EXPECTED)
    assert_preserved(approved,parent);assert state_sha256(parent['state_dict'])==native.PARENT_STATE
    torch.manual_seed(8383);torch.cuda.manual_seed_all(8383)
    model=CorrosionContextMain(parent['reference_counts'],parent['native_specs'],parent['wear_variant'],parent['surface_reference_specs'])
    missing,extra=model.load_state_dict(parent['state_dict'],strict=False);assert not extra and all(n.startswith('corrosion_context_head.') for n in missing)
    init=torch.load(previous.INIT,map_location='cpu',weights_only=True)
    head={n.removeprefix('corrosion_native_head.'):v for n,v in init['added_state_dict'].items() if n.startswith('corrosion_native_head.')}
    missing,extra=model.corrosion_context_head.load_state_dict(head,strict=False);assert missing==['context_projection.weight'] and not extra
    assert all(torch.equal(v,model.corrosion_context_head.state_dict()[n]) for n,v in head.items()) and not model.corrosion_context_head.context_projection.weight.count_nonzero()
    new(RUN/'initializer.json',{'selected_failed_v81_SHA256':sha(previous.INIT),'old_head_state_SHA256':state_sha256(head),'old_tensor_buffers_exact':True,'context_projection_zero':True,'failed_initializer_not_promoted':True})
    MODEL=model.eval().requires_grad_(False).cuda();return MODEL,parent,original


def teachers(model,phase):
    rows,values=ORIGINAL_TEACHERS(model,phase)
    WHOLE.update({r['id']:p for r,p in zip(rows,values)})
    return rows,values


def native_pair(archives,row,stage):
    item=ORIGINAL_PAIR(archives,row,stage)
    if row['id'] in WHOLE:
        p=WHOLE[row['id']];cached=True
    else:
        g=whole_geometry(row['size']);view=Image.new('RGB',(512,512),(124,116,104))
        view.paste(Image.fromarray(item['rgb']).resize(tuple(g['resized_size']),Image.Resampling.BICUBIC),(g['left'],g['top']))
        with torch.no_grad():
            f=native.features(MODEL,torch.from_numpy(np.array(view)).permute(2,0,1)[None],stage+'_whole_context',[row['id']])
            p=MODEL.forward_ostrava_features(**f)[0].sigmoid().cpu().numpy()
        cached=False
    item['whole512']=p;item['context']=native_context(p,row['size'])
    native.event('context-events.jsonl',{'stage':stage,'id':row['id'],'whole_cached_FIT_VAL_teacher':cached,'new_whole_encoder_inputs':int(not cached),'native_context_SHA256':hashlib.sha256(item['context'].tobytes()).hexdigest()})
    return item


def prepared(phase):
    data,used=ORIGINAL_PREPARED(phase);contextbytes=sum(d['context'].nbytes for d in data)
    new(RUN/(phase+'-context-binding.json'),{'ids':[d['row']['id'] for d in data],'context_RAM_bytes':contextbytes,'RGB_GT_coverage_context_RAM_bytes':used+contextbytes,'whole_encoder_inputs':len(data),'context_not_new_GT':True})
    return data,used+contextbytes


def contexts(item,boxes,model):
    return torch.from_numpy(np.stack([context_tile128(item['context'],b) for b in boxes])).to(next(model.parameters()).device)


def photo_objective(model,item,stage,training=False):
    boxes=item['boxes']
    if training:boxes=[b for b in boxes if np.any(item['truth'][b['top']:b['top']+b['height'],b['left']:b['left']+b['width']]!=255)]
    value=0.;micro=tiles=0
    for start in range(0,len(boxes),4):
        chunk=boxes[start:start+4];x,y,w=native.batch_for(item,chunk,True)
        f=native.features(model,x,stage,[item['row']['id']]*len(chunk))
        loss=native_loss(model.forward_corrosion_context_features(**f,context=contexts(item,chunk,model)),y.cuda(),w.cuda());assert torch.isfinite(loss)
        if training:loss.backward()
        value+=float(loss.detach());micro+=1;tiles+=len(chunk)
        native.event('microbatches.jsonl',{'stage':stage,'id':item['row']['id'],'tile_count':len(chunk),'loss_contribution':float(loss.detach()),'backward':training})
    return value,micro,tiles


def retention_objective(model,item,teacher512,stage,training=False):
    tnative=native_probability(torch.from_numpy(teacher512),whole_geometry(item['row']['size'])).numpy();value=0.;micro=tiles=0
    for start in range(0,len(item['boxes']),4):
        chunk=item['boxes'][start:start+4];x=native.batch_for(item,chunk,False)
        w=torch.from_numpy(np.stack([retained_weights(item['truth'],item['coverage'],b)[1] for b in chunk])).cuda()
        t=torch.from_numpy(np.stack([teacher_tile(tnative,b) for b in chunk])).cuda()
        f=native.features(model,x,stage,[item['row']['id']]*len(chunk))
        loss=teacher_kl(model.forward_corrosion_context_features(**f,context=contexts(item,chunk,model)),t,w);assert torch.isfinite(loss)
        if training:(.5*loss).backward()
        value+=float(loss.detach());micro+=1;tiles+=len(chunk)
        native.event('teacher-microbatches.jsonl',{'stage':stage,'id':item['row']['id'],'tile_count':len(chunk),'KL_contribution':float(loss.detach()),'backward':training})
    return value,micro,tiles


def validation(model,rios,ost,t512,archives,stage):
    model.corrosion_context_head.eval();rvalues=[];ovalues=[]
    with torch.no_grad():
        for item in rios:rvalues.append(photo_objective(model,item,stage+'_Rios')[0])
        for row,t in zip(ost,t512):ovalues.append(retention_objective(model,native_pair(archives,row,stage+'_Ostrava'),t,stage+'_Ostrava')[0])
    a,b=float(np.mean(rvalues)),float(np.mean(ovalues));return .5*(a+b),a,b


def initial_identity(model,rios,ost,archives):
    rows=sorted(rios,key=lambda d:d['row']['id'])[:3]+[native_pair(archives,r,'FIT_initial_identity') for r in sorted(ost,key=lambda r:r['id'])[:3]]
    total=0;maximum=0.
    with torch.no_grad():
        for item in rows:
            boxes=fixed_probe_tiles(tuple(item['row']['size']));f=native.features(model,native.batch_for(item,boxes,False),'FIT_context_identity',[item['row']['id']]*len(boxes))
            old=RustMaskHead.forward(model.corrosion_context_head,**f)
            actual=model.forward_corrosion_context_features(**f,context=contexts(item,boxes,model))
            maximum=max(maximum,float((old-actual).abs().max()));assert torch.equal(old,actual);total+=len(boxes)
    new(RUN/'initial-output-identity.json',{'fixed_FIT_photos':6,'new_native_encoder_inputs':total,'logit_pairs':total,'old88_path_zero_context_output_torch_equal':True,'maximum_difference':maximum,'new_source_gradient_optimizer':0,'not_accuracy_validation':True})


def train(model,parent,original):
    ost,t=teachers(model,'fit');vost,vt=teachers(model,'validation')
    rios,used=prepared('fit');vrios,vused=prepared('validation');assert used+vused<=512*1024**2
    head=model.corrosion_context_head
    with ExitStack() as stack:
        archives=native.archives_for(ost+vost,stack);initial_identity(model,rios,ost,archives)
        head.requires_grad_(True);assert sum(v.numel() for v in model.parameters() if v.requires_grad)==101169
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=1e-6)
        orders=np.load(RUN/'paired-orders.npz',allow_pickle=False);actual=0;tr=to=micro=0
        best,vr,vo=validation(model,vrios,vost,vt,archives,'initial_VAL');bestepoch=0
        new(RUN/'initial-VAL.json',{'objective':best,'Rios_BCE':vr,'Ostrava_teacher_KL':vo});torch.save({'head':head.state_dict(),'epoch':0,'VAL':best,'actual_updates':0},RUN/'head-epoch-00.pt')
        print({'initial_VAL':best,'Rios_BCE':vr,'Ostrava_KL':vo},flush=True)
        with (RUN/'optimizer.jsonl').open('x',encoding='utf-8',buffering=1) as journal:
            try:
                for e in range(30):
                    assert shutil.disk_usage(ROOT).free>1024**3+16*1024**2;head.train();losses=[]
                    for ir,io in zip(orders['rios'][e],orders['ostrava'][e]):
                        opt.zero_grad(set_to_none=True);a,ma,ta=photo_objective(model,rios[ir],'FIT_Rios',True)
                        for p in head.parameters():
                            if p.grad is not None:p.grad.mul_(.5)
                        b,mb,tb=retention_objective(model,native_pair(archives,ost[io],'FIT_Ostrava'),t[io],'FIT_Ostrava',True)
                        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5);assert torch.isfinite(norm);opt.step();actual+=1;tr+=ta;to+=tb;micro+=ma+mb;losses.append(.5*(a+b))
                        journal.write(__import__('json').dumps({'actual_update':actual,'epoch':e+1,'photo_id':rios[ir]['row']['id'],'Ostrava_id':ost[io]['id'],'Rios_BCE':a,'Ostrava_teacher_KL':b,'paired_objective':losses[-1],'Rios_encoder_tiles':ta,'Ostrava_encoder_tiles':tb,'microbatches':ma+mb,'gradient_norm_before_clip':float(norm)})+'\n')
                    scheduler.step();score,vr,vo=validation(model,vrios,vost,vt,archives,'VAL_epoch_'+str(e+1))
                    if score<best:best,bestepoch=score,e+1
                    torch.save({'head':head.state_dict(),'epoch':e+1,'VAL':score,'actual_updates':actual},RUN/f'head-epoch-{e+1:02d}.pt')
                    torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':e+1,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/f'optimizer-epoch-{e+1:02d}.pt')
                    new(RUN/f'epoch-{e+1:02d}.json',{'epoch':e+1,'VAL':score,'Rios_BCE':vr,'Ostrava_teacher_KL':vo,'train_objective':float(np.mean(losses)),'best_epoch':bestepoch,'actual_updates':actual})
                    print({'epoch':e+1,'VAL':score,'Rios_BCE':vr,'Ostrava_KL':vo,'best_epoch':bestepoch,'actual_optimizer':actual},flush=True)
            except BaseException:
                torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'epoch':e+1,'actual_updates':actual,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},RUN/'partial-failure.pt');raise
    chosen=torch.load(RUN/f'head-epoch-{bestepoch:02d}.pt',map_location='cpu',weights_only=True);head.load_state_dict(chosen['head']);head.eval().requires_grad_(False)
    info={'completed_epochs':30,'actual_optimizer_updates':actual,'selected_epoch':bestepoch,'selected_actual_updates':chosen['actual_updates'],'selected_VAL':best,'initial_VAL':read(RUN/'initial-VAL.json')['objective'],'Rios_original_FIT':15,'Ostrava_original_FIT':60,'Rios_photo_exposures':450,'Ostrava_photo_exposures':450,'Rios_encoder_tiles':tr,'Ostrava_encoder_tiles':to,'microbatches':micro,'new_natural_photos':0,'teacher_outputs_are_not_new_GT':True,'trainable_parameters':101169,'context_is_frozen_response_feature':True}
    candidate=deepcopy(parent);candidate['state_dict']={n:v.detach().cpu() for n,v in model.state_dict().items()};candidate.update(corrosion_context_protocol_SHA256=sha(PROTOCOL),corrosion_context_training=info,corrosion_context_scope='wholeRGB_context_native_research_not_apphealth')
    assert_preserved(parent,candidate);payload=pack_delta(original,candidate,base.EXPECTED,base.START.relative_to(ROOT).as_posix(),PREFIXES);(RUN/'trained').mkdir();torch.save(payload,RUN/'trained/final-delta.pt')
    restored=restore_delta(original,torch.load(RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),base.EXPECTED);assert_preserved(parent,restored)
    assert all(torch.equal(v,restored['state_dict'][n]) for n,v in candidate['state_dict'].items()) and all(v==restored[n] for n,v in candidate.items() if n!='state_dict')
    info['serialized_delta_restore_exact']=True;new(RUN/'trained/training.json',info);return info


def predict(model,item,stage,reference=False):
    totals=[np.zeros(item['truth'].shape,np.float32) for _ in range(2 if reference else 1)]
    with torch.no_grad():
        for start in range(0,len(item['boxes']),4):
            boxes=item['boxes'][start:start+4];f=native.features(model,native.batch_for(item,boxes,False),stage,[item['row']['id']]*len(boxes))
            probabilities=[model.forward_corrosion_context_features(**f,context=contexts(item,boxes,model)).sigmoid().cpu().numpy()]
            if reference:probabilities.append(model.forward_ostrava_features(**f).sigmoid().cpu().numpy())
            for total,values in zip(totals,probabilities):
                for b,p in zip(boxes,values):add_probability(total,p,b)
    return [v/item['coverage'] for v in totals]


def calibrate(model):
    rows=native.rows_for('cal');totals=np.zeros((19,4),np.int64);tiles=0
    with ExitStack() as stack:
        archives=native.archives_for(rows,stack)
        for r in rows:
            item=native_pair(archives,r,'cal');p=predict(model,item,'cal')[0];gt=torch.from_numpy(item['truth'])
            totals+=np.stack([known_confusion(torch.from_numpy(p),gt,c).numpy() for c in native.CUTS]);tiles+=len(item['boxes'])
    values=[pixel_rates(v) for v in totals];index=min(range(19),key=lambda i:(.5*(values[i]['fpr']+values[i]['fnr']),i))
    new(RUN/'calibration.json',{'cut':native.CUTS[index],'cuts':native.CUTS,'annotated_native_ROI_metrics':values,'CAL_photos':6,'native_encoder_tile_inputs':tiles,'whole512_context_encoder_inputs':6,'TEST_features_scores_before_this':0,'wholephoto_calibration':False,'delta_SHA256':sha(RUN/'trained/final-delta.pt')});return native.CUTS[index]


def evaluate(model,rows,cut,stage,retention=False):
    assert (RUN/'calibration.json').exists();photos=[];refs=[];totals={'candidate':np.zeros(4,np.int64),'parent':np.zeros(4,np.int64)}
    with ExitStack() as stack:
        archives=native.archives_for(rows,stack)
        for r in rows:
            item=native_pair(archives,r,stage);ps=predict(model,item,stage,not retention);p=ps[0]
            old=native_probability(torch.from_numpy(item['whole512']),whole_geometry(r['size'])).numpy() if retention else ps[1]
            gt=torch.from_numpy(item['truth']);c=known_confusion(torch.from_numpy(p),gt,cut).numpy();b=known_confusion(torch.from_numpy(old),gt,.20).numpy();totals['candidate']+=c;totals['parent']+=b;unknown=item['truth']==255
            photos.append({'id':r['id'],'group_id':r['group_id'],'candidate':pixel_rates(c),'parent':pixel_rates(b),'native_grid_encoder_inputs':len(item['boxes']),'whole512_context_encoder_inputs':1,'additional_parent_whole_encoder_inputs':0,'parent_whole_response_shared_with_context':retention,'unknown_positive_coverage_not_FPR':float((p[unknown]>cut).mean()) if unknown.any() else None});refs.append(p)
    return {'metrics':{k:pixel_rates(v) for k,v in totals.items()},'photos':photos,'parent_input':'whole512' if retention else 'same_native_grid','candidate_input':'native512_stride384_plus_whole512_context'},refs


# Reuse the frozen v83 runner's selection/retention/device/preservation gates.
# Overrides are scoped to this Python process and all code hashes are bound in v85.
previous.RUN=RUN;previous.PROTOCOL=PROTOCOL;previous.SOURCES=SOURCES;previous.PREFIXES=PREFIXES
previous.load=load;previous.train=train
native.RUN=RUN;native.native_pair=native_pair;native.predict=predict;native.calibrate=calibrate;native.evaluate=evaluate


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text().strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
        try:previous.run()
        except BaseException as e:new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':datetime.now(timezone.utc).isoformat(),'error':repr(e)});raise
        finally:(RUN/'operation.lock').unlink()
