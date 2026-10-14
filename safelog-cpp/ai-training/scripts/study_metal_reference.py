"""Prespecified normal-only calibration and reference fit on one prototype."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.metal_reference import ARCH,CATEGORY,MetalReferenceMain,patch_features,normal_budget_threshold
from safelog_ai.equipment_expansion import ExpandedMain,CATEGORIES,SIZE,photo_score,stage_gate
from safelog_ai.equipment_anomaly import classification_rates,rank_auroc
from safelog_ai.retention_distillation import state_sha256
from scripts.study_equipment_expansion import images,read,MANIFEST
from scripts.prepare_equipment_expansion import new
from scripts.finish_equipment_expansion import percentile995
from scripts.fetch_equipment_expansion import sha
from scripts.study_unified_belt_tiles import snapshot
RUN=ROOT/'runs/equipment-metal-reference-v1';REPORTS=ROOT/'reports';PROTOCOL=REPORTS/'equipment-metal-reference-v1-protocol.json'
START=ROOT/'runs/equipment-expansion-v1/train/final.pt';EXPECTED='c7e5b5d5f0d30473211dedf3b1d15624501a0110826d0e17db60913ed4f7a3bc'
VARIANTS=('calibration_only20','normal_reference20')
SOURCES=('safelog_ai/metal_reference.py','scripts/study_metal_reference.py','tests/test_metal_reference.py')


def freeze():
    assert not RUN.exists() and sha(START)==EXPECTED;RUN.mkdir()
    ledger=dict(read(ROOT/'runs/equipment-expansion-v1/protected-inputs.json'))
    for directory in (ROOT/'reports',ROOT/'runs/equipment-expansion-v1'):
        for p in directory.rglob('*'):
            if p.is_file() and not p.name.endswith(('.lock','.log')):ledger[p.relative_to(ROOT).as_posix()]=snapshot(p)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda x:check(*x),ledger.items()))
    new(RUN/'protected-inputs.json',ledger)
    rows=[r for r in read(MANIFEST)['rows'] if r['category']==CATEGORY];counts={phase:sum(r['phase']==phase for r in rows) for phase in ('fit','cal','test')}
    assert counts=={'fit':34,'cal':20,'test':97}
    new(PROTOCOL,{'schema':'safelog-metal-reference-fixed-v1','declared_before_fit_CAL_rescoring_or_TEST_rescoring':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for folder in ('scripts','tests','safelog_ai') for p in (ROOT/folder).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES},
        'start_sha256':EXPECTED,'manifest_sha256':sha(MANIFEST),'protected_inputs_sha256':sha(RUN/'protected-inputs.json'),'variants':list(VARIANTS),
        'counts':counts,'baseline':read(REPORTS/'equipment-expansion-v1-results.json')['categories'][CATEGORY],
        'calibration_only20':'Same weights, scores, heatmap and pixel threshold; photo threshold from existing normalCAL20 scores at ceil((n+1)*.8) order statistic',
        'normal_reference20':'Same ONE frozen common MobileNet; low40/high960 branch-normalized features16x16; avgpool3 stride1 count_include_padFalse and unit normalize; all8704 normalFIT patches, no coreset/search; exact nearest cosine with bounded chunks512/2048; interpolate to256; photo top1percent mean',
        'reference_source':'Only34 verified publisher-normal FIT photos, no masks or CAL/TEST patches in reference bank',
        'thresholds':'Photo normalCAL20 rank17/alpha.20; pixel normalCAL pooled linear quantile.995; no TEST threshold tuning or sweep',
        'neural_optimizer_updates_planned':0,'new_normal_reference_fit_photos':34,'new_source_photos_or_labels':0,
        'prespecified_selection_guards':['photo_FPR<=.20','photo_FNR<=.20','photo_FNR<prior','pixel_IoU>=prior','pixel_FNR<=prior'],
        'selection':'Onlyall5 pass; prefer smaller max(photoFPR,photoFNR), then lowerFNR, then calibration-only; otherwise prior prototype remains',
        'all_other_tensor_and_inference_outputs_preserved':True,'ksdd2_surface_preserved_exactly':True,
        'method_inspiration':'https://arxiv.org/abs/2106.08265','exact_PatchCore_reproduction':False,
        'TEST_reused_known_benchmark':True,'new_independent_factory_validation':False,'app_default_promoted':False,'unit_tests_passed':4})
    print(json.dumps({'frozen':True,'fit':34,'cal':20,'known_TEST':97,'variants':list(VARIANTS),'new_neural_updates':0,'protected_files':len(ledger)}),flush=True)


def check(n,v):assert snapshot(ROOT/n)==v,n


def validate():
    p=read(PROTOCOL)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in [(START,EXPECTED),(MANIFEST,p['manifest_sha256']),(RUN/'protected-inputs.json',p['protected_inputs_sha256'])]:assert sha(path)==h,path
    return p


def load(name,device='cuda'):
    state=torch.load(RUN/name/'final.pt',map_location='cpu',weights_only=True)
    model=ExpandedMain() if name==VARIANTS[0] else MetalReferenceMain(state['reference_patches'])
    model.load_state_dict(state['state_dict'],strict=True);model=model.to(device).eval().requires_grad_(False)
    return model,state


def fit():
    p=validate();repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();source=torch.load(START,map_location='cpu',weights_only=True)
    rows=read(MANIFEST)['rows'];fit=[r for r in rows if r['category']==CATEGORY and r['phase']=='fit'];cal=[r for r in rows if r['category']==CATEGORY and r['phase']=='cal']
    for name in VARIANTS:
        folder=RUN/name;assert not folder.exists();folder.mkdir();start=time.monotonic();state=deepcopy(source)
        if name==VARIANTS[0]:
            threshold,rank=normal_budget_threshold(state['equipment_operating_points'][CATEGORY]['scores'],.20)
            state['equipment_operating_points'][CATEGORY]['photo_threshold']=threshold
            state.update(metal_refinement_variant=name,neural_updates_this_refinement=0,normalCAL_alpha=.20,normalCAL_rank=rank)
        else:
            model=MetalReferenceMain(34*256).cuda().eval().requires_grad_(False)
            missing,extra=model.load_state_dict(source['state_dict'],strict=False);assert missing==['metal_reference'] and not extra
            patches=[]
            with torch.inference_mode():
                for i in range(0,len(fit),8):
                    f=patch_features(model.backbone(images(fit[i:i+8])));patches.append(f.permute(0,2,3,1).reshape(-1,1000))
                bank=torch.cat(patches);assert bank.shape==(8704,1000) and torch.isfinite(bank).all();model.metal_reference.copy_(bank)
                maps=[]
                for i in range(0,len(cal),8):maps.append(model.forward_equipment(images(cal[i:i+8]),CATEGORY).cpu())
                maps=torch.cat(maps);scores=photo_score(maps).double().tolist();threshold,rank=normal_budget_threshold(scores,.20)
            state['equipment_operating_points'][CATEGORY].update(photo_threshold=threshold,pixel_threshold=percentile995(maps),scores=scores)
            state.update(architecture=ARCH,state_dict={n:v.cpu() for n,v in model.state_dict().items()},reference_patches=8704,
                reference_source_ids=[r['id'] for r in fit],reference_feature_sha256=state_sha256({'normal_reference':model.metal_reference}),
                metal_refinement_variant=name,neural_updates_this_refinement=0,normalCAL_alpha=.20,normalCAL_rank=rank)
        state.update(parent_prototype_sha256=EXPECTED,refinement_protocol_sha256=sha(PROTOCOL),refinement_protocol_commit=commit)
        assert all(torch.equal(v,state['state_dict'][n]) for n,v in source['state_dict'].items())
        with (folder/'final.pt').open('xb') as f:torch.save(state,f)
        proofs={};model,saved=load(name);cpu,_=load(name,'cpu');prior=ExpandedMain().eval();prior.load_state_dict(source['state_dict'],strict=True)
        with torch.inference_mode():
            for category in CATEGORIES:
                row=next(r for r in rows if r['category']==category and r['phase']=='fit');x=images([row]);a=model.forward_equipment(x,category);b=cpu.forward_equipment(x.cpu(),category)
                assert torch.allclose(a.cpu(),b,atol=2e-4,rtol=.002),(name,category)
                if category!=CATEGORY:assert torch.equal(b,prior.forward_equipment(x.cpu(),category)),category
                proofs[category]={'CPU_GPU_close':True,'prior_CPU_exact':category!=CATEGORY}
            if name==VARIANTS[0]:
                row=fit[0];x=images([row],'cpu');assert torch.equal(cpu.forward_equipment(x,CATEGORY),prior.forward_equipment(x,CATEGORY))
        new(folder/'fit.json',{'status':'normal_only_fit_and_calibrated_before_TEST','variant':name,'weights_sha256':sha(folder/'final.pt'),
            'parent_sha256':EXPECTED,'all_prior_neural_tensors_exact':True,'new_neural_optimizer_updates':0,'normal_reference_photos':34 if name==VARIANTS[1] else 0,
            'reference_patches':8704 if name==VARIANTS[1] else 0,'normalCAL_rank':rank,'normalCAL_photos':20,
            'photo_threshold':threshold,'pixel_threshold':state['equipment_operating_points'][CATEGORY]['pixel_threshold'],'CPU_GPU_and_other_category_proofs':proofs,
            'elapsed_fit_and_proof_seconds':time.monotonic()-start,'TEST_used_for_fit_or_CAL':False})
        print(json.dumps({'fitted':name,'neural_updates':0,'threshold':threshold,'SHA':sha(folder/'final.pt')}),flush=True)


def evaluate():
    p=validate();rows=[r for r in read(MANIFEST)['rows'] if r['category']==CATEGORY and r['phase']=='test'];all_results={}
    for name in VARIANTS:
        model,state=load(name);point=state['equipment_operating_points'][CATEGORY];scores=[];pixels=dict(tp=0,fp=0,tn=0,fn=0);observations=[]
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];maps=model.forward_equipment(images(batch),CATEGORY).cpu();batch_scores=photo_score(maps).tolist();scores.extend(batch_scores)
                for row,heat,score in zip(batch,maps,batch_scores):
                    mask=np.zeros((SIZE,SIZE),dtype=bool)
                    for path in row['masks']:
                        with Image.open(ROOT/path) as im:mask|=np.array(im.convert('L').resize((SIZE,SIZE),Image.Resampling.NEAREST))>0
                    pred=heat.numpy()>point['pixel_threshold'];counts={k:int(v.sum()) for k,v in [('tp',pred&mask),('fp',pred&~mask),('tn',~pred&~mask),('fn',~pred&mask)]}
                    for k,v in counts.items():pixels[k]+=v
                    observations.append({'id':row['id'],'label':row['label'],'defect':row['defect'],'score':score,'candidate':score>point['photo_threshold'],'pixels':counts})
        rates=classification_rates([r['label'] for r in rows],scores,point['photo_threshold']);iou=pixels['tp']/max(1,pixels['tp']+pixels['fp']+pixels['fn']);fnr=pixels['fn']/max(1,pixels['tp']+pixels['fn'])
        baseline=p['baseline'];checks={'photo_FPR_within20':rates['fpr']<=.20,'photo_FNR_within20':rates['fnr']<=.20,
            'photo_FNR_better_than_prior':rates['fnr']<baseline['photo_rates']['fnr'],'pixel_IoU_not_worse':iou>=baseline['pixel_iou'],
            'pixel_FNR_not_worse':fnr<=baseline['pixel_fnr']}
        if name==VARIANTS[0]:
            prior=read(REPORTS/'equipment-expansion-v1-results.json');old={r['id']:r for r in prior['rows'] if r['category']==CATEGORY}
            assert pixels==baseline['pixel_counts_model256'] and all(abs(r['score']-old[r['id']]['score'])<1e-6 for r in observations)
        value={'variant':name,'photo_rates':rates,'pixel_counts_model256':pixels,'pixel_iou':iou,'pixel_fnr':fnr,'normal_test':26,'defect_test':71,
            'normalCAL_photo_threshold':point['photo_threshold'],'normalCAL_pixel_threshold':point['pixel_threshold'],'guards':checks,'selected_eligible':all(checks.values()),
            'stage_gate':stage_gate(rates['fpr'],rates['fnr'],26,71),'observations':observations,'checkpoint_sha256':sha(RUN/name/'final.pt')}
        new(REPORTS/f'equipment-metal-reference-v1-{name}-validation.json',value);all_results[name]=value
        print(json.dumps({'variant':name,'FPR':rates['fpr'],'FNR':rates['fnr'],'IoU':iou,'pixel_FNR':fnr,'eligible':all(checks.values())}),flush=True)
    eligible=[n for n,r in all_results.items() if r['selected_eligible']]
    chosen=min(eligible,key=lambda n:(max(all_results[n]['photo_rates']['fpr'],all_results[n]['photo_rates']['fnr']),all_results[n]['photo_rates']['fnr'],VARIANTS.index(n))) if eligible else 'prior_prototype'
    path=START if not eligible else RUN/chosen/'final.pt'
    new(REPORTS/'equipment-metal-reference-v1-selection.json',{'selected':chosen,'checkpoint_path':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),
        'all_other_categories_preserved_exactly':True,'ksdd2_surface_target15_preserved':True,'new_neural_updates_performed':0,
        'known_TEST_reused':True,'independent_field_validated':False,'app_default_promoted':False,'variant_guards':{n:r['guards'] for n,r in all_results.items()}})


def verify():
    p=validate();ledger=read(RUN/'protected-inputs.json')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda x:check(*x),ledger.items()))
    for n,h in p['legacy_source_sha256'].items():assert sha(ROOT/n)==h,n
    previous=torch.load(START,map_location='cpu',weights_only=True)['state_dict']
    for name in VARIANTS:
        state=torch.load(RUN/name/'final.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,state['state_dict'][n]) for n,v in previous.items())
    choice=read(REPORTS/'equipment-metal-reference-v1-selection.json')
    new(REPORTS/'equipment-metal-reference-v1-verification.json',{'status':'passed','protected_files':len(ledger),'legacy_sources':len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved':True,'all_prior_neural_tensors_exact':True,'new_neural_optimizer_updates':0,
        'source_FIT_CAL_TEST_split_preserved':True,'normal_reference_contains_FIT_only':True,'selected':choice['selected']})
    print(json.dumps({'verified':True,'selected':choice['selected'],'preserved_files':len(ledger)}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['freeze','fit','eval','verify'],required=True);args=parser.parse_args()
    {'freeze':freeze,'fit':fit,'eval':evaluate,'verify':verify}[args.mode]()
