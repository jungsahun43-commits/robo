"""Two frozen inference policies, no fit, and a reused-VAL error audit."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from safelog_ai.belt_fine_tiles import MODES, BASELINE, policy, components, fuse, guards
from safelog_ai.belt_evaluation import summarize
from safelog_ai.belt_segmentation import pixel_counts
from safelog_ai.unified_main import UnifiedMain
from safelog_ai.process_probe import process_exists
from scripts.unified_main_data import DATA, Cache, read, sha
from scripts.study_unified_belt_tiles import native, snapshot, START, START_SHA

BASE=ROOT/'runs/unified-belt-fine-tiles-v1'
PROTOCOL=ROOT/'reports/unified-belt-fine-protocol.json'
SOURCES=('safelog_ai/belt_fine_tiles.py','scripts/study_unified_belt_fine.py','tests/test_belt_fine_tiles.py')
PRIOR=ROOT/'reports/unified-belt-tiles-accepted_tiles-validation.json'


def new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def slices(rows):
    groups=defaultdict(list)
    for row in rows:
        c=row['pixels']; fraction=(c['tp']+c['fn'])/sum(c.values())
        label='<=0.25%' if fraction<=.0025 else '0.25-1%' if fraction<=.01 else '1-3%' if fraction<=.03 else '>3%'
        groups['sequence:'+row['sequence']].append(row)
        groups['GT-area:'+label].append(row)
    return {k:summarize(v) for k,v in sorted(groups.items())}


def freeze():
    assert not BASE.exists() and sha(START)==START_SHA
    BASE.mkdir(); prior=read(PRIOR)
    new(ROOT/'reports/unified-belt-fine-baseline-error-audit.json',{'status':'descriptive_known_reused_VAL_only',
        'source_sha256':sha(PRIOR),'summary':prior['summary'],'slices':slices(prior['rows']),
        'slice_membership':'Publisher GT area fraction and author sequence; descriptive only, never inference input',
        'training_performed':False,'normal_photo_FPR':None,'independent_factory_verified':False})
    files=dict(read(ROOT/'runs/unified-belt-uniform-tiles-v1/protected-inputs.json'))
    for folder in [ROOT/'reports',ROOT/'runs/unified-belt-uniform-tiles-v1',ROOT/'runs/unified-belt-uniform-tiles-v1/train']:
        for path in folder.iterdir():
            if path.is_file() and not path.name.endswith(('.log','.lock')): files[path.relative_to(ROOT).as_posix()]=snapshot(path)
    def check(item):
        name,value=item
        assert snapshot(ROOT/name)==value,name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check,files.items()))
    new(BASE/'protected-inputs.json',files)
    legacy={p.relative_to(ROOT).as_posix():sha(p) for d in ('safelog_ai','scripts','tests') for p in (ROOT/d).glob('*.py') if p.relative_to(ROOT).as_posix() not in SOURCES}
    new(PROTOCOL,{'schema':'unified-belt-fine-tiles-v1','declared_before_new_policy_VAL':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'legacy_source_sha256':legacy,'protected_ledger_sha256':sha(BASE/'protected-inputs.json'),
        'manifest_sha256':sha(DATA/'manifest.json'),'checkpoint_sha256':START_SHA,'baseline_report_sha256':sha(PRIOR),
        'baseline':prior['summary'],'baseline_slices':slices(prior['rows']),
        'policies':{m:policy(m) for m in MODES},'neural_models':1,'new_neural_updates':0,'new_independent_photos':0,'new_labels':0,
        'evaluation':'Known reused author VAL353/4 sequences, same publisher640 raster truth, threshold.5; reserved TEST untouched',
        'selection':'IoU>=currentaccepted+.005; pixelFPR/FNR, photoFNR, zeroGT no worse; smallGT<=.25% IoU/FNR no worse; each sequence pixelFNR <= baseline+.01. Rank IoU then FPR then fewer views. Otherwise retain previous ZIP.',
        'normal_photo_FPR':None,'independent_factory_verified':False,'app_default_promoted':False,'tests_passed':4})
    print(json.dumps({'frozen':True,'protected_files':len(files),'legacy_sources':len(legacy),'new_neural_updates':0}),flush=True)


def validate():
    p=read(PROTOCOL)
    for name,value in p['source_sha256'].items(): assert sha(ROOT/name)==value,name
    for path,value in [(START,START_SHA),(PRIOR,p['baseline_report_sha256']),(DATA/'manifest.json',p['manifest_sha256']),(BASE/'protected-inputs.json',p['protected_ledger_sha256'])]: assert sha(path)==value,path
    return p


def evaluate():
    p=validate(); repo=ROOT.parents[1]
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])).hexdigest()==sha(ROOT/name),name
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    lock=BASE/'evaluation.lock'
    if lock.exists():
        if process_exists(int(lock.read_text())): raise RuntimeError('Study evaluation already running')
        lock.unlink()
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        model=UnifiedMain().cuda().eval(); model.load_state_dict(torch.load(START,map_location='cpu',weights_only=True)['state_dict'],strict=True)
        rows=read(DATA/'manifest.json')['belt_known_validation']; assert len(rows)==353
        prior={r['id']:r for r in read(PRIOR)['rows']}; truth=Cache(rows,range(len(rows))); started=time.monotonic()
        for index,row in enumerate(rows):
            dest=BASE/'observations'/f'{index:04d}.json'
            if dest.exists():
                cached=read(dest); assert cached['id']==row['id'] and cached['protocol_sha256']==sha(PROTOCOL)
                continue
            rgb,_,_=native(row)
            g,c,f,box,counts=components(model,rgb,'cuda'); _,t=truth.batch([index]); observations={}
            for mode in MODES:
                probability=fuse(g,c,f,box,mode)
                item={'id':row['id'],'sequence':row['sequence'],'photo_positive':row['photo_positive'],
                    'detail_views':counts['coarse'] if mode==BASELINE else counts['fine'] if mode=='global_fine_mean' else sum(counts.values()),
                    'score':float(probability[t['valid'].bool()].max()),'pixels':pixel_counts(probability,t['mask'],t['valid'])}
                if mode==BASELINE:
                    assert item['pixels']==prior[row['id']]['pixels'],row['id']
                    assert abs(item['score']-prior[row['id']]['score'])<1e-6,row['id']
                observations[mode]=item
            new(dest,{'id':row['id'],'protocol_sha256':sha(PROTOCOL),'policies':observations,
                'wall_seconds':time.monotonic()-started})
            if (index+1)%50==0: print(json.dumps({'evaluated_photos':index+1,'total':len(rows),'elapsed_seconds':time.monotonic()-started}),flush=True)
        records=[read(BASE/'observations'/f'{i:04d}.json') for i in range(len(rows))]
        for mode in MODES:
            dest=ROOT/f'reports/unified-belt-fine-{mode}-validation.json'
            if dest.exists(): continue
            observations=[r['policies'][mode] for r in records]; summary=summarize(observations)
            summary['photo_score_definition']='Maximum fused probability over known non-padding pixels; not calibrated'
            new(dest,{'status':'known_reused_VAL353','mode':mode,'policy':policy(mode),'checkpoint_sha256':START_SHA,
                'new_neural_updates':0,'new_independent_photos':0,'new_labels':0,'rows':observations,'summary':summary,'slices':slices(observations),
                'protocol_commit':commit,'normal_photo_FPR_available':False,'reserved_TEST_used':False})
            print(json.dumps({'mode':mode,'summary':summary}),flush=True)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()): lock.unlink()


def select():
    p=validate(); results={m:read(ROOT/f'reports/unified-belt-fine-{m}-validation.json') for m in MODES}
    checks={m:guards(results[m]['summary'],p['baseline'],results[m]['slices'],p['baseline_slices']) for m in MODES[1:]}
    accepted=[m for m in MODES[1:] if all(checks[m].values())]
    selected=min(accepted,key=lambda m:(-results[m]['summary']['pixel_iou'],results[m]['summary']['pixel_fpr'],m!='global_fine_mean')) if accepted else BASELINE
    new(ROOT/'reports/unified-belt-fine-selection.json',{'selected':selected,'guards':checks,'summary':results[selected]['summary'],
        'checkpoint_sha256':START_SHA,'policy':policy(selected),'new_neural_updates':0,'normal_photo_FPR_available':False,
        'known_VAL_reused':True,'reserved_TEST_used':False,'independent_factory_verified':False,'app_default_promoted':False})
    print(json.dumps({'selected':selected,'guards':checks}),flush=True)


def verify():
    p=validate(); ledger=read(BASE/'protected-inputs.json')
    def check(item):
        name,value=item
        assert snapshot(ROOT/name)==value,name
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check,ledger.items()))
    for name,value in p['legacy_source_sha256'].items(): assert sha(ROOT/name)==value,name
    assert sha(START)==START_SHA
    new(ROOT/'reports/unified-belt-fine-verification.json',{'protected_files':len(ledger),'legacy_sources':len(p['legacy_source_sha256']),
        'old_bytes_and_mtimes_preserved':True,'one_neural_model':True,'weights_unchanged':True,
        'new_neural_updates':0,'baseline353_pixel_counts_exact_reproduced':True,'normal_photo_FPR':None,'reserved_TEST_used':False})
    print(json.dumps({'verified':True,'old_files':len(ledger),'weights_unchanged':True,'new_neural_updates':0}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['freeze','eval','select','verify'],required=True)
    args=parser.parse_args(); {'freeze':freeze,'eval':evaluate,'select':select,'verify':verify}[args.mode]()
