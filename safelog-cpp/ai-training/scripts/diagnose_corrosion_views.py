"""A fixed 2x2 FIT-only diagnostic: parent/student heads, whole/native inputs."""
import argparse
from contextlib import ExitStack
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_fit_diagnosis import summary,balanced_difference
from safelog_ai.corrosion_background_roi import target512
from safelog_ai.corrosion_partial_mask import native_probability
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_corrosion_native as native
from scripts import study_ostrava_rust_mask as base
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-corrosion-views-v82'
POLICY=ROOT/'reports/industrial-corrosion-views-v82-policy.json'
DELTA=native.RUN/'trained/final-delta.pt'
DELTA_SHA='f5938468e7acb4b5dfbb7025b0c986c8e5a2cfc90bfe3c0f5ec6431fa983cf10'
COMBINED='91180ead165861bb4856e8e251843c766c2ca6890cf3067d623cb7e07724afc5'
CACHE=base.RUN/'fit-features.pt'
SOURCES=('scripts/diagnose_corrosion_views.py','safelog_ai/corrosion_fit_diagnosis.py','tests/test_corrosion_fit_diagnosis.py')


def freeze():
    assert not RUN.exists() and not POLICY.exists() and shutil.disk_usage(ROOT).free>1024**3+32*1024**2
    assert sha(DELTA)==DELTA_SHA and sha(CACHE)==read(base.RUN/'fit-cache.json')['sha256'];RUN.mkdir()
    files=[DELTA,CACHE,base.RUN/'fit-cache.json',base.PROTOCOL,native.PARENT,native.MANIFEST,base.MANIFEST]
    files += [ROOT/n for n in ('scripts/study_corrosion_native.py','scripts/study_corrosion_partial.py','scripts/study_ostrava_rust_mask.py','scripts/prepare_corrosion_partial.py','safelog_ai/native_corrosion_grid.py','safelog_ai/corrosion_native_mask.py')]
    new(POLICY,{'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_FIT_features_scores':True,'sources':{n:sha(ROOT/n) for n in SOURCES},'inputs':{v.relative_to(ROOT).as_posix():sha(v) for v in files},
        'scope':'All and only locked FIT: Rios15+Ostrava60; full input grids from sizes, no content/GT-selected inference tiles. Fixed 2x2 views originalv62/selectedfailedv81 x whole512/native512stride384. Same two frozen heads, no model selection or promotion. Oldv81remains rejected.',
        'statistics':'Native known-label class-balanced BCE, probability means, class-balanced probability MAE across input/head changes; no new cut/grid/pooling search or threshold selection. Unknown255 excluded, fullnegative uses onlypresentclass. No images visually selected.',
        'reuse':'Ostrava whole512 FIT FP16 features/IDs/rawSHA/maskSHA/order/protocol/encoder binding from completed v62 cache exact, no old whole extraction rerun. New native features online; Rioswhole view fresh. Share each encoder view across bothheads. No feature disk cache.',
        'preserve':'All model state before/after exact; cache and bound code/file SHA unchanged. New source gradient/optimizer/VAL_CAL_TEST/visual views/API_CPP_ZIP0. Not source/test improvement/causal proof for unseen field errors. No new natural photos, no GT changes or new audit/split.',
        'future':'If FIT view/head drift gives evidence, separately preregister ONE teacher-retention training hypothesis using existing FIT only. Teacher soft probabilities are regularization targets, not new natural annotations. No same TEST threshold or retention-guard changes.',
        'storage':'Sequential native maps and bounded RAM, output metadata<32MiB/>=1GiBreserve; retain all models/data/failedruns/caches; no full parent file/dense feature cache/deletion.',
        'licenses':'RiosCC BY4.0 DOI10.17632/kcyn4nhv2c.1/OstravaCC BY4.0 DOI10.5281/zenodo.11235637/InsPLADCC BY-NC3.0/VTCC0/allparentterms retained','new_units_passed':2,'independent_field_validation':False})
    print({'policy_SHA256':sha(POLICY),'new_optimizer':0},flush=True)


def run():
    policy=read(POLICY)
    for name,h in {**policy['sources'],**policy['inputs']}.items():assert sha(ROOT/name)==h,name
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    native.RUN=RUN;model,_,_=native.load()
    payload=torch.load(DELTA,map_location='cpu',weights_only=True)
    head={n.removeprefix('corrosion_native_head.'):v for n,v in payload['added_state_dict'].items() if n.startswith('corrosion_native_head.')}
    model.corrosion_native_head.load_state_dict(head);before=state_sha256(model.state_dict());assert before==COMBINED
    cache=torch.load(CACHE,map_location='cpu',weights_only=True);ost=base.rows_for('fit');rios=native.rows_for('fit')
    assert cache['ids']==[r['id'] for r in ost] and cache['raw_sha256']==[r['encoded_sha256'] for r in ost] and cache['mask_sha256']==[r['mask_encoded_sha256'] for r in ost]
    assert cache['manifest_sha256']==base.MANIFEST_SHA and cache['encoder_parent_sha256']==base.EXPECTED and cache['approved_three_head_delta_sha256']==base.PRIOR_SHA and cache['protocol_sha256']==sha(base.PROTOCOL)
    results=[]
    with ExitStack() as stack,torch.inference_mode():
        archives=native.archives_for(rios+ost,stack)
        for source,rows in (('rios',rios),('ostrava',ost)):
            for index,row in enumerate(rows):
                assert row['phase']=='fit';item=native.native_pair(archives,row,'FIT_'+source)
                pn,qn=native.predict(model,item,'FIT_'+source+'_native',True)
                if source=='ostrava':f={n:v[index:index+1].cuda() for n,v in cache['features'].items()}
                else:
                    _,g=target512(item['truth']);view=Image.new('RGB',(512,512),(124,116,104));view.paste(Image.fromarray(item['rgb']).resize(tuple(g['resized_size']),Image.Resampling.BICUBIC),(g['left'],g['top']))
                    f=native.features(model,torch.from_numpy(np.array(view)).permute(2,0,1)[None],'FIT_rios_whole',[row['id']])
                _,g=target512(item['truth'])
                pw=native_probability(model.forward_corrosion_native_features(**f)[0].sigmoid(),g).cpu().numpy()
                qw=native_probability(model.forward_ostrava_features(**f)[0].sigmoid(),g).cpu().numpy()
                maps={'v62_whole':qw,'v62_native':qn,'v81_whole':pw,'v81_native':pn}
                stats={name:summary(p,item['truth']) for name,p in maps.items()}
                changes={name:balanced_difference(a,b,item['truth']) for name,a,b in (('v62_input_MAE',qn,qw),('v81_input_MAE',pn,pw),('whole_head_MAE',pw,qw),('native_head_MAE',pn,qn))}
                results.append({'source':source,'id':row['id'],'group':row['group_id'],'stats':stats,'probability_changes':changes,'native_tiles':len(item['boxes'])})
                print({'source':source,'photo':index+1,'total':len(rows)},flush=True)
    assert state_sha256(model.state_dict())==before and sha(CACHE)==policy['inputs'][CACHE.relative_to(ROOT).as_posix()]
    for n,h in {**policy['sources'],**policy['inputs']}.items():assert sha(ROOT/n)==h,n
    aggregates={}
    for source in ('rios','ostrava'):
        rows=[r for r in results if r['source']==source]
        aggregates[source]={'photos':len(rows),'mean_photo_balanced_BCE':{name:float(np.mean([r['stats'][name]['balanced_known_BCE'] for r in rows])) for name in results[0]['stats']},'mean_balanced_probability_MAE':{name:float(np.mean([r['probability_changes'][name] for r in rows])) for name in results[0]['probability_changes']},'native_encoder_inputs':sum(r['native_tiles'] for r in rows)}
    output={'UTC':datetime.now(timezone.utc).isoformat(),'FIT_aggregates':aggregates,'rows':results,'model_state_before_after_exact':True,'model_state_SHA256':before,'old_FIT_cache_SHA_exact':True,'new_optimizer_gradient_VAL_CAL_TEST_visual_API_CPP_ZIP':0,'new_natural_photos':0,'new_feature_cache_files':0,'not_independent_performance_or_causal_TEST_diagnosis':True}
    new(RUN/'diagnosis.json',output);new(ROOT/'reports/industrial-corrosion-views-v82-diagnosis.json',output)
    print(aggregates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text().strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x') as out:out.write(str(os.getpid()))
        try:run()
        except BaseException as e:new(RUN/'failure.json',{'error':repr(e),'UTC':datetime.now(timezone.utc).isoformat()});raise
        finally:(RUN/'operation.lock').unlink()
