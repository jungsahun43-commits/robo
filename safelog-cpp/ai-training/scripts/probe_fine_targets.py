"""Fixed FIT-pair feature investigation only, zero optimizer/CAL/TEST work."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.native_component_views import normalized
from safelog_ai.conditioned_surface import frozen_residual
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_target_conditioned as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
RUN=ROOT/'runs/industrial-fine-probe-v20';PROTOCOL=ROOT/'reports/industrial-fine-probe-v20-protocol.json'
CHECKPOINT=study.RUN/'trained/final.pt';EXPECTED='ae0654d54e4f465e1d3826eade7173fa82d3dfb46fab4287b0c27766afe75255'
MANIFEST=study.DATA/'manifest.json';SOURCES=('safelog_ai/fine_feature_probe.py','scripts/probe_fine_targets.py')


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists() and sha(CHECKPOINT)==EXPECTED
    assert study.read(study.RUN/'verification.json')['status']=='passed'
    rows=study.read(MANIFEST)['rows'];pairs=[]
    for c in study.CATEGORIES:
        positives=[r for r in rows if r['phase']=='fit' and r['category']==c and r['label']]
        groups=sorted({r['group_id'] for r in positives},key=lambda g:hashlib.sha256(('fineprobe20:'+g).encode()).hexdigest())[:6]
        for g in groups:
            r=min((r for r in positives if r['group_id']==g),key=lambda r:r['id'])
            base=next(b for b in rows if b['phase']=='fit' and b['label']==0 and b['category']==c and b['base_normal_id']==r['base_normal_id'])
            pairs.append({'category':c,'base_id':base['id'],'synthetic_id':r['id']})
    assert len(pairs)==12;RUN.mkdir()
    new(PROTOCOL,{'schema':'safelog-fine-tap-FIT-probe-v20','UTC':datetime.now(timezone.utc).isoformat(),'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
        'checkpoint_sha256':EXPECTED,'manifest_sha256':sha(MANIFEST),'pairs':pairs,'selection_rule':'six FIT positive-background groups per target by SHA(fineprobe20:+group), first synthetic id; no model-score-based pair selection',
        'tap':'existing backbone numeric stage3,24channels,stride4; no hooks/weights/return_layers mutation; existing low/high must be bitwise exact',
        'metrics':'raw per-cell feature difference RMS in pooled edit-support cells, relative to base RMS; descriptive cross-representation statistics, not detection accuracy',
        'head_scores':'current frozen v19 head paired top32 sigmoid scores, no new calibration or target TEST scores',
        'actual_RGB_inputs':24,'comparison_backbone_traversals':48,'optimizer_updates':0,'VAL_CAL_TEST_photos_used':False,'not_a_new_training_protocol':True,'independent_field_validated':False})
    print({'frozen':True,'FIT_pairs':12,'optimizer_updates':0},flush=True)


def run():
    p=study.read(PROTOCOL);assert sha(CHECKPOINT)==EXPECTED and sha(MANIFEST)==p['manifest_sha256']
    repo=ROOT.parents[1];git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(lock))
    model,_=study.load(CHECKPOINT,'cuda');before=state_sha256(model);rows={r['id']:r for r in study.read(MANIFEST)['rows']};results=[]
    with torch.inference_mode():
        for pair in p['pairs']:
            base,syn=(rows[pair[k]] for k in ('base_id','synthetic_id'));assert base['phase']==syn['phase']=='fit'
            observed=[];scores=[]
            for r in (base,syn):
                assert sha(ROOT/r['image'])==r['image_sha256']
                with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
                image=normalized(rgb)[None].cuda();normal=model.backbone(image);tapped=forward_with_fine(model.backbone,image)
                assert all(torch.equal(normal[k],tapped[k]) for k in ('low','high')),r['id']
                observed.append({k:v.detach().cpu() for k,v in tapped.items()})
                residual=frozen_residual(model,tapped['low'],tapped['high'],[pair['category']])
                prob=model.forward_conditioned_features(tapped['low'],tapped['high'],residual).sigmoid().cpu()
                scores.append(float(ksdd_photo_score(prob,torch.ones_like(prob,dtype=torch.bool))[0]))
            assert sha(ROOT/syn['mask'])==syn['mask_sha256']
            with Image.open(ROOT/syn['mask']) as im:mask=torch.from_numpy(np.array(im.convert('L'))>0)[None,None].float()
            metrics={}
            for name in ('fine','low','high'):
                a,b=observed[0][name],observed[1][name];support=F.adaptive_max_pool2d(mask,a.shape[-2:])[:,0]>0
                delta=(b-a).square().mean(1).sqrt();base_rms=a.square().mean(1).sqrt()
                metrics[name]={'shape':list(a.shape),'support_cells':int(support.sum()),'mean_edit_cell_delta_RMS':float(delta[support].mean()),'mean_other_cell_delta_RMS':float(delta[~support].mean()),'mean_relative_edit_cell_delta':float((delta/(base_rms+1e-6))[support].mean())}
            results.append({**pair,'synthetic_mask_pixels':int(mask.sum()),'metrics':metrics,'v19_base_photo_score':scores[0],'v19_synthetic_photo_score':scores[1],'v19_paired_score_delta':scores[1]-scores[0]})
            print({'FIT_pair_checked':len(results),'total':12},flush=True)
    after=state_sha256(model);assert before==after
    summary={}
    for c in study.CATEGORIES:
        selected=[r for r in results if r['category']==c]
        summary[c]={'pairs':len(selected),'median_v19_paired_score_delta':float(np.median([r['v19_paired_score_delta'] for r in selected])),
            'synthetic_score_above_own_base':sum(r['v19_paired_score_delta']>0 for r in selected),
            'median_edit_cell_relative_delta':{n:float(np.median([r['metrics'][n]['mean_relative_edit_cell_delta'] for r in selected])) for n in ('fine','low','high')}}
    value={'status':'passed_existing_low_high_bitwise_exact_all24_RGB_inputs','UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL),
        'actual_FIT_pairs':12,'actual_RGB_inputs':24,'actual_backbone_traversals':48,'all_parameters_buffers_hash_unchanged':True,'state_sha256':before,
        'fine_shape':[1,24,128,128],'old_low_shape':[1,40,64,64],'old_high_shape':[1,960,32,32],
        'summary':summary,'observations':results,'optimizer_updates':0,'new_VAL_CAL_TEST_photo_scores':0,
        'fine_feature_response_is_not_detection_performance':True,'synthetic_edits_not_natural_fault_GT':True,'v20_training_protocol_fixed':False,'independent_field_validated':False}
    new(RUN/'probe.json',value);new(ROOT/'reports/industrial-fine-probe-v20.json',value);print({k:v for k,v in value.items() if k!='observations'},flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));args=parser.parse_args();globals()[args.mode]()
