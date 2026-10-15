"""Eight immutable checkpoint views on VAL only; no optimizer or new TEST score."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.group_rank_metrics import label_group_weights,weighted_auc
from safelog_ai.equipment_anomaly import rank_auroc
from safelog_ai.process_probe import process_exists
from scripts import study_gear_only as global_study
from scripts import study_gear_spatial_mil as spatial
from scripts import study_gear_residual_mil as residual
from scripts import study_gear_task_adapter as adapter
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-gear-selection-v32-diagnostic'
POLICY=ROOT/'reports/industrial-gear-selection-v32-policy.json'
SOURCES=('safelog_ai/group_rank_metrics.py','scripts/diagnose_gear_selection.py','tests/test_group_rank_metrics.py')
MODELS={'v28':global_study.RUN,'v29':spatial.RUN,'v30':residual.RUN,'v31':adapter.RUN}


def freeze():
    assert not RUN.exists() and not POLICY.exists() and read(adapter.RUN/'verification.json')['status']=='passed'
    RUN.mkdir();paths={}
    for name,folder in MODELS.items():
        selected=folder/('trained/final-delta.pt' if name=='v31' else 'trained/final.pt')
        paths[name]={'selected':str(selected),'selected_sha256':sha(selected),'last':str(folder/'progress.pt'),'last_sha256':sha(folder/'progress.pt'),'training_sha256':sha(folder/'trained/training.json')}
    p={'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_diagnostic_VAL_scores':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'manifest_sha256':sha(spatial.MANIFEST),'models':paths,
       'fine_VAL_sha256':sha(spatial.RUN/'validation-fine.pt'),
       'near_pairs_sha256':sha(ROOT/'data/industrial-gearbox-source-v25/near-pairs.json'),
       'scope':'exact8views selected and lastcompleteepoch for v28/v29/v30/v31 on same VAL1440 only; no training,checkpointpromotion,thresholdselection,CAL/TESTscoring,newRGB,or labelchanges',
       'metrics':'same macro perlabel/group BCE reproduce selected storedvalue abs<=2e-6; unweightedAUROC and weightedAUROC with perlabel/group inversecounts, fixed metrics no variants',
       'label_audit':'FIT/VAL gear metadata only: exactRGB contradictorylabels, withinphase opposite-label pHash<=2 pairs, no automatic reannotation; report componentclassimbalance',
       'source_note':'author paper https://arxiv.org/pdf/2511.15440 explicitly discusses percomponent classimbalance and componentappearance shortcuts; labels are manually assigned patchwear acceptance, not assumed labelerrors',
       'actual_new_optimizer_updates':0,'new_TEST_scores':0,'new_unit_tests_passed':3,'future_training_protocol_fixed':False}
    new(POLICY,p)


def run():
    p=read(POLICY);assert not (RUN/'result.json').exists()
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h
    assert sha(spatial.MANIFEST)==p['manifest_sha256'] and sha(spatial.RUN/'validation-fine.pt')==p['fine_VAL_sha256']
    allrows=read(spatial.MANIFEST)['rows'];rows=[r for r in allrows if r['phase']=='validation'];labels=[r['label'] for r in rows];groups=[r['group_id'] for r in rows]
    value=torch.load(spatial.RUN/'validation-fine.pt',map_location='cpu',weights_only=True);assert value['ids']==[r['id'] for r in rows]
    fine=value['fine'];drows,descriptor=global_study.cache('validation');assert [r['id'] for r in drows]==value['ids']
    weights=label_group_weights(labels,groups);truth=torch.tensor(labels,dtype=torch.float32);results={}
    for name,folder in MODELS.items():
        spec=p['models'][name];assert sha(Path(spec['selected']))==spec['selected_sha256'] and sha(Path(spec['last']))==spec['last_sha256'] and sha(folder/'trained/training.json')==spec['training_sha256']
        if name=='v28':model,state=global_study.load(Path(spec['selected']))
        elif name=='v29':model,state=spatial.load(Path(spec['selected']))
        elif name=='v30':model,state=residual.load(Path(spec['selected']))
        else:model,state=adapter.load(Path(spec['selected']))
        progress=torch.load(spec['last'],map_location='cpu',weights_only=True);results[name]={}
        for stage in ('selected','last'):
            if stage=='last':
                if name=='v28':model.gearbox_head.load_state_dict(progress['head'])
                elif name in ('v29','v30'):model.gear_spatial_head.load_state_dict(progress['head'])
                else:
                    _,extra=model.load_state_dict(progress['head'],strict=False);assert not extra
            model.eval();output=[]
            with torch.inference_mode():
                for i in range(0,len(rows),16):
                    if name=='v28':v=model.forward_gearbox_descriptor(descriptor[i:i+16])
                    elif name=='v29':v=model.forward_gear_spatial_features(fine[i:i+16].cuda())
                    elif name=='v30':v=model.forward_gear_residual_features(fine[i:i+16].cuda(),descriptor[i:i+16])
                    else:v=model.forward_gear_adapter_features(fine[i:i+16].cuda())
                    output.append(v.cpu())
            logits=torch.cat(output);scores=logits.sigmoid().tolist();bce=float(global_study.prior.macro_loss(F.binary_cross_entropy_with_logits(logits,truth,reduction='none'),rows))
            if stage=='selected':assert abs(bce-read(folder/'trained/training.json')['VAL_macro_group_BCE'])<=2e-6
            results[name][stage]={'epoch':read(folder/'trained/training.json')['selected_epoch'] if stage=='selected' else progress['epoch'],
                'VAL_macro_label_group_BCE':bce,'VAL_AUROC':rank_auroc(labels,scores),'VAL_label_group_weighted_AUROC':weighted_auc(labels,scores,weights)}
            new(RUN/(name+'-'+stage+'-VAL-scores.json'),{'ids':value['ids'],'scores':scores})
        results[name]['last_minus_selected_weighted_AUROC']=results[name]['last']['VAL_label_group_weighted_AUROC']-results[name]['selected']['VAL_label_group_weighted_AUROC']
        del model,state,progress;torch.cuda.empty_cache()
    eligible=[r for r in allrows if r['phase'] in ('fit','validation')];byrgb=defaultdict(list)
    for row in eligible:byrgb[row['rgb_sha256']].append(row)
    exact_conflicts=[{'ids':[r['id'] for r in values],'phases':sorted({r['phase'] for r in values})} for values in byrgb.values() if len({r['label'] for r in values})>1]
    indexed={r['idx']:r for r in eligible};near_path=ROOT/'data/industrial-gearbox-source-v25/near-pairs.json';assert sha(near_path)==p['near_pairs_sha256'];opposite=[];cross=[]
    for a,b in read(near_path):
        if a not in indexed or b not in indexed:continue
        left,right=indexed[a],indexed[b]
        if left['phase']!=right['phase']:cross.append([a,b])
        if left['label']!=right['label']:opposite.append({'left':left['id'],'right':right['id'],'phase':left['phase']})
    assert not cross
    counts={phase:{group:dict(Counter(str(r['label']) for r in eligible if r['phase']==phase and r['group_id']==group)) for group in sorted({r['group_id'] for r in eligible if r['phase']==phase})} for phase in ('fit','validation')}
    result={'UTC':datetime.now(timezone.utc).isoformat(),'status':'completed_VAL_only_diagnostic','policy_sha256':sha(POLICY),'results':results,
        'VAL_patches':len(rows),'VAL_groups':len(set(groups)),'new_diagnostic_VAL_forward_inputs':8*len(rows),
        'FIT_VAL_group_label_counts':counts,'development_exact_RGB_opposite_label_groups':exact_conflicts,
        'development_near_opposite_label_pairs':opposite,'cross_FIT_VAL_near_pairs':cross,
        'near_opposite_labels_are_not_confirmed_label_errors':True,'no_labels_changed':True,
        'new_RGB_features':0,'actual_new_optimizer_updates':0,'new_CAL_TEST_scores':0,'checkpoint_promotions':0,
        'new_unit_tests_passed':3,'next_training_protocol_fixed':False,'independent_field_validated':False}
    for name,spec in p['models'].items():assert sha(Path(spec['selected']))==spec['selected_sha256'] and sha(Path(spec['last']))==spec['last_sha256']
    new(RUN/'result.json',result);new(ROOT/'reports/industrial-gear-selection-v32-diagnostic.json',result)
    print({'results':results,'exact_conflicting_groups':len(exact_conflicts),'near_opposite_pairs':len(opposite),'new_optimizer':0,'new_TEST_scores':0},flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='run':
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work '+str(path))
    globals()[args.stage]()
