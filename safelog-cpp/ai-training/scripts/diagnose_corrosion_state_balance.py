"""FIT exposure/resize and VAL-only selected-vs-final diagnosis; no new TEST."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_corrosion_state_mask as study
from safelog_ai.corrosion_state_mask import state_view,native_probabilities,confusion,rates
from safelog_ai.corrosion_json_pairs import embedded_image,polygon_labels
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-corrosion-state-balance-v70'
POLICY=ROOT/'reports/industrial-corrosion-state-balance-v70-policy.json'
SCRIPT='scripts/diagnose_corrosion_state_balance.py'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not POLICY.exists()
    assert read(study.RUN/'verification.json')['status']=='passed'
    assert not read(study.RUN/'selection.json')['source_four_state_research_gate_passed']
    files=[study.PROTOCOL,study.MANIFEST,study.RUN/'head-epoch-06.pt',study.RUN/'head-epoch-30.pt',study.RUN/'fit-input-binding.json',study.RUN/'draws.npy',study.RUN/'trained/final-delta.pt',study.RUN/'verification.json',ROOT/'scripts/study_corrosion_state_mask.py',ROOT/'safelog_ai/corrosion_state_mask.py',ROOT/'safelog_ai/corrosion_json_pairs.py']
    p={'UTC':utc(),'fixed_before_new_FIT_resize_or_VAL_forward_diagnosis':True,'script_sha256':sha(ROOT/SCRIPT),'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},
       'FIT':'Recreate onlyfixed276FIT uint8letterbox inputs/targets; SHA must equal originalv69FIT inputbinding. Count native-vs512classpresence and actual7680draw exposures. Compute originalperphoto1/n_present coefficient sum byclass; this is coefficientbudget, not gradientimportance or causalproof. NoGT relabel.',
       'VAL':'Only60internalVAL, oneencoder forward perphoto shared by2fixedheadviews: selectedepoch6 andfinalepoch30. Compute native4stateconfusion and512Severe_GTmeanprob/logitmargin. NoCAL_TESTnewforward/scores, no epochpromotion or conditionchange.',
       'reference':'Reset selectedhead6 aftertwo views and require fullmodeltensor/buffer SHAexact beforeafter. Allparents unchanged. No new optimizer or weightupdate.',
       'future':'If VALandFITdiagnosis supportclassunderweighting, preregister onlyone lossweight candidate fromFIT expectedpresentclasscoefficients; no multiple weights/samplers/architecture/TESTthreshold sweep. Do not claim this proves a cause.',
       'storage':'Metadata only<8MiB andstreamnativeRAM; no newdensefeatures/fullmodels or imagecontentviews; >=1GiBreserve',
       'new_optimizer':0,'new_CAL_TEST_scores':0,'independent_field_validation':False}
    RUN.mkdir();new(POLICY,p);print({'policy_sha256':sha(POLICY)},flush=True)


def pair(archive,r):
    raw=archive.read(r['source_json_member']);assert hashlib.sha256(raw).hexdigest()==r['source_json_encoded_sha256']
    a=json.loads(raw);rgb,encoded,_=embedded_image(a);y,_=polygon_labels(a,rgb.size)
    assert hashlib.sha256(encoded).hexdigest()==r['embedded_encoded_sha256'] and hashlib.sha256(y.tobytes()).hexdigest()==r['class_raster_sha256']
    x,target,g=state_view(rgb,y);return x,target,g,y


def run():
    p=read(POLICY);assert sha(ROOT/SCRIPT)==p['script_sha256']
    for n,h in p['files_sha256'].items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fit=study.rows_for('fit');native=[];presence=[];pixels=[];xh=hashlib.sha256();yh=hashlib.sha256()
    with zipfile.ZipFile(ROOT/fit[0]['archive']) as z:
        for r in fit:
            x,y,_,_=pair(z,r);xh.update(x.transpose(2,0,1).tobytes());yh.update(y.tobytes())
            presence.append([bool(np.any(y==k)) for k in range(4)]);pixels.append([int(np.count_nonzero(y==k)) for k in range(4)])
            native.append([r['class_pixels'].get(str(k),0)>0 for k in range(4)])
    binding=read(study.RUN/'fit-input-binding.json');assert xh.hexdigest()==binding['uint8_RGB512_sha256'] and yh.hexdigest()==binding['uint8_target512_sha256']
    present=np.asarray(presence);native=np.asarray(native);draws=np.load(study.RUN/'draws.npy',allow_pickle=False).flatten()
    coefficient=present/present.sum(1,keepdims=True);drawn=coefficient[draws]
    fit_result={'native_positive_photo_counts':native.sum(0).tolist(),'view512_positive_photo_counts':present.sum(0).tolist(),
                'view512_class_pixels':np.asarray(pixels).sum(0).tolist(),'actual_draw_photo_exposures_by_class':present[draws].sum(0).tolist(),
                'actual_draw_presentclass_loss_coefficient_sums':drawn.sum(0).tolist(),
                'FIT_group_uniform_expected_coefficients':coefficient.mean(0).tolist(),
                'candidate_inverse_expected_coefficient_weights_not_trained':(1/(4*coefficient.mean(0))).tolist(),
                'original_v69_input_bytes_recreated_exact':True,'not_gradient_or_causal_importance':True}
    model,parent,original=study.load();heads={str(e):torch.load(study.RUN/f'head-epoch-{e:02d}.pt',map_location='cpu',weights_only=True)['head'] for e in (6,30)}
    model.corrosion_state_head.load_state_dict(heads['6']);model.eval();before=state_sha256(model.state_dict())
    val=study.rows_for('validation');matrix={n:np.zeros((4,4),dtype=np.int64) for n in heads};diagnostics={n:[] for n in heads}
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    from safelog_ai.fine_feature_probe import forward_with_fine
    with zipfile.ZipFile(ROOT/val[0]['archive']) as z,torch.inference_mode():
        for r in val:
            x,target,g,truth=pair(z,r);tensor=torch.from_numpy(x).permute(2,0,1)[None].cuda().float()/255
            f={n:v.half() for n,v in forward_with_fine(model.backbone,(tensor-mean)/std).items() if n in ('fine','low','high')}
            severe=torch.from_numpy(target==3).cuda();support=int(severe.sum())
            for n,head in heads.items():
                model.corrosion_state_head.load_state_dict(head);logits=model.forward_corrosion_state_features(**f)[0];prob=logits.softmax(0)
                matrix[n]+=confusion(native_probabilities(prob,g),torch.from_numpy(truth).cuda()).cpu().numpy()
                if support:diagnostics[n].append({'id':r['id'],'Severe_GT512_pixels':support,'Severe_GT_mean_probability':float(prob[3][severe].mean()),'Severe_GT_mean_logit_margin':float((logits[3]-logits[:3].max(0).values)[severe].mean())})
    model.corrosion_state_head.load_state_dict(heads['6']);assert state_sha256(model.state_dict())==before
    result={'UTC':utc(),'policy_sha256':sha(POLICY),'FIT':fit_result,'VAL_head_views':{n:{'metrics':rates(matrix[n]),'Severe_GT512_diagnostics':diagnostics[n]} for n in heads},
            'new_FIT_embedded_preprocesses':len(fit),'new_VAL_embedded_preprocesses_and_encoder_inputs':len(val),'head_map_forward_views':2*len(val),
            'new_CAL_TEST_scores':0,'new_optimizer_updates':0,'new_visual_content_views':0,'selected6_model_state_before_after_exact':True,
            'lasthead_promoted':False,'independent_field_validation':False}
    new(RUN/'diagnosis.json',result);new(ROOT/'reports/industrial-corrosion-state-balance-v70-diagnosis.json',result)
    print({'FIT':fit_result,'VAL_selected_recall':rates(matrix['6'])['per_class_recall'],'VAL_last_recall':rates(matrix['30'])['per_class_recall'],'new_optimizer':0,'new_CAL_TEST':0},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
        try:run()
        except BaseException as e:new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(e),'pid':os.getpid()});raise
        finally:(RUN/'operation.lock').unlink()
