"""VAL-only coverage diagnosis; an unknown-region prediction is not a false positive."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_corrosion_partial as study
from safelog_ai.corrosion_partial_mask import native_probability
from safelog_ai.research_delta import restore_delta
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

PRIOR_RUN=study.RUN
RUN=ROOT/'runs/industrial-corrosion-partial-coverage-v74'
POLICY=ROOT/'reports/industrial-corrosion-partial-v74-diagnostic-policy.json'
SCRIPT='scripts/diagnose_corrosion_partial.py'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    assert not RUN.exists() and not POLICY.exists();assert read(PRIOR_RUN/'verification.json')['status']=='passed'
    assert not read(PRIOR_RUN/'selection.json')['source_partial_research_gate_passed'];RUN.mkdir()
    files=[PRIOR_RUN/'trained/final-delta.pt',PRIOR_RUN/'validation-input-binding.json',PRIOR_RUN/'calibration.json',study.MANIFEST,study.PROTOCOL,ROOT/'scripts/study_corrosion_partial.py',ROOT/'safelog_ai/corrosion_partial_mask.py',ROOT/'safelog_ai/corrosion_partial_data.py']
    new(POLICY,{'UTC':utc(),'fixed_before_new_VAL_diagnostic_forward':True,'source_sha256':{SCRIPT:sha(ROOT/SCRIPT)},'inputs_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},
        'scope':'OnlyexistingVAL75(15Ostrava+60VT); frozenv62 andselectedv73head8 once onsharedfeatures. No FITnewfeatures, newCAL_TESTfeatures/scores, optimizer/gradients, relabel/visualcontentview/modeladoption.',
        'prediction':'Use originalfixedparentcut.20 andv73CALcut.30 only. Aggregate nativepositive/knownnegative/unknown predictedforeground coverage. VTunknown coverage isNOTFPR orlabelerror. No cutoffvariation/selection.',
        'input':'RecreateVALRGB512+derivedGT bindingexactv73. Onlineoneencoder inputperVALphoto, twoexistingheadviews. Redirecthelper journals tothisnewRUN, neverappendoldcompletedlogs. Sequentialnativebuffers/no newfeaturecache.',
        'preservation':'Fullselectedv73tensor-buffer stateSHA before/afterexact. Oldparent/heads/meta stayexact; no lastheadpromotion. This describespredictions; itcannotprove sourcecausalshortcut orbackgroundtruth.',
        'next':'If unknowncoverage high, do not convertunknown to negatives or tune same loss/TESTcut. Seek independentlyannotatedknownbackground in matchingphotodomain, withsource/license/groupchecks; humanlabelsneedreview.',
        'new_optimizer':0,'new_CAL_TEST':0,'new_source_visual_views':0,'independent_field_validation':False})
    print({'policy_sha256':sha(POLICY)},flush=True)


def run():
    p=read(POLICY)
    for n,h in {**p['source_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (SCRIPT,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    study.RUN=RUN # New output namespace only; the saved v73 module is not modified.
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,original=study.load();chosen=restore_delta(original,torch.load(PRIOR_RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True),study.base.EXPECTED)
    model.load_state_dict(chosen['state_dict'],strict=True);before=state_sha256(model.state_dict());assert before==read(PRIOR_RUN/'verification.json')['combined_state_sha256']
    cut=read(PRIOR_RUN/'calibration.json')['cut'];assert cut==.30
    rows=study.rows_for('validation');xhash=hashlib.sha256();yhash=hashlib.sha256();totals=defaultdict(lambda:[0,0]);photos=[]
    with ExitStack() as stack,torch.inference_mode():
        archives=study.archives_for(rows,stack)
        for r in rows:
            view,target,g,truth,_=study.pair(archives,r,'VAL_diagnosis');x=torch.from_numpy(view).permute(2,0,1).contiguous()[None]
            xhash.update(x[0].numpy().tobytes());yhash.update(target.tobytes());f=study.features(model,x,'VAL_diagnosis',[r['id']])
            predictions={'parent':native_probability(model.forward_ostrava_features(**f)[0].sigmoid(),g)>.20,'selected_v73':native_probability(model.forward_corrosion_partial_features(**f)[0].sigmoid(),g)>cut}
            t=torch.from_numpy(truth).cuda();item={'id':r['id'],'source':r['source'],'regions':{}}
            for region,label in (('positive',1),('known_negative',0),('unknown',255)):
                valid=t==label;number=int(valid.sum());item['regions'][region]={'pixels':number}
                for name,pred in predictions.items():
                    hit=int((pred&valid).sum());key=r['source']+'/'+region+'/'+name;totals[key][0]+=hit;totals[key][1]+=number
                    item['regions'][region][name+'_predicted_foreground_fraction']=hit/number if number else None
            photos.append(item)
    binding=read(PRIOR_RUN/'validation-input-binding.json');assert binding['ids']==[r['id'] for r in rows] and xhash.hexdigest()==binding['uint8_RGB512_sha256'] and yhash.hexdigest()==binding['uint8_target512_sha256']
    assert state_sha256(model.state_dict())==before
    result={'UTC':utc(),'VAL_inputs':len(rows),'encoder_inputs':len(rows),'head_views':2*len(rows),'coverage':{k:{'predicted_foreground_pixels':v[0],'region_pixels':v[1],'fraction':v[0]/v[1] if v[1] else None} for k,v in totals.items()},'photos':photos,'VAL_RGB_target_binding_exact_v73':True,'full_selected_v73_state_before_after_exact':True,'new_CAL_TEST_features_scores':0,'new_optimizer':0,'new_visual_content_views':0,'unknown_coverage_is_not_FPR':True,'causal_shortcut_proven':False,'no_model_promotion':True}
    new(RUN/'diagnosis.json',result);new(ROOT/'reports/industrial-corrosion-partial-v74-diagnosis.json',result)
    print({k:v for k,v in result.items() if k!='photos'},flush=True)


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
