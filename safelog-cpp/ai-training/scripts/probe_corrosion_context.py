"""FIT-only coordinate and numeric feasibility, no candidate training or accuracy gate."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
from io import BytesIO
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
from safelog_ai.corrosion_context_input import whole_geometry,native_context,context_tile512,context_tile128,fixed_probe_tiles
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_ostrava_rust_mask as base
from scripts import study_corrosion_native as native
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
RUN=ROOT/'runs/industrial-corrosion-context-probe-v84'
POLICY=ROOT/'reports/industrial-corrosion-context-v84-policy.json'
SOURCES=('scripts/probe_corrosion_context.py','safelog_ai/corrosion_context_input.py','tests/test_corrosion_context_input.py')


def selected():
    return {'rios':sorted(native.rows_for('fit'),key=lambda r:r['id'])[:3],'ostrava':sorted(base.rows_for('fit'),key=lambda r:r['id'])[:3]}


def freeze():
    assert not RUN.exists() and not POLICY.exists() and shutil.disk_usage(ROOT).free>1024**3+32*1024**2;RUN.mkdir()
    files=[native.MANIFEST,base.MANIFEST,native.PARENT,base.RUN/'fit-features.pt',base.RUN/'fit-cache.json',base.PROTOCOL,ROOT/'reports/industrial-corrosion-views-v82-diagnosis.json',ROOT/'reports/industrial-corrosion-teacher-retention-v83-results.json',ROOT/'scripts/study_corrosion_native.py',ROOT/'safelog_ai/corrosion_partial_mask.py']
    new(POLICY,{'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_FIT_context_features':True,'sources':{n:sha(ROOT/n) for n in SOURCES},'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'fixed_FIT_ids':{s:[r['id'] for r in rows] for s,rows in selected().items()},
        'scope':'6FIT originals only, Rios/Ostrava sortedIDs first3each. For eachimage size-only grid first/middle/last distincttiles. NoGT/capture/category chosen crop. v83excluded unchanged; no learning, labelmetric, modelselection, newVAL_CAL_TEST or visualview.',
        'feature':'Originalfrozenv62 whole512 probability. Roundaspectresize/centerfloorletterbox geometry; remove padding, bilinearprobability to native alignFalse, clampP[1e-6,1-1e-6], logit/8, clamp[-1,1], nativecrop512 withzero/neutralcontextpad, bilineardown128 alignFalse. Responsefeature notactualGT. OriginalRGB/GT unmodified.',
        'reuse':'CompletedOstravaFITwholecache verified ID/raw/mask/encoder/protocol/SHA; useonlyselectedFIT3cachedheadviews. RiosFIT3wholeRGBfreshforward from originalencoded/rgbSHA. No nativeencoderneededforthisgeometryprobe. Newencoder3/parenthead6views,18contexttilefields, notnewnaturalphotos.',
        'checks':'Everyvalidnativecrop roundtripexact, finite128feature and[-1,1]range, paddingzero, modelentirestatebeforeafterexact/cacheSHAunchanged. No GT/TEST scored; no newtrainablehead/gradient/optimizer. Metadataonlyoutput<32MiB/>=1GiBreserve.',
        'next':'If compatible, separatelyfreeze ONE context-conditioned model/code/initializer/sampling/loss/VAL/CAL/oldguards/device/delta beforegradient. Additionalcontext weightszero andoriginalinitialoutputpreservation require their own numericcheck. Thisprobe isnotperformance improvement.',
        'licenses':'RiosCC BY4.0 DOI10.17632/kcyn4nhv2c.1/OstravaCC BY4.0 DOI10.5281/zenodo.11235637/InsPLADCC BY-NC3.0/VTCC0/allparentterms retained','synthetic_units_passed':3,'independent_field_validation':False})
    print({'policy_SHA256':sha(POLICY),'new_optimizer':0},flush=True)


def run():
    p=read(POLICY)
    for n,h in {**p['sources'],**p['inputs']}.items():assert sha(ROOT/n)==h,n
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,_=base.load(delta=True);before=state_sha256(model.state_dict());assert before==native.PARENT_STATE;native.RUN=RUN
    cachepath=base.RUN/'fit-features.pt';cache=torch.load(cachepath,map_location='cpu',weights_only=True);allost=base.rows_for('fit')
    assert cache['ids']==[r['id'] for r in allost] and cache['raw_sha256']==[r['encoded_sha256'] for r in allost] and cache['mask_sha256']==[r['mask_encoded_sha256'] for r in allost]
    assert cache['protocol_sha256']==sha(base.PROTOCOL) and cache['manifest_sha256']==base.MANIFEST_SHA and cache['encoder_parent_sha256']==base.EXPECTED and cache['approved_three_head_delta_sha256']==base.PRIOR_SHA
    results=[]
    with ExitStack() as stack,torch.inference_mode():
        archives={n:stack.enter_context(zipfile.ZipFile(ROOT/n)) for n in {r['archive'] for rows in selected().values() for r in rows}}
        for source,rows in selected().items():
            for r in rows:
                assert r['phase']=='fit';g=whole_geometry(r['size'])
                if source=='ostrava':
                    index=cache['ids'].index(r['id']);f={n:v[index:index+1].cuda() for n,v in cache['features'].items()}
                else:
                    raw=archives[r['archive']].read(r['original']['member']);assert hashlib.sha256(raw).hexdigest()==r['original']['encoded_SHA256']
                    with Image.open(BytesIO(raw)) as image:
                        assert getattr(image,'n_frames',1)==1;rgb=image.convert('RGB')
                    assert list(rgb.size)==r['size'] and hashlib.sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256'] and g==r['geometry']
                    view=Image.new('RGB',(512,512),(124,116,104));view.paste(rgb.resize(tuple(g['resized_size']),Image.Resampling.BICUBIC),(g['left'],g['top']))
                    f=native.features(model,torch.from_numpy(np.array(view)).permute(2,0,1)[None],'FIT_whole_context',[r['id']])
                probability=model.forward_ostrava_features(**f)[0].sigmoid().cpu().numpy();field=native_context(probability,r['size']);tiles=[]
                for b in fixed_probe_tiles(tuple(r['size'])):
                    tile=context_tile512(field,b);x,y,w,h=(b[k] for k in ('left','top','width','height'))
                    assert np.array_equal(tile[:h,:w],field[y:y+h,x:x+w]) and np.all(tile[h:]==0) and np.all(tile[:,w:]==0)
                    small=context_tile128(field,b);assert small.shape==(1,128,128) and np.isfinite(small).all() and np.max(np.abs(small))<=1
                    tiles.append({'box':b,'context128_SHA256':hashlib.sha256(small.tobytes()).hexdigest(),'minimum':float(small.min()),'maximum':float(small.max()),'mean':float(small.mean()),'valid_native_crop_roundtrip_exact':True,'padding_zero':True})
                results.append({'source':source,'id':r['id'],'size':r['size'],'geometry':g,'whole_probability512_SHA256':hashlib.sha256(probability.tobytes()).hexdigest(),'native_context_SHA256':hashlib.sha256(field.tobytes()).hexdigest(),'tiles':tiles})
    assert state_sha256(model.state_dict())==before
    for n,h in {**p['sources'],**p['inputs']}.items():assert sha(ROOT/n)==h,n
    out={'UTC':datetime.now(timezone.utc).isoformat(),'FIT_photos':6,'new_RGB_redecodes':3,'new_encoder_inputs':3,'cached_FIT_parent_head_views':3,'fresh_parent_head_views':3,'context_tile_fields':sum(len(r['tiles']) for r in results),'rows':results,'parent_all_tensor_buffer_state_exact':True,'parent_state_SHA256':before,'coordinate_checks_passed':True,'new_head_gradient_optimizer_VAL_CAL_TEST_visual_accuracy_scores':0,'new_natural_photos':0,'new_feature_cache_files':0,'performance_improvement_proven':False}
    new(RUN/'probe.json',out);new(ROOT/'reports/industrial-corrosion-context-v84-probe.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text().strip())):raise RuntimeError('Live job '+str(path))
        with (RUN/'operation.lock').open('x') as f:f.write(str(os.getpid()))
        try:run()
        except BaseException as e:new(RUN/'failure.json',{'error':repr(e),'UTC':datetime.now(timezone.utc).isoformat()});raise
        finally:(RUN/'operation.lock').unlink()
