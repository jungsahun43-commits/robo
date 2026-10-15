"""Fresh original RGB CPU/GPU inference for the research-only native head."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_commutator_surface as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from safelog_ai.commutator_surface import commutator_view
from safelog_ai.ksdd_supervised import ksdd_photo_score
from safelog_ai.native_component_views import normalized
from safelog_ai.process_probe import process_exists


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    study.validate();assert study.read(study.RUN/'verification.json')['status']=='passed'
    output=ROOT/'reports/industrial-commutator-v18-CPU.json';assert not output.exists()
    for lock in (ROOT/'runs').glob('**/*.lock'):
        if process_exists(int(lock.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work process: '+str(lock))
    checkpoint=study.RUN/'trained/final.pt';cpu,_=study.load(checkpoint);gpu,_=study.load(checkpoint,'cuda')
    rows=study.rows_for('test');observations=[r for r in study.read(study.RUN/'candidate-evaluation.json')['rows'] if r['category']==study.CATEGORY and r['variant']=='candidate']
    assert [r['id'] for r in rows]==[r['id'] for r in observations]
    cut=study.read(study.RUN/'calibration.json')['categories'][study.CATEGORY]['candidate']['photo_threshold']
    values=[];max_prob=0.;start=time.monotonic()
    with torch.inference_mode():
        for i,r in enumerate(rows):
            assert sha(ROOT/r['image'])==r['image_sha256']
            with Image.open(ROOT/r['image']) as im:rgb=im.convert('RGB')
            view,_,valid,geometry=commutator_view(rgb);native=normalized(view)[None]
            cp=cpu.forward_commutator_mask(native).sigmoid();gp=gpu.forward_commutator_mask(native.cuda()).sigmoid().cpu()
            cs=float(ksdd_photo_score(cp,valid[None])[0]);gs=float(ksdd_photo_score(gp,valid[None])[0]);cached=observations[i]['score']
            assert (cs>cut)==(gs>cut)==observations[i]['candidate_detected'],r['id']
            assert np.isclose(cs,gs,rtol=1e-4,atol=1e-5) and np.isclose(gs,cached,rtol=1e-4,atol=1e-5),r['id']
            max_prob=max(max_prob,float((cp-gp).abs().max()))
            values.append({'id':r['id'],'label':r['label'],'CPU_score':cs,'fresh_GPU_score':gs,'cached_GPU_score':cached,'detected':cs>cut})
            if i%25==0:print({'fresh_RGB_CPU_GPU_checked':i+1,'total':len(rows)},flush=True)
    result={'status':'passed','UTC':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':sha(checkpoint),'actual_current_photos':len(rows),
        'all_CPU_fresh_GPU_cached_GPU_decisions_equal':True,'max_CPU_GPU_photo_difference':max(abs(r['CPU_score']-r['fresh_GPU_score']) for r in values),
        'max_fresh_GPU_cached_photo_difference':max(abs(r['fresh_GPU_score']-r['cached_GPU_score']) for r in values),
        'max_native_probability_CPU_GPU_difference':max_prob,'elapsed_both_devices_whole_benchmark_seconds':time.monotonic()-start,
        'fresh_original_RGB_native512x1408_preprocessing':True,'cached_features_not_used_for_either_device_forward':True,
        'source_research_head_only':True,'new_API_aliases_tested':0,'new_CPP_operations':0,'app_category_registered':False,
        'photo_threshold_unchanged':cut,'TEST_defect_views':18,'photo20percent_PASS_claimed':False,'independent_field_validated':False,
        'Wilson_photo_intervals_assume_independent_views_not_cluster_adjusted':True,'observations':values}
    new(output,result);new(study.RUN/'CPU-verification.json',result)
    print({k:v for k,v in result.items() if k!='observations'},flush=True)


if __name__=='__main__':main()
