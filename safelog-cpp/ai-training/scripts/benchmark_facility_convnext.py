"""Bounded local research inference timing on one original TRAIN photo."""
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_convnext import NAME,CONTROL,read,sha,require
from scripts.fetch_rc2119 import write_new
from safelog_ai.convnext_facility import ConvnextResearchPresence
from safelog_ai.presence_classifier import PresenceClassifier,image_transform
def main():
    torch.set_num_threads(4);require(torch.cuda.is_available(),'GPU required')
    proof=read(ROOT/'reports/facility-convnext-study-verification.json');require(proof['status']=='passed','Verified training required')
    photo=read(ROOT/'data/facility-spatial-training/train.json')['items'][0]['image']
    with Image.open(ROOT/photo)as image:inputs=image_transform(640)(image.convert('RGB'))[None].to('cuda')
    entries=[]
    for name,factory in((CONTROL,PresenceClassifier),(NAME,ConvnextResearchPresence)):
        path=ROOT/'runs'/name/'best.pt';adapter=factory(path,'cuda');model=adapter.model
        with torch.inference_mode():
            for _ in range(10):model(inputs)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[]
            for _ in range(40):
                start=time.perf_counter();output=model(inputs);torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
        require(output.shape==(1,7)and torch.isfinite(output).all(),'Public output differs')
        entries.append({'run':name,'weights_sha256':sha(path),'checkpoint_bytes':path.stat().st_size,
            'parameter_count':sum(p.numel()for p in model.parameters()),'timed_calls':len(timings),
            'median_milliseconds':float(np.median(timings)),'p90_milliseconds':float(np.quantile(timings,.9)),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'all_outputs_finite':True})
        del model,adapter,output;gc.collect();torch.cuda.empty_cache()
    report={'schema':'facility_convnext_local_timing_v1','torch':torch.__version__,'device':torch.cuda.get_device_name(0),
        'batch_size':1,'imgsz':640,'dtype':'float32','source_split':'train','warmup_calls':10,'timed_calls_per_model':40,
        'source_test_used':False,'includes_file_io_or_transfer':False,'end_to_end_app_latency_measured':False,
        'clock':'perf_counter with CUDA synchronize','source_photo_sha256':sha(ROOT/photo),'script_sha256':sha(Path(__file__)),
        'experiments':entries,'limitation':'One local GPU session and TRAIN photo; model-only latency, not mobile/server end-to-end or representative fleet measurements'}
    write_new(ROOT/'reports/facility-convnext-local-timing.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
