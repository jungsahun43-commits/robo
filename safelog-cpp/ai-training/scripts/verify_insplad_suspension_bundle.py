"""Import only an extracted overlay's code for fresh original-RGB device checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safelog_ai.insplad_suspension_server import SuspensionEngine
from safelog_ai.native_component_views import normalized


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    inputs=json.loads(Path(args.inputs).read_text(encoding='utf-8'))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    engine=SuspensionEngine(**inputs['engine']);cpu=[];gpu=[]
    for device,values in (('cpu',cpu),('cuda',gpu)):
        engine.model.to(device)
        with torch.inference_mode():
            for i in range(0,len(inputs['rows']),16):
                batch=[]
                for r in inputs['rows'][i:i+16]:
                    raw=Path(r['image']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['image_sha256']
                    with Image.open(r['image']) as im:
                        assert min(im.size)>=64 and im.getexif().get(274) in (None,1)
                        batch.append(normalized(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC)))
                values.extend(engine.model.forward_insplad_suspension(torch.stack(batch).to(device)).sigmoid().tolist())
    cut=engine.suspension_cut;cached=inputs['cached_scores']
    assert len(cpu)==len(gpu)==len(cached)==251
    assert np.allclose(cpu,gpu,atol=5e-4,rtol=1e-3)
    assert np.allclose(cpu,cached,atol=5e-4,rtol=1e-3)
    assert [v>cut for v in cpu]==[v>cut for v in gpu]==[v>cut for v in cached]
    result={'status':'passed','original_RGB_photos':len(cpu),'CPU_GPU_cached_decisions_exact':True,
            'max_CPU_GPU_score_difference':float(np.max(np.abs(np.array(cpu)-gpu))),
            'max_CPU_cached_score_difference':float(np.max(np.abs(np.array(cpu)-cached))),
            'fixed_atol':5e-4,'fixed_rtol':1e-3,'new_optimizer':0,
            'effective_combined_state_sha256':engine.digest,'CPU_scores':cpu,'GPU_scores':gpu,
            'new_field_test':False,'Android_speed_measured':False}
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if not k.endswith('_scores')},flush=True)


if __name__=='__main__':main()
