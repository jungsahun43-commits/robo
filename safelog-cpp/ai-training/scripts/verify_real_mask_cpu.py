"""Run from the extracted ZIP; actual CPU API path versus fresh GPU forward79."""
import base64
import json
from pathlib import Path
import sys
import time
import torch
from PIL import Image
ROOT=Path(sys.argv[1]).resolve();STAGE=Path.cwd().resolve();sys.path.insert(0,str(STAGE))
from safelog_ai.real_mask_server import RealMaskEngine
from safelog_ai.loco_context_server import ContextEngine
from safelog_ai.magnetic_surface import RealMaskMain,surface_view,photo_score
from safelog_ai.native_component_views import normalized
from safelog_ai.equipment_expansion import photo_score as local_photo_score


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import safelog_ai.real_mask_server as runtime
    assert Path(runtime.__file__).resolve().is_relative_to(STAGE)
    info=read(STAGE/'model.json');engine=RealMaskEngine(STAGE/'model.pt',info['sha256'],STAGE/'evaluation.json')
    parent=ContextEngine(ROOT/'runs/industrial-loco-context-v13/selected/final.pt','b3e0f1cc8c487f45795b60d4bd67e2a8a03d7095d3844c6e2f42ba14875b65b0',ROOT/'runs/industrial-loco-context-v13/evaluation.json')
    state=torch.load(STAGE/'model.pt',map_location='cpu',weights_only=True)
    gpu=RealMaskMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs']);gpu.load_state_dict(state['state_dict'],strict=True);gpu.eval().requires_grad_(False).cuda()
    rows=[r for r in read(ROOT/'data/industrial-neighbor-surfaces-v15/manifest.json')['rows'] if r['category']=='mpdd_bracket_black' and r['phase']=='test']
    expected={r['id']:r for r in read(ROOT/'runs/industrial-magnetic-tiles-v16/candidate-evaluation.json')['rows'] if r['category']=='mpdd_bracket_black' and r.get('method')=='max_transfer_local'}
    point=state['transfer_surface_points']['mpdd_bracket_black'];maxdelta=0;freshdelta=0;records=[];started=time.monotonic()
    for row in rows:
        image=ROOT/row['image'];encoded=base64.b64encode(image.read_bytes()).decode();reply=engine.analyze('mpdd_bracket_black',encoded);original=parent.analyze('mpdd_bracket_black',encoded)
        with Image.open(image) as im:rgb=im.convert('RGB')
        native,_,valid,_=surface_view(rgb)
        with torch.inference_mode():
            probability=gpu.forward_surface(normalized(native)[None].cuda()).sigmoid();new=float(photo_score(probability,valid[None].cuda())[0])
            location=gpu.forward_equipment(normalized(rgb.resize((256,256),Image.Resampling.BICUBIC))[None].cuda(),'mpdd_bracket_black');old=float(local_photo_score(location)[0])
        score=max(new/point['transfer_divisor'],old/point['local_divisor']);cpu_score=reply['classResults'][0]['score'];maxdelta=max(maxdelta,abs(cpu_score-score));freshdelta=max(freshdelta,abs(score-expected[row['id']]['score']))
        assert reply['candidateDetected']==(score>point['photo_threshold'])==expected[row['id']]['candidate_detected'],row['id']
        assert reply['heatmapPngBase64']==original['heatmapPngBase64'],row['id']
        assert reply['originalPixelThreshold']==original.get('originalPixelThreshold',engine.points['mpdd_bracket_black']['pixel_threshold'])
        records.append({'id':row['id'],'CPU_detected':reply['candidateDetected'],'GPU_detected':score>point['photo_threshold'],'CPU_score':cpu_score,'GPU_score':score,'primary_heatmap_exact_to_parent':True})
    result={'status':'passed','actual_photos':len(rows),'CPU_actual_analyze_path':True,'fresh_GPU_full_preprocess_forward':True,'all_photo_decisions_equal_to_frozen_evaluation':True,'all79_original_location_PNG_exact_to_parent_CPU':True,'max_CPU_GPU_score_difference':maxdelta,'max_fresh_GPU_cached_score_difference':freshdelta,'total_benchmark_seconds_not_single_photo_latency':time.monotonic()-started,'independent_field_validated':False,'records':records}
    target=Path(sys.argv[2])
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print({k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':main()
