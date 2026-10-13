"""Finish normal calibration of the completed2640 steps, with no extra fit.

PyTorch quantile limits flattened inputs to2**24 elements. Use the same linear
percentile definition with NumPy for the large normalCAL pixel array. Frozen
training sources/protocol, saved weights, journal and epoch records stay intact.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.study_equipment_expansion import validate,read,RUN,MANIFEST,images,initialize,old_state
from scripts.study_equipment_expansion import ARCH,CATEGORIES,SEED,EPOCHS,START_SHA,OLD_DATA,Cache,TASKS,UnifiedMain,ExpandedMain,state_sha256,new,sha


def percentile995(maps):
    values=maps.detach().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():raise ValueError('Finite normalCAL residuals required')
    return float(np.quantile(values,.995,method='linear'))


def main():
    p=validate();run=RUN/'train';assert not (run/'final.pt').exists()
    progress=torch.load(run/'progress.pt',map_location='cpu',weights_only=True)
    assert progress['epoch']==31 and progress['cursor']==0 and progress['updates']==2640
    journal=[json.loads(x) for x in (run/'update-journal.jsonl').read_text().splitlines()]
    assert len(journal)==2640 and [x['update'] for x in journal]==list(range(1,2641))
    new(ROOT/'reports/equipment-expansion-v1-calibration-repair-v2.json',{'failure':'torch.quantile input tensor too large at normalCAL pixel threshold computation',
        'stage':'after30epochs and2640 completed optimizer updates, before final checkpoint export and TEST evaluation',
        'saved_progress_sha256':sha(run/'progress.pt'),'failed_optimizer_calls':0,'extra_optimizer_calls_for_repair':0,
        'same_percentile_definition':'0.995 linear interpolation on normalCAL residual pixels, NumPy supports >2**24 inputs',
        'previous_finalize_attempt':'Stopped at CPU/CUDA exact-equality check, before final export or TEST scoring; no new optimizer calls',
        'device_check_repair':'Compare saved old tensors on CPU with accepted checkpoint tensors',
        'frozen_training_source_unchanged':True,'TEST_scores_used_for_repair':False,'repair_source_sha256':sha(Path(__file__))})
    torch.manual_seed(SEED);model,initial=initialize();model.load_state_dict(progress['state_dict'],strict=True);model.eval()
    fixed=state_sha256(old_state(model));assert all(torch.equal(v.cpu(),initial['state_dict'][n]) for n,v in old_state(model).items())
    ready=p['trained_categories'];rows=read(MANIFEST)['rows'];points={}
    with torch.inference_mode():
        from safelog_ai.equipment_expansion import photo_score
        for cat in ready:
            cal=[r for r in rows if r['category']==cat and r['phase']=='cal'];maps=[]
            for i in range(0,len(cal),16):maps.append(model.forward_equipment(images(cal[i:i+16]),cat).cpu())
            maps=torch.cat(maps);scores=photo_score(maps).double().tolist();rank=min(len(scores),int(np.ceil((len(scores)+1)*.85)))
            points[cat]={'photo_threshold':sorted(scores)[rank-1],'pixel_threshold':percentile995(maps),'normal_CAL_photos':len(cal),'ids':[r['id'] for r in cal],'scores':scores}
    host=ExpandedMain().eval();host.load_state_dict({n:v.cpu() for n,v in model.state_dict().items()},strict=True)
    old=UnifiedMain().eval();old.load_state_dict(initial['state_dict'],strict=True);manifest=read(OLD_DATA/'manifest.json');proof={}
    with torch.inference_mode():
        for task in TASKS:
            x,_=Cache(manifest['fit'][task],{0,1}).batch([0,1]);gpu=model.forward_task(x,task);cpu=host.forward_task(x.cpu(),task);prior=old.forward_task(x.cpu(),task)
            if task=='facility_surface':gpu,cpu,prior=gpu[0],cpu[0],prior[0]
            assert torch.equal(cpu,prior) and torch.allclose(gpu.cpu(),cpu,atol=1e-4,rtol=.001);proof[task]={'prior_CPU_exact':True,'CPU_GPU_close':True}
        fit=[r for r in rows if r['category']==ready[0] and r['phase']=='fit'];x=images(fit[:2])
        assert torch.allclose(model.forward_equipment(x,ready[0]).cpu(),host.forward_equipment(x.cpu(),ready[0]),atol=1e-4,rtol=.001)
    import subprocess
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip();assert commit.startswith('d0a48ab')
    meta={'architecture':ARCH,'tasks':list(TASKS)+['equipment_surface'],'one_shared_backbone':True,'neural_model_count':1,
        'thresholds':initial['thresholds'],'equipment_operating_points':points,'equipment_categories':list(CATEGORIES),'trained_categories':ready,
        'initial_checkpoint_sha256':START_SHA,'protocol_sha256':sha(ROOT/'reports/equipment-expansion-v1-protocol.json'),'frozen_before_fit_commit':commit,
        'epochs':EPOCHS,'optimizer_updates':2640,'normal_FIT_draws':42240,'unique_normal_FIT_photos':p['unique_normal_fit'],
        'old_tensors_sha256':fixed,'old_CPU_GPU_proofs':proof,'equipment_CPU_GPU_close':True,
        'elapsed_optimizer_training_minutes':read(run/'epoch-30.json')['elapsed_minutes'],
        'normal_calibration_finalization_repaired_without_new_fit':True,'research_prototype_only':True,'fieldPerformanceValidated':False,
        'publisher_TEST_used_for_fit_or_calibration':False,'app_default_promoted':False,
        'trained_dataset_sources':['DACL10k','DamSegment','CODEBRIM','BeltCrack','Mendeley-s25x2bnshz','MVTec AD','MVTec LOCO AD','MPDD','KolektorSDD2']}
    with (run/'final.pt').open('xb') as f:torch.save({**meta,'state_dict':{n:v.cpu() for n,v in model.state_dict().items()}},f)
    new(ROOT/'reports/equipment-expansion-v1-training.json',{**meta,'weights_sha256':sha(run/'final.pt')})
    print(json.dumps({'completed_neural_updates':2640,'extra_fit_for_repair':0,'checkpoint_sha256':sha(run/'final.pt')}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;main()
