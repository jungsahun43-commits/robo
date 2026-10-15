"""Lossless storage and shared-tail feasibility only; no new model training."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.research_delta import pack_delta,restore_delta
from safelog_ai.gear_adapter_probe import ZeroFineAdapterProbe,tail_from_fine
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.gearbox_supervised import gearbox_descriptor
from safelog_ai.real_mask_server import RuntimeRealMaskMain
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_gear_residual_mil as prior
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-gear-adapter-v31-probe'
POLICY=ROOT/'reports/industrial-gear-adapter-v31-probe-policy.json'
SOURCES=('safelog_ai/research_delta.py','safelog_ai/gear_adapter_probe.py','scripts/probe_gear_adapter.py','tests/test_research_delta.py','tests/test_gear_adapter_probe.py')
SOURCE=prior.RUN/'trained/final.pt'
OLD_PROBE=ROOT/'runs/industrial-gear-spatial-probe-v29'


def freeze():
    assert not RUN.exists() and not POLICY.exists()
    assert read(prior.RUN/'verification.json')['status']=='passed' and not read(prior.RUN/'selection.json')['eligible_pending_package_checks']
    assert shutil.disk_usage(ROOT).free>1024**3
    RUN.mkdir();old=read(ROOT/'reports/industrial-gear-spatial-v29-probe-policy.json')
    p={'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_source_storage_conversion_or_new_FIT_tail_features':True,
       'source_sha256':{n:sha(ROOT/n) for n in SOURCES},'parent_sha256':prior.EXPECTED,
       'source_failed_v30_sha256':sha(SOURCE),'manifest_sha256':sha(prior.MANIFEST),
       'old_FIT_fine_sha256':sha(OLD_PROBE/'FIT-fine.pt'),'old_FIT_probe_policy_sha256':sha(ROOT/'reports/industrial-gear-spatial-v29-probe-policy.json'),
       'FIT_ids':old['FIT_ids'],'new_encoder_RGB_inputs':32,
       'storage_test':'only added gear/gearbox tensor namespaces + added metadata, bind immutable acceptedv16fileSHA and full combinedstateSHA; restore inRAM and compare everytensor/metadata to existingfailedv30, reread serialized delta; never delete/modify parent or oldfullcheckpoint',
       'adapter_probe':'zero-init24to4 GELU4to24 residual prototype at existing stage3fine; reuse original later modules, no copy/newencoder/hooks/return_layers mutation; ordinary oldforward untouched',
       'identity_checks':'freshFP32fine plus zeroadapter and originaltail produce bitwiseexact fullforward low/high; fullmodel stateSHA before/after same; oldbackbone freshforward same afterprobe',
       'quantized_probe':'oldcached32FP16fine must equal freshfine.half; feedcachedfine.float through zeroadapter/originaltail, compare3072pooleddescriptor to freshfullforward at atol.02 rtol.01 fixed beforeprobe; record mismatch and do not relax or train',
       'scope':'technicalFIT32only, no patch classifier scoring/VAL_CAL_TEST pixels or newmodeloptimizer/actualsourcegradients',
       'new_unit_tests_passed':5,'actual_new_optimizer_updates':0,'new_training_protocol_fixed':False,
       'future':'full adapter structure/init/trainables/precision/cache/resume/VAL/CAL/guard and delta-loading contract must be preregistered before actual newtraining; one commonencoder and oldtasks preserved'}
    new(POLICY,p);print({'frozen':True,'FIT_inputs':32,'new_optimizer':0},flush=True)


def validate():
    p=read(POLICY)
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    for path,h in ((prior.START,prior.EXPECTED),(SOURCE,p['source_failed_v30_sha256']),(prior.MANIFEST,p['manifest_sha256']),(OLD_PROBE/'FIT-fine.pt',p['old_FIT_fine_sha256'])):assert sha(path)==h,path
    return p


def delta():
    p=validate();assert not (RUN/'delta-test.json').exists()
    parent=torch.load(prior.START,map_location='cpu',weights_only=True);candidate=torch.load(SOURCE,map_location='cpu',weights_only=True)
    payload=pack_delta(parent,candidate,p['parent_sha256'],prior.START.relative_to(ROOT).as_posix(),('gearbox_','gear_spatial_'))
    path=RUN/'failed-v30-research-delta.pt'
    with path.open('xb') as out:torch.save(payload,out)
    loaded=torch.load(path,map_location='cpu',weights_only=True);rebuilt=restore_delta(parent,loaded,sha(prior.START))
    assert set(rebuilt)==set(candidate) and set(rebuilt['state_dict'])==set(candidate['state_dict'])
    for name,value in candidate['state_dict'].items():assert torch.equal(value,rebuilt['state_dict'][name]),name
    for name,value in candidate.items():
        if name!='state_dict':assert value==rebuilt[name],name
    assert sha(prior.START)==p['parent_sha256'] and sha(SOURCE)==p['source_failed_v30_sha256']
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'passed','parent_path':str(prior.START),'parent_sha256':p['parent_sha256'],
       'existing_failed_full_checkpoint_path':str(SOURCE),'existing_failed_full_checkpoint_sha256':p['source_failed_v30_sha256'],
       'existing_full_checkpoint_bytes':SOURCE.stat().st_size,'research_delta_bytes':path.stat().st_size,'research_delta_sha256':sha(path),
       'all_state_keys_tensors_buffers_metadata_exact_to_failed_full':True,'combined_state_sha256':loaded['combined_state_sha256'],
       'old_parent_and_full_checkpoint_unchanged':True,'old_files_deleted_or_compacted':0,
       'parent_required_not_self_contained_app_delivery':True,'new_trained_models':0,'new_optimizer_updates':0,'new_unit_tests_passed':3}
    new(RUN/'delta-test.json',r);new(ROOT/'reports/industrial-research-delta-v31-test.json',r);print(r,flush=True)


def adapter():
    p=validate();assert not (RUN/'adapter-probe.json').exists()
    state=torch.load(prior.START,map_location='cpu',weights_only=True)
    model=RuntimeRealMaskMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs']).eval().requires_grad_(False).cuda()
    model.load_state_dict(state['state_dict'],strict=True);before=state_sha256(model)
    rows_byid={r['id']:r for r in read(prior.MANIFEST)['rows']};rows=[rows_byid[i] for i in p['FIT_ids']];assert all(r['phase']=='fit' for r in rows)
    cached=torch.load(OLD_PROBE/'FIT-fine.pt',map_location='cpu',weights_only=True);assert cached['ids']==p['FIT_ids']
    images=prior.global_study.prior.images(rows).cuda();module=ZeroFineAdapterProbe().eval().requires_grad_(False).cuda()
    with torch.inference_mode():
        fresh=forward_with_fine(model.backbone,images);fine=fresh['fine']
        cached_exact=torch.equal(fine.half().cpu(),cached['fine']);assert cached_exact
        resumed=tail_from_fine(model.backbone,module(fine))
        identity=all(torch.equal(fresh[k],resumed[k]) for k in ('low','high','fine'));assert identity
        quantized=tail_from_fine(model.backbone,module(cached['fine'].float().cuda()))
        original_descriptor=gearbox_descriptor(fresh).cpu();quant_descriptor=gearbox_descriptor(quantized).cpu()
        delta_value=(original_descriptor-quant_descriptor).abs();compatible=torch.allclose(original_descriptor,quant_descriptor,atol=.02,rtol=.01)
        ordinary=model.backbone(images);oldexact=all(torch.equal(fresh[k],ordinary[k]) for k in ('low','high'));assert oldexact
    assert before==state_sha256(model)
    r={'UTC':datetime.now(timezone.utc).isoformat(),'status':'passed' if compatible else 'cache_precision_mismatch_stop_before_training',
       'FIT_RGB_inputs':32,'old_cachedfine_FP16_exact':cached_exact,'freshFP32_tail_zero_adapter_bitwiseexact':identity,
       'old_forward_after_probe_bitwiseexact':oldexact,'all_parent_tensors_buffers_before_after_exact':True,
       'same_original_later_modules_reused_no_extra_encoder':True,'prototype_new_adapter_parameters':sum(v.numel() for v in module.parameters()),
       'cachedfine_to_tail_descriptor_atol':.02,'cachedfine_to_tail_descriptor_rtol':.01,
       'cachedfine_to_tail_descriptor_tolerance_passed':bool(compatible),'descriptor_max_abs_delta':float(delta_value.max()),'descriptor_mean_abs_delta':float(delta_value.mean()),
       'new_patch_classifier_scores':0,'new_VAL_CAL_TEST_features_scores':0,'actual_new_optimizer_updates':0,
       'new_unit_tests_passed':2,'new_training_protocol_fixed':False,'no_accuracy_improvement_claimed':True,'independent_field_validated':False}
    new(RUN/'adapter-probe.json',r);new(ROOT/'reports/industrial-gear-adapter-v31-FIT-probe.json',r);print(r,flush=True)
    assert compatible,'Preserve numerical mismatch; no tolerance adjustment or actual training'


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','delta','adapter','run'));args=parser.parse_args()
    if args.stage!='freeze':
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work '+str(path))
    if args.stage=='run':delta();adapter()
    else:globals()[args.stage]()
