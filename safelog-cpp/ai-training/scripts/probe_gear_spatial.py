"""FIT-only preserved-encoder feasibility probe; no head training or new TEST score."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.fine_feature_probe import forward_with_fine
from safelog_ai.gearbox_supervised import gearbox_descriptor
from safelog_ai.gear_only_cache import select_descriptors
from safelog_ai.real_mask_server import RuntimeRealMaskMain
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_gear_only as prior
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-gear-spatial-probe-v29'
POLICY=ROOT/'reports/industrial-gear-spatial-v29-probe-policy.json'


def freeze():
    assert not RUN.exists() and not POLICY.exists()
    assert read(prior.RUN/'verification.json')['status']=='passed'
    rows=[r for r in read(prior.MANIFEST)['rows'] if r['phase']=='fit'];selected=[]
    for y in (0,1):
        groups={g:sorted([r for r in rows if r['label']==y and r['group_id']==g],key=lambda r:r['idx']) for g in sorted({r['group_id'] for r in rows if r['label']==y})}
        picks=[];offset=0
        while len(picks)<16:
            for g,values in groups.items():
                if offset<len(values) and len(picks)<16:picks.append(values[offset])
            offset+=1
        selected.extend(picks)
    RUN.mkdir()
    p={'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_new_probe_features':True,
       'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(prior.MANIFEST),'parent_sha256':prior.prior.EXPECTED,
       'FIT_ids':[r['id'] for r in selected],'FIT_labels':{str(y):16 for y in (0,1)},
       'sampling':'newv28FIT only, fixed sortedgroup roundrobin lowestidx,16perlabel; no VAL/CAL/TEST RGB read or head scores',
       'input':'same RGBbicubic256 ImageNet normalization; same frozen acceptedencoder forward_with_fine',
       'checks':'full model stateSHA before/after exact; freshfine24@64x64; freshlow/high bitwiseexact ordinaryforward; first72FP16pooledfine descriptor vs canonical oldsourcecache atol2e-3 rtol2e-3, no relax after scores',
       'planned_new_encoder_inputs':32,'new_optimizer_or_model_scores':0,
       'future_not_fixed':'one spatial convolutional MIL head on same fine24 features could learn from actual patchlabels; no newpixelGT/masks or independent localization. Full architecture/cache/training/VAL/CAL/guard must be fixed before actual new training.',
       'model_count':1,'shared_encoder_count':1}
    new(POLICY,p)


def run():
    p=read(POLICY);assert sha(Path(__file__))==p['script_sha256'] and sha(prior.MANIFEST)==p['manifest_sha256']
    assert not (RUN/'probe.json').exists() and sha(prior.prior.START)==p['parent_sha256']
    state=torch.load(prior.prior.START,map_location='cpu',weights_only=True)
    model=RuntimeRealMaskMain(state['reference_counts'],state['native_specs'],state['wear_variant'],state['surface_reference_specs']).eval().requires_grad_(False).cuda()
    model.load_state_dict(state['state_dict'],strict=True);before=state_sha256(model)
    byid={r['id']:r for r in read(prior.MANIFEST)['rows']};rows=[byid[i] for i in p['FIT_ids']];assert all(r['phase']=='fit' for r in rows)
    source_ids=[];source=[]
    for phase in ('fit','validation','cal'):
        path=prior.prior.RUN/(phase+'-features.pt');assert sha(path)==read(prior.PROTOCOL)['cached_features_SHA256'][phase]
        value=torch.load(path,map_location='cpu',weights_only=True);source_ids.extend(value['ids']);source.append(value['features'])
    expected=select_descriptors(p['FIT_ids'],source_ids,torch.cat(source)).float()[:,:72]
    batch=prior.prior.images(rows).cuda()
    with torch.inference_mode():
        features=forward_with_fine(model.backbone,batch);ordinary=model.backbone(batch)
        exact=all(torch.equal(features[k],ordinary[k]) for k in ('low','high'));assert exact
        actual=gearbox_descriptor(features).cpu()[:,:72]
        delta=float((actual-expected).abs().max());compatible=torch.allclose(actual,expected,atol=2e-3,rtol=2e-3)
        fine=features['fine'].half().cpu();assert fine.shape==(32,24,64,64)
    assert state_sha256(model)==before
    with (RUN/'FIT-fine.pt').open('xb') as out:torch.save({'ids':p['FIT_ids'],'fine':fine,'manifest_sha256':sha(prior.MANIFEST),'policy_sha256':sha(POLICY)},out)
    audit={'UTC':datetime.now(timezone.utc).isoformat(),'source_policy_sha256':sha(POLICY),'actual_encoder_inputs':32,'FIT_only':True,
       'fine_shape':list(fine.shape),'fine_dtype':str(fine.dtype),'fine_cache_sha256':sha(RUN/'FIT-fine.pt'),
       'fresh_low_high_bitwiseexact':exact,'all_parent_tensor_buffer_before_after_exact':True,
       'canonical_pooledfine_atol2e3_rtol2e3_passed':bool(compatible),'maximum_fine_descriptor_abs_delta':delta,
       'fine_spatial_std_mean':float(fine.float().flatten(2).std(2,correction=0).mean()),
       'new_VAL_CAL_TEST_model_scores':0,'actual_new_optimizer_updates':0,'full_training_protocol_fixed':False,
       'spatial_variance_is_not_accuracy_or_localization_truth':True,'weak_patch_label_is_not_pixel_mask':True,
       'previous_v28_kept':True,'independent_field_validation':False}
    new(RUN/'probe.json',audit);new(ROOT/'reports/industrial-gear-spatial-v29-FIT-probe.json',audit)
    print(audit,flush=True)
    assert compatible,'Preserve numerical mismatch; do not relax cache tolerance or start training'


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='run':
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Existing work '+str(path))
    globals()[args.stage]()
