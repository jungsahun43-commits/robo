"""Fixed native ROI pair: change detail pixels, preserve every existing target."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safelog_ai.auxiliary_classifier import ARCH, AUX_CLASSES
from scripts.facility_resolution_study import (CLASSES, INITIAL_SHA, build_sampling,
    prepare_data as original_data, validate_items)
from scripts.train_facility_target import DOMAINS, read, sha
from scripts.verify_facility_resolution import RUNTIME_SOURCES as PREVIOUS_SOURCES
from scripts.report_facility_detail import GATE

PROTOCOL = 'reports/facility-native-roi-study-protocol.json'
AUDIT = 'reports/facility-native-roi-data-audit.json'
MANIFESTS = {v:f'data/facility-native-roi-training/{v}/train.json' for v in ('control','native')}
REFERENCE = 'facility-presence-target-roi-control'
NAMES = {'control':'facility-presence-target-native-roi-control',
         'native':'facility-presence-target-native-roi-native'}
MIN_TEST_COUNTS = {'tests/test_facility_native_roi_data.py':10,
                   'tests/test_facility_native_roi_study.py':8,
                   'tests/test_facility_native_roi_report.py':12}
RECIPE = {'version':'native_roi_pixels_v1', 'input_size':640, 'output_format':'PNG',
          'control':'same native RGB -> existing processed-parent dimensions -> existing crop box -> 640 square',
          'native':'same native RGB -> equivalent floating native crop box -> 640 square',
          'resampling':'Pillow LANCZOS', 'reducing_gap':None, 'encoding_metadata':'pixels_only',
          'historical_thumbnail_jpeg_pixel_replay':False,
          'eligibility':'geometry only; native width/height >= processed, at least one greater',
          'new_labels':0, 'new_masks':0, 'full_rows_changed':0,
          'scope':'Detail-only pre-crop downsampling intervention; no native full-photo inference'}
SOURCE_FILES = tuple(dict.fromkeys(PREVIOUS_SOURCES + (
    'scripts/audit_facility_resolution_sources.py', 'scripts/prepare_facility_native_roi.py',
    'scripts/facility_native_roi_study.py', 'scripts/train_facility_native_roi.py',
    'scripts/preflight_facility_native_roi.py', 'scripts/verify_facility_native_roi.py',
    'scripts/report_facility_native_roi.py', 'scripts/run_facility_native_roi_pair.py',
    'scripts/prepare_facility_data.py', 'scripts/test_facility_native_roi.py',
    'scripts/verify_facility_resolution.py', 'scripts/report_facility_resolution.py',
)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def fixed_values():
    return {'schema':'facility_native_roi_study_protocol_v1', 'declared_before_training':True,
            'reference':REFERENCE, 'control':NAMES['control'], 'treatment':NAMES['native'],
            'architecture_by_variant':{'control':ARCH,'native':ARCH},
            'imgsz_by_variant':{'control':640,'native':640},
            'classes':CLASSES,'auxiliary_classes':list(AUX_CLASSES),
            'seed':56,'requested_epochs':6,'patience':6,'batch_size':8,
            'draws_per_epoch':14248,'backbone_lr':.00004,'head_lr':.00025,
            'auxiliary_weight':.5,'target_ranking_weight':0.,'domain_proportions':[.7,.1,.2],
            'loader_randomness':{'sampler_seed':56,'training_worker_seed':57,'post_model_seed':58,
                                'validation_worker_seeds':{d:156+k for d,k in DOMAINS.items()}},
            'initial_weights_sha256':INITIAL_SHA,'native_roi_recipe':RECIPE,
            'paired_manifest_paths':MANIFESTS,'research_candidate_gate':GATE,
            'source_test_inference_executed':False,'app_model_promoted':False}


def validate_protocol(protocol, root=ROOT, args=None):
    root=Path(root).resolve()
    for key,value in fixed_values().items():
        require(protocol.get(key)==value,f'Declared native ROI condition differs: {key}')
    require(set(protocol.get('source_sha256',{}))==set(SOURCE_FILES),'Runtime source inventory differs')
    for name,expected in protocol['source_sha256'].items():
        require(sha(root/name)==expected,f'Frozen runtime source changed: {name}')
    protected={'core_spatial_manifest_sha256':'data/facility-spatial-training/train.json',
               'auxiliary_manifest_sha256':'data/facility-auxiliary-training/train.json',
               'paired_data_audit_sha256':AUDIT,'app_profile_sha256':'reports/facility-inference-profile.json',
               'source_records_sha256':'data/dacl10k-yolo/records.json',
               'source_preparation_sha256':'data/dacl10k-yolo/PREPARATION.json'}
    for field,path in protected.items():
        require(sha(root/path)==protocol.get(field),f'Protected original/paired data changed: {field}')
    require(sha(root/f'runs/{REFERENCE}/best.pt')==INITIAL_SHA,'Frozen initializer changed')
    require(set(protocol.get('paired_manifest_sha256',{}))==set(MANIFESTS),'Paired manifests missing')
    for variant,path in MANIFESTS.items():
        require(sha(root/path)==protocol['paired_manifest_sha256'][variant],'Derived manifest bytes changed')
    if args is not None:
        require(args.variant in NAMES and args.name==NAMES[args.variant],'Run name differs from fixed pair')
        fields={'seed':'seed','epochs':'requested_epochs','patience':'patience','batch':'batch_size',
                'draws_per_epoch':'draws_per_epoch','backbone_lr':'backbone_lr','head_lr':'head_lr',
                'auxiliary_weight':'auxiliary_weight'}
        for argument,field in fields.items():
            require(getattr(args,argument,None)==protocol[field],f'CLI condition differs: {argument}')
        require(Path(args.initial).resolve()==(root/f'runs/{REFERENCE}/best.pt').resolve(),
                'Use the original fixed initializer path')
        require(Path(args.auxiliary_manifest).resolve()==(root/'data/facility-auxiliary-training/train.json').resolve(),
                'Use the original auxiliary labels')
        return {'imgsz':640,'architecture':ARCH}
    return deepcopy(protocol)


def prepare_data(root, protocol, variant, auxiliary_path=None):
    from scripts.prepare_facility_native_roi import validate_pair
    require(variant in MANIFESTS,'Unknown paired dataset')
    root=Path(root).resolve()
    data=original_data(root,protocol,auxiliary_path)
    pair={v:read(root/path) for v,path in MANIFESTS.items()}
    validate_pair(data['manifest'],pair['control'],pair['native'])
    original=data['items']; full=data['full_count']
    changed=[i for i,(a,b) in enumerate(zip(original,pair[variant]['items'])) if a['image']!=b['image']]
    require(changed and all(i>=full and original[i]['domain']=='dacl' for i in changed),
            'Only existing DACL detail pixels can change')
    audit=read(root/AUDIT)
    require(audit.get('status')=='prepared','Actual paired data preparation must complete')
    for v in MANIFESTS:
        require(sha(root/MANIFESTS[v])==protocol['paired_manifest_sha256'][v],'Paired manifest changed')
    items=validate_items(pair[variant],root=root)
    # The two manifests retain the original audit's pixel counts. Loss weights
    # are deliberately calculated from the original source supervision above.
    sampling=build_sampling(items,full,CLASSES,DOMAINS,tuple(protocol['domain_proportions']))
    import torch
    require(all(torch.equal(sampling[k],data['sampling_data'][k]) for k in sampling),
            'Pixel intervention cannot change sampling or photo weights')
    data.update(items=items,manifest=pair[variant],manifest_path=root/MANIFESTS[variant],
                sampling_data=sampling,pair_proof={'full_rows_equal':True,'photo_targets_equal':True,
                    'pixel_target_paths_equal':True,'sample_weights_equal':True,
                    'row_count':len(items),'changed_dacl_detail_rows':len(changed),
                    'auxiliary_crop_targets_remain_unknown':True})
    return data


def protected_hashes(root=ROOT):
    paths=['data/facility-spatial-training/train.json','data/facility-auxiliary-training/train.json',
           'data/dacl10k-yolo/records.json','data/dacl10k-yolo/PREPARATION.json',
           'reports/facility-inference-profile.json',f'runs/{REFERENCE}/best.pt',
           'reports/facility-resolution-study-protocol.json',PROTOCOL,AUDIT]+list(MANIFESTS.values())
    return {path:sha(Path(root)/path) for path in paths}


def declare():
    path=ROOT/PROTOCOL
    require(not path.exists(),'Preserve the predeclared protocol')
    require(not any((ROOT/'runs'/name).exists() for name in NAMES.values()),'Cannot declare after training starts')
    p=fixed_values()
    p.update(declared_utc=datetime.now(timezone.utc).isoformat(),
             core_spatial_manifest_sha256=sha(ROOT/'data/facility-spatial-training/train.json'),
             auxiliary_manifest_sha256=sha(ROOT/'data/facility-auxiliary-training/train.json'),
             paired_data_audit_sha256=sha(ROOT/AUDIT),
             app_profile_sha256=sha(ROOT/'reports/facility-inference-profile.json'),
             source_records_sha256=sha(ROOT/'data/dacl10k-yolo/records.json'),
             source_preparation_sha256=sha(ROOT/'data/dacl10k-yolo/PREPARATION.json'),
             paired_manifest_sha256={v:sha(ROOT/file) for v,file in MANIFESTS.items()},
             source_sha256={file:sha(ROOT/file) for file in SOURCE_FILES})
    validate_protocol(p,ROOT)
    for v in MANIFESTS:prepare_data(ROOT,p,v)
    write(path,p)
    print(json.dumps({'status':'declared','runtime_sources':len(SOURCE_FILES),'epochs':12,
                      'protocol_sha256':sha(path),'new_labels':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare',action='store_true',required=True)
    parser.parse_args();declare()
