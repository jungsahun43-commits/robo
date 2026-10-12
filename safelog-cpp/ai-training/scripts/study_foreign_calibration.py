"""Calibrate a fixed trained network, then score new frames once; no neural updates."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_foreign_calibration_check import DATA, PRIOR, read, new, sha
from safelog_ai.conveyor_foreign import create_model, normalize, summarize
from safelog_ai.foreign_calibration import fit_threshold
from safelog_ai.retention_distillation import state_sha256

RUN = ROOT/'runs/conveyor-foreign-calibrated-v3'
PROTOCOL = ROOT/'reports/foreign-calibration-protocol.json'
LEDGER = ROOT/'runs/foreign-calibration-ledger.json'
RAW = ROOT/'runs/conveyor-foreign-expansion-v2/final.pt'
BASELINE = ROOT/'runs/conveyor-foreign-grouped-v1/final.pt'
CAL = ROOT/'runs/conveyor-foreign-expansion-v2/val-predictions.json'
SOURCES = ('scripts/prepare_foreign_calibration_check.py','scripts/study_foreign_calibration.py',
           'safelog_ai/foreign_calibration.py','tests/test_foreign_calibration.py')


def sources():
    return {name:sha(ROOT/name) for name in SOURCES}


def snap(path):
    stat = path.stat()
    return {'sha256':sha(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def freeze():
    previous = read(ROOT/'reports/conveyor-foreign-expansion-verification.json')
    assert previous['status']=='passed' and previous['weights_sha256']==sha(RAW)
    prior_ledger = ROOT/'runs/conveyor-foreign-expansion-input-ledger.json'
    inherited = read(prior_ledger)
    old = {**inherited['previous_artifacts'],**inherited['expanded_inputs']}
    for name,record in old.items():
        stat = (ROOT/name).stat()
        assert stat.st_size==record['size_bytes'] and stat.st_mtime_ns==record['mtime_ns']
    legacy = {name:sha(path) for folder in ('scripts','safelog_ai','tests')
              for path in (ROOT/folder).glob('*.py') if (name:=path.relative_to(ROOT).as_posix()) not in SOURCES}
    prior = read(ROOT/'reports/conveyor-foreign-expansion-protocol.json')
    for name,digest in {**prior['legacy_source_sha256'],**prior['source_sha256']}.items():
        assert legacy[name]==digest
    protected = {}
    for folder in ('runs/conveyor-foreign-grouped-v1','runs/conveyor-foreign-expansion-v2'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = snap(path)
    for pattern in ('conveyor-foreign-expansion*','conveyor-expansion*','CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md'):
        for path in (ROOT/'reports').glob(pattern):
            protected[path.relative_to(ROOT).as_posix()] = snap(path)
    inputs = {p.relative_to(ROOT).as_posix():snap(p) for p in DATA.rglob('*') if p.is_file()}
    new(LEDGER,{'fresh_inputs':inputs,'protected_recent_artifacts':protected,
        'inherited_ledger_sha256':sha(prior_ledger),'inherited_full_verification_sha256':sha(ROOT/'reports/conveyor-foreign-expansion-verification.json'),
        'old_size_mtime_entries_checked':len(old)})
    new(PROTOCOL,{'schema':'foreign-decision-calibration-v3','declared_before_threshold_fit':True,
        'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources(),'legacy_source_sha256':legacy,
        'ledger_sha256':sha(LEDGER),'fresh_manifest_sha256':sha(DATA/'manifest.json'),
        'raw_weights_sha256':sha(RAW),'baseline_weights_sha256':sha(BASELINE),'calibration_scores_sha256':sha(CAL),
        'calibration_dataset':'Original320 VAL photos, now explicitly CAL, known results; TRAIN weights unchanged',
        'algorithm':'Minimize max(FPR,FNR) then sum on CAL; unique score logit midpoints; ties closest0.5',
        'fresh_selection':read(DATA/'selection.json')['rule'],'fresh_counts':read(DATA/'manifest.json')['counts'],
        'same_known_test_recordings':True,'independent_new_recording_factory_or_object':False,
        'fresh_photo_scores_previously_seen':False,'test_scores_or_labels_used_to_fit_threshold':False,
        'calibration_method_chosen_after_known_raw_validation_failure':True,
        'known_prior_test_results_exist':True,'raw_checkpoint_final30_fixed':True,
        'new_neural_epochs':0,'new_optimizer_updates':0,'trained_source_epochs':30,'trained_source_updates':4290,
        'tests_expected':5,'normal_definition':'No foreign object in iron-ore flow, no mechanical-health verdict',
        'mechanical_health_ground_truth':False,'new_ai_or_human_labels':0,'app_default_promoted':False})
    print({'frozen_before_fit':True,'counts':read(DATA/'manifest.json')['counts'],'legacy_sources':len(legacy)},flush=True)


def validate():
    protocol = read(PROTOCOL)
    assert protocol['source_sha256']==sources() and protocol['ledger_sha256']==sha(LEDGER)
    assert protocol['raw_weights_sha256']==sha(RAW) and protocol['calibration_scores_sha256']==sha(CAL)
    assert protocol['fresh_manifest_sha256']==sha(DATA/'manifest.json')
    return protocol


def fit():
    validate()
    repo = ROOT.parents[1]
    git = ['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+repo.as_posix(),'-C',str(repo)]
    for name in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        raw = subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+name])
        assert hashlib.sha256(raw).hexdigest()==sha(ROOT/name)
    assert not RUN.exists();RUN.mkdir()
    rows = [{**r,'split':'calibration'} for r in read(CAL)]
    fitted = fit_threshold(rows)
    state = torch.load(RAW,map_location='cpu',weights_only=True)
    source_model = create_model(False).eval();source_model.load_state_dict(state['state_dict'],strict=True)
    before = state_sha256(source_model)
    assert before==state['final_state_sha256']
    metadata = {**state,'threshold':fitted['threshold'],'decision_schema':'safelog.foreign-calibrated.v3',
        'threshold_source':'OriginalVAL320 explicitly CAL; supervised threshold fit',
        'calibration_protocol_sha256':sha(PROTOCOL),'source_weights_sha256':sha(RAW),
        'calibration_score_sha256':sha(CAL),'threshold_fitted_using_test':False,
        'new_neural_epochs':0,'new_optimizer_updates':0}
    torch.save(metadata,RUN/'final.pt')
    reloaded = create_model(False).eval()
    reloaded.load_state_dict(torch.load(RUN/'final.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    assert state_sha256(reloaded)==before
    new(ROOT/'reports/foreign-calibration-fit.json',{**fitted,'weights_sha256':sha(RUN/'final.pt'),
        'source_weights_sha256':sha(RAW),'neural_state_sha256':before,'neural_weights_unchanged':True,
        'new_neural_epochs':0,'new_optimizer_updates':0,'strict_cpu_reload_passed':True,
        'calibration_resubstitution':summarize(rows,fitted['threshold']),
        'fit_before_fresh_scores':True,'protocol_sha256':sha(PROTOCOL),
        'frozen_before_fit_commit':subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()})
    print({'threshold':fitted['threshold'],'neural_updates':0,'source_actual_epochs':30},flush=True)


def infer(checkpoint,rows,images):
    model = create_model(False).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    observations = []
    with torch.inference_mode():
        for start in range(0,len(rows),32):
            scores = model(normalize(images[start:start+32],'cuda')).flatten().sigmoid().cpu().double().tolist()
            observations.extend({'id':r['id'],'group':r['recording_group'],'label':r['foreign_object_photo_label'],'score':s}
                                for r,s in zip(rows[start:start+32],scores))
    return observations


def evaluate():
    validate()
    fitted = read(ROOT/'reports/foreign-calibration-fit.json')
    assert fitted['weights_sha256']==sha(RUN/'final.pt')
    assert not (ROOT/'reports/foreign-calibration-results.json').exists()
    rows = read(DATA/'manifest.json')['rows']
    tensors = []
    for row in rows:
        assert sha(DATA/row['image'])==row['image_sha256']
        with Image.open(DATA/row['image']) as image:
            tensors.append(torch.from_numpy(np.array(image)).permute(2,0,1))
    images = torch.stack(tensors)
    expanded = infer(RAW,rows,images); baseline = infer(BASELINE,rows,images)
    new(RUN/'fresh-predictions.json',{'expanded':expanded,'baseline':baseline})
    threshold = fitted['threshold']
    result = {'status':'evaluated','threshold':threshold,
        'calibration_resubstitution_is_performance_test':False,
        'fresh':{'baseline_v1_fixed0_5':summarize(baseline),'expanded_raw_fixed0_5':summarize(expanded),
                 'expanded_calibrated':summarize(expanded,threshold)},
        'fresh_per_group':{g:summarize([r for r in expanded if r['group']==g],threshold) for g in sorted({r['group'] for r in expanded})},
        'known_audit_val':summarize(read(ROOT/'runs/conveyor-foreign-expansion-v2/audit_val-predictions.json'),threshold),
        'known_test':summarize(read(ROOT/'runs/conveyor-foreign-expansion-v2/test-predictions.json'),threshold),
        'fresh_photos_count':len(rows),'fresh_photos_same_known_recordings':True,
        'independent_new_factory_recording_or_physical_objects':False,'threshold_fitted_using_fresh':False,
        'known_prior_test_results_exist':True,'new_neural_epochs':0,'new_optimizer_updates':0,
        'weights_sha256':sha(RUN/'final.pt'),'protocol_sha256':sha(PROTOCOL),'app_default_promoted':False}
    new(ROOT/'reports/foreign-calibration-results.json',result)
    print(json.dumps(result['fresh']),flush=True)


def verify():
    protocol = validate()
    for name,digest in protocol['legacy_source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    ledger = read(LEDGER)
    for key in ('fresh_inputs','protected_recent_artifacts'):
        for name,record in ledger[key].items():
            assert snap(ROOT/name)==record,name
    fitted,result = read(ROOT/'reports/foreign-calibration-fit.json'),read(ROOT/'reports/foreign-calibration-results.json')
    state = torch.load(RUN/'final.pt',map_location='cpu',weights_only=True)
    model = create_model(False).eval();model.load_state_dict(state['state_dict'],strict=True)
    assert state_sha256(model)==fitted['neural_state_sha256'] and fitted['weights_sha256']==sha(RUN/'final.pt')
    observations = read(RUN/'fresh-predictions.json')
    assert summarize(observations['expanded'],fitted['threshold'])==result['fresh']['expanded_calibrated']
    new(ROOT/'reports/foreign-calibration-verification.json',{'status':'passed','tests_passed':5,
        'new_neural_epochs':0,'new_optimizer_updates':0,'source_actual_neural_epochs':30,'source_updates':4290,
        'neural_state_unchanged':True,'threshold_fit_before_fresh_scores':True,
        'legacy_sources_sha_preserved':len(protocol['legacy_source_sha256']),
        'fresh_inputs_sha_size_mtime_preserved':len(ledger['fresh_inputs']),
        'protected_recent_artifacts_sha_size_mtime_preserved':len(ledger['protected_recent_artifacts']),
        'old_full_verification_inherited_from':'conveyor-foreign-expansion-verification.json',
        'old_size_mtime_entries_checked':ledger['old_size_mtime_entries_checked'],
        'weights_sha256':sha(RUN/'final.pt'),'protocol_sha256':sha(PROTOCOL),'app_default_promoted':False})
    print('Calibration provenance, unchanged neural state and one fresh scoring pass verified',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','fit','evaluate','verify'))
    args = parser.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'freeze':freeze,'fit':fit,'evaluate':evaluate,'verify':verify}[args.mode]()
