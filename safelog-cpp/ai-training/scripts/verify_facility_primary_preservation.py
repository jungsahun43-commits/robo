"""Verify real primary regularization and preserve all prior study artifacts."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_primary_preservation import NAME,PROTOCOL,DRAWS,MANIFEST,INITIAL,CONTROL,IDEA,TEACHER,WEIGHT,TEMPERATURE,EPOCHS,DRAW_COUNT,NEW_SOURCES,PRIOR,AUDIT,load_data,validate_protocol,read,sha,require
from scripts.verify_facility_rc_positive import git,snapshots,verify_git_sources
from scripts.verify_facility_idea_research import verify_sampling
from scripts.fetch_rc2119 import write_new
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier,ARCH
from safelog_ai.frozen_batchnorm import batchnorm_state_sha256
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.presence_classifier import image_transform
BEFORE='runs/facility-primary-preservation-before.json'
LEDGER='runs/facility-primary-preservation-input-ledger.json'
TESTS='runs/facility-primary-preservation-tests.json'
PREFLIGHT='runs/facility-primary-preservation-preflight/preflight.json'
OUTPUT='reports/facility-primary-preservation-study-verification.json'
OLD_LEDGER='runs/facility-idea-input-ledger.json'

def prerequisites(protocol):
    prior=read(ROOT/PRIOR);require(prior['status']=='passed'and len(prior['source_sha256'])==163,'Prior163 frozen sources required')
    require(set(protocol['source_sha256'])==set(prior['source_sha256'])|set(NEW_SOURCES)and all(protocol['source_sha256'][p]==h for p,h in prior['source_sha256'].items()),'Previous runtime source bytes changed')
    tests=read(ROOT/TESTS);require(tests['tests_run']>=15 and all(tests[k]==0 for k in('failures','errors','skipped'))and tests['source_sha256']==protocol['source_sha256']and tests['protocol_sha256']==sha(ROOT/PROTOCOL),'Actual frozen tests required')
    flight=read(ROOT/PREFLIGHT)
    require(flight['status']=='passed'and flight['actual_disposable_fp32_updates']==1 and flight['actual_training_epochs']==0 and flight['protocol_sha256']==sha(ROOT/PROTOCOL)
        and all(flight[k]is True for k in('strict_cpu_324_reload','teacher_and_bn_preserved','strict_fp32_diagnostic_flags_restored','primary_teacher_preserved'))
        and all(flight[k]==0 for k in('new_pixel_known_cells','new_aux_known_cells','new_other5_known_targets'))
        and flight['primary_preservation_diagnostics']=={'original_rows':4,'teacher_batches':1,'known_primary_entries':8}and flight['primary_preservation_loss']>0,'Actual primary TRAIN-only preflight failed')
    return prior,tests,flight

def before(protocol):
    require(not(ROOT/BEFORE).exists()and not(ROOT/LEDGER).exists()and not(ROOT/'runs'/NAME).exists(),'Preserve prior proof/run')
    prior,tests,flight=prerequisites(protocol);require(sha(ROOT/OLD_LEDGER)==prior['input_ledger_sha256'],'Prior input ledger differs')
    old=read(ROOT/OLD_LEDGER)['snapshots_before'];paths=set(old)|set(protocol['source_sha256'])|set(protocol['input_sha256'])
    paths.update((PROTOCOL,TESTS,PREFLIGHT,'runs/facility-primary-preservation-preflight/disposable.pt',PRIOR,OLD_LEDGER))
    paths.update(p.relative_to(ROOT).as_posix()for p in(ROOT/'runs'/IDEA).rglob('*')if p.is_file())
    paths.update(p.relative_to(ROOT).as_posix()for p in(ROOT/'runs/facility-idea-transition-audit').rglob('*')if p.is_file())
    paths.update(('reports/facility-idea-study-comparison.json','reports/FACILITY_IDEA_STUDY_RESULTS_KO.md','reports/facility-idea-study-comparison.png','reports/facility-idea-study-comparison.plot.json'))
    current=snapshots(paths);require(all(current[p]==v for p,v in old.items()),'Prior protected SHA/size/mtime changed')
    commit=git(['rev-parse','HEAD']).decode().strip();verify_git_sources({**protocol['source_sha256'],PROTOCOL:sha(ROOT/PROTOCOL)},commit)
    write_new(ROOT/LEDGER,{'snapshots_before':current,'prior_protected_files':len(old),'previous_ledger_sha256':sha(ROOT/OLD_LEDGER)})
    write_new(ROOT/BEFORE,{'status':'passed','protocol_sha256':sha(ROOT/PROTOCOL),'source_git_commit':commit,'source_sha256':protocol['source_sha256'],
        'input_ledger_sha256':sha(ROOT/LEDGER),'protected_files':len(current),'test_record_sha256':sha(ROOT/TESTS),'preflight_sha256':sha(ROOT/PREFLIGHT),'git_blob_bytes_verified':True})
    print(json.dumps({'status':'passed_before_training','protected_files':len(current)}),flush=True)

def after(protocol):
    require(not(ROOT/OUTPUT).exists(),'Preserve actual verification')
    before_record,ledger=read(ROOT/BEFORE),read(ROOT/LEDGER)
    require(before_record['protocol_sha256']==sha(ROOT/PROTOCOL)and before_record['input_ledger_sha256']==sha(ROOT/LEDGER)and before_record['test_record_sha256']==sha(ROOT/TESTS)and before_record['preflight_sha256']==sha(ROOT/PREFLIGHT),'Before proof bindings differ')
    current=snapshots(ledger['snapshots_before']);require(current==ledger['snapshots_before'],'Protected input SHA/size/mtime changed')
    verify_git_sources({**protocol['source_sha256'],PROTOCOL:sha(ROOT/PROTOCOL)},before_record['source_git_commit'])
    prior,tests,flight=prerequisites(protocol);data=load_data();run=ROOT/'runs'/NAME;training,history=read(run/'TRAINING.json'),read(run/'history.json')
    require(training['status']=='complete'and training['actual_epochs']==EPOCHS and training['study_protocol_sha256']==sha(ROOT/PROTOCOL)and training['source_sha256']==protocol['source_sha256']
        and training['source_git_commit']==before_record['source_git_commit']and training['original_labels_changed']is False and training['human_feedback_used']is False and training['ai_pseudo_labels_created']==0
        and training['source_test_used']is False and training['historical_control_retrained']is False and training['matched_data_control_retrained']is False
        and training['teacher_probabilities_are_new_truth']is False and training['val_cases_used_in_training_or_sampling']is False and training['local_noncommercial_research_only']is True
        and training['supplement_manifest_sha256']==sha(ROOT/MANIFEST)and training['primary_photo_loss_recipe']==protocol['primary_photo_loss_recipe']
        and training['primary_preservation_weight']==WEIGHT and training['primary_preservation_temperature']==TEMPERATURE and training['primary_teacher_weights_sha256']==sha(ROOT/TEACHER),'Actual training conditions differ')
    measured=verify_sampling(history,data);require(training['attempted_batches']==measured['attempted_batches']and training['actual_optimizer_steps']==measured['actual_optimizer_steps'],'Update summary differs')
    matched=read(ROOT/f'runs/{IDEA}/history.json');core=len(data['items']);labels=np.asarray([r['targets']for r in data['items']],np.int64);scope=[]
    for e,(row,comparison)in enumerate(zip(history,matched)):
        for key in('sampled_row_indices_sha256','sampled_photo_target_counts','sampled_domain_counts','supplement_exposure','optimizer_learning_rates'):
            require(row[key]==comparison[key],'Matched IDEA sampling/labels/LR differs: '+key)
        draws=data['candidate_draws'][e];original=draws<core
        expected={'original_rows':int(original.sum()),'teacher_batches':int(original.reshape(-1,8).any(axis=1).sum()),
            'known_primary_entries':int((labels[draws[original],:2]>=0).sum()),'supplement_rows':0}
        require(row['primary_teacher_scope']==expected and row['primary_teacher_unchanged']is True and row['loss_components_mean_per_batch']['primary_preservation']>=0,'Observed teacher source scope differs')
        scope.append(expected)
    require(training['actual_primary_teacher_rows']==sum(r['original_rows']for r in scope)and training['actual_primary_teacher_forward_batches']==sum(r['teacher_batches']for r in scope),'Primary teacher forward accounting differs')
    selected=min(history,key=lambda r:(r['worst_target_error'],r['sum_target_errors']));checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    require(training['weights_sha256']==sha(run/'best.pt')and checkpoint['epoch']==selected['epoch']and checkpoint['study_protocol_sha256']==sha(ROOT/PROTOCOL)and checkpoint['architecture']==ARCH
        and len(checkpoint['state_dict'])==324 and read(run/'VALIDATION.json')==selected,'Actual selected model differs')
    model=AuxiliaryClassifier(7,pretrained=False);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    original=AuxiliaryClassifier(7,pretrained=False);original.load_state_dict(torch.load(ROOT/INITIAL,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    control=AuxiliaryClassifier(7,pretrained=False);control.load_state_dict(torch.load(ROOT/TEACHER,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    require(training['initial_bn_buffer_sha256']==training['final_bn_buffer_sha256']==batchnorm_state_sha256(original)==batchnorm_state_sha256(model)
        and training['teacher_state_sha256']==training['final_teacher_state_sha256']==state_sha256(original.state_dict())
        and training['primary_teacher_state_sha256']==training['final_primary_teacher_state_sha256']==state_sha256(control.state_dict())==flight['primary_teacher_state_sha256'],'Frozen teacher/BN state differs')
    for key,expected in(('original_photo_positive_weights',data['sampling_data']['photo_weights']),('original_pixel_positive_weights',data['supervision_weights']['pixel_weights']),('original_auxiliary_positive_weights',data['supervision_weights']['auxiliary_weights'])):
        require(training[key]==expected.tolist(),'Original loss weights differ')
    require(all(torch.isfinite(v).all()for v in checkpoint['state_dict'].values()),'Nonfinite selected state')
    with Image.open(ROOT/data['items'][0]['image'])as image:inputs=image_transform(640)(image.convert('RGB'))[None]
    with torch.no_grad():outputs=model.forward_training(inputs);public=model(inputs)
    require([list(x.shape)for x in outputs]==[[1,7],[1,7,80,80],[1,19]]and all(torch.isfinite(x).all()for x in outputs)and torch.equal(public,outputs[0]),'Strict CPU/public contract differs')
    require(not(run/'TARGET-TEST.json').exists()and not list(run.glob('target-test-*.json')),'SourceTEST forbidden')
    proof={'schema':'facility_primary_preservation_verification_v1','status':'passed','protocol_sha256':sha(ROOT/PROTOCOL),'source_git_commit':before_record['source_git_commit'],
        'source_sha256':protocol['source_sha256'],'runtime_source_count':len(protocol['source_sha256']),'git_blob_bytes_verified':True,'input_ledger_sha256':sha(ROOT/LEDGER),
        'protected_file_count':len(current),'prior_protected_files_preserved':ledger['prior_protected_files'],'all_input_sha_size_mtime_preserved':True,'tests':tests,'preflight':flight,
        'actual_sampling':measured,'primary_teacher_scope':scope,'both_teacher_states_preserved':True,'actual_completed_training_epochs':EPOCHS,'weights_sha256':sha(run/'best.pt'),
        'selected_epoch':selected['epoch'],'strict_cpu_324_reload':True,'all_outputs_finite':True,'frozen_teacher_and_bn_preserved':True,'human_feedback_used':False,'new_ai_labels':0,
        'app_model_promoted':False,'source_test_used':False,'val_cases_used_in_training_or_sampling':False,'local_noncommercial_research_only':True,'checkpoint_redistributed':False,
        'app_profile_unchanged':True,'app_profile_sha256':sha(ROOT/'reports/facility-inference-profile.json'),'independent_industrial_accuracy_verified':False}
    write_new(ROOT/OUTPUT,proof);print(json.dumps({k:proof[k]for k in('status','actual_completed_training_epochs','weights_sha256','protected_file_count')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');args=parser.parse_args();torch.set_num_threads(4)
    protocol=validate_protocol(read(ROOT/PROTOCOL));(before if args.before else after)(protocol)
