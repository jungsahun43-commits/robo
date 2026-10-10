"""Verify committed recipe, unchanged files, actual full-encoder updates."""
from pathlib import Path
import argparse
import json
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_convnext import NAME,PROTOCOL,INITIAL,PRETRAINED,CONTROL,EPOCHS,NEW_SOURCES,load_data,validate_protocol,read,sha,require
from scripts.verify_facility_rc_positive import git,snapshots,verify_git_sources
from scripts.verify_facility_primary_asymmetric import verify_sampling
from scripts.fetch_rc2119 import write_new
from safelog_ai.convnext_facility import ConvnextFacility,ARCH,load_imagenet,inventory
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier
from safelog_ai.frozen_batchnorm import batchnorm_state_sha256
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.presence_classifier import image_transform
BEFORE='runs/facility-convnext-before.json';LEDGER='runs/facility-convnext-input-ledger.json'
TESTS='runs/facility-convnext-tests.json';PREFLIGHT='runs/facility-convnext-preflight/preflight.json'
OUTPUT='reports/facility-convnext-study-verification.json'
PRIOR='reports/facility-primary-asymmetric-study-verification.json';OLD_LEDGER='runs/facility-primary-asymmetric-input-ledger.json'
def prerequisites(protocol):
    prior=read(ROOT/PRIOR);require(prior['status']=='passed'and len(prior['source_sha256'])==143,'Prior143 source inventory required')
    require(set(protocol['source_sha256'])==set(prior['source_sha256'])|set(NEW_SOURCES)
        and all(protocol['source_sha256'][p]==h for p,h in prior['source_sha256'].items()),'Previous source bytes changed')
    tests=read(ROOT/TESTS);require(tests['tests_run']>=9 and all(tests[k]==0 for k in('failures','errors','skipped'))
        and tests['source_sha256']==protocol['source_sha256']and tests['protocol_sha256']==sha(ROOT/PROTOCOL),'Actual frozen tests required')
    preflight=read(ROOT/PREFLIGHT);require(preflight['status']=='passed'and preflight['actual_disposable_fp32_updates']==1
        and preflight['actual_training_epochs']==0 and preflight['protocol_sha256']==sha(ROOT/PROTOCOL)
        and all(preflight[k]is True for k in('strict_cpu_new_graph_reload','teacher_and_bn_preserved','strict_fp32_diagnostic_flags_restored')),'Actual preflight failed')
    return prior,tests,preflight
def before(protocol):
    require(not(ROOT/BEFORE).exists()and not(ROOT/LEDGER).exists()and not(ROOT/'runs'/NAME).exists(),'Preserve before-proof')
    prior,tests,preflight=prerequisites(protocol);require(sha(ROOT/OLD_LEDGER)==prior['input_ledger_sha256'],'Prior ledger differs')
    old=read(ROOT/OLD_LEDGER)['snapshots_before'];paths=set(old)|set(protocol['source_sha256'])|set(protocol['input_sha256'])
    paths.update((PROTOCOL,TESTS,PREFLIGHT,'runs/facility-convnext-preflight/disposable.pt',PRIOR,OLD_LEDGER,
        'reports/facility-primary-asymmetric-study-comparison.json','reports/FACILITY_PRIMARY_ASYMMETRIC_STUDY_RESULTS_KO.md'))
    paths.update(p.relative_to(ROOT).as_posix()for p in(ROOT/'runs/facility-presence-target-primary-asymmetric').rglob('*')if p.is_file())
    current=snapshots(paths);require(all(current[p]==v for p,v in old.items()),'Previous input SHA/size/mtime changed')
    commit=git(['rev-parse','HEAD']).decode().strip();verify_git_sources({**protocol['source_sha256'],PROTOCOL:sha(ROOT/PROTOCOL)},commit)
    write_new(ROOT/LEDGER,{'snapshots_before':current,'prior_protected_files':len(old),'previous_ledger_sha256':sha(ROOT/OLD_LEDGER)})
    write_new(ROOT/BEFORE,{'status':'passed','protocol_sha256':sha(ROOT/PROTOCOL),'source_git_commit':commit,
        'source_sha256':protocol['source_sha256'],'input_ledger_sha256':sha(ROOT/LEDGER),'protected_files':len(current),
        'test_record_sha256':sha(ROOT/TESTS),'preflight_sha256':sha(ROOT/PREFLIGHT),'git_blob_bytes_verified':True})
    print(json.dumps({'status':'passed_before_training','protected_files':len(current)}),flush=True)
def after(protocol):
    require(not(ROOT/OUTPUT).exists(),'Preserve completed verification')
    bound,ledger=read(ROOT/BEFORE),read(ROOT/LEDGER)
    require(bound['protocol_sha256']==sha(ROOT/PROTOCOL)and bound['input_ledger_sha256']==sha(ROOT/LEDGER)
        and bound['test_record_sha256']==sha(ROOT/TESTS)and bound['preflight_sha256']==sha(ROOT/PREFLIGHT),'Before-proof bindings differ')
    current=snapshots(ledger['snapshots_before']);require(current==ledger['snapshots_before'],'Protected SHA/size/mtime changed')
    verify_git_sources({**protocol['source_sha256'],PROTOCOL:sha(ROOT/PROTOCOL)},bound['source_git_commit'])
    prior,tests,preflight=prerequisites(protocol);data=load_data();run=ROOT/'runs'/NAME
    training,history=read(run/'TRAINING.json'),read(run/'history.json')
    require(training['status']=='complete'and training['actual_epochs']==EPOCHS and training['study_protocol_sha256']==sha(ROOT/PROTOCOL)
        and training['source_sha256']==protocol['source_sha256']and training['source_git_commit']==bound['source_git_commit']
        and not any(training[k]for k in('human_feedback_used','original_labels_changed','historical_control_retrained','source_test_used'))
        and training['ai_pseudo_labels_created']==0 and training['primary_photo_loss_recipe']==protocol['primary_photo_loss_recipe'],'Actual training provenance differs')
    measured=verify_sampling(history,data,read(ROOT/'runs'/CONTROL/'history.json'))
    require(training['attempted_batches']==measured['attempted_batches']and training['actual_optimizer_steps']==measured['actual_optimizer_steps']
        and all(r['encoder_updated']for r in history),'Actual full encoder update missing')
    selected=min(history,key=lambda r:(r['worst_target_error'],r['sum_target_errors']))
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    require(training['weights_sha256']==sha(run/'best.pt')and checkpoint['epoch']==selected['epoch']
        and checkpoint['study_protocol_sha256']==sha(ROOT/PROTOCOL)and checkpoint['architecture']==ARCH
        and read(run/'VALIDATION.json')==selected,'Selected checkpoint differs')
    model=ConvnextFacility();model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    original=ConvnextFacility();transfer=load_imagenet(original,ROOT/PRETRAINED)
    require(inventory(model)==training['model_inventory']==preflight['model_inventory']
        and inventory(model)['student_batchnorm_modules']==0 and training['encoder_initial_sha256']==transfer['encoder_state_sha256']
        and training['encoder_final_sha256']!=training['encoder_initial_sha256']
        and state_sha256(model.encoder_state())!=training['encoder_initial_sha256'],'Student inventory/encoder update differs')
    teacher=AuxiliaryClassifier(7,pretrained=False);teacher.load_state_dict(torch.load(ROOT/INITIAL,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    require(training['teacher_state_sha256']==training['final_teacher_state_sha256']==state_sha256(teacher.state_dict())
        and training['initial_bn_buffer_sha256']==training['final_bn_buffer_sha256']==batchnorm_state_sha256(teacher),'Teacher state/BN changed')
    for key,expected in(('original_photo_positive_weights',data['sampling_data']['photo_weights']),('original_pixel_positive_weights',data['supervision_weights']['pixel_weights']),('original_auxiliary_positive_weights',data['supervision_weights']['auxiliary_weights'])):
        require(training[key]==expected.tolist(),'Original loss weights differ')
    require(all(torch.isfinite(v).all()for v in checkpoint['state_dict'].values()),'Nonfinite selected state')
    with Image.open(ROOT/data['items'][0]['image'])as image:inputs=image_transform(640)(image.convert('RGB'))[None]
    with torch.no_grad():outputs=model.forward_training(inputs);public=model(inputs)
    require([list(x.shape)for x in outputs]==[[1,7],[1,7,80,80],[1,19]]and all(torch.isfinite(x).all()for x in outputs)
        and torch.equal(public,outputs[0]),'CPU/public contract differs')
    require(not(run/'TARGET-TEST.json').exists()and not list(run.glob('target-test-*.json')),'SourceTEST forbidden')
    proof={'schema':'facility_convnext_verification_v1','status':'passed','protocol_sha256':sha(ROOT/PROTOCOL),
        'source_git_commit':bound['source_git_commit'],'source_sha256':protocol['source_sha256'],'runtime_source_count':len(protocol['source_sha256']),
        'git_blob_bytes_verified':True,'input_ledger_sha256':sha(ROOT/LEDGER),'protected_file_count':len(current),
        'prior_protected_files_preserved':ledger['prior_protected_files'],'all_input_sha_size_mtime_preserved':True,
        'tests':tests,'preflight':preflight,'actual_sampling':measured,'actual_completed_training_epochs':EPOCHS,
        'weights_sha256':sha(run/'best.pt'),'selected_epoch':selected['epoch'],'strict_cpu_new_graph_reload':True,'all_outputs_finite':True,
        'model_inventory':inventory(model),'encoder_actually_updated':True,'frozen_teacher_and_bn_preserved':True,
        'human_feedback_used':False,'new_ai_labels':0,'app_model_promoted':False,'source_test_used':False,
        'app_profile_unchanged':True,'app_profile_sha256':sha(ROOT/'reports/facility-inference-profile.json'),
        'independent_industrial_accuracy_verified':False}
    write_new(ROOT/OUTPUT,proof);print(json.dumps({k:proof[k]for k in('status','actual_completed_training_epochs','weights_sha256','protected_file_count')}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);protocol=validate_protocol(read(ROOT/PROTOCOL));(before if args.before else after)(protocol)
