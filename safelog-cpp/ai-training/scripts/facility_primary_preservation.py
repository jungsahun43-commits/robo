"""Fixed original-prediction regularizer with exact completed IDEA data/order."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_idea_research import load_data as original_data,dataset,INITIAL,INITIAL_SHA,CONTROL,CONTROL_SHA,REFERENCE,DRAWS,MANIFEST,EPOCHS,DRAW_COUNT,SEED,REPLACEMENTS,NEW_DOMAINS,read,sha,require
from scripts.facility_head_lr_study import RETENTION_GATE
from scripts.report_facility_detail import GATE
from scripts.fetch_rc2119 import write_new
from safelog_ai.auxiliary_classifier import ARCH
NAME='facility-presence-target-primary-preservation'
IDEA='facility-presence-target-idea-building'
PROTOCOL='reports/facility-primary-preservation-study-protocol.json'
PRIOR='reports/facility-idea-study-verification.json'
TEACHER='runs/'+CONTROL+'/best.pt'
WEIGHT,TEMPERATURE=1.,2.
AUDIT='reports/facility-idea-transition-audit.json'
NEW_SOURCES=('safelog_ai/primary_preservation.py','scripts/audit_facility_idea_transitions.py','scripts/facility_primary_preservation.py',
    'scripts/train_facility_primary_preservation.py','scripts/verify_facility_primary_preservation.py','scripts/report_facility_primary_preservation.py','tests/test_primary_preservation.py')

def load_data():
    data=original_data();arrays=np.load(ROOT/DRAWS,allow_pickle=False)
    require(np.array_equal(arrays['candidate'],data['candidate_draws'])and np.array_equal(arrays['control'],data['epoch_draws']),'Exact previous IDEA draw arrays required')
    return data

def fixed():
    return {'schema':'facility_primary_preservation_study_v1','declared_before_training':True,'run':NAME,'control':CONTROL,'reference':REFERENCE,
        'matched_data_control':IDEA,'architecture':ARCH,'epochs':EPOCHS,'batch_size':8,'imgsz':640,'draws_per_epoch':DRAW_COUNT,
        'replacement_draws_per_epoch':REPLACEMENTS,'seed':SEED,'primary_preservation_weight':WEIGHT,'primary_preservation_temperature':TEMPERATURE,
        'primary_teacher_path':TEACHER,'primary_teacher_weights_sha256':CONTROL_SHA,'primary_teacher_classes':[0,1],
        'primary_teacher_on_original_known_rows_only':True,'supplement_primary_teacher_rows':0,
        'backbone_lr':4e-5,'head_lr':1e-4,'distillation_weight':4.,'auxiliary_photo_weight':.5,'original_labels_changed':False,
        'human_feedback_used':False,'ai_pseudo_labels_created':0,'source_test_used':False,'local_noncommercial_research_only':True,
        'distribute_source_images_or_checkpoint':False,'research_candidate_gate':GATE,'retention_candidate_gate':RETENTION_GATE}

def validate_protocol(protocol):
    for k,v in fixed().items():require(protocol.get(k)==v,'Declared condition differs: '+k)
    for key in('source_sha256','input_sha256'):
        for path,digest in protocol[key].items():require(sha(ROOT/path)==digest,'Frozen source/input differs: '+path)
    require(protocol['input_sha256'][INITIAL]==INITIAL_SHA and protocol['input_sha256'][TEACHER]==CONTROL_SHA,'Initializer/primary teacher differs')
    return protocol

def declare():
    require(not(ROOT/PROTOCOL).exists()and not(ROOT/'runs'/NAME).exists(),'Declare before actual training')
    prior=read(ROOT/PRIOR);require(prior['status']=='passed'and len(prior['source_sha256'])==163,'Prior163 source verification required')
    data=load_data();audit=read(ROOT/AUDIT)
    require(audit['source_test_used']is False and audit['new_labels_created']==0 and audit['val_cases_used_in_training_or_sampling']is False,'Read-only diagnostic required')
    sources={**prior['source_sha256'],**{p:sha(ROOT/p)for p in NEW_SOURCES}}
    inputs={p:sha(ROOT/p)for p in(INITIAL,TEACHER,DRAWS,MANIFEST,PRIOR,AUDIT,'data/facility-spatial-training/train.json',
        'data/facility-auxiliary-training/train.json','data/codebrim-training/val.json','runs/'+IDEA+'/best.pt',
        'runs/'+IDEA+'/history.json','reports/FACILITY_IDEA_TRANSITIONS_KO.md')}
    protocol={**fixed(),'declared_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sources,'input_sha256':inputs,
        'selected_supplement_photos':len(data['supplement']),'primary_photo_loss_recipe':'Original IDEA photo/spatial/aux19/other5KD4 recipe plus primary Bernoulli KL1/T2 on original known TRAIN only',
        'comparison_scope':'Only added primary regularization vs completed IDEA data/order arm. Historical low-LR control and0773 reference remain acceptance comparators; same6epoch/draw budget, extra teacher compute differs.',
        'teacher_probabilities_are_new_truth':False,'val_cases_used_in_training_or_sampling':False,'new_training_epochs_planned':6,
        'method_sources':['https://arxiv.org/abs/1503.02531','https://arxiv.org/abs/1606.09282'],
        'method_scope':'Bernoulli known-label/original-row preservation adaptation; original TRAIN is available. Not a reproduction of new-data-only LwF or evidence that forgetting caused observed transitions.',
        'app_model_promoted':False,'independent_industrial_accuracy_verified':False}
    validate_protocol(protocol);write_new(ROOT/PROTOCOL,protocol);print('Fixed primary-preservation study declared',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare',action='store_true');args=parser.parse_args();require(args.declare,'Use --declare');declare()
