"""Fixed full-encoder ConvNeXt experiment; original sampling and losses."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_head_lr_study import prepare_data,RETENTION_GATE
from scripts.facility_rc_positive import INITIAL,INITIAL_SHA,CONTROL,CONTROL_SHA,REFERENCE
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.train_facility_target import read,sha
from scripts.report_facility_detail import GATE
from scripts.fetch_rc2119 import write_new
from safelog_ai.convnext_facility import ARCH,OFFICIAL_SHA

NAME='facility-presence-target-convnext-finetune'
PROTOCOL='reports/facility-convnext-study-protocol.json'
DRAWS='runs/facility-spalling-sampler-plan/paired-draws.npz'
PRETRAINED='data/pretrained/convnext-tiny-imagenet1k-v1.pth'
EPOCHS,DRAW_COUNT,SEED=6,14248,56
RECIPE={'photo':'original masked focal gamma1','emphasis':[2.,2.,1.,1.,1.,1.,1.],
    'spatial':'original masked focal plus Dice','auxiliary_photo_weight':.5,'distillation_weight':4.,'distillation_temperature':2.}
NEW_SOURCES=('safelog_ai/convnext_facility.py','scripts/facility_convnext.py','scripts/train_facility_convnext.py',
    'scripts/evaluate_facility_convnext.py','scripts/verify_facility_convnext.py','scripts/report_facility_convnext.py','tests/test_convnext_facility.py')
def require(condition,message):
    if not condition:raise ValueError(message)
def load_data():
    torch.set_num_threads(4)
    data=prepare_data(ROOT,read(ROOT/'reports/facility-head-lr-study-protocol.json'),'low_lr',ROOT/'data/facility-auxiliary-training/train.json')
    data['candidate_draws']=data['epoch_draws'];return data
def dataset(data):return ResolutionPhotos(data['items'],data['auxiliary'],data['full_count'])
def validate_protocol(protocol):
    fixed={'schema':'facility_convnext_finetune_study_v1','run':NAME,'architecture':ARCH,'epochs':EPOCHS,
        'draws_per_epoch':DRAW_COUNT,'seed':SEED,'backbone_lr':4e-5,'head_lr':1e-4,'distillation_weight':4.,
        'auxiliary_photo_weight':.5,'primary_photo_loss_recipe':RECIPE,'original_sampling_unchanged':True,
        'original_labels_changed':False,'human_feedback_used':False,'ai_pseudo_labels_created':0,'source_test_used':False,
        'research_candidate_gate':GATE,'retention_candidate_gate':RETENTION_GATE,'activation_checkpointing':True,
        'all_encoder_parameters_trainable':True,'student_batchnorm_modules':0,'batch_size':8}
    for k,v in fixed.items():require(protocol.get(k)==v,'Declared condition changed: '+k)
    for key in('source_sha256','input_sha256'):
        for path,digest in protocol[key].items():require(sha(ROOT/path)==digest,'Frozen bytes changed: '+path)
    require(protocol['input_sha256'][INITIAL]==INITIAL_SHA and protocol['input_sha256'][PRETRAINED]==OFFICIAL_SHA
        and protocol['input_sha256']['runs/'+CONTROL+'/best.pt']==CONTROL_SHA,'Teacher/pretrained/control changed')
    return protocol
def declare():
    require(not(ROOT/PROTOCOL).exists()and not(ROOT/'runs'/NAME).exists(),'Declare before new training')
    prior=read(ROOT/'reports/facility-primary-asymmetric-study-verification.json')
    require(prior['status']=='passed'and len(prior['source_sha256'])==143,'Prior143 frozen sources required')
    sources={**prior['source_sha256'],**{p:sha(ROOT/p)for p in NEW_SOURCES}}
    inputs={p:sha(ROOT/p)for p in(INITIAL,'runs/'+CONTROL+'/best.pt',DRAWS,PRETRAINED,
        'reports/facility-semantic-pretrained-weights.json','data/facility-spatial-training/train.json',
        'data/facility-auxiliary-training/train.json','data/codebrim-training/val.json')}
    protocol={'schema':'facility_convnext_finetune_study_v1','run':NAME,'architecture':ARCH,'epochs':EPOCHS,
        'draws_per_epoch':DRAW_COUNT,'seed':SEED,'backbone_lr':4e-5,'head_lr':1e-4,'distillation_weight':4.,
        'auxiliary_photo_weight':.5,'primary_photo_loss_recipe':RECIPE,'original_sampling_unchanged':True,
        'original_labels_changed':False,'human_feedback_used':False,'ai_pseudo_labels_created':0,'source_test_used':False,
        'research_candidate_gate':GATE,'retention_candidate_gate':RETENTION_GATE,'activation_checkpointing':True,
        'all_encoder_parameters_trainable':True,'student_batchnorm_modules':0,'batch_size':8,
        'source_sha256':sources,'input_sha256':inputs,'declared_utc':datetime.now(timezone.utc).isoformat(),
        'comparison_scope':'Architecture and initialization package: trainable ImageNet ConvNeXt encoder plus new random FPN/photo/aux heads, LayerNorm/GroupNorm and stochastic depth .1. Original full640 transform, draw order, labels, losses, teacher and LRs retained. Historical MobileNet control reused; not an isolated encoder causal comparison; repeated public VAL is exploratory.',
        'method_sources':['https://docs.pytorch.org/vision/master/models/generated/torchvision.models.convnext_tiny.html',
        'https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html']}
    validate_protocol(protocol);write_new(ROOT/PROTOCOL,protocol);print('Full-encoder ConvNeXt study declared',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare',action='store_true');args=parser.parse_args();require(args.declare,'Use --declare');declare()
