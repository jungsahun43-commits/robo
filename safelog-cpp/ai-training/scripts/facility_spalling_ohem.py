"""Fixed negative-pixel mining candidate with original TRAIN draws."""
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
from safelog_ai.auxiliary_classifier import ARCH
from safelog_ai.spalling_ohem import HARD_FRACTION
NAME='facility-presence-target-spalling-ohem'
PROTOCOL='reports/facility-spalling-ohem-study-protocol.json'
DRAWS='runs/facility-spalling-sampler-plan/paired-draws.npz'
EPOCHS,DRAW_COUNT,SEED=6,14248,56
RECIPE={'photo':'original masked focal gamma1','emphasis':[2.,2.,1.,1.,1.,1.,1.],
    'spatial':'original focal plus Dice except spalling known-background focal tail',
    'hard_background_fraction':HARD_FRACTION,'selection':'per-photo ceil(fraction * asserted background cells)',
    'negative_normalization':'background_count / selected_count; original global denominator',
    'original_spalling_foreground_and_soft_target_terms_preserved':True,'original_other_six_spatial_terms_preserved':True,
    'original_all_seven_dice_terms_preserved':True,'unknown_regions_excluded':True,'auxiliary_photo_weight':.5,
    'distillation_weight':4.,'distillation_temperature':2.}
NEW_SOURCES=('safelog_ai/spalling_ohem.py','scripts/facility_spalling_ohem.py','scripts/train_facility_spalling_ohem.py',
    'scripts/verify_facility_spalling_ohem.py','scripts/report_facility_spalling_ohem.py','tests/test_spalling_ohem.py')
def require(condition,message):
    if not condition:raise ValueError(message)
def load_data():
    torch.set_num_threads(4)
    data=prepare_data(ROOT,read(ROOT/'reports/facility-head-lr-study-protocol.json'),'low_lr',ROOT/'data/facility-auxiliary-training/train.json')
    data['candidate_draws']=data['epoch_draws'];return data
def dataset(data):return ResolutionPhotos(data['items'],data['auxiliary'],data['full_count'])
def fixed():
    return {'schema':'facility_spalling_ohem_study_v1','run':NAME,'architecture':ARCH,'epochs':EPOCHS,
        'draws_per_epoch':DRAW_COUNT,'seed':SEED,'backbone_lr':4e-5,'head_lr':1e-4,'distillation_weight':4.,
        'auxiliary_photo_weight':.5,'primary_photo_loss_recipe':RECIPE,'original_sampling_unchanged':True,
        'original_labels_changed':False,'human_feedback_used':False,'ai_pseudo_labels_created':0,'source_test_used':False,
        'research_candidate_gate':GATE,'retention_candidate_gate':RETENTION_GATE}
def validate_protocol(protocol):
    for k,v in fixed().items():require(protocol.get(k)==v,'Declared condition changed: '+k)
    for key in('source_sha256','input_sha256'):
        for path,digest in protocol[key].items():require(sha(ROOT/path)==digest,'Frozen bytes changed: '+path)
    require(protocol['input_sha256'][INITIAL]==INITIAL_SHA
        and protocol['input_sha256']['runs/'+CONTROL+'/best.pt']==CONTROL_SHA,'Initializer/control changed')
    return protocol
def declare():
    require(not(ROOT/PROTOCOL).exists()and not(ROOT/'runs'/NAME).exists(),'Declare before training')
    prior=read(ROOT/'reports/facility-convnext-study-verification.json')
    require(prior['status']=='passed'and len(prior['source_sha256'])==150,'Prior150 frozen sources required')
    sources={**prior['source_sha256'],**{p:sha(ROOT/p)for p in NEW_SOURCES}}
    inputs={p:sha(ROOT/p)for p in(INITIAL,'runs/'+CONTROL+'/best.pt',DRAWS,'data/facility-spatial-training/train.json',
        'data/facility-auxiliary-training/train.json','data/codebrim-training/val.json')}
    protocol={**fixed(),'source_sha256':sources,'input_sha256':inputs,'declared_utc':datetime.now(timezone.utc).isoformat(),
        'client_date':'2026-10-11','method_sources':['https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html'],
        'comparison_scope':'Only spalling asserted-background spatial focal uses its hardest10% per photo, retaining original global normalization. Original foreground/soft targets, other-six spatial focal and all Dice, photo/auxiliary/KD objectives,324-state architecture,0773 initializer, frozen BN, LRs and draw order retained. Historical low-LR control reused. Pixel adaptation of hard mining, not original detector reproduction; repeated public VAL is exploratory.'}
    validate_protocol(protocol);write_new(ROOT/PROTOCOL,protocol);print('Spalling negative-pixel OHEM study declared',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare',action='store_true');args=parser.parse_args();require(args.declare,'Use --declare');declare()
