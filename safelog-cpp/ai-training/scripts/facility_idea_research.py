"""One predeclared IDEA data/exposure study; no source TEST or model promotion."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_head_lr_study import prepare_data,RETENTION_GATE
from scripts.facility_rc_positive import INITIAL,INITIAL_SHA,CONTROL,CONTROL_SHA,REFERENCE
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.train_facility_target import DOMAINS,read,sha
from scripts.report_facility_detail import GATE
from scripts.fetch_rc2119 import write_new
from scripts.prepare_idea_research import MANIFEST
from safelog_ai.auxiliary_classifier import ARCH

NAME='facility-presence-target-idea-building'
PROTOCOL='reports/facility-idea-study-protocol.json'
DRAWS='data/idea-research/draws.npz'
EPOCHS,DRAW_COUNT,SEED,REPLACEMENTS=6,14248,56,1784
NEW_DOMAINS={**DOMAINS,'idea':3}
NEW_SOURCES=('scripts/fetch_idea_research.py','scripts/prepare_idea_research.py','scripts/facility_idea_research.py',
    'scripts/train_facility_idea_research.py','scripts/verify_facility_idea_research.py','scripts/report_facility_idea_research.py','tests/test_idea_research.py')
PRIOR='reports/facility-spalling-ohem-study-verification.json'

def require(condition,message):
    if not condition:raise ValueError(message)

def fixed_draws(original,core_count,items):
    require(original.shape==(EPOCHS,DRAW_COUNT)and original.dtype==np.int64 and 0<=original.min()and original.max()<core_count,'Original draw budget differs')
    groups=[np.asarray([i for i,r in enumerate(items)if r['stratum']==s],np.int64)for s in('rc_spalling','author_no_damage')]
    require(all(len(g)>0 for g in groups),'Two publisher strata required')
    rng=np.random.default_rng(61);result=original.copy()
    for epoch in range(EPOCHS):
        positions=np.sort(rng.choice(DRAW_COUNT,REPLACEMENTS,replace=False));chosen=[]
        for group in groups:
            order=rng.permutation(group);chosen.extend(order[np.arange(REPLACEMENTS//2)%len(order)])
        chosen=np.asarray(chosen,np.int64);rng.shuffle(chosen);result[epoch,positions]=core_count+chosen
    return result

def valid_targets(row):
    values=row['targets'];require(isinstance(values,list)and len(values)==7 and all(type(v)is int and v in(-1,0,1)for v in values)and values[2:]==[-1]*5,'Invalid publisher primary targets')
    if row['stratum']=='rc_spalling':require(values[1]==1 and values[0]in(-1,1),'Unasserted damaged-photo negative')
    elif row['stratum']=='author_no_damage':require(values[:2]==[0,0],'Normal source presence tag required')
    else:raise ValueError('Unknown publisher stratum')

class StudyPhotos:
    def __init__(self,data):
        self.core=ResolutionPhotos(data['items'],data['auxiliary'],data['full_count'])
        self.supplement=ResolutionPhotos(data['supplement'],{},len(data['supplement']),NEW_DOMAINS,640)
    def __len__(self):return len(self.core)+len(self.supplement)
    def __getitem__(self,index):
        if index<len(self.core):return self.core[index]
        row=self.supplement[index-len(self.core)];return row[:9]+(index,)

def dataset(data):return StudyPhotos(data)

def load_data():
    torch.set_num_threads(4)
    data=prepare_data(ROOT,read(ROOT/'reports/facility-head-lr-study-protocol.json'),'low_lr',ROOT/'data/facility-auxiliary-training/train.json')
    manifest=read(ROOT/MANIFEST);require(manifest['split']=='train'and manifest['classes']==data['classes'],'Supplement ontology differs')
    items=manifest['items'];require(len({r['group_id']for r in items})==len(items)and 2<=len(items)<=400,'Invalid group representative count')
    for row in items:
        valid_targets(row);require(row['domain']=='idea'and row['source_split']=='train_only_unsplit','Source provenance differs')
        for key,digest in(('image','image_sha256'),('pixel_target','pixel_target_sha256'),('source_image','source_image_sha256'),('source_annotation','source_annotation_sha256')):
            path=(ROOT/row[key]).resolve();require(path.is_relative_to((ROOT/'data/idea-research').resolve())and sha(path)==row[digest],'Source bytes/path differ')
        with np.load(ROOT/row['pixel_target'],allow_pickle=False)as fields:
            require(fields['mask'].shape==(7,80,80)and fields['known'].shape==(7,)and not fields['mask'].any()and not fields['known'].any(),'Bounding boxes must not become pixel truth')
    data['supplement']=items;data['candidate_draws']=fixed_draws(data['epoch_draws'],len(data['items']),items);return data

def fixed():
    return {'schema':'facility_idea_research_study_v1','declared_before_training':True,'run':NAME,'control':CONTROL,'reference':REFERENCE,'architecture':ARCH,
        'epochs':EPOCHS,'batch_size':8,'imgsz':640,'draws_per_epoch':DRAW_COUNT,'replacement_draws_per_epoch':REPLACEMENTS,
        'supplement_positive_and_normal_draws_per_epoch':REPLACEMENTS//2,'seed':SEED,'selection_seed':61,
        'backbone_lr':4e-5,'head_lr':1e-4,'distillation_weight':4.,'auxiliary_photo_weight':.5,
        'original_labels_changed':False,'human_feedback_used':False,'ai_pseudo_labels_created':0,'source_test_used':False,
        'local_noncommercial_research_only':True,'distribute_source_images_or_checkpoint':False,
        'research_candidate_gate':GATE,'retention_candidate_gate':RETENTION_GATE}

def validate_protocol(protocol):
    for k,v in fixed().items():require(protocol.get(k)==v,'Declared condition differs: '+k)
    for key in('source_sha256','input_sha256'):
        for path,digest in protocol[key].items():require(sha(ROOT/path)==digest,'Frozen bytes changed: '+path)
    require(protocol['input_sha256'][INITIAL]==INITIAL_SHA and protocol['input_sha256']['runs/'+CONTROL+'/best.pt']==CONTROL_SHA,'Initializer/control differs')
    return protocol

def declare():
    require(not(ROOT/PROTOCOL).exists()and not(ROOT/'runs'/NAME).exists(),'Declare before new training')
    prior=read(ROOT/PRIOR);require(prior['status']=='passed'and len(prior['source_sha256'])==156,'Prior156 frozen sources required')
    data=load_data();np.savez_compressed(ROOT/DRAWS,control=data['epoch_draws'],candidate=data['candidate_draws'])
    sources={**prior['source_sha256'],**{p:sha(ROOT/p)for p in NEW_SOURCES}}
    inputs={p:sha(ROOT/p)for p in(INITIAL,'runs/'+CONTROL+'/best.pt',DRAWS,MANIFEST,PRIOR,
        'data/facility-spatial-training/train.json','data/facility-auxiliary-training/train.json','data/codebrim-training/val.json',
        'data/idea-research/record.json','data/idea-research/annotation-index.json','data/idea-research/zip-directory.json',
        'data/idea-research/acquisition.json','data/idea-research/source-audit.json','data/idea-research/crop-review.json',
        'reports/facility-idea-data.json')}
    protocol={**fixed(),'declared_utc':datetime.now(timezone.utc).isoformat(),'client_date':'2026-10-11','source_sha256':sources,'input_sha256':inputs,
        'selected_photos':len(data['supplement']),'selected_strata':read(ROOT/MANIFEST)['audit']['strata'],
        'comparison_scope':'One building IDEA photo-data +12.52% balanced source-exposure package vs historical low-LR control. Original recipe/architecture/initializer retained, core draws replaced; not isolated data-only causality.',
        'unknown_policy':'Explicit whole-photo no-damage building tag provides primary0/0. Damage absence unknown; RC object Crack/Concrete spalling provides positive. Other5/pixel/aux unknown. Normal material unverified.',
        'loss_recipe':'Original masked photo focal gamma1 primary emphasis2, spatial focal+Dice, auxiliary19*.5, known other5 Bernoulli KD4/T2. New-domain photo positive weight1.',
        'training_recipe':'Original640,324states,0773 initializer, frozen47BN, AdamW.0002/Cosine6 eta5e-6,8batch. One fixed6epoch candidate; old control reused.',
        'source':'https://zenodo.org/records/15120522','license':'CC BY-NC-ND 4.0','source_test_used':False,
        'independent_industrial_accuracy_verified':False,'app_model_promoted':False}
    validate_protocol(protocol);write_new(ROOT/PROTOCOL,protocol);print('IDEA building photo research recipe declared',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare',action='store_true');args=parser.parse_args();require(args.declare,'Use --declare');declare()
