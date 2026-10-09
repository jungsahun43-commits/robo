"""Original TRAIN sampling with one predeclared primary photo-loss change."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
from collections import Counter
import sys
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_head_lr_study import prepare_data,RETENTION_GATE
from scripts.facility_rc_positive import INITIAL,INITIAL_SHA,CONTROL,CONTROL_SHA,REFERENCE
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.train_facility_target import read,sha
from scripts.report_facility_detail import GATE
from scripts.fetch_rc2119 import write_new
from safelog_ai.auxiliary_classifier import ARCH
from safelog_ai.primary_asymmetric_loss import GAMMA_POS,GAMMA_NEG,NEGATIVE_CLIP

NAME="facility-presence-target-primary-asymmetric"
PROTOCOL="reports/facility-primary-asymmetric-study-protocol.json"
BASE="runs/facility-primary-asymmetric-20261010"
AUDIT=BASE+"/NATIVE-REVIEW-OBSERVATIONS.json"
DRAWS="runs/facility-spalling-sampler-plan/paired-draws.npz"
EPOCHS,DRAW_COUNT,SEED=6,14248,56
RECIPE={"primary_columns":[0,1],"gamma_positive":GAMMA_POS,"gamma_negative":GAMMA_NEG,
    "negative_probability_shift":NEGATIVE_CLIP,"focal_weight_detached":True,"primary_computation":"float32 stable logsigmoid positive term",
    "other_five_photo_terms":"original gamma1 masked focal","positive_weights":"original source-specific weights",
    "emphasis":[2.,2.,1.,1.,1.,1.,1.],"reduction":"original per-photo known count then batch mean",
    "unknown_photo_labels":"zero contribution and gradient, excluded from denominator"}
NEW_SOURCES=("safelog_ai/primary_asymmetric_loss.py","scripts/facility_primary_asymmetric.py",
    "scripts/train_facility_primary_asymmetric.py","scripts/verify_facility_primary_asymmetric.py",
    "scripts/report_facility_primary_asymmetric.py","tests/test_primary_asymmetric_loss.py")

def require(condition,message):
    if not condition:raise ValueError(message)

def load_data():
    torch.set_num_threads(4)
    data=prepare_data(ROOT,read(ROOT/"reports/facility-head-lr-study-protocol.json"),"low_lr",ROOT/"data/facility-auxiliary-training/train.json")
    data["candidate_draws"]=data["epoch_draws"]
    return data

def dataset(data):return ResolutionPhotos(data["items"],data["auxiliary"],data["full_count"])

def validate_protocol(protocol):
    fixed={"schema":"facility_primary_asymmetric_study_v1","run":NAME,"architecture":ARCH,"epochs":EPOCHS,
        "draws_per_epoch":DRAW_COUNT,"seed":SEED,"backbone_lr":4e-5,"head_lr":1e-4,"distillation_weight":4.,
        "auxiliary_photo_weight":.5,"primary_photo_loss_recipe":RECIPE,"original_sampling_unchanged":True,
        "original_labels_changed":False,"human_feedback_used":False,"ai_pseudo_labels_created":0,
        "source_test_used":False,"research_candidate_gate":GATE,"retention_candidate_gate":RETENTION_GATE}
    for k,v in fixed.items():require(protocol.get(k)==v,"Declared condition changed: "+k)
    for key in("source_sha256","input_sha256"):
        for path,digest in protocol[key].items():require(sha(ROOT/path)==digest,"Frozen bytes changed: "+path)
    require(protocol["input_sha256"][INITIAL]==INITIAL_SHA and protocol["input_sha256"]["runs/"+CONTROL+"/best.pt"]==CONTROL_SHA,"Initializer/control changed")
    return protocol

def declare():
    require(not(ROOT/PROTOCOL).exists()and not(ROOT/"runs"/NAME).exists(),"Declare before any new training")
    cases=read(ROOT/BASE/"NATIVE-REVIEW-CASES.json");decisions=read(ROOT/BASE/"native-decisions.json")
    require(len(cases)==len(decisions)==12 and len({r["case_id"]for r in cases})==12,"Twelve unique TRAIN native reviews required")
    observed=[]
    for case,row in zip(cases,decisions):
        require(case["case_id"]==row[0]and all(j in("present","absent","uncertain")for j in row[1:3]),"Actual native observation identity differs")
        path=ROOT/case["original_source"];require(sha(path)==case["original_source_sha256"],"Original TRAIN photo changed")
        with Image.open(path)as image:width,height=image.size
        observed.append({"case_id":case["case_id"],"source_photo_sha256":sha(path),"source_dimensions":[width,height],
            "source_full_photo_viewed":True,"judgements":row[1:3],"evidence":row[3],"annotation_mask_viewed":False})
    audit={"schema":"facility_ai_native_diagnostic_review_v1","observer_type":"ai","expert_confirmed":False,
        "human_feedback_used":False,"new_training_labels":0,"source_test_used":False,"split":"train",
        "review_scope":"12 previously uncertain TRAIN error candidates opened as original source photos; UI may downsize; source tags were known during review; not blind and no expert truth",
        "cases":observed,"both_definite_photos":sum("uncertain"not in r["judgements"]for r in observed),
        "interpretation":"Native views resolve some cues but joints, coarse aggregate and tiny damage remain ambiguous. No observation changes source labels or sampling."}
    write_new(ROOT/AUDIT,audit)
    prior=read(ROOT/"reports/facility-ai-agreement-study-verification.json")
    require(prior["status"]=="passed"and len(prior["source_sha256"])==137,"Prior frozen137 sources required")
    sources={**prior["source_sha256"],**{p:sha(ROOT/p)for p in NEW_SOURCES}}
    inputs={p:sha(ROOT/p)for p in (INITIAL,"runs/"+CONTROL+"/best.pt",DRAWS,AUDIT,
        BASE+"/NATIVE-REVIEW-CASES.json",BASE+"/native-decisions.json","data/facility-spatial-training/train.json",
        "data/facility-auxiliary-training/train.json","data/codebrim-training/val.json")}
    protocol={"schema":"facility_primary_asymmetric_study_v1","run":NAME,"architecture":ARCH,"epochs":EPOCHS,
        "draws_per_epoch":DRAW_COUNT,"seed":SEED,"backbone_lr":4e-5,"head_lr":1e-4,"distillation_weight":4.,
        "auxiliary_photo_weight":.5,"primary_photo_loss_recipe":RECIPE,"original_sampling_unchanged":True,
        "original_labels_changed":False,"human_feedback_used":False,"ai_pseudo_labels_created":0,"source_test_used":False,
        "research_candidate_gate":GATE,"retention_candidate_gate":RETENTION_GATE,"source_sha256":sources,"input_sha256":inputs,
        "declared_utc":datetime.now(timezone.utc).isoformat(),"native_diagnostic_review":{"opened_train_photos":12,"both_definite":audit["both_definite_photos"],"expert_confirmed":False,"observations_used_as_training_labels":False},
        "method_sources":["https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html","https://github.com/Alibaba-MIIL/ASL/blob/main/train.py"],
        "comparison_scope":"Only primary photo objective becomes ASL-style gamma_pos0/gamma_neg4/clip0.05 with detached focusing and stable FP32; original draw order/labels/other losses/architecture/initial state/LRs/BN policy retained; historical control reused, repeated public VAL is exploratory"}
    validate_protocol(protocol);write_new(ROOT/PROTOCOL,protocol)
    print("Primary ASL study declared; native observations remain unconfirmed diagnostics; zero new labels",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--declare",action="store_true");args=p.parse_args();require(args.declare,"Use --declare");declare()
