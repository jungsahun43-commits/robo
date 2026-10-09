"""Score unchanged full TRAIN and prepare exactly200 human-review photos."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_facility_train_review import canonical_training,validate_training_manifest,enrich_cases,verify_cache
from scripts.train_facility_target import FacilityPhotos,read,sha,predict
from scripts.build_facility_train_review import CLASSES,TASKS,LICENSE
from scripts.facility_rc_positive import CONTROL,CONTROL_SHA
from safelog_ai.presence_classifier import PresenceClassifier
from safelog_ai.review_feedback import canonical_sha256,validate_package

OUTPUT="runs/facility-human-review-20261009"
COUNT=200
# Full TRAIN census has only4 Dam error candidates; keep all4 and use36
# distinct Dam comparisons. No heldout image is substituted to fill a quota.
SOURCE_BUDGETS={"dacl":(92,28),"damsegment":(4,36),"codebrim":(24,16)}


def require(condition,message):
    if not condition:raise ValueError(message)


def write_new(path,document):
    with Path(path).open("x",encoding="utf-8",newline="\n")as stream:
        stream.write(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def outcomes(item,probability,thresholds):
    result=[]
    for k,task in enumerate(TASKS):
        if item["targets"][k]<0:continue
        prediction=probability[k]>=thresholds[task]
        actual=item["targets"][k]==1
        result.append({"task":task,"outcome":("TP" if prediction else "FN") if actual else ("FP" if prediction else "TN"),
                       "probability":probability[k],"threshold":thresholds[task]})
    return result


def select_cases(items,scores,thresholds,seed=71):
    """120 error candidates+80 comparisons; source-balanced distinct photos.

Half of the error quota in each source is ranked by mistake margin, the rest
is uniformly sampled. A parent group is represented once where it is known.
Only original TRAIN publisher labels classify the sampling strata.
    """
    rng=random.Random(seed);selected=[];audit=[];used_groups=set()
    for source,(error_budget,correct_budget)in SOURCE_BUDGETS.items():
        buckets={True:[],False:[]}
        for item in items:
            if item["domain"]!=source:continue
            marks=outcomes(item,scores[item["image"]],thresholds)
            if len(marks)!=2:continue
            errors=[m for m in marks if m["outcome"]in("FP","FN")]
            severity=max((abs(m["probability"]-m["threshold"])for m in errors),default=0.)
            buckets[bool(errors)].append((item,marks,severity))
        for error,budget in((True,error_budget),(False,correct_budget)):
            rows=sorted(buckets[error],key=lambda r:(-r[2],r[0]["image"]))
            hard_count=(budget+1)//2 if error else 0
            chosen=[]
            def take(row):
                group=(source,row[0].get("group_id",row[0]["image"]))
                if group in used_groups:return False
                used_groups.add(group);chosen.append(row);return True
            for row in rows:
                if len(chosen)>=hard_count:break
                take(row)
            chosen_images={r[0]["image"]for r in chosen}
            rest=[r for r in rows if r[0]["image"]not in chosen_images];rng.shuffle(rest)
            for row in rest:
                if len(chosen)>=budget:break
                take(row)
            require(len(chosen)==budget,"Not enough distinct TRAIN cases in the fixed source stratum")
            for item,marks,severity in chosen:
                selected.append({"image":item["image"],"domain":source,"original_targets":item["targets"][:],
                    "probabilities":scores[item["image"]][:],"selected_for":marks,
                    "review_stratum":"error_candidate"if error else"correct_comparison"})
            audit.append({"domain":source,"stratum":"error_candidate"if error else"correct_comparison",
                          "available":len(rows),"sampled":len(chosen)})
    require(len(selected)==COUNT and len({r["image"]for r in selected})==COUNT,"Expected exactly200 unique TRAIN photos")
    rng.shuffle(selected)
    return selected,audit


def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--device",default="cuda");args=parser.parse_args()
    destination=ROOT/OUTPUT;destination.mkdir(exist_ok=True)
    require(not(destination/"TRAIN-REVIEW.json").exists(),"Preserve existing human review package")
    weights=ROOT/"runs"/CONTROL/"best.pt";require(sha(weights)==CONTROL_SHA,"Fixed baseline checkpoint changed")
    manifest_path=ROOT/"data/facility-spatial-training/train.json";manifest=read(manifest_path)
    canonical,records,membership_sources=canonical_training(ROOT)
    detail=read(ROOT/"data/facility-detail-training/manifest.json")
    full,_=validate_training_manifest(manifest,canonical,detail)
    require(len(full)==14248,"Original full TRAIN count changed")
    cache_path=destination/"TRAIN-SCORES.json"
    if cache_path.exists():
        cache=read(cache_path)
        require(cache["weights_sha256"]==CONTROL_SHA and cache["manifest_sha256"]==sha(manifest_path)
                and cache["split"]=="train"and cache["images"]==[r["image"]for r in full],"Existing scoring cache does not bind the current TRAIN inputs")
    else:
        classifier=PresenceClassifier(weights,args.device)
        require(classifier.classes==CLASSES,"Original7 class order changed")
        loader=DataLoader(FacilityPhotos(full,640),batch_size=16,num_workers=4)
        targets,probabilities=predict(classifier.model,loader,args.device)
        require(np.array_equal(targets,np.asarray([r["targets"]for r in full])),"Scoring changed original labels")
        cache={"split":"train","classes":CLASSES,"weights_sha256":CONTROL_SHA,"manifest_sha256":sha(manifest_path),
               "images":[r["image"]for r in full],"targets":targets.tolist(),"probabilities":probabilities.tolist(),
               "training_epochs":0,"heldout_inference":False}
        write_new(cache_path,cache)
    selection_path=weights.parent/"TARGET-SELECTION.json";selection=read(selection_path)
    require(selection["weights_sha256"]==CONTROL_SHA and selection["selected"]["grid"]==1,"Frozen baseline cutoff changed")
    thresholds={c:selection["selected"]["per_class"][c]["threshold"]for c in TASKS}
    scores=verify_cache(cache,full,CLASSES,CONTROL_SHA)
    selected,sampling=select_cases(full,scores,thresholds)
    cases=enrich_cases(selected,canonical,records,ROOT,destination)
    package={"schema":"facility_train_review_v1","split":"train","created_utc":datetime.now(timezone.utc).isoformat(),
        "classes":CLASSES,"weights_sha256":CONTROL_SHA,"thresholds":thresholds,"cases":cases,"sampling":sampling,
        "license":LICENSE,"policy":"사람이 직접 입력한 판단을 별도 TRAIN 정답 후보로 저장. 원본·VAL·TEST 정답은 유지.",
        "original_full_train_count":len(full),"scored_cache_count":len(full),"detail_rows_not_reviewed":0,
        "scope":"Exactly200 original TRAIN photos.120 publisher-defined error candidates and80 comparison cases; no heldout prediction mining",
        "provenance":{"cache_sha256":sha(cache_path),"manifest_sha256":sha(manifest_path),
            "thresholds_sha256":sha(selection_path),"membership_sources":membership_sources,"script_sha256":sha(Path(__file__))}}
    package["package_content_sha256"]=canonical_sha256(package);validate_package(package)
    write_new(destination/"TRAIN-REVIEW.json",package)
    audit={"schema":"facility_human_review_preparation_v1","status":"ready_for_real_human_input","photos":COUNT,
        "primary_tasks_per_photo":2,"error_candidate_photos":120,"correct_comparison_photos":80,"sampling":sampling,
        "weights_sha256":CONTROL_SHA,"package_content_sha256":package["package_content_sha256"],
        "package_file_sha256":sha(destination/"TRAIN-REVIEW.json"),"source_core_manifest_sha256":sha(manifest_path),
        "source_train_photos_scored":len(full),"actual_new_training_epochs":0,"actual_human_judgements":0,
        "source_test_inference":False,"source_validation_inference":False,"original_labels_changed":False,
        "existing_app_model_promoted":False,"individual_source_paths_published":False}
    write_new(ROOT/"reports/facility-human-review-preparation.json",audit)
    print(json.dumps({"status":audit["status"],"photos":COUNT,"actual_human_judgements":0,"actual_new_training_epochs":0}),flush=True)


if __name__=="__main__":main()
