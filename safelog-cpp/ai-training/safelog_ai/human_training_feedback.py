"""Explicit user-submitted TRAIN opinions become a separate label overlay.

Original annotations and seven-channel mask files are never rewritten.
The reviewer is not treated as a verified domain expert. Unknown judgements
remove affected supervision, including derived crops of the same parent.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy

from safelog_ai.review_feedback import CLASSES,TASKS,validate_package,validate_feedback,canonical_sha256

SCHEMA="facility_human_training_submission_v1"
AUXILIARY_PRIMARY={"concrete_crack":"Crack","concrete_spalling":"Spalling"}


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_submission(submission,package):
    package=validate_package(package)
    require(isinstance(submission,dict)and set(submission)=={"schema","purpose","source_package_sha256","feedback","submitted_for_training","test_fixture"},"Invalid human training submission fields")
    require(submission["schema"]==SCHEMA and submission["purpose"]=="human_reviewed_train_photo_labels"
            and submission["submitted_for_training"]is True and submission["test_fixture"]is False,
            "Actual submitted human feedback is required; test fixtures/AI observations are not training labels")
    require(submission["source_package_sha256"]==package["package_content_sha256"],"Human submission belongs to a different package")
    feedback=validate_feedback(submission["feedback"],package)
    reviewed=sum(t["judgement"]!="unreviewed"for c in feedback["cases"]for t in c["tasks"])
    known=sum(t["judgement"]in("present","absent")for c in feedback["cases"]for t in c["tasks"])
    require(reviewed>0 and known>0,"No definite human judgement is available for training")
    return feedback


def build_overlay(submission,package,core_manifest):
    """Correct full-photo tags; never propagate positive parent labels to crops."""
    feedback=validate_submission(submission,package)
    require(core_manifest.get("split")=="train"and core_manifest.get("classes")==CLASSES,"Only original7 TRAIN is eligible")
    result=deepcopy(core_manifest);items=result["items"];full_count=result["full_count"]
    full={r["image"]:r for r in items[:full_count]}
    require(len(full)==full_count,"Duplicate full TRAIN image")
    cases={c["case_id"]:c for c in package["cases"]};affected={};verified={};counts=Counter()
    for opinion in feedback["cases"]:
        case=cases[opinion["case_id"]];row=full.get(case["image"])
        require(row is not None and row["domain"]==case["domain"]and row["targets"]==case["original_targets"],"Review source is stale, heldout or mismatched")
        for task in opinion["tasks"]:
            judgement=task["judgement"];k=TASKS.index(task["task"])
            counts[judgement]+=1
            if judgement=="unreviewed":continue
            old=row["targets"][k];new={"present":1,"absent":0,"uncertain":-1}[judgement]
            row["targets"][k]=new
            if new>=0:verified.setdefault(row["image"],[]).append(k)
            if old!=new or new<0:
                counts["changed_full_photo_targets"]+=int(new>=0)
                counts["uncertain_full_photo_targets"]+=int(new<0)
                affected.setdefault(row["image"],set()).add(k)
                row.setdefault("human_pixel_unknown_columns",[]).append(k)
                row.setdefault("human_auxiliary_unknown_tags",[]).append(AUXILIARY_PRIMARY[task["task"]])
    for row in items[full_count:]:
        columns=affected.get(row.get("parent_image"),set())
        if not columns:continue
        for k in sorted(columns):row["targets"][k]=-1
        row["human_pixel_unknown_columns"]=sorted(columns)
        row["human_auxiliary_unknown_tags"]=[AUXILIARY_PRIMARY[TASKS[k]]for k in sorted(columns)]
        counts["derived_crop_channels_masked_unknown"]+=len(columns)
    stats={k:int(counts[k])for k in("present","absent","uncertain","unreviewed","changed_full_photo_targets","uncertain_full_photo_targets","derived_crop_channels_masked_unknown")}
    stats.update(reviewed_photos=sum(any(t["judgement"]!="unreviewed"for t in c["tasks"])for c in feedback["cases"]),
                 verified_photo_tasks=sum(len(v)for v in verified.values()),expert_qualification_verified=False,
                 original_mask_files_changed=False,source_validation_or_test_labels_changed=False)
    result["human_review_overlay"]={"schema":"facility_human_training_overlay_v1","submission_content_sha256":canonical_sha256(submission),
        "source_package_sha256":package["package_content_sha256"],"reviewer_role":feedback["reviewer"]["role"],
        "source_core_manifest_content_sha256":canonical_sha256(core_manifest),"stats":stats,"verified_photo_columns":verified,
        "policy":"User-reviewed photo labels only. Changed/uncertain parent classes disable original pixel and auxiliary supervision and crop class labels; no new masks or expert certification"}
    return result,stats
