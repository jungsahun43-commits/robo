"""AI visual observations select original TRAIN examples; never human truth."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.facility_head_lr_study import prepare_data, RETENTION_GATE
from scripts.facility_rc_positive import INITIAL, INITIAL_SHA, CONTROL, CONTROL_SHA, REFERENCE
from scripts.facility_resolution_study import ResolutionPhotos
from scripts.train_facility_target import read, sha, DOMAINS
from scripts.report_facility_detail import GATE
from scripts.fetch_rc2119 import write_new
from safelog_ai.auxiliary_classifier import ARCH

NAME = "facility-presence-target-ai-agreement"
PROTOCOL = "reports/facility-ai-agreement-study-protocol.json"
BASE = "runs/facility-ai-visual-review-20261010"
NOTES = BASE + "/AI-VISUAL-OBSERVATIONS.json"
DRAWS = BASE + "/agreement-draws.npz"
PACKAGE = "runs/facility-human-review-20261009/TRAIN-REVIEW.json"
EPOCHS, DRAW_COUNT, SEED = 6, 14248, 56
EXTRA, SAMPLING_SEED = 2, 79
NEW_SOURCES = ("scripts/facility_ai_agreement.py", "scripts/train_facility_ai_agreement.py",
    "scripts/verify_facility_ai_agreement.py", "scripts/report_facility_ai_agreement.py",
    "tests/test_ai_agreement.py")

def require(condition, message):
    if not condition: raise ValueError(message)

def select_agreement(notes, cases):
    require(notes.get("schema") == "facility_ai_visual_selection_v1"
        and notes.get("observer_type") == "ai" and notes.get("expert_confirmed") is False
        and notes.get("human_feedback_used") is False and notes.get("split") == "train",
        "AI observations must remain unconfirmed, TRAIN-only and separate from human feedback")
    observations = notes.get("cases", [])
    require(len(observations) == len(cases) == 200, "Exactly200 actually viewed TRAIN cases required")
    eligible, audit = [], Counter()
    ids = set()
    for observation, case in zip(observations, cases):
        require(observation["case_id"] == case["case_id"] and case["case_id"] not in ids
            and observation["image_sha256"] == case["image_sha256"]
            and observation["annotation_sha256"] == case["annotation"].get("sha256")
            and observation.get("annotation_kind") == case["annotation"].get("kind")
            and observation["contact_photo_viewed"] is True and observation["evidence"].strip(),
            "Viewed photo/annotation identity or evidence differs")
        ids.add(case["case_id"])
        judgements = observation["judgements"]
        require(len(judgements) == 2 and all(j in ("present", "absent", "uncertain") for j in judgements),
            "Two independent visual primary judgements required")
        if "uncertain" in judgements:
            audit["uncertain_photo"] += 1
        elif [int(j == "present") for j in judgements] != case["original_targets"][:2]:
            audit["definite_disagreement_photo"] += 1
        else:
            eligible.append(case["image"]); audit["both_definite_agreement_photo"] += 1
    require(eligible and len(set(eligible)) == len(eligible), "At least one unique original agreement required")
    return sorted(eligible), dict(audit)

def agreement_draws(original, items, full_count, eligible, seed=SAMPLING_SEED):
    """Add exactly2 reviewed exposures per epoch while preserving every tag count.

Each replaced full-photo position has the same source and original seven
asserted/unknown targets. All crop positions remain unchanged. No labels are
created from AI predictions. Reviewed positions cannot replace each other.
    """
    require(original.dtype == np.int64 and original.shape == (6,14248)
        and original.min() >= 0 and original.max() < len(items), "Original six-epoch TRAIN draw contract required")
    full = {r["image"]: i for i,r in enumerate(items[:full_count])}
    require(eligible and len(set(eligible)) == len(eligible) and all(p in full for p in eligible),
        "Only unique original full TRAIN photos can receive extra exposure")
    selected = {full[p] for p in eligible}
    groups = defaultdict(list)
    keys = [(r["domain"], tuple(r["targets"])) for r in items]
    for index in sorted(selected):
        require(items[index]["targets"][0] in (0,1) and items[index]["targets"][1] in (0,1), "Unknown primary label")
        groups[keys[index]].append(index)
    result = original.copy(); rng = np.random.default_rng(seed); changed = []
    for epoch, draws in enumerate(original):
        actual = 0
        for key, pool in sorted(groups.items()):
            positions = np.asarray([i for i,j in enumerate(draws) if j < full_count and j not in selected and keys[j] == key], dtype=np.int64)
            additions = np.tile(np.asarray(pool,dtype=np.int64), EXTRA)
            require(len(positions) >= len(additions), "Extra exposure exceeds unchanged source/full/tag budget")
            replace = rng.choice(positions, len(additions), replace=False); rng.shuffle(additions)
            result[epoch, replace] = additions; actual += len(additions)
        require(actual == EXTRA*len(selected), "Exact bounded extra exposure not achieved")
        changed.append(actual)
    return result, changed

def load_data():
    torch.set_num_threads(4)
    data = prepare_data(ROOT, read(ROOT/"reports/facility-head-lr-study-protocol.json"),
        "low_lr", ROOT/"data/facility-auxiliary-training/train.json")
    notes, package = read(ROOT/NOTES), read(ROOT/PACKAGE)
    eligible,audit = select_agreement(notes, package["cases"])
    draws, changed = agreement_draws(data["epoch_draws"], data["items"], data["full_count"], eligible)
    if (ROOT/DRAWS).exists():
        with np.load(ROOT/DRAWS, allow_pickle=False) as arrays:
            require(np.array_equal(arrays["candidate"],draws) and np.array_equal(arrays["control"],data["epoch_draws"]), "Frozen draw arrays differ")
    data.update(candidate_draws=draws, agreement_photos=eligible, agreement_audit=audit, changed_draws=changed)
    return data

def dataset(data):
    return ResolutionPhotos(data["items"], data["auxiliary"], data["full_count"])

def validate_protocol(protocol):
    fixed = {"schema":"facility_ai_agreement_study_v1", "run":NAME, "epochs":EPOCHS,
        "draws_per_epoch":DRAW_COUNT, "seed":SEED, "architecture":ARCH,
        "extra_exposures_per_selected_photo":EXTRA, "sampling_seed":SAMPLING_SEED,
        "backbone_lr":4e-5, "head_lr":1e-4, "distillation_weight":4., "auxiliary_photo_weight":.5,
        "research_candidate_gate":GATE, "retention_candidate_gate":RETENTION_GATE,
        "original_labels_changed":False, "human_feedback_used":False, "source_test_used":False}
    for k,v in fixed.items(): require(protocol.get(k) == v, "Declared condition changed: "+k)
    for field in ("source_sha256","input_sha256"):
        for path,digest in protocol[field].items(): require(sha(ROOT/path) == digest, "Frozen bytes changed: "+path)
    require(protocol["input_sha256"][INITIAL] == INITIAL_SHA and protocol["input_sha256"]["runs/"+CONTROL+"/best.pt"] == CONTROL_SHA,
        "Original initializer or reused control changed")
    return protocol

def prepare():
    require(not (ROOT/PROTOCOL).exists() and not (ROOT/NOTES).exists(), "Preserve an existing declared study")
    package=read(ROOT/PACKAGE); mapping={"P":"present","A":"absent","U":"uncertain"}
    raw=[]
    for page in range(1,11):
        rows=read(ROOT/BASE/f"page-{page:02d}.json")
        require(len(rows)==20, "Actually viewed20-case contact page required")
        raw.extend(rows)
    cases=[]
    for row,case in zip(raw,package["cases"]):
        require(case["case_id"]==f"train-review-{row[0]:04d}", "Review order changed")
        cases.append({"case_id":case["case_id"],"image_sha256":case["image_sha256"],
            "annotation_sha256":case["annotation"].get("sha256"),"annotation_kind":case["annotation"].get("kind"),"contact_photo_viewed":True,
            "contact_page":f"contact-{(row[0]-1)//20+1:02d}.jpg",
            "judgements":[mapping[row[1]],mapping[row[2]]],"evidence":row[3].replace("表面","표면")})
    notes={"schema":"facility_ai_visual_selection_v1","split":"train","observer_type":"ai",
        "expert_confirmed":False,"human_feedback_used":False,"label_changes":0,
        "created_utc":datetime.now(timezone.utc).isoformat(),"package_sha256":sha(ROOT/PACKAGE),
        "review_scope":"200 contact-sheet RGB photos; no model scores or publisher truth displayed during visual judgement; no masks visually reviewed",
        "criteria":"Visible concrete fracture or material-loss spalling. Joint/shadow/chalk/rust/coating/rough aggregate alone is insufficient. Ambiguous material, blur or boundary becomes uncertain. Absence is only absence of visible indicators in this view.",
        "field_provenance":"Public bridge/dam/concrete datasets; factory provenance and structural safety not inferred",
        "cases":cases}
    select_agreement(notes,package["cases"]); write_new(ROOT/NOTES,notes)
    data=load_data()
    np.savez_compressed(ROOT/DRAWS, control=data["epoch_draws"], candidate=data["candidate_draws"])
    prior=read(ROOT/"reports/facility-dense-auxiliary-study-verification.json")
    sources={**prior["source_sha256"],**{p:sha(ROOT/p) for p in NEW_SOURCES}}
    inputs={p:sha(ROOT/p) for p in (INITIAL,"runs/"+CONTROL+"/best.pt",PACKAGE,NOTES,DRAWS,
        "data/facility-spatial-training/train.json","data/facility-auxiliary-training/train.json",
        "runs/facility-spalling-sampler-plan/paired-draws.npz","data/codebrim-training/val.json")}
    for p in (ROOT/BASE).glob("contact-*.jpg"): inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (ROOT/BASE).glob("page-*.json"): inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    protocol={"schema":"facility_ai_agreement_study_v1","run":NAME,"architecture":ARCH,
        "epochs":EPOCHS,"draws_per_epoch":DRAW_COUNT,"seed":SEED,"sampling_seed":SAMPLING_SEED,
        "extra_exposures_per_selected_photo":EXTRA,"backbone_lr":4e-5,"head_lr":1e-4,
        "distillation_weight":4.,"auxiliary_photo_weight":.5,"source_sha256":sources,"input_sha256":inputs,
        "visual_review_audit":data["agreement_audit"],"selected_full_train_photos":len(data["agreement_photos"]),
        "selected_by_domain":dict(Counter(r["domain"] for r in package["cases"] if r["image"] in data["agreement_photos"])),
        "changed_draws_by_epoch":data["changed_draws"],"original_labels_changed":False,"human_feedback_used":False,
        "source_test_used":False,"research_candidate_gate":GATE,"retention_candidate_gate":RETENTION_GATE,
        "package_comparison_scope":"AI agreement selection + bounded original TRAIN example exposure; source/full/crop/seven known/unknown tag counts preserved; unchanged labels, architecture, losses, initializer and recipe; reused historical six-epoch control; exploratory repeated public VAL, not independent industrial-site validation"}
    validate_protocol(protocol);write_new(ROOT/PROTOCOL,protocol)
    print(json.dumps({"status":"prepared","selected":protocol["selected_full_train_photos"],"audit":data["agreement_audit"]}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args(); require(args.prepare,"Use --prepare");prepare()
