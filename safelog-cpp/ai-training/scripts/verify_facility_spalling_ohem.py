"""Verify real spalling-OHEM updates and preserved original samples and targets."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_spalling_ohem import NAME,PROTOCOL,DRAWS,INITIAL,CONTROL,EPOCHS,DRAW_COUNT,NEW_SOURCES,load_data,validate_protocol,read,sha,require
from scripts.verify_facility_rc_positive import git,snapshots,verify_git_sources
from scripts.fetch_rc2119 import write_new
from scripts.train_facility_target import DOMAINS
from safelog_ai.auxiliary_classifier import AuxiliaryClassifier,ARCH
from safelog_ai.frozen_batchnorm import batchnorm_state_sha256
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.presence_classifier import image_transform

BEFORE="runs/facility-spalling-ohem-before.json"
LEDGER="runs/facility-spalling-ohem-input-ledger.json"
TESTS="runs/facility-spalling-ohem-tests.json"
PREFLIGHT="runs/facility-spalling-ohem-preflight/preflight.json"
OUTPUT="reports/facility-spalling-ohem-study-verification.json"
PRIOR="reports/facility-convnext-study-verification.json"
OLD_LEDGER="runs/facility-convnext-input-ledger.json"

def prerequisites(protocol):
    prior=read(ROOT/PRIOR);require(prior["status"]=="passed"and len(prior["source_sha256"])==150,"Prior150 source inventory required")
    require(set(protocol["source_sha256"])==set(prior["source_sha256"])|set(NEW_SOURCES)
        and all(protocol["source_sha256"][p]==h for p,h in prior["source_sha256"].items()),"Previous source bytes changed")
    tests=read(ROOT/TESTS);require(type(tests["tests_run"])is int and tests["tests_run"]>=11
        and all(tests[k]==0 for k in("failures","errors","skipped"))and tests["source_sha256"]==protocol["source_sha256"]
        and tests["protocol_sha256"]==sha(ROOT/PROTOCOL),"Actual frozen tests required")
    preflight=read(ROOT/PREFLIGHT);require(preflight["status"]=="passed"and preflight["actual_disposable_fp32_updates"]==1
        and preflight["actual_training_epochs"]==0 and preflight["protocol_sha256"]==sha(ROOT/PROTOCOL)
        and all(preflight[k]is True for k in("strict_cpu_324_reload","teacher_and_bn_preserved","strict_fp32_diagnostic_flags_restored")),"Actual preflight failed")
    return prior,tests,preflight

def before(protocol):
    require(not(ROOT/BEFORE).exists()and not(ROOT/LEDGER).exists()and not(ROOT/"runs"/NAME).exists(),"Preserve previous proof/run")
    prior,tests,preflight=prerequisites(protocol);require(sha(ROOT/OLD_LEDGER)==prior["input_ledger_sha256"],"Prior immutable ledger differs")
    old=read(ROOT/OLD_LEDGER)["snapshots_before"]
    paths=set(old)|set(protocol["source_sha256"])|set(protocol["input_sha256"])
    paths.update((PROTOCOL,TESTS,PREFLIGHT,"runs/facility-spalling-ohem-preflight/disposable.pt",PRIOR,OLD_LEDGER,
        "reports/facility-convnext-study-comparison.json","reports/FACILITY_CONVNEXT_STUDY_RESULTS_KO.md"))
    paths.update(p.relative_to(ROOT).as_posix()for p in(ROOT/"runs/facility-presence-target-convnext-finetune").rglob("*")if p.is_file())
    paths.update(('reports/FACILITY_CONVNEXT_HANDOFF_KO.md','reports/facility-convnext-handoff-manifest.json',
        'reports/facility-convnext-local-timing.json','reports/facility-convnext-cli-smoke.json','runs/facility-convnext-research-handoff.zip'))
    current=snapshots(paths);require(all(current[p]==v for p,v in old.items()),"An existing input SHA/size/mtime changed")
    commit=git(["rev-parse","HEAD"]).decode().strip();verify_git_sources({**protocol["source_sha256"],PROTOCOL:sha(ROOT/PROTOCOL)},commit)
    write_new(ROOT/LEDGER,{"snapshots_before":current,"prior_protected_files":len(old),"previous_ledger_sha256":sha(ROOT/OLD_LEDGER)})
    write_new(ROOT/BEFORE,{"status":"passed","protocol_sha256":sha(ROOT/PROTOCOL),"source_git_commit":commit,
        "source_sha256":protocol["source_sha256"],"input_ledger_sha256":sha(ROOT/LEDGER),"protected_files":len(current),
        "test_record_sha256":sha(ROOT/TESTS),"preflight_sha256":sha(ROOT/PREFLIGHT),"git_blob_bytes_verified":True})
    print(json.dumps({"status":"passed_before_training","protected_files":len(current)}),flush=True)

def verify_sampling(history,data,control_history):
    require(len(history)==len(control_history)==EPOCHS,"Six actual candidate and historical epochs required")
    draws=data["candidate_draws"];original=data["epoch_draws"]
    labels=np.asarray([r["targets"]for r in data["items"]],np.int64)
    domains=np.asarray([DOMAINS[r["domain"]]for r in data["items"]],np.int64)
    require(np.array_equal(draws,original),"Exact original control draw arrays required")
    for e,(row,control)in enumerate(zip(history,control_history)):
        require(row["epoch"]==e+1 and row["sampled_row_indices_sha256"]==hashlib.sha256(draws[e].astype("<i8").tobytes()).hexdigest()
            and control["sampled_row_indices_sha256"]==hashlib.sha256(original[e].astype("<i8").tobytes()).hexdigest(),"Actual candidate/control draw digest differs")
        expected={c:{"positive":int((labels[draws[e],k]==1).sum()),"negative":int((labels[draws[e],k]==0).sum()),"unknown":int((labels[draws[e],k]==-1).sum())}for k,c in enumerate(data["classes"])}
        domain_counts={d:int((domains[draws[e]]==k).sum())for d,k in DOMAINS.items()}
        require(row["sampled_photo_target_counts"]==control["sampled_photo_target_counts"]==expected
            and row["sampled_domain_counts"]==control["sampled_domain_counts"]==domain_counts,"Observed photo target/domain tally differs")
        counts=row["optimizer_step_diagnostics"]
        require(all(type(v)is int and v>=0 for v in counts.values())and counts["attempted_batches"]==DRAW_COUNT//8==row["teacher_forward_batches"]
            and counts["actual_optimizer_steps"]>0 and counts["actual_optimizer_steps"]+counts["amp_skipped_steps"]==counts["attempted_batches"],"Actual optimizer accounting differs")
        rates=[5e-6+(lr-5e-6)*(1+math.cos(math.pi*e/EPOCHS))/2 for lr in(4e-5,1e-4)]
        require(all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15)for a,b in zip(row["optimizer_learning_rates"],rates))
            and row["teacher_state_unchanged"]and row["bn_buffers_unchanged"]and math.isfinite(row["train_loss"]),"Recipe/frozen state differs")
    return {"completed_epochs":len(history),"draws":int(draws.size),"attempted_batches":sum(r["optimizer_step_diagnostics"]["attempted_batches"]for r in history),
        "actual_optimizer_steps":sum(r["optimizer_step_diagnostics"]["actual_optimizer_steps"]for r in history),
        "amp_skipped_steps":sum(r["optimizer_step_diagnostics"]["amp_skipped_steps"]for r in history),
        "exact_original_control_draw_order_and_photo_targets_preserved":True}

def after(protocol):
    require(not(ROOT/OUTPUT).exists(),"Preserve existing verification")
    before_record,ledger=read(ROOT/BEFORE),read(ROOT/LEDGER)
    require(before_record["protocol_sha256"]==sha(ROOT/PROTOCOL)and before_record["input_ledger_sha256"]==sha(ROOT/LEDGER)
        and before_record["test_record_sha256"]==sha(ROOT/TESTS)and before_record["preflight_sha256"]==sha(ROOT/PREFLIGHT),"Before-proof bindings differ")
    current=snapshots(ledger["snapshots_before"]);require(current==ledger["snapshots_before"],"Protected SHA/size/mtime changed")
    verify_git_sources({**protocol["source_sha256"],PROTOCOL:sha(ROOT/PROTOCOL)},before_record["source_git_commit"])
    prior,tests,preflight=prerequisites(protocol);data=load_data();run=ROOT/"runs"/NAME
    training,history=read(run/"TRAINING.json"),read(run/"history.json")
    require(training["status"]=="complete"and training["actual_epochs"]==EPOCHS and training["study_protocol_sha256"]==sha(ROOT/PROTOCOL)
        and training["source_sha256"]==protocol["source_sha256"]and training["source_git_commit"]==before_record["source_git_commit"]
        and training["human_feedback_used"]is False and training["ai_pseudo_labels_created"]==0 and training["original_labels_changed"]is False
        and training["historical_control_retrained"]is False and training["source_test_used"]is False
        and training["original_sampling_unchanged"]is True and training["ai_native_observations_used_as_labels"]is False
        and training["primary_photo_loss_recipe"]==protocol["primary_photo_loss_recipe"],"Actual training/provenance differs")
    measured=verify_sampling(history,data,read(ROOT/"runs"/CONTROL/"history.json"))
    require(training["attempted_batches"]==measured["attempted_batches"]and training["actual_optimizer_steps"]==measured["actual_optimizer_steps"],"Update summary differs")
    mining=[row['spalling_pixel_mining']for row in history]
    require(all(0<r['selected_background_cells']<=r['known_background_cells']and r['foreground_cells_retained']>0 for r in mining),'Actual mining missing')
    selected=min(history,key=lambda r:(r["worst_target_error"],r["sum_target_errors"]));checkpoint=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
    require(training["weights_sha256"]==sha(run/"best.pt")and checkpoint["epoch"]==selected["epoch"]and checkpoint["study_protocol_sha256"]==sha(ROOT/PROTOCOL)
        and checkpoint["architecture"]==ARCH and len(checkpoint["state_dict"])==324 and read(run/"VALIDATION.json")==selected,"Selected actual checkpoint differs")
    model=AuxiliaryClassifier(7,pretrained=False);model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
    original=AuxiliaryClassifier(7,pretrained=False);original.load_state_dict(torch.load(ROOT/INITIAL,map_location="cpu",weights_only=True)["state_dict"],strict=True)
    require(training["initial_bn_buffer_sha256"]==training["final_bn_buffer_sha256"]==batchnorm_state_sha256(original)==batchnorm_state_sha256(model)
        and training["teacher_state_sha256"]==training["final_teacher_state_sha256"]==state_sha256(original.state_dict()),"Original teacher or BN changed")
    for key,expected in(("original_photo_positive_weights",data["sampling_data"]["photo_weights"]),("original_pixel_positive_weights",data["supervision_weights"]["pixel_weights"]),("original_auxiliary_positive_weights",data["supervision_weights"]["auxiliary_weights"])):
        require(training[key]==expected.tolist(),"Original loss weights differ")
    require(all(torch.isfinite(v).all()for v in checkpoint["state_dict"].values()),"Nonfinite selected state")
    with Image.open(ROOT/data["items"][0]["image"])as image:inputs=image_transform(640)(image.convert("RGB"))[None]
    with torch.no_grad():outputs=model.forward_training(inputs);public=model(inputs)
    require([list(x.shape)for x in outputs]==[[1,7],[1,7,80,80],[1,19]]and all(torch.isfinite(x).all()for x in outputs)and torch.equal(public,outputs[0]),"CPU/public contract differs")
    require(not(run/"TARGET-TEST.json").exists()and not list(run.glob("target-test-*.json")),"SourceTEST forbidden in repeated selection")
    proof={"schema":"facility_spalling_ohem_verification_v1","status":"passed","protocol_sha256":sha(ROOT/PROTOCOL),"source_git_commit":before_record["source_git_commit"],
        "source_sha256":protocol["source_sha256"],"runtime_source_count":len(protocol["source_sha256"]),"git_blob_bytes_verified":True,
        "input_ledger_sha256":sha(ROOT/LEDGER),"protected_file_count":len(current),"prior_protected_files_preserved":ledger["prior_protected_files"],
        "all_input_sha_size_mtime_preserved":True,"tests":tests,"preflight":preflight,"actual_sampling":measured,"actual_pixel_mining":mining,"actual_completed_training_epochs":EPOCHS,
        "weights_sha256":sha(run/"best.pt"),"selected_epoch":selected["epoch"],"strict_cpu_324_reload":True,"all_outputs_finite":True,
        "frozen_teacher_and_bn_preserved":True,"human_feedback_used":False,"new_ai_labels":0,"app_model_promoted":False,"source_test_used":False,
        "app_profile_unchanged":True,"app_profile_sha256":sha(ROOT/"reports/facility-inference-profile.json"),"independent_industrial_accuracy_verified":False}
    write_new(ROOT/OUTPUT,proof);print(json.dumps({k:proof[k]for k in("status","actual_completed_training_epochs","weights_sha256","protected_file_count")}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--before",action="store_true");args=parser.parse_args()
    torch.set_num_threads(4);protocol=validate_protocol(read(ROOT/PROTOCOL));(before if args.before else after)(protocol)
