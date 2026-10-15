"""One frozen rust-mask transfer hypothesis, with current TEST-safe calibration."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from safelog_ai.corrosion_review_gate import mask_photo_score,corroborated,passes_guard
from safelog_ai.insplad_pixel_groups import Groups
from safelog_ai.native_component_views import normalized
from safelog_ai.metal_reference import normal_budget_threshold
from safelog_ai.equipment_anomaly import classification_rates
from safelog_ai.retention_distillation import state_sha256
from safelog_ai.process_probe import process_exists
from scripts import study_ostrava_rust_mask as source
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read

RUN=ROOT/'runs/industrial-corrosion-review-v63'
PROTOCOL=ROOT/'reports/industrial-corrosion-review-v63-protocol.json'
MANIFEST=RUN/'manifest.json'
OLD=ROOT/'data/industrial-insplad-grouped-v44'
V57=ROOT/'data/industrial-insplad-lower-shackle-v57'
META=ROOT/'data/industrial-insplad-unsupervised-v56-metadata/all-rows.json'
ELIGIBLE=ROOT/'runs/industrial-insplad-remaining-labels-v59/retained-supervised-originals.json'
BASE_SCORES=ROOT/'runs/industrial-insplad-corrosion-v45/test-scores.json'
BASE_CAL=ROOT/'runs/industrial-insplad-corrosion-v45/calibration.json'
SOURCES=('scripts/study_corrosion_review_gate.py','safelog_ai/corrosion_review_gate.py','tests/test_corrosion_review_gate.py')


def utc():return datetime.now(timezone.utc).isoformat()


def old_groups():
    rows=read(META)['rows'];old=read(OLD/'grouped-all-source.json')['rows'];held=read(OLD/'retained-TEST-fingerprints.json')['rows']
    a,b=len(rows),len(rows)+len(old);g=Groups(b+len(held));oldgroup={};newgroup={}
    for j,r in enumerate(old):
        key=r['group_id']
        if key in oldgroup:g.union(a+j,oldgroup[key])
        else:oldgroup[key]=a+j
    for i,r in enumerate(rows):
        key=r['group_proxy']
        if key in newgroup:g.union(i,newgroup[key])
        else:newgroup[key]=i
        for key in r['old_source_group_links']:g.union(i,oldgroup[key])
    edges=[json.loads(v) for v in (V57/'pixel-joining-edges.jsonl').read_text(encoding='utf-8').splitlines()];assert len(edges)==94
    for i,j,*_ in edges:g.union(i,j)
    keys=['unsupervised:'+r['member'] for r in rows]+['supervised:'+r['member'] for r in old]+['held:'+r['image'] for r in held];canonical={}
    for i,key in enumerate(keys):root=g.find(i);canonical[root]=min(canonical.get(root,key),key)
    heldroots={g.find(i) for i,r in enumerate(rows) if r['publisher_role']=='test'}|{g.find(a+j) for j,r in enumerate(old) if r['publisher_role']=='val'}|{g.find(i) for i in range(b,b+len(held))}
    return {r['member']:canonical[g.find(a+j)] for j,r in enumerate(old)},{canonical[v] for v in heldroots}


def freeze():
    assert not RUN.exists() and not PROTOCOL.exists()
    assert read(source.RUN/'verification.json')['status']=='passed' and read(source.RUN/'device-compatibility.json')['status']=='passed'
    assert read(source.RUN/'selection.json')['source_research_gate_passed']
    assert sha(source.RUN/'trained/final-delta.pt')=='26c26bc5739930f153d1553b9b22840f4e83ba925cddeca5f46eee4f9f25a01b'
    groups,held=old_groups();prior=read(OLD/'manifest.json')['rows'];byid={r['id']:r for r in prior}
    eligible=[r for r in read(ELIGIBLE)['rows'] if r['asset']=='polymer-insulator-upper-shackle' and r['condition']=='normal' and r['prior_role']=='cal']
    assert len(eligible)==75
    cal=[]
    for r in eligible:
        row=byid[r['id']];assert row['phase']=='cal' and row['label']==0 and groups[row['member']]==r['current_pixel_group_id'] and groups[row['member']] not in held
        cal.append({**row,'phase':'cal','current_group_id':groups[row['member']]})
    test=[{**r,'current_group_id':groups[r['member']]} for r in prior if r['phase']=='test']
    assert len(test)==64 and sum(r['label']==0 for r in test)==31 and sum(r['label']==1 for r in test)==33
    assert {r['current_group_id'] for r in cal}.isdisjoint({r['current_group_id'] for r in test})
    scores=read(BASE_SCORES);assert set(scores['ids'])=={r['id'] for r in test}
    oldscore=dict(zip(scores['ids'],scores['scores']));cut=read(BASE_CAL)['cut']
    base=classification_rates([r['label'] for r in test],[oldscore[r['id']] for r in test],cut)
    assert base['fp']==6 and base['fn']==0
    RUN.mkdir();new(MANIFEST,{'rows':sorted(cal+test,key=lambda r:(r['phase'],r['id'])),'original_TEST64_unchanged':True})
    files=(ELIGIBLE,OLD/'manifest.json',OLD/'grouped-all-source.json',OLD/'retained-TEST-fingerprints.json',META,V57/'pixel-joining-edges.jsonl',BASE_SCORES,BASE_CAL,
           source.PROTOCOL,source.RUN/'trained/final-delta.pt',source.RUN/'verification.json',source.RUN/'device-compatibility.json')
    p={'UTC':utc(),'fixed_before_new_target_features_scores':True,'source_sha256':{n:sha(ROOT/n) for n in SOURCES},
       'files_sha256':{v.relative_to(ROOT).as_posix():sha(v) for v in files},'manifest_sha256':sha(MANIFEST),
       'trained_source_delta_sha256':sha(source.RUN/'trained/final-delta.pt'),'selected_source_epoch':read(source.RUN/'trained/training.json')['selected_epoch'],
       'target':'only original cropped InsPLAD polymer-insulator-upper-shackle visible normal/rust, no genericmachine/electricalhealth or targetpixelGT',
       'source':'v62 actualcorrosionmask58positiveFITphotos, selectedbyOstravaVAL, no new target training/optimizer or normalizationrefit',
       'data':'alloriginal64publisherVAL TEST(31normal/33rust) kept. CurrentnewTEST/oldVAL/old8995TEST-linkedcal excluded percompletedv59graph;only75currenteligible originalnormalCAL, no rolechanges or finalFIT',
       'input':'newmask fullRGB bicubic512square ImageNet, frozenv62 normalization; oldparent freshRGBbicubic256 ImageNet. Two views/encoderpasses share ONE encoder instance/model, no newbackbone/embedding/crop fromGT.',
       'mask_score':'mean topceil(.02*512*512)=5243 sigmoid512pixels; no sourcecut reusedasphoto cut, no targetmaskGT, no new defaultlocationmap',
       'CAL':'normal75 only alpha.15 rankceil((75+1)*.85)=65 strictscore>cut savedbeforeTESTphotos/features/scores; no other quantile/pooling trial',
       'one_candidate':'ONLY AND(parentoriginalsigmoid>originalcut.015575625002384186,maskscore>newCALcut). Existingparentalias/cut/weights/outputs stay available. No standalone newmask or multiple combination TEST search.',
       'parent_validation':'fresh64parent256scores must match storedv45scores atol5e-4/rtol1e-3 and decisions exact, baseline6FP/31normal and0FN/33rust; stop onmismatch, not retune',
       'guards':['source selectedepoch>0','TESTeachlabel>=20 andcurrentgroups>=3','candidatephotoFPandFN<=20percent','FPR decrease>=5percentagepoints vsparent','FNR increase<=5percentagepoints vsparent'],
       'nondefault':'even ifpointguardpasses, no automaticoldaliasreplacement; optionalresearchalias onlyafter ownnewZIP CPU_freshGPU_alloldalias_oldPNG_actualCPP checks',
       'location':'sourceRustmap maycorroboratevisualevidence but targetlocationGTabsent; no validatedtargetlocation or defaultprimaryPNG replacement',
       'preservation':'v62allstateSHA before/aftertargetinference exact; originalv16+approvedthreehead buffers/meta remainexact, modelparameter updates0; allsource andoldhandoffZIP preserved',
       'storage':'scorejson/manifest only<8MiB, no targetdensecache/fullmodelduplicate,>=1GiBreserve',
       'license':'InsPLADMendeleyV1 CC BY-NC3.0 author/source/version/changes plusOstravaZenodoV2 CC BY4.0, otherparentlicensesunchanged',
       'limits':'publishedTEST seenbefore forparent andpublicdevelopment; newTEST priorFITproxyexposure unresolved, date/background correlation, no independentphysical/siteproof or0FNguarantee',
       'new_units_passed':3,'new_optimizer_updates':0,'independent_field_validation':False,'automatic_app_adoption':False}
    assert shutil.disk_usage(ROOT).free>1024**3+8*1024**2
    new(PROTOCOL,p);print({'protocol_sha256':sha(PROTOCOL),'manifest_sha256':sha(MANIFEST),'normalCAL':75,'originalTEST':64},flush=True)


def validate():
    p=read(PROTOCOL)
    for n,h in {**p['source_sha256'],**p['files_sha256']}.items():assert sha(ROOT/n)==h,n
    assert sha(MANIFEST)==p['manifest_sha256']
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,PROTOCOL.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:safelog-cpp/ai-training/'+n])).hexdigest()==sha(ROOT/n),n
    return p


def infer(rows,model,parent=False):
    maskvalues=[];parentvalues=[]
    with torch.inference_mode():
        for i in range(0,len(rows),4):
            images=[];old=[]
            for r in rows[i:i+4]:
                assert sha(ROOT/r['image'])==r['image_sha256']
                with Image.open(ROOT/r['image']) as im:
                    rgb=im.convert('RGB');images.append(normalized(rgb.resize((512,512),Image.Resampling.BICUBIC)))
                    if parent:old.append(normalized(rgb.resize((256,256),Image.Resampling.BICUBIC)))
            probability=model.forward_ostrava_mask(torch.stack(images).cuda()).sigmoid();maskvalues.extend(mask_photo_score(probability).cpu().tolist())
            if parent:parentvalues.extend(model.forward_insplad_upper_shackle(torch.stack(old).cuda()).sigmoid().cpu().tolist())
    return maskvalues,parentvalues


def run():
    p=validate();assert not (RUN/'selection.json').exists()
    model,_,_=source.load(delta=True);before=state_sha256(model);rows=read(MANIFEST)['rows'];cal=[r for r in rows if r['phase']=='cal'];test=[r for r in rows if r['phase']=='test']
    if (RUN/'calibration.json').exists():calibration=read(RUN/'calibration.json')
    else:
        scores,_=infer(cal,model);cut,rank=normal_budget_threshold(scores,alpha=.15)
        new(RUN/'calibration-scores.json',{'ids':[r['id'] for r in cal],'scores':scores})
        calibration={'cut':cut,'rank':rank,'normal_photos':75,'source_delta_sha256':p['trained_source_delta_sha256'],'protocol_sha256':sha(PROTOCOL),'new_TEST_scores_before_this':0}
        new(RUN/'calibration.json',calibration)
    assert calibration['source_delta_sha256']==sha(source.RUN/'trained/final-delta.pt') and calibration['protocol_sha256']==sha(PROTOCOL)
    scores,parent=infer(test,model,parent=True);stored=read(BASE_SCORES);baseline=dict(zip(stored['ids'],stored['scores']));expected=torch.tensor([baseline[r['id']] for r in test]);actual=torch.tensor(parent);oldcut=read(BASE_CAL)['cut']
    assert torch.allclose(actual,expected,atol=5e-4,rtol=1e-3) and torch.equal(actual>oldcut,expected>oldcut),'Preserve parentmismatch; no cutadjustment'
    labels=[r['label'] for r in test];base=classification_rates(labels,parent,oldcut);assert base['fp']==6 and base['fn']==0
    decisions=corroborated(actual,oldcut,torch.tensor(scores),calibration['cut']);rates=classification_rates(labels,decisions.float().tolist(),.5)
    groups={str(y):len({r['current_group_id'] for r in test if r['label']==y}) for y in (0,1)};passed=passes_guard(base,rates,groups,p['selected_source_epoch'])
    after=state_sha256(model);assert before==after
    result={'UTC':utc(),'source_delta_sha256':p['trained_source_delta_sha256'],'model_state_before_after_exact':True,'combined_state_sha256':before,
        'baseline_photo_rates':base,'candidate_photo_rates':rates,'groups_each_label':groups,'parent_stored_fresh_decisions_exact':True,
        'parent_maximum_score_difference':float((actual-expected).abs().max()),'new_CAL_cut':calibration['cut'],'one_AND_candidate_point_guard_passed':passed,
        'FPR_reduction_percentagepoints':100*(base['fpr']-rates['fpr']),'FNR_increase_percentagepoints':100*(rates['fnr']-base['fnr']),
        'new_optimizer_updates':0,'fresh_target_mask_encoder_inputs':len(cal)+len(test),'fresh_parent_encoder_inputs':len(test),
        'encoder_instances':1,'new_API_CPP_ZIP_checks':0,'new_CPU_GPU_package_checks':0,'automatic_old_alias_replacement':False,
        'target_pixel_localization_verified':False,'independent_field_validation':False}
    new(RUN/'test-scores.json',{'ids':[r['id'] for r in test],'parent_scores':parent,'mask_scores':scores,'AND_decisions':decisions.tolist()})
    new(RUN/'selection.json',result);new(ROOT/'reports/industrial-corrosion-review-v63-selection.json',result);print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job '+str(path))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:run()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'UTC':utc(),'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
