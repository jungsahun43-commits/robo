"""Reuse the immutable prior smoke procedure with explicit v13 differences."""
from pathlib import Path
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_loco_context as study
from scripts import release_ksdd_photo_only as prior
from scripts.fetch_equipment_expansion import sha


def main():
    p=study.validate();assert sha(ROOT/'scripts/release_ksdd_photo_only.py')==p['legacy_source_sha256']['scripts/release_ksdd_photo_only.py']
    target=ROOT/'scripts/release_loco_context.py';assert not target.exists()
    smoke=inspect.getsource(prior.smoke)
    smoke=smoke.replace('safelog_ai.ksdd_photo_only_server','safelog_ai.loco_context_server')
    smoke=smoke.replace('    process = subprocess.Popen(', "    server_env = dict(env); server_env['PATH'] = os.environ.get('PATH', '')\n    process = subprocess.Popen(")
    smoke=smoke.replace('cwd=ISOLATED, env=env, stdout=log', 'cwd=ISOLATED, env=server_env, stdout=log')
    smoke=smoke.replace('study.VALIDATION',"(REPORTS/'industrial-ksdd-supervised-v10-validation.json')")
    smoke=smoke.replace("REPORTS/'industrial-night-v9-release.json'","REPORTS/'industrial-ksdd-photo-only-v11-release.json'")
    smoke=smoke.replace("elif 'inferencePolicySha256' in old_replies[task]:","elif task != 'loco_splicing_connectors' and 'inferencePolicySha256' in old_replies[task]:")
    smoke=smoke.replace("('ksdd2_surface', 'ball_screw_pitting', 'mpdd_bracket_white')","('ksdd2_surface', 'ball_screw_pitting', 'loco_splicing_connectors')")
    smoke=smoke.replace("assert policies['supervisedPhoto']['ksdd2_surface']['originalLocalizationChanged'] is False", "assert policies['supervisedPhoto']['ksdd2_surface']['originalLocalizationChanged'] is False\n        assert policies['globalContext']['loco_splicing_connectors']['semanticClampWireRelationsVerified'] is False")
    smoke=smoke.replace("'industrial-ksdd-photo-only-v11-release.json'", "'industrial-loco-context-v13-release.json'")
    # The baseline lookup must still read the immutable prior, not our new report.
    smoke=smoke.replace("old_replies = {r['task']: r['result'] for r in study.read(REPORTS/'industrial-loco-context-v13-release.json')['hazard_replies']}", "old_replies = {r['task']: r['result'] for r in study.read(REPORTS/'industrial-ksdd-photo-only-v11-release.json')['hazard_replies']}")
    source='''"""Selected context handoff; all prior API aliases and real C++ calls."""
import argparse,base64,hashlib,io,json,os,shutil,socket,subprocess,sys,time,zipfile
from pathlib import Path
from urllib.error import HTTPError
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_loco_context as study
from scripts import study_ksdd_supervised as trained
from scripts.fetch_equipment_expansion import sha
from scripts.release_unified_main import local_links
from scripts.release_metal_calibration import write,http
from safelog_ai.equipment_expansion import CATEGORIES
from safelog_ai.industrial_supervised import WEAR_CATEGORY
RUN,REPORTS=study.RUN,study.REPORTS;REPO=ROOT.parents[1];OUTPUTS=REPO.parents[1]/'outputs'
STAGE=RUN/'handoff-stage';ISOLATED=RUN/'isolated-zip'
ARCHIVE=OUTPUTS/'unified-loco-context-role4-handoff-2026-10-15-v13.zip'

def checked():
    study.validate();choice=study.read(REPORTS/'industrial-loco-context-v13-selection.json')
    assert choice['selected'] and sha(ROOT/choice['checkpoint_path'])==choice['checkpoint_sha256']
    assert study.read(REPORTS/'industrial-loco-context-v13-verification.json')['status']=='passed'
    assert all(study.read(REPORTS/'industrial-loco-context-v13-validation.json')['candidates'][choice['selected']]['guards'].values())
    return choice

def package():
    choice=checked();assert not STAGE.exists() and not ARCHIVE.exists();STAGE.mkdir()
    shutil.copy2(ROOT/choice['checkpoint_path'],STAGE/'model.pt');shutil.copy2(RUN/'evaluation.json',STAGE/'evaluation.json')
    write(STAGE/'model.json',{'path':'model.pt','sha256':choice['checkpoint_sha256'],'checkpoint_count':1,'backbone_count':1,
        'context_method':choice['selected'],'actual_context_optimizer_updates':1920,'selected_epoch_by_validation':29,
        'semantic_clamp_wire_relations_verified':False,'known_TEST_reused_for_development':True,'independent_field_validated':False,'app_default_promoted':False})
    write(STAGE/'safelog_ai/__init__.py','')
    previous=ROOT/'runs/industrial-ksdd-photo-only-v11/handoff-stage'
    for name in [p.name for p in (previous/'safelog_ai').glob('*.py') if p.name!='__init__.py']+['loco_context.py','loco_context_server.py']:
        shutil.copy2(ROOT/'safelog_ai'/name,STAGE/'safelog_ai'/name)
    shutil.copy2(previous/'requirements-api.txt',STAGE/'requirements-api.txt')
    write(STAGE/'README_KO.md','# SafeLog 전선 연결부 전체 문맥 보강\\n\\n한 모델·한 공통 분석부. 사진 미탐39.90%는 여전히20% 목표 미달이다. 기존 위치 지도와 다른 설비를 보존한다.\\n\\n- [역할4 실행 안내](safelog-cpp/docs/LOCO_CONTEXT_HANDOFF_KO.md)\\n- [실제 학습과 결과](safelog-cpp/ai-training/reports/LOCO_CONTEXT_RESULTS_KO.md)\\n')
    queue=[REPO/'safelog-cpp/docs/LOCO_CONTEXT_HANDOFF_KO.md',REPORTS/'LOCO_CONTEXT_RESULTS_KO.md'];seen=set()
    while queue:
        source=queue.pop().resolve()
        if source in seen:continue
        assert source.is_file() and source.is_relative_to(REPO);seen.add(source)
        output=STAGE/source.relative_to(REPO);output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,output)
        if source.suffix=='.md':queue.extend(local_links(source))
    for path in STAGE.rglob('*.md'):
        assert all(p.is_file() and p.is_relative_to(STAGE) for p in local_links(path))
    write(STAGE/'package-sha256.json',{p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()})
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as out:
        for path in sorted(STAGE.rglob('*')):
            if path.is_file():out.write(path,path.relative_to(STAGE).as_posix())
    print({'archive':str(ARCHIVE),'sha256':sha(ARCHIVE),'bytes':ARCHIVE.stat().st_size},flush=True)

'''+smoke+'''
def cpu():
    checked();assert ISOLATED.exists()
    copied=RUN/'cpu-known-test.py';assert not copied.exists();shutil.copy2(ROOT/'scripts/verify_loco_context_cpu.py',copied)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,str(copied),str(ROOT),str(REPORTS/'industrial-loco-context-v13-CPU.json')],cwd=ISOLATED,env=env,capture_output=True,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    with (RUN/'cpu-known-test.log').open('x',encoding='utf-8') as out:out.write(result.stdout.decode('utf-8',errors='replace')+result.stderr.decode('utf-8',errors='replace'))
    print(result.stdout.decode('utf-8',errors='replace'),flush=True);assert result.returncode==0,result.stderr.decode(errors='replace')

if __name__=='__main__':
    torch.set_num_threads(4);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['package','smoke','cpu']);args=parser.parse_args();globals()[args.mode]()
'''
    compile(source,str(target),'exec')
    with target.open('x',encoding='utf-8',newline='\n') as out:out.write(source)
    print({'release_script_created':True,'sha256':sha(target)},flush=True)


if __name__=='__main__':main()
