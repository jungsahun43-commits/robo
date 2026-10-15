"""Small parent-bound research overlay; fresh device/old-route/C++ proofs."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import study_insplad_corrosion as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from safelog_ai.process_probe import process_exists
from safelog_ai.insplad_server import CATEGORY
from safelog_ai.real_mask_server import RealMaskEngine
from safelog_ai.equipment_expansion import CATEGORIES
from safelog_ai.industrial_supervised import WEAR_CATEGORY

RUN=ROOT/'runs/industrial-insplad-corrosion-release-v46-r1'
POLICY=ROOT/'reports/industrial-insplad-corrosion-v46-release-policy-r1.json'
OUTPUTS=ROOT.parents[3]/'outputs'
ARCHIVE=OUTPUTS/'insplad-corrosion-role4-research-overlay-2026-10-15-v45-r1.zip'
BASE=ROOT/'runs/industrial-magnetic-tiles-v16/isolated-zip-r1'
ISOLATED=RUN/'isolated-overlay'
SOURCES=('safelog_ai/insplad_server.py','scripts/release_insplad_corrosion_r1.py','scripts/verify_insplad_bundle.py','tests/test_insplad_server_contract.py','../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp','../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp','../apps/mobile-qt/tests/http_ai_safety_analyzer_tests.cpp','../apps/mobile-qt/tools/unified_main_probe.cpp')


def freeze():
    assert not RUN.exists() and not POLICY.exists() and not ARCHIVE.exists()
    study.validate();selection=read(study.RUN/'selection.json')
    assert selection['source_candidate_eligible_pending_package_checks'] and read(study.RUN/'verification.json')['status']=='passed'
    assert sha(BASE/'model.pt')==study.EXPECTED
    RUN.mkdir()
    policy={'UTC':datetime.now(timezone.utc).isoformat(),'fixed_before_packaged_CPU_GPU_or_API_checks':True,
            'sources':{n:sha(ROOT/n) for n in SOURCES},'training_protocol_sha256':sha(study.PROTOCOL),
            'delta_sha256':sha(study.RUN/'trained/final-delta.pt'),'parent_sha256':study.EXPECTED,
            'selection_sha256':sha(study.RUN/'selection.json'),'calibration_sha256':sha(study.RUN/'calibration.json'),
            'overlay_only':'small delta+Python+calibration+source evaluation; originalv16-r1 model.pt/evaluation.json required, no duplicate435MBmodel/densecache',
            'identity':'API modelSha256 is delta-bound combinedtensorstateSHA, NOT model.pt fileSHA; parentCheckpointSha256 anddeltaSha256 separate; still one active sharedencoder/model',
            'device_guard':'all64originalRGB freshCPU/freshGPU/cachedGPU decisions exact; probability atol5e-4 rtol1e-3 fixed before newCPU/GPU comparisons; unchanged threshold',
            'old_aliases':'all15 existingtaskaliases one firstFIT each CPU; exact classResults,policySHA andallprimary/auxiliaryPNG strings compared with unchangedparentCPU; no claim these are15accuracy evaluations',
            'CPP':'fresh separately compiledQtprobe with explicit upper-shackle allowlist, currentoldKS2/BS/LOCO/blackbracket andnewupper-shackle cases,3operations each=15newoperations; outputfiles andexit0 required; no pastchecks reused',
            'input_guard':'explicitnewtaskalias or matched equipment_surface/category; genericunknown/mismatchedcategory422; cropminside64. Domain manuallyselected, not automaticallyverified.',
            'scope':'sourcephoto20percent benchmark only; date-labelcorrelation/physicalparts unverified; no overallmachinehealth/location/independentfieldclaim',
            'location':'newclassifier no map; allold exposedmap outputs preserved; no synthetic/CAM/mask promotion',
            'release_guard':'ZIPCRC/memberSHA,sourceCPU_GPU,currentCPU16aliases,CPP15operations andoldPNGexact allpass before researchhandoff; existingdefaultv16-r1 not automaticallyswitched',
            'storage':'newoverlayZIP/isolatedcode each<32MiB,sourceproofs<8MiB,existingparentusedread-only,>=1GiBfree; no deletions',
            'new_optimizer':0,'new_units_passed':3,'APK_or_fresh_pip_install_tested':False}
    new(POLICY,policy);print({'policy_sha256':sha(POLICY),'new_optimizer':0},flush=True)


def validate():
    p=read(POLICY)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    assert sha(study.RUN/'trained/final-delta.pt')==p['delta_sha256']
    assert sha(study.RUN/'calibration.json')==p['calibration_sha256'] and sha(BASE/'model.pt')==study.EXPECTED
    git=['C:/Program Files/Git/cmd/git.exe','-c','safe.directory='+ROOT.parents[1].as_posix(),'-C',str(ROOT.parents[1])]
    for n in (*SOURCES,POLICY.relative_to(ROOT).as_posix()):
        assert hashlib.sha256(subprocess.check_output(git+['show','HEAD:'+(ROOT/n).resolve().relative_to(ROOT.parents[1]).as_posix()])).hexdigest()==sha(ROOT/n),n
    return p


def package():
    validate();assert not ARCHIVE.exists() and not ISOLATED.exists()
    assert shutil.disk_usage(ROOT).free>1024**3+100*1024**2
    payload=torch.load(study.RUN/'trained/final-delta.pt',map_location='cpu',weights_only=True)
    files={f'safelog_ai/{path.name}':path for path in (ROOT/'safelog_ai').glob('*.py')}
    files.update({'insplad-delta.pt':study.RUN/'trained/final-delta.pt','calibration.json':study.RUN/'calibration.json',
                  'source-evaluation.json':study.RUN/'selection.json','training-protocol.json':study.PROTOCOL,
                  'release-policy.json':POLICY,'verify_bundle.py':ROOT/'scripts/verify_insplad_bundle.py',
                  'requirements-api.txt':BASE/'requirements-api.txt'})
    for cpp in ('adapters/ai/http_ai_safety_analyzer.cpp','adapters/ai/http_ai_safety_analyzer.hpp','tests/http_ai_safety_analyzer_tests.cpp','tools/unified_main_probe.cpp'):
        files['safelog-cpp/apps/mobile-qt/'+cpp]=ROOT.parent/'apps/mobile-qt'/cpp
    metadata={'schema':'safelog.research-overlay.v45','parent_file_sha256':study.EXPECTED,
              'required_parent_bundle_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
              'delta_sha256':sha(files['insplad-delta.pt']),'combined_tensor_state_sha256':payload['combined_state_sha256'],
              'one_shared_encoder':True,'automatic_default_replacement':False,'source_scope':'cropped_polymer_insulator_upper_shackle_visible_rust',
              'source_fpr':6/31,'source_fnr':0.0,'independent_field_validation':False}
    guide='''# 역할4: 연결고리 부식 연구 추가 기능 v45

기존 v16-r1 ZIP의 model.pt와 evaluation.json이 필요합니다. 이 ZIP은 작은 추가 분류부이며 단독 앱/단독 모델이 아닙니다. 같은 공통 encoder를 사용합니다.

기존 ZIP과 이 ZIP을 각각 다른 폴더에 풉니다. 이 ZIP 폴더에서 기존 Python 환경을 사용해 실행합니다. 아래 /BASE는 실제 기존 ZIP 폴더로 바꿉니다. 설치 의존성은 requirements-api.txt이며 이번 검사는 새 pip 설치나 Android APK 빌드를 포함하지 않습니다.

```powershell
$overlay = Get-Content -Raw overlay.json | ConvertFrom-Json
python -m safelog_ai.insplad_server --parent /BASE/model.pt --parent-sha $overlay.parent_file_sha256 --delta insplad-delta.pt --delta-sha $overlay.delta_sha256 --calibration calibration.json --evaluation /BASE/evaluation.json --port 8775
```

C++ 앱에서는 이 ZIP의 safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp를 같은 프로젝트 경로에 반영하고 다시 빌드해야 합니다. 새 항목을 허용 목록에 추가한 변경이며 기존 15개 항목은 유지합니다. 테스트 파일도 함께 제공합니다.

C++의 SAFELOG_AI_BASE_URL은 http://127.0.0.1:8775, inspectionTask는 insplad_upper_shackle_rust입니다. equipment_surface + 같은 equipmentCategory도 가능합니다. modelSha256은 기존 파일 SHA가 아니라 overlay.json의 combined_tensor_state_sha256입니다. 원본 파일과 추가 파일 SHA는 별도 필드로 기록합니다. 기존 15개 alias도 같은 서버에서 유지합니다.

입력은 고분자 애자의 상부 금속 연결고리를 자른 사진이어야 합니다. 일반 컨베이어·다른 금속 부품·전력선 전체 사진의 부식 검출 모델로 쓰지 마세요. 입력 종류의 자동 판별은 구현하지 않았습니다. 결과는 사진상 부식 후보이며 점검자가 확인합니다. confirmed pixel 위치나 기계/전기 건강 판단은 제공하지 않습니다. 보고서 요약은 기존 규칙 템플릿입니다.

원저자 평가 정상31/부식33장에서 FP6/FN0, 즉 오탐19.35%/미탐0%였습니다. CAL 정상108장의 고정 rank93 기준을 사용합니다. 촬영 시기와 라벨의 상관, 다른 프레임의 같은 실제 부품 가능성이 남아 있습니다. 작은 공개 출처 평가 결과를 실제 현장 오차율이나 0미탐 보장으로 설명하지 않습니다. 기존 검정 브래킷/튜브 성능은 개선했다고 측정하지 않았습니다.

InsPLAD Mendeley V1 DOI10.17632/5n3fjgvfyz.1, 원저자 논문 DOI10.1080/01431161.2023.2283900, CC BY-NC3.0 비상업 학술 연구/인용/출처·버전·변경 표시를 유지합니다. 기존 모델의 다른 자료 이용조건도 유지합니다. 이 전달물은 추가 분류부와 명시적인 API 경로를 추가한 변경입니다.
'''
    # rank is computed from the stored calibration, never from evaluation scores.
    guide=guide.replace('rank93',f"rank{read(study.RUN/'calibration.json')['rank']}")
    new(RUN/'overlay.json',metadata);(RUN/'README_KO.md').write_text(guide,encoding='utf-8')
    files.update({'overlay.json':RUN/'overlay.json','README_KO.md':RUN/'README_KO.md'})
    hashes={n:sha(path) for n,path in files.items()};new(RUN/'package-sha256.json',hashes);files['package-sha256.json']=RUN/'package-sha256.json'
    with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for name,path in sorted(files.items()):archive.write(path,name)
    assert ARCHIVE.stat().st_size<32*1024**2
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(archive.read(n)).hexdigest()==h
        for name in archive.namelist():assert (ISOLATED/name).resolve().is_relative_to(ISOLATED.resolve())
        archive.extractall(ISOLATED)
    new(RUN/'package.json',{'archive':str(ARCHIVE),'sha256':sha(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'members':len(files),
                          'no_parent_checkpoint_duplicated':True,'CPU_CPP_checked_yet':False,'combined_tensor_state_sha256':metadata['combined_tensor_state_sha256']})


def smoke():
    p=validate();package_record=read(RUN/'package.json');assert sha(ARCHIVE)==package_record['sha256']
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    engine={'parent':str(BASE/'model.pt'),'parent_sha':study.EXPECTED,'delta':str(ISOLATED/'insplad-delta.pt'),
            'delta_sha':p['delta_sha256'],'calibration':str(ISOLATED/'calibration.json'),'evaluation':str(BASE/'evaluation.json')}
    rows=study.rows_for('test')
    new(RUN/'device-inputs.json',{'engine':engine,'rows':[{**r,'image':str(ROOT/r['image'])} for r in rows],
                                'cached_scores':read(study.RUN/'test-scores.json')['scores']})
    with (RUN/'device-check.log').open('x',encoding='utf-8') as log:
        device=subprocess.run([sys.executable,'verify_bundle.py','--inputs',str(RUN/'device-inputs.json'),'--output',str(RUN/'device-proof.json')],cwd=ISOLATED,env=env,stdout=log,stderr=log,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert device.returncode==0,(RUN/'device-check.log').read_text(encoding='utf-8')
    deviceproof=read(RUN/'device-proof.json');assert deviceproof['status']=='passed'
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8775))!=0,'Preserve unrelated server'
    command=[sys.executable,'-m','safelog_ai.insplad_server']
    for key,value in engine.items():command.extend(['--'+key.replace('_','-'),value])
    command.extend(['--port','8775'])
    log=(RUN/'server.log').open('x',encoding='utf-8')
    server=subprocess.Popen(command,cwd=ISOLATED,env=env,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    def http(path,body=None):
        req=Request('http://127.0.0.1:8775'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=45) as response:return json.loads(response.read())
    proofs=[];cpp=[];pngs=0
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError((RUN/'server.log').read_text(encoding='utf-8'))
            try:health=http('/health');break
            except (OSError,ValueError):time.sleep(.25)
        else:raise RuntimeError('Packaged server timed out')
        assert health['modelSha256']==package_record['combined_tensor_state_sha256'] and health['neuralModelCount']==health['sharedBackboneCount']==1
        parent=RealMaskEngine(BASE/'model.pt',study.EXPECTED,BASE/'evaluation.json')
        old=read(ROOT/'data/unified-main-v1/manifest.json')
        photos={'facility_surface':ROOT/old['fit']['facility_surface'][0]['image'],
                'belt_damage':ROOT/'runs/belt-author-embedded-sample.jpg','conveyor_foreign':ROOT/old['fit']['conveyor_foreign'][0]['image']}
        oldrows=read(ROOT/'data/equipment-expansion-v1/manifest.json')['rows']
        for c in CATEGORIES:photos[c]=ROOT/next(r['image'] for r in oldrows if r['phase']=='fit' and r['category']==c)
        photos[WEAR_CATEGORY]=ROOT/next(r['image'] for r in read(ROOT/'data/industrial-bsclass-v1/manifest.json')['rows'] if r['phase']=='fit' and not r['label'])
        assert len(photos)==15
        for task,photo in photos.items():
            encoded=base64.b64encode(photo.read_bytes()).decode();prior=parent.analyze(task,encoded)
            current=http('/v1/analyze-hazard',{'inspectionTask':task,'image':encoded,'promptVersion':'insplad-v46-compat'})
            assert current['classResults']==prior['classResults'] and current['candidateDetected']==prior['candidateDetected'],task
            assert current.get('inferencePolicySha256')==prior.get('inferencePolicySha256'),task
            for key,value in prior.items():
                if key.endswith('PngBase64'):assert current[key]==value,(task,key);pngs+=1
            proofs.append({'task':task,'classResults_and_policy_exact':True,'original_PNG_exact':True})
            print({'old_CPU_alias':task},flush=True)
        del parent
        selected=next(r for r in study.rows_for('fit') if r['label']==1);photo=ROOT/selected['image'];encoded=base64.b64encode(photo.read_bytes()).decode()
        newreply=http('/v1/analyze-hazard',{'inspectionTask':CATEGORY,'image':encoded,'promptVersion':'insplad-v46'})
        explicit=http('/v1/main/analyze',{'inspectionTask':'equipment_surface','equipmentCategory':CATEGORY,'image':encoded,'promptVersion':'insplad-v46'})
        assert newreply['classResults']==explicit['classResults'] and not newreply['localizationAvailable'] and not newreply['inputDomainAutoVerified']
        assert newreply['modelSha256']==package_record['combined_tensor_state_sha256'] and newreply['humanConfirmationRequired']
        for body in ({'inspectionTask':'all_rust'},{'inspectionTask':CATEGORY,'equipmentCategory':'ad_screw'}):
            try:http('/v1/analyze-hazard',{**body,'image':encoded,'promptVersion':'negative-contract'})
            except HTTPError as error:assert error.code==422
            else:raise AssertionError('Unsupported input must be rejected')
        cppenv=dict(env);cppenv.update(QT_QPA_PLATFORM='offscreen',PATH='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH',''))
        photos[CATEGORY]=photo
        for task in ('ksdd2_surface',WEAR_CATEGORY,'loco_splicing_connectors','mpdd_bracket_black',CATEGORY):
            output=RUN/(task+'-cpp.json')
            cp=subprocess.run([str(RUN/'qt-probe/build/unified_probe.exe'),task,str(photos[task]),package_record['combined_tensor_state_sha256'],str(output)],cwd=ISOLATED,env=cppenv,capture_output=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
            if cp.returncode!=0:
                new(RUN/(task+'-cpp-failure.json'),{'returncode':cp.returncode,'stdout':cp.stdout.decode(errors='replace'),'stderr':cp.stderr.decode(errors='replace')})
            assert cp.returncode==0 and output.exists(),task
            result=read(output);assert result['operations']==3 and result['samePhotoDoesNotResolve'];cpp.append(result)
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close()
    result={'status':'passed','overlay_archive':str(ARCHIVE),'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
            'effective_combined_tensor_state_sha256':package_record['combined_tensor_state_sha256'],'parent_sha256':study.EXPECTED,'delta_sha256':p['delta_sha256'],
            'fresh_packaged_RGB_CPU_GPU_cached_decisions_exact':64,'all_old_CPU_aliases_checked':15,'new_CPU_aliases_checked':1,'explicit_new_equipment_category_checked':True,
            'old_classResults_and_policy_exact':True,'old_PNG_strings_exact':pngs,'current_real_CPP_operations':sum(v['operations'] for v in cpp),
            'CPP_cases':cpp,'old_alias_proofs':proofs,'unknown_or_mismatched_inputs_rejected422':True,'owned_server_stopped':server.poll() is not None,
            'previous_v46_failure_preserved':True,'new_optimizer':0,'one_shared_encoder':True,'parent_checkpoint_duplicated':False,'default_v16_automatically_replaced':False,
            'independent_field_validation':False,'APK_built':False,'fresh_pip_install_tested':False}
    new(RUN/'release.json',result);new(ROOT/'reports/industrial-insplad-corrosion-v46-release-r1.json',result)
    print({k:v for k,v in result.items() if k not in ('CPP_cases','old_alias_proofs')},flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','run'));args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        validate()
        for path in (ROOT/'runs').glob('**/*.lock'):
            if process_exists(int(path.read_text(encoding='utf-8').strip())):raise RuntimeError('Live job: '+str(path))
        lock=RUN/'operation.lock'
        with lock.open('x',encoding='utf-8') as out:out.write(str(os.getpid()))
        try:package();smoke()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
