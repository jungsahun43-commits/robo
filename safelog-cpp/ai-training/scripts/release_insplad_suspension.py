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
from scripts import study_insplad_suspension as study
from scripts.fetch_equipment_expansion import sha
from scripts.prepare_equipment_expansion import new
from scripts.study_equipment_expansion import read
from safelog_ai.process_probe import process_exists
from safelog_ai.insplad_suspension_server import CATEGORY
from safelog_ai.insplad_glass_server import CATEGORY as GLASS_CATEGORY,GlassEngine
from safelog_ai.insplad_server import CATEGORY as CORROSION_CATEGORY,InspladEngine
from safelog_ai.equipment_expansion import CATEGORIES
from safelog_ai.industrial_supervised import WEAR_CATEGORY

RUN=ROOT/'runs/industrial-insplad-suspension-release-v52'
POLICY=ROOT/'reports/industrial-insplad-suspension-v52-release-policy.json'
OUTPUTS=ROOT.parents[3]/'outputs'
ARCHIVE=OUTPUTS/'insplad-three-tasks-role4-research-overlay-2026-10-16-v51.zip'
CORROSION_DELTA=ROOT/'runs/industrial-insplad-corrosion-v45/trained/final-delta.pt'
CORROSION_SHA='6c1fec4fd86cfcd9ac42f8196c449879ad090b2bafdee045fd9269542e25124d'
BASE=ROOT/'runs/industrial-magnetic-tiles-v16/isolated-zip-r1'
ISOLATED=RUN/'isolated-overlay'
SOURCES=('safelog_ai/insplad_suspension_server.py','scripts/release_insplad_suspension.py','scripts/verify_insplad_suspension_bundle.py','tests/test_insplad_suspension_server.py','../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp','../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp','../apps/mobile-qt/tests/http_ai_safety_analyzer_tests.cpp','../apps/mobile-qt/tools/unified_main_probe.cpp')


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
            'device_guard':'all251originalRGB freshCPU/freshGPU/cachedGPU decisions exact; probability atol5e-4 rtol1e-3 fixed before newCPU/GPU comparisons; unchanged threshold',
            'old_aliases':'all17 existingtaskaliases one firstFIT each CPU; exact classResults,policySHA andallprimary/auxiliaryPNG strings compared with approved glass+corrosion parentCPU; no claim these are15accuracy evaluations',
            'CPP':'fresh separately compiledQtprobe with explicit suspension+glass+upper-shackle allowlist, currentoldKS2/BS/LOCO/blackbracket andpreservedcorrosion andpreservedglass andnewsuspension cases,3operations each=21newoperations; outputfiles andexit0 required; no pastchecks reused',
            'input_guard':'explicitnewtaskalias or matched equipment_surface/category; genericunknown/mismatchedcategory422; cropminside64. Domain manuallyselected, not automaticallyverified.',
            'scope':'sourcephoto20percent benchmark only; date-labelcorrelation/physicalparts unverified; no overallmachinehealth/location/independentfieldclaim',
            'location':'newclassifier no map; allold exposedmap outputs preserved; no synthetic/CAM/mask promotion',
            'release_guard':'ZIPCRC/memberSHA,sourceCPU_GPU,currentCPU18aliases,CPP21operations andoldPNGexact allpass before researchhandoff; existingdefaultv16-r1 not automaticallyswitched',
            'storage':'newoverlayZIP/isolatedcode each<32MiB,sourceproofs<8MiB,existingparentusedread-only,>=1GiBfree; no deletions',
            'new_optimizer':0,'new_units_passed':3,'Qt_unit_gate':'freshbuild all10actualtests including suspension forward, plusinitcleanup12functions, allpass required','APK_or_fresh_pip_install_tested':False}
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
    files.update({'insplad-suspension-delta.pt':study.RUN/'trained/final-delta.pt','calibration.json':study.RUN/'calibration.json',
                  'source-evaluation.json':study.RUN/'selection.json','training-protocol.json':study.PROTOCOL,
                  'release-policy.json':POLICY,'verify_bundle.py':ROOT/'scripts/verify_insplad_suspension_bundle.py',
                  'requirements-api.txt':BASE/'requirements-api.txt'})
    for cpp in ('adapters/ai/http_ai_safety_analyzer.cpp','adapters/ai/http_ai_safety_analyzer.hpp','tests/http_ai_safety_analyzer_tests.cpp','tools/unified_main_probe.cpp'):
        files['safelog-cpp/apps/mobile-qt/'+cpp]=ROOT.parent/'apps/mobile-qt'/cpp
    files.update({'corrosion-delta.pt':CORROSION_DELTA,'corrosion-calibration.json':ROOT/'runs/industrial-insplad-corrosion-v45/calibration.json',
                  'glass-delta.pt':study.PRIOR_DELTA,'glass-calibration.json':ROOT/'runs/industrial-insplad-glass-v48/calibration.json',
                  'corrosion-source-evaluation.json':ROOT/'runs/industrial-insplad-corrosion-v45/selection.json',
                  'glass-source-evaluation.json':ROOT/'runs/industrial-insplad-glass-v48/selection.json'})
    metadata={'schema':'safelog.research-overlay.v51','parent_file_sha256':study.EXPECTED,
              'required_parent_bundle_sha256':'40109dd30dc1778abc780a53e2aa502ac895fce45f66e9b9a9f46af60dd80a49',
              'delta_sha256':sha(files['insplad-suspension-delta.pt']),'preserved_corrosion_delta_sha256':CORROSION_SHA,'preserved_glass_delta_sha256':study.PRIOR_SHA,'combined_tensor_state_sha256':payload['combined_state_sha256'],
              'one_shared_encoder':True,'automatic_default_replacement':False,'source_scope':'cropped_lightning_rod_suspension_rust',
              'source_fpr':24/231,'source_fnr':0.0,'independent_field_validation':False}
    guide='''# 역할4: 현수 연결구 부식 + 유리 애자 캡 누락 + 상부 연결고리 부식

기존 v16-r1 ZIP의 model.pt/evaluation.json이 필요합니다. 새 ZIP에는 세 추가 분류부의 결합 delta와 이전 두 기능의 보존 검사용 delta·기준·코드가 들어 있습니다. 하나의 공통 encoder를 쓰는 한 모델이며 단독 앱이나 단독 모델 파일이 아닙니다. 기본 앱을 자동 교체하지 않습니다.

각 ZIP을 별도 폴더에 풀고 이 ZIP 폴더에서 기존 Python 환경으로 실행합니다. /BASE는 원래 v16-r1 폴더로 바꿉니다.

```powershell
$overlay = Get-Content -Raw overlay.json | ConvertFrom-Json
python -m safelog_ai.insplad_suspension_server --parent "/BASE/model.pt" --parent-sha $overlay.parent_file_sha256 --corrosion-delta corrosion-delta.pt --corrosion-delta-sha $overlay.preserved_corrosion_delta_sha256 --corrosion-calibration corrosion-calibration.json --glass-delta glass-delta.pt --glass-delta-sha $overlay.preserved_glass_delta_sha256 --glass-calibration glass-calibration.json --delta insplad-suspension-delta.pt --delta-sha $overlay.delta_sha256 --calibration calibration.json --evaluation "/BASE/evaluation.json" --port 8775
```

함께 든 C++ 어댑터와 테스트를 반영하고 재빌드합니다. 새 inspectionTask=insplad_lightning_suspension_rust이며 equipment_surface와 동일 equipmentCategory도 가능합니다. 기존 insplad_glass_missing_cap, insplad_upper_shackle_rust 및 15개 항목을 보존합니다. SAFELOG_AI_BASE_URL=http://127.0.0.1:8775입니다.

새 입력은 가공지선 현수 연결구를 자른 사진이며 256 RGB 입력으로 사진상 부식 후보를 제안합니다. 유리 애자는 384 종횡비 보존, 상부 연결고리 부식은 256 입력의 기존 가중치·기준을 보존합니다. 입력 자산의 종류는 사용자가 선택합니다. 자산 자동 판별·전체 설비 안전·전기/기계 작동·새 결함 위치 지도를 제공하지 않습니다. 점검자가 최종 확인합니다. 요약과 위험 등급은 기존 규칙 템플릿입니다.

API modelSha256은 overlay.json의 combined_tensor_state_sha256으로 부모 파일 SHA와 다릅니다. parentCheckpointSha256과 deltaSha256은 따로 제공합니다.

새 원저자 평가 정상231/부식20에서 오탐24/231=10.39%, 미탐0/20=0%였습니다. 정상CAL51 rank45의 고정 기준을 사용했습니다. 평가 날짜 중 정상/부식이 함께 있는 날짜는0개라 날짜·배경 혼동을 배제하지 못하며 물리 부품·촬영·현장 독립성도 검증 전입니다. 0미탐 보장이나 독립 현장 성능이 아닙니다. 기존 두 기능의 공개 평가 수치나 검사를 새 검사의 수량으로 합치지 않습니다.

InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1, 논문10.1080/01431161.2023.2283900, CC BY-NC3.0 비상업 학술 연구와 원저자·출처·버전·변경 표시를 유지합니다. 기존 자료 이용조건도 유지합니다. 새 pip 설치나 Android APK 검증은 포함하지 않습니다.
'''
    assert read(study.RUN/'calibration.json')['rank']==45
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


def build_probe():
    import xml.etree.ElementTree as ET
    probe=RUN/'qt-probe';probe.mkdir()
    shutil.copyfile(ROOT/'runs/industrial-surfaces-v9/qt-probe/CMakeLists.txt',probe/'CMakeLists.txt')
    env=dict(os.environ);env['QT_QPA_PLATFORM']='offscreen';env['PATH']='C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;C:/Qt/Tools/Ninja;'+env.get('PATH','')
    cmake='C:/Qt/Tools/CMake_64/bin/cmake.exe'
    commands=[[cmake,'-S',str(probe),'-B',str(probe/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH=C:/Qt/6.11.2/mingw_64','-DCMAKE_CXX_COMPILER=C:/Qt/Tools/mingw1310_64/bin/c++.exe','-DCMAKE_MAKE_PROGRAM=C:/Qt/Tools/Ninja/ninja.exe'],
              [cmake,'--build',str(probe/'build'),'--parallel','2'],
              [str(probe/'build/adapter_tests.exe'),'-o',str(RUN/'adapter-tests.xml')+',xml']]
    with (RUN/'fresh-CPP-build.log').open('x',encoding='utf-8') as log:
        for command in commands:
            done=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=log,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
            assert done.returncode==0,command
    functions=ET.parse(RUN/'adapter-tests.xml').getroot().findall('TestFunction')
    assert len(functions)==12 and all(len(v.findall('Incident'))==1 and v.find('Incident').attrib['type']=='pass' for v in functions)
    assert any(v.attrib['name']=='forwardsSuspensionResearchTask' for v in functions)
    new(RUN/'fresh-Qt-unit-proof.json',{'status':'passed','test_functions_including_init_cleanup':12,'actual_tests':10,'fresh_build':True})


def smoke():
    p=validate();package_record=read(RUN/'package.json');assert sha(ARCHIVE)==package_record['sha256']
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    engine={'parent':str(BASE/'model.pt'),'parent_sha':study.EXPECTED,'corrosion_delta':str(ISOLATED/'corrosion-delta.pt'),'corrosion_delta_sha':CORROSION_SHA,'corrosion_calibration':str(ISOLATED/'corrosion-calibration.json'),'glass_delta':str(ISOLATED/'glass-delta.pt'),'glass_delta_sha':study.PRIOR_SHA,'glass_calibration':str(ISOLATED/'glass-calibration.json'),'delta':str(ISOLATED/'insplad-suspension-delta.pt'),
            'delta_sha':p['delta_sha256'],'calibration':str(ISOLATED/'calibration.json'),'evaluation':str(BASE/'evaluation.json')}
    rows=study.rows_for('test')
    new(RUN/'device-inputs.json',{'engine':engine,'rows':[{**r,'image':str(ROOT/r['image'])} for r in rows],
                                'cached_scores':read(study.RUN/'test-scores.json')['scores']})
    with (RUN/'device-check.log').open('x',encoding='utf-8') as log:
        device=subprocess.run([sys.executable,'verify_bundle.py','--inputs',str(RUN/'device-inputs.json'),'--output',str(RUN/'device-proof.json')],cwd=ISOLATED,env=env,stdout=log,stderr=log,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert device.returncode==0,(RUN/'device-check.log').read_text(encoding='utf-8')
    deviceproof=read(RUN/'device-proof.json');assert deviceproof['status']=='passed'
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8775))!=0,'Preserve unrelated server'
    command=[sys.executable,'-m','safelog_ai.insplad_suspension_server']
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
        parent=GlassEngine(BASE/'model.pt',study.EXPECTED,CORROSION_DELTA,CORROSION_SHA,ROOT/'runs/industrial-insplad-corrosion-v45/calibration.json',study.PRIOR_DELTA,study.PRIOR_SHA,ROOT/'runs/industrial-insplad-glass-v48/calibration.json',BASE/'evaluation.json')
        old=read(ROOT/'data/unified-main-v1/manifest.json')
        photos={'facility_surface':ROOT/old['fit']['facility_surface'][0]['image'],
                'belt_damage':ROOT/'runs/belt-author-embedded-sample.jpg','conveyor_foreign':ROOT/old['fit']['conveyor_foreign'][0]['image']}
        oldrows=read(ROOT/'data/equipment-expansion-v1/manifest.json')['rows']
        for c in CATEGORIES:photos[c]=ROOT/next(r['image'] for r in oldrows if r['phase']=='fit' and r['category']==c)
        photos[WEAR_CATEGORY]=ROOT/next(r['image'] for r in read(ROOT/'data/industrial-bsclass-v1/manifest.json')['rows'] if r['phase']=='fit' and not r['label'])
        photos[CORROSION_CATEGORY]=ROOT/next(r['image'] for r in read(ROOT/'data/industrial-insplad-grouped-v44/manifest.json')['rows'] if r['phase']=='fit' and r['label']==1)
        photos[GLASS_CATEGORY]=ROOT/next(r['image'] for r in read(ROOT/'data/industrial-insplad-glass-v47/manifest.json')['rows'] if r['phase']=='fit' and r['label']==1)
        assert len(photos)==17
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
        for task in ('ksdd2_surface',WEAR_CATEGORY,'loco_splicing_connectors','mpdd_bracket_black',CORROSION_CATEGORY,GLASS_CATEGORY,CATEGORY):
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
            'fresh_packaged_RGB_CPU_GPU_cached_decisions_exact':251,'all_old_CPU_aliases_checked':17,'new_CPU_aliases_checked':1,'explicit_new_equipment_category_checked':True,
            'old_classResults_and_policy_exact':True,'old_PNG_strings_exact':pngs,'current_real_CPP_operations':sum(v['operations'] for v in cpp),
            'CPP_cases':cpp,'old_alias_proofs':proofs,'unknown_or_mismatched_inputs_rejected422':True,'owned_server_stopped':server.poll() is not None,
            'previous_v46_and_v49_proofs_preserved_not_recounted':True,'fresh_Qt_unit_tests':read(RUN/'fresh-Qt-unit-proof.json'),'new_optimizer':0,'one_shared_encoder':True,'parent_checkpoint_duplicated':False,'default_v16_automatically_replaced':False,
            'independent_field_validation':False,'APK_built':False,'fresh_pip_install_tested':False}
    new(RUN/'release.json',result);new(ROOT/'reports/industrial-insplad-suspension-v52-release.json',result)
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
        try:build_probe();package();smoke()
        except BaseException as error:
            new(RUN/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'error':repr(error),'pid':os.getpid()});raise
        finally:lock.unlink()
