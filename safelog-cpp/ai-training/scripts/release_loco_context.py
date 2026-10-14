"""Selected context handoff; all prior API aliases and real C++ calls."""
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
    write(STAGE/'README_KO.md','# SafeLog 전선 연결부 전체 문맥 보강\n\n한 모델·한 공통 분석부. 사진 미탐39.90%는 여전히20% 목표 미달이다. 기존 위치 지도와 다른 설비를 보존한다.\n\n- [역할4 실행 안내](safelog-cpp/docs/LOCO_CONTEXT_HANDOFF_KO.md)\n- [실제 학습과 결과](safelog-cpp/ai-training/reports/LOCO_CONTEXT_RESULTS_KO.md)\n')
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

def smoke():
    choice = checked(); assert not ISOLATED.exists()
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1', 8775)) != 0
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and sum(n.endswith('.pt') for n in z.namelist()) == 1
        for name in z.namelist(): assert (ISOLATED/name).resolve().is_relative_to(ISOLATED.resolve())
        z.extractall(ISOLATED)
    digest = choice['checkpoint_sha256']; assert sha(ISOLATED/'model.pt') == digest
    old = study.read(ROOT/'data/unified-main-v1/manifest.json'); rows = study.read(trained.previous.MANIFEST)['rows']
    photos = {'facility_surface': ROOT/old['fit']['facility_surface'][0]['image'], 'belt_damage': ROOT/'runs/belt-author-embedded-sample.jpg', 'conveyor_foreign': ROOT/old['fit']['conveyor_foreign'][0]['image']}
    photos.update({c: ROOT/next(r['image'] for r in rows if r['phase'] == 'fit' and r['category'] == c) for c in CATEGORIES})
    data = study.read(ROOT/'data/industrial-bsclass-v1/manifest.json')['rows']; photos[WEAR_CATEGORY] = ROOT/next(r['image'] for r in data if r['phase'] == 'fit' and not r['label'])
    point = study.read(trained.RUN/'calibration.json')['point']; expected = next(r for r in study.read((REPORTS/'industrial-ksdd-supervised-v10-validation.json'))['observations'] if r['label'] and r['score'] > point['photo_threshold']+.2)
    photos['ksdd2_surface'] = ROOT/next(r['image'] for r in rows if r['id'] == expected['id'])
    env = dict(os.environ); env.pop('PYTHONPATH', None); env.update(PYTHONIOENCODING='utf-8', QT_QPA_PLATFORM='offscreen')
    env['PATH'] = 'C:/Qt/6.11.2/mingw_64/bin;C:/Qt/Tools/mingw1310_64/bin;'+env.get('PATH', '')
    log = (ISOLATED/'server.log').open('x', encoding='utf-8')
    server_env = dict(env); server_env['PATH'] = os.environ.get('PATH', '')
    process = subprocess.Popen([sys.executable, '-m', 'safelog_ai.loco_context_server', '--checkpoint', 'model.pt', '--sha256', digest, '--evaluation', 'evaluation.json', '--port', '8775'], cwd=ISOLATED, env=server_env, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    proofs = []; replies = []
    try:
        deadline = time.monotonic()+50
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError((ISOLATED/'server.log').read_text())
            try: health = http('/health'); break
            except (OSError, ValueError): time.sleep(.25)
        else: raise RuntimeError('Packaged CPU server timed out')
        assert health['modelSha256'] == digest and health['neuralModelCount'] == health['sharedBackboneCount'] == 1
        policies = http('/v1/inference-policy'); assert policies['supervisedPhoto']['ksdd2_surface']['originalLocalizationChanged'] is False
        assert policies['globalContext']['loco_splicing_connectors']['semanticClampWireRelationsVerified'] is False
        categories = http('/v1/equipment-categories')['categories']; assert len(categories) == 12 and sum(c['benchmarkStage']['ready_for_next_stage'] for c in categories) == 8
        old_replies = {r['task']: r['result'] for r in study.read(REPORTS/'industrial-ksdd-photo-only-v11-release.json')['hazard_replies']}
        for task, photo in photos.items():
            encoded = base64.b64encode(photo.read_bytes()).decode(); reply = http('/v1/analyze-hazard', {'inspectionTask': task, 'image': encoded, 'promptVersion': 'ksdd-photo-only-v11-smoke'})
            assert reply['modelSha256'] == digest and reply['inputSha256'] == sha(photo) and reply['neuralModelCount'] == reply['sharedBackboneCount'] == 1
            assert reply['humanConfirmationRequired'] and not reply['fieldPerformanceValidated']
            if task == 'ksdd2_surface':
                assert reply['candidateDetected'] and abs(reply['classResults'][0]['score']-expected['score']) < 2e-4
                from safelog_ai.industrial_batch_server import IndustrialEngine
                parent = IndustrialEngine(trained.START, trained.EXPECTED, trained.PARENT_EVALUATION); prior = parent.analyze(task, encoded)
                assert reply['heatmapPngBase64'] == prior['heatmapPngBase64'] and not reply['originalLocalizationChanged']
                assert reply['auxiliaryLocalizationWorseFNR'] and not reply['auxiliaryIsPrimaryLocationOutput']
                for key in ('heatmapPngBase64', 'supervisedAuxiliaryHeatmapPngBase64', 'supervisedAuxiliaryMaskPngBase64'):
                    with Image.open(io.BytesIO(base64.b64decode(reply[key]))) as im: assert im.size == (reply['inputWidth'], reply['inputHeight'])
                del parent
            elif task != 'loco_splicing_connectors' and 'inferencePolicySha256' in old_replies[task]: assert reply['inferencePolicySha256'] == old_replies[task]['inferencePolicySha256'], task
            for key in [k for k in reply if k.endswith('PngBase64')]: reply.pop(key)
            replies.append({'task': task, 'result': reply})
            if task in ('ksdd2_surface', 'ball_screw_pitting', 'loco_splicing_connectors'):
                output = ISOLATED/(task+'-cpp.json')
                cp = subprocess.run([str(trained.previous.RUN/'qt-probe/build/unified_probe.exe'), task, str(photo), digest, str(output)], cwd=ISOLATED, env=env, capture_output=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
                assert cp.returncode == 0 and output.exists(), cp.stderr.decode(errors='replace')
                proof = study.read(output); assert proof['operations'] == 3 and proof['samePhotoDoesNotResolve']; proofs.append(proof)
            print({'CPU_alias_checked': task}, flush=True)
        buffer = io.BytesIO(); Image.new('RGB', (32, 32), 'white').save(buffer, format='PNG')
        try: http('/v1/analyze-hazard', {'inspectionTask': 'ksdd2_surface', 'image': base64.b64encode(buffer.getvalue()).decode(), 'promptVersion': 'too-small'})
        except HTTPError as error: assert error.code == 422
        else: raise AssertionError('Too-small input must require review')
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        log.close()
    with zipfile.ZipFile(ARCHIVE) as z:
        hashes = json.loads(z.read('package-sha256.json')); assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items())
    write(REPORTS/'industrial-loco-context-v13-release.json', {'status': 'passed', 'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'model_sha256': digest, 'all15_CPU_task_aliases_checked': True, 'actual_current_CPP_operations': 9, 'CPP_cases': proofs,
        'prior_v9_CPP45_not_counted_as_current': True, 'primary_KSDD_heatmap_PNG_exact_to_prior_CPU': True,
        'all_other_exposed_policy_SHA_preserved': True, 'auxiliary_worse_localization_explicit': True, 'old_and_auxiliary_masks_original_sizes_checked': True,
        'too_small_input_rejected422': True, 'hazard_replies': replies, 'ZIP_CRC_members_SHA_links_verified': True,
        'owned_server_stopped': process.poll() is not None, 'one_checkpoint_one_backbone': True, 'fresh_pip_install_tested': False, 'APK_built': False, 'independent_field_validated': False})
    print({'release_passed': True, 'actual_CPP_operations': 9, 'CPU_aliases': 15}, flush=True)

def cpu():
    checked();assert ISOLATED.exists()
    copied=RUN/'cpu-known-test.py';assert not copied.exists();shutil.copy2(ROOT/'scripts/verify_loco_context_cpu.py',copied)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run([sys.executable,str(copied),str(ROOT),str(REPORTS/'industrial-loco-context-v13-CPU.json')],cwd=ISOLATED,env=env,capture_output=True,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    with (RUN/'cpu-known-test.log').open('x',encoding='utf-8') as out:out.write(result.stdout.decode('utf-8',errors='replace')+result.stderr.decode('utf-8',errors='replace'))
    print(result.stdout.decode('utf-8',errors='replace'),flush=True);assert result.returncode==0,result.stderr.decode(errors='replace')

if __name__=='__main__':
    torch.set_num_threads(4);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['package','smoke','cpu']);args=parser.parse_args();globals()[args.mode]()
