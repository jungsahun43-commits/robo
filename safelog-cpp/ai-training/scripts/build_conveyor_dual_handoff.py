"""New combined handoff; preserve both completed experiments and older ZIPs."""
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[3]/'outputs'
NAME = 'equipment-ai-role4-handoff-2026-10-12-v3.zip'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files, models = {}, {}
    for prefix, folder in [('beltcrack-expansion', 'beltcrack-cached-expansion-v2'),
                           ('conveyor-foreign-grouped', 'conveyor-foreign-grouped-v1')]:
        training = read(ROOT/f'reports/{prefix}-training.json')
        result = read(ROOT/f'reports/{prefix}-results.json')
        verification = read(ROOT/f'reports/{prefix}-verification.json')
        smoke = read(ROOT/f'reports/{prefix}-api-smoke.json')
        assert verification['status']==smoke['status']=='passed'
        model = ROOT/'runs'/folder/'final.pt'
        digest = sha(model)
        assert digest==training['weights_sha256']==result['weights_sha256']==verification['weights_sha256']
        assert digest==(smoke.get('weights_sha256') or smoke['result']['modelSha256'])
        files[folder+'/final.pt'] = model
        models[training['class']] = {'path':folder+'/final.pt', 'sha256':digest,
            'actual_epochs':training['epochs'], 'actual_optimizer_updates':training['optimizer_updates']}
        for suffix in ('protocol', 'training', 'results', 'verification', 'api-smoke', 'plot'):
            files[f'reports/{prefix}-{suffix}.json'] = ROOT/f'reports/{prefix}-{suffix}.json'
    for name in ('belt_inference.py', 'belt_segmentation.py', 'equipment_server.py', 'conveyor_foreign.py',
                 'conveyor_foreign_inference.py', 'conveyor_foreign_server.py'):
        files['safelog_ai/'+name] = ROOT/'safelog_ai'/name
    for name in ('BELTCRACK_EXPANSION_RESULTS_KO.md', 'BELT_NORMAL_DATA_AUDIT_KO.md',
                 'CONVEYOR_DATA_ATTRIBUTION_KO.md', 'CONVEYOR_FOREIGN_ACQUISITION_KO.md',
                 'CONVEYOR_FOREIGN_ATTRIBUTION_KO.md', 'CONVEYOR_FOREIGN_RESULTS_KO.md',
                 'conveyor-foreign-acquisition.json', 'beltcrack-expansion-comparison.png',
                 'beltcrack-expansion-examples.png', 'conveyor-foreign-grouped-rates.png',
                 'conveyor-foreign-grouped-examples.png'):
        files['reports/'+name] = ROOT/'reports'/name
    portable = {}
    for name in ('EQUIPMENT_API_HANDOFF_KO.md', 'CONVEYOR_FOREIGN_API_HANDOFF_KO.md'):
        content = (ROOT.parent/'docs'/name).read_text(encoding='utf-8')
        content = content.replace('../ai-training/reports/', '../reports/').replace('.venv/Scripts/python.exe', 'python')
        for model in models.values():
            content = content.replace('runs/'+model['path'], model['path'])
        content = content.replace('`safelog-cpp/ai-training`', '압축 해제한 폴더')
        portable['docs/'+name] = content.encode('utf-8')
    # Portable report link rewrites change only package copies, never completed sources.
    for name in ('BELTCRACK_EXPANSION_RESULTS_KO.md', 'CONVEYOR_FOREIGN_RESULTS_KO.md'):
        key = 'reports/'+name
        content = files.pop(key).read_text(encoding='utf-8').replace('../../docs/', '../docs/')
        portable[key] = content.encode('utf-8')
    versions = {name:importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'fastapi', 'uvicorn')}
    belt, foreign = models['belt_crack'], models['conveyor_foreign_object']
    starter = f'''# 역할4 전달: 컨베이어 두 AI 모델

이 ZIP에는 실제 학습한 모델2개와 실행 가능한 로컬 CPU HTTP 서버가 있다.
코드/모델만 압축 해제하면 실행할 수 있으며 학습 데이터·API 키·개인 사진은 없다.
검증 환경의 Python3.12와 패키지 버전은 environment.json에 기록했다.

## 1. 사용 순서

1. ZIP을 압축 해제한다. 기존 SafeLog AI .venv 또는 같은 패키지 환경을 사용한다.
2. 터미널1에서 표면 균열 API를 실행한다:

python -m safelog_ai.equipment_server --checkpoint {belt['path']} --sha256 {belt['sha256']} --port 8771

3. 터미널2에서 이물질 API를 실행한다:

python -m safelog_ai.conveyor_foreign_server --checkpoint {foreign['path']} --sha256 {foreign['sha256']} --port 8772

4. 각 /docs에서 확인하고 Qt/C++ 앱의 해당 점검 화면을 연결한다.
   Android USB는 adb reverse tcp:8771 tcp:8771 및 adb reverse tcp:8772 tcp:8772를 설정한다.
5. 역할2에 사진/모델 SHA와 AI 제안·사람 확인·조치 이력을 저장하도록 전달한다.

## 2. 모델을 선택할 기준

- belt_crack: 벨트 표면 균열/찢어짐 후보와 heatmap. 실제40epoch/7,000업데이트.
  같은127장 미탐5.51%, 전체353장 미탐5.95%, 정상 사진 오탐은 미측정.
- conveyor_foreign_object: 철광석 흐름의 근접/잘린 벨트 사진에서 이물질 유무.
  실제30epoch/1,170업데이트. TEST448장 오탐0.39%·미탐0.52%, VAL 미탐6.25%.
  위치 정답이 없어 이물질 위치/heatmap은 없다. 다른 운반물/공장 성능은 미검증.

두 모델의 정상을 섞거나 전체 설비 오차5% 미만이라고 소개하지 않는다.
사진의 미검출 결과는 정상 가동·기계 고장 없음의 판정이 아니다.
완성 앱 연결·APK는 역할4가 진행하며, 이 ZIP은 앱 전체 빌드 결과물이 아니다.

상세: docs/EQUIPMENT_API_HANDOFF_KO.md, docs/CONVEYOR_FOREIGN_API_HANDOFF_KO.md.
실제 결과와 자료별 오류 사진: reports/의 두 결과 문서.
사진 출처·변경·CC BY-SA4.0/CC BY4.0 조건: reports/의 두 출처 문서.
models.json과 file-sha256.json으로 받은 파일을 확인한다.
'''
    portable['README_RUN_KO.md'] = starter.encode('utf-8')
    portable['safelog_ai/__init__.py'] = b''
    portable['environment.json'] = json.dumps({'python':'3.12', 'verified_versions':versions}, indent=2).encode()
    portable['models.json'] = json.dumps(models, indent=2).encode()
    manifest = {name:sha(path) for name,path in files.items()}
    manifest.update({name:hashlib.sha256(payload).hexdigest() for name,payload in portable.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    package = OUT/NAME
    assert not package.exists()
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name,path in files.items():
            archive.write(path, name)
        for name,payload in portable.items():
            archive.writestr(name, payload)
        archive.writestr('file-sha256.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        for name,digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    release = {'status':'created_and_verified', 'package':str(package.resolve()),
        'bytes':package.stat().st_size, 'package_sha256':sha(package), 'models':models,
        'file_sha256':manifest, 'private_photos_or_api_keys_included':False,
        'builder_sha256':sha(Path(__file__)), 'existing_role4_v2_package_untouched':True,
        'standalone_unpack_execution_check_pending':True}
    with (ROOT/'reports/conveyor-dual-handoff-release-v3.json').open('x', encoding='utf-8') as handle:
        json.dump(release, handle, indent=2)
    print(json.dumps({k:v for k,v in release.items() if k!='file_sha256'}, indent=2))


if __name__=='__main__':
    main()
