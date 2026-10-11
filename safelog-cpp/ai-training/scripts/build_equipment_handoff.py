"""Package the exact research checkpoint and local HTTP service for role4."""
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = REPO.parents[1]/'outputs'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    training = read(ROOT/'reports/beltcrack-expansion-training.json')
    result = read(ROOT/'reports/beltcrack-expansion-results.json')
    verified = read(ROOT/'reports/beltcrack-expansion-verification.json')
    smoke = read(ROOT/'reports/beltcrack-expansion-api-smoke.json')
    model = ROOT/'runs/beltcrack-cached-expansion-v2/final.pt'
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert verified['status']=='passed' and smoke['status']=='passed'
    assert digest==training['weights_sha256']==result['weights_sha256']==verified['weights_sha256']==smoke['result']['modelSha256']
    OUT.mkdir(parents=True,exist_ok=True)
    package = OUT/'equipment-ai-role4-handoff-2026-10-12.zip'
    assert not package.exists()
    versions = {name:importlib.metadata.version(name) for name in
                ('torch','torchvision','numpy','Pillow','fastapi','uvicorn')}
    starter = f'''# 역할4 전달: 실제 컨베이어 AI와 로컬 API

이 ZIP은 연구 후보 모델과 앱 연결용 서버다. 실제40epoch 학습, 원본 검증,
CPU/GPU 재로드, 실제HTTP 사진 분석을 통과한 동일 파일을 포함한다.
모델 SHA256: {digest}
동일127장 수치와 전체353장 수치는 reports/BELTCRACK_EXPANSION_RESULTS_KO.md에 있다.
정상 사진 오탐률은 미검증이며 모델은 고장/안전을 확정하지 않는다.

1. ZIP을 압축 해제한다.
2. Python3.12와 torch/torchvision/numpy/Pillow/fastapi/uvicorn 환경을 준비한다.
   기존 SafeLog AI 학습 .venv가 있으면 그대로 사용한다.
   검증 환경의 실제 버전은 environment.json을 참고한다.
3. 압축 해제한 폴더에서 실행한다:

python -m safelog_ai.equipment_server --checkpoint beltcrack-cached-expansion-v2/final.pt --sha256 {digest} --port 8771

4. http://127.0.0.1:8771/docs에서 확인하고 Qt/C++ 사진 화면을 연결한다.
   요청/응답/DB 필드와 Qt 예시는 docs/EQUIPMENT_API_HANDOFF_KO.md에 있다.
   Android USB 연결은 adb reverse tcp:8771 tcp:8771을 이용한다.
5. 역할2에 modelSha256/inputSha256/AI 제안/사람 확인 이력을 저장하도록 전달한다.

모델은 로컬 CPU에서 실행된다. 사용자 원본 사진과 API 키는 이 ZIP에 없다.
기존 앱의 기본 서버를 자동 교체하지 않는다. 최종 점검은 사람이 확인한다.
모델 학습 데이터: BeltCrack V1 https://doi.org/10.57760/sciencedb.31181 (CC BY-SA4.0).
추가 출처와 정상 데이터 접근 조건은 reports/BELT_NORMAL_DATA_AUDIT_KO.md에 있다.
'''
    files = {
        'beltcrack-cached-expansion-v2/final.pt':model,
        'reports/BELTCRACK_EXPANSION_RESULTS_KO.md':ROOT/'reports/BELTCRACK_EXPANSION_RESULTS_KO.md',
        'reports/BELT_NORMAL_DATA_AUDIT_KO.md':ROOT/'reports/BELT_NORMAL_DATA_AUDIT_KO.md'}
    for name in ('belt_inference.py','belt_segmentation.py','equipment_server.py'):
        files['safelog_ai/'+name] = ROOT/'safelog_ai'/name
    files['scripts/audit_belt_normal_coco.py'] = ROOT/'scripts/audit_belt_normal_coco.py'
    for name in ('protocol','training','results','verification','api-smoke','plot'):
        files[f'reports/beltcrack-expansion-{name}.json'] = ROOT/f'reports/beltcrack-expansion-{name}.json'
    for name in ('comparison','examples'):
        files[f'reports/beltcrack-expansion-{name}.png'] = ROOT/f'reports/beltcrack-expansion-{name}.png'
    manifest = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    handoff_doc = (ROOT.parent/'docs/EQUIPMENT_API_HANDOFF_KO.md').read_text(encoding='utf-8')
    handoff_doc = handoff_doc.replace('../ai-training/reports/','../reports/').replace('.venv/Scripts/python.exe','python')
    handoff_doc = handoff_doc.replace('runs/beltcrack-cached-expansion-v2/final.pt','beltcrack-cached-expansion-v2/final.pt')
    handoff_doc = handoff_doc.replace('`safelog-cpp/ai-training` 폴더에서 실행한다.','이 ZIP을 압축 해제한 폴더에서 실행한다.')
    handoff_bytes = handoff_doc.encode('utf-8')
    manifest['docs/EQUIPMENT_API_HANDOFF_KO.md'] = hashlib.sha256(handoff_bytes).hexdigest()
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as packed:
        for name,path in files.items():
            packed.write(path,name)
        packed.writestr('docs/EQUIPMENT_API_HANDOFF_KO.md',handoff_bytes)
        packed.writestr('safelog_ai/__init__.py','')
        packed.writestr('README_RUN_KO.md',starter)
        packed.writestr('environment.json',json.dumps({'python':'3.12','actual_verified_versions':versions},indent=2))
        packed.writestr('file-sha256.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(package) as packed:
        assert packed.testzip() is None
        for name,expected in manifest.items():
            assert hashlib.sha256(packed.read(name)).hexdigest()==expected
    release = {'status':'created_and_verified','package':str(package.resolve()),'bytes':package.stat().st_size,
        'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'model_sha256':digest,
        'model_training_epochs':40,'model_optimizer_updates':training['optimizer_updates'],
        'whole_photo_normal_fpr_verified':False,'actual_loopback_http_smoke_passed':True,
        'private_photos_or_api_keys_included':False,'file_sha256':manifest,
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (ROOT/'reports/beltcrack-expansion-release.json').open('x',encoding='utf-8') as handle:
        json.dump(release,handle,indent=2)
    print(json.dumps({k:v for k,v in release.items() if k!='file_sha256'},indent=2))


if __name__=='__main__':
    main()
