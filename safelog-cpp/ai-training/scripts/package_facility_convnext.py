"""Local research handoff: checkpoint, validated results and exact checksums."""
from pathlib import Path
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.facility_convnext import NAME,read,sha,require
from scripts.fetch_rc2119 import write_new
def main():
    comparison=read(ROOT/'reports/facility-convnext-study-comparison.json')
    timing=read(ROOT/'reports/facility-convnext-local-timing.json')
    proof=comparison['technical_verification'];require(proof['status']=='passed','Verified training required')
    entries=comparison['experiments'];model=entries[-1];require(model['weights_sha256']==proof['weights_sha256']==timing['experiments'][-1]['weights_sha256'],'Artifact binding differs')
    lines=['# ConvNeXt 연구 모델 전달 안내','',
        '역할3이 준비한 시설 사진 분류 연구 후보다. 아래 지표는 같은 공개 VAL의 반복 탐색 결과이며 독립 산업현장 정확도가 아니다. AP는 양성 사진을 앞쪽에 배치하는 순위 지표이며 정확도 백분율과 다르다.','',
        f"실제 학습6epoch, 선택{proof['selected_epoch']}epoch. 최대 오탐·미탐률: 기존{entries[1]['worst_error']*100:.4f}% → 신규{model['worst_error']*100:.4f}%.",
        f"연구 후보 기준: {'통과'if comparison['research_gate']['research_candidate_nominated']else'미달'}. 기존 항목 보존 기준: {'통과'if comparison['retention_gate']['retention_candidate_nominated']else'미달'}. 각 비율5% 미만: {'통과'if model['target_passed']else'미달'}.",'',
        '## 시설 항목별 AP','',
        '| 출처·항목 | 기존 대조군 | 신규 모델 | 차이 |','|---|---:|---:|---:|']
    for row in comparison['retention_gate']['per_source_other_class_ap']:
        lines.append(f"| {row['domain']} · {row['class']} | {row['control_ap']:.4f} | {row['treatment_ap']:.4f} | {row['treatment_minus_control_ap']:+.4f} |")
    lines +=['','## 로컬 GPU 추론 비용','',
        '같은 TRAIN 사진1장,640픽셀,FP32,batch1. 모델별10회 예열 후40회 측정했다. 파일 읽기·전송·앱 네트워크·초기 모델 로딩은 제외한다.','',
        '| 모델 | 파라미터 | 추론 중앙값 | p90 | checkpoint |','|---|---:|---:|---:|---:|']
    for row in timing['experiments']:
        lines.append(f"| {row['run']} | {row['parameter_count']:,} | {row['median_milliseconds']:.2f}ms | {row['p90_milliseconds']:.2f}ms | {row['checkpoint_bytes']/1024**2:.2f}MiB |")
    lines +=['','## 파일 사용','',
        '1. GitHub의 feature/ai-engine 브랜치에서 코드를 받는다. 기존 AI Python 환경의 Torch/Torchvision을 사용한다.',
        '2. 이 ZIP을 safelog-cpp/ai-training 폴더에 압축 해제한다. runs/facility-presence-target-convnext-finetune/best.pt가 있어야 한다. 가중치는 Git에 포함하지 않으며 별도 ZIP으로 전달한다.',
        '3. 사진 경로를 지정해 아래 명령을 실행하면 검증된 가중치 해시를 확인하고7종 확률을 JSON으로 출력한다. GPU가 없으면 --device cpu를 사용한다.','',
        '```powershell','cd safelog-cpp/ai-training',r'.\.venv\Scripts\python.exe scripts/predict_facility_convnext.py --image "사진경로.jpg" --device cuda','```','',
        '199개 state tensor의 새 모델이며 이전324개 MobileNet 가중치와 구조가 다르다. 기존 PresenceClassifier에 새 가중치 파일을 넣는 방식으로 연결할 수 없다. ConvnextResearchPresence가 strict 재로딩과 기존640픽셀 전처리를 제공한다.',
        '역할4의 앱 화면·공용 API 계약은 기존대로 사용한다. 본 ZIP과 명령은 별도 모델 검토용이며 앱 기본 모델이나 서버 설정을 교체하지 않는다. 앱 연결 시 역할3이 서버의 모델 로더와 프로필을 별도 구현·검증해야 한다.','',
        '사람 검토가 필요한 AI 제안이다. 사진 점수는 구조 안전성이나 산업현장 검증 정확도 자체를 의미하지 않는다. 원본 데이터·사람 검수 기록·sourceTEST는 ZIP에 포함하지 않는다.','']
    doc=ROOT/'reports/FACILITY_CONVNEXT_HANDOFF_KO.md';require(not doc.exists(),'Preserve handoff');doc.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    paths=[f'runs/{NAME}/best.pt','reports/facility-convnext-study-protocol.json','reports/facility-convnext-study-verification.json',
        'reports/facility-convnext-study-comparison.json','reports/FACILITY_CONVNEXT_STUDY_RESULTS_KO.md',
        'reports/facility-convnext-local-timing.json',doc.relative_to(ROOT).as_posix()]
    manifest={'schema':'facility_convnext_research_handoff_v1','weights_sha256':proof['weights_sha256'],
        'app_model_promoted':False,'human_review_required':True,'independent_industrial_accuracy_verified':False,
        'files':{p:{'sha256':sha(ROOT/p),'size_bytes':(ROOT/p).stat().st_size}for p in paths},'script_sha256':sha(Path(__file__))}
    package=ROOT/'runs/facility-convnext-research-handoff.zip';require(not package.exists(),'Preserve ZIP')
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_STORED)as bundle:
        for path in paths:bundle.write(ROOT/path,path)
        bundle.writestr('CONVNEXT-HANDOFF-MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(package)as bundle:
        require(bundle.testzip()is None,'ZIP integrity failed')
        import hashlib
        for path,record in manifest['files'].items():
            with bundle.open(path)as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            require(digest==record['sha256']and bundle.getinfo(path).file_size==record['size_bytes'],'Bundled bytes differ')
    write_new(ROOT/'reports/facility-convnext-handoff-manifest.json',{**manifest,'zip_sha256':sha(package),'zip_size_bytes':package.stat().st_size,'zip_members_verified':True})
    print(json.dumps({'status':'passed','zip':str(package),'sha256':sha(package),'bytes':package.stat().st_size}),flush=True)
if __name__=='__main__':main()
