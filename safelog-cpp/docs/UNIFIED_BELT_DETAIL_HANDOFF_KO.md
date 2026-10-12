# 역할4 · 벨트 출력 추가 학습 통합 모델

연구 후보 `native_mix`. 파일 하나·공통 분석부 하나·3개 출력이며 기존 시설·이물질 가중치는 v1과 정확히 동일하다. 모델 SHA는 `4b6d91c045cf6a33631ed5542c81d8e116166f541e97e6d745e1dfbe3f7fe5e5`다. API modelName은 구조 계열 unified-main-v1이며 실제 학습 버전은 modelSha256 및 model.json으로 구분한다.

## CPU 실행

`unified-main-belt-detail-role4-handoff-2026-10-13-v1.zip`을 새 폴더에 풀고 Python 3.12에서 실행한다. 제작 환경의 실제 실행과 새 환경의 pip 설치 시험은 구분한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$beltDetailModelRecord = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.unified_main_server --checkpoint $beltDetailModelRecord.path --sha256 $beltDetailModelRecord.sha256 --port 8775
```

서버는127.0.0.1에서 실행한다. 실제 스마트폰은 별도 서버 주소·인증·HTTPS를 팀에서 구성한다. Android APK를 만든 전달물이 아니다.

## C++ 앱 연결

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="belt_damage"
```

[기존 어댑터 설정](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp)과 [요청 코드](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp)를 사용한다. 점검 항목은 facility_surface·belt_damage·conveyor_foreign이다. 환경변수는 실행 모드를 선택하며 촬영 화면의 항목 선택 UI·DB 저장은 팀 앱 흐름에 연결한다.

- GET /health: ready, neuralModelCount=1, sharedBackboneCount=1, modelSha256 확인.
- POST /v1/analyze-hazard: image, inspectionTask, userMemo, promptVersion.
- POST /v1/compare-action: beforeImage, afterImage, inspectionTask, actionNote, promptVersion.
- POST /v1/summarize: inspectionContext, promptVersion.

점검 항목 누락·미지원은422다. 요청하지 않은 다른 항목은 정상으로 판단하지 않는다. 벨트 heatmap은 원본 크기 확인 자료이며 확정 결함 위치가 아니다. 점수는 보정된 고장 확률이 아니고 같은 사진으로 수리 완료가 나오면 안 된다. 내부 기계 고장은 not_assessed다. 위험 등급·조치·요약은 규칙 템플릿이며 점검자 최종 확인이 필요하다.

[실제 학습 성능](../ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md), [기존 다른 기능의 수치](../ai-training/reports/UNIFIED_MAIN_RESULTS_KO.md), [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)을 함께 전달한다. 기존 앱 기본 모델은 자동 교체하지 않았다.
