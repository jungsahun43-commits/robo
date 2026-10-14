# 역할4 · 금속판20% 목표 보강 시제품

기존 범위 확장 시제품과 같은 신경망 가중치이며 금속판의 사진 경고 기준만 정상CAL20장으로 조정했다. 이 전달물은 금속판 오탐3.85%·미탐18.31%를 측정한 연구 연결용 버전이다. 부품 표면의15%통과 상태는 유지했다. 나머지9유형은 보강/평가부족 상태다. 기존 ZIP과 앱 기본 모델은 자동 교체하지 않았다.

## CPU 실행

ZIP을 새 폴더에 풀고 Python3.12에서 실행한다. 이전 시제품과 설치 의존성은 같다. 새 pip설치를 이번검사로 검증한 것은 아니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$metalPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.metal_calibration_server --checkpoint $metalPrototype.path --sha256 $metalPrototype.sha256 --evaluation evaluation.json --port 8775
```

새 서버 모듈은금속판의20%정상경고예산/rank17정책을명시한다. 이전서버의85%정상분위수정책으로표시하지않는다. 모델SHA와금속판정책SHA를함께기록한다. 서버는127.0.0.1에바인딩하며스마트폰의서버주소·HTTPS·인증은팀앱흐름에연결한다. APK를빌드한전달물이아니다.

## C++ 설정

기존 확장 C++ 어댑터의 유형 ID를 그대로 사용한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="mpdd_metal_plate"
```

기존3항목과새11유형을같은모델파일/공통분석부하나로처리한다. 전체유형ID·역할별연결·DB계약은[기존전체틀가이드](EQUIPMENT_EXPANSION_HANDOFF_KO.md)를참고하되실행서버와모델파일은이번ZIP의값을사용한다.

- GET /health: 모델수1·공통분석부수1·새모델SHA.
- GET /v1/equipment-categories: 부품표면15%목표와금속판20%목표상태.
- GET /v1/inference-policy: 기존belt/equipment정책과이번metalPlate정책.
- POST /v1/analyze-hazard, /v1/compare-action, /v1/summarize: 기존계약유지.

새UI는inspectionTask=equipment_surface와equipmentCategory=mpdd_metal_plate를명시할수있다. 금속판사진응답의normalCALAlpha=.20·normalCALRank=17을확인한다. 다른설비는이전정책과기준을유지한다.

동일사진으로조치완료가나와서는안된다. 사진경고기준을조정해미탐을줄인것이며위치pixel미탐27.96%는그대로다. 위치heatmap은점검자확인자료이고확정고장위치가아니다. 내부기계고장은not_assessed,최종확인은사람이수행한다. 위험등급·조치·요약은규칙템플릿이다.

[금속판비교](../ai-training/reports/METAL_REFERENCE_RESULTS_KO.md), [자료출처·사용조건](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md).
