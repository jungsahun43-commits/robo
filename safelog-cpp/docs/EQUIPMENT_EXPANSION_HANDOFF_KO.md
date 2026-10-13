# 역할4 · 산업 설비 범위 확장 시제품 연결

기존 시설·벨트·이물질과 새 설비11유형을 같은 공통 분석부 하나·모델 파일 하나로 처리한다. 이 전달물은 전체 앱 흐름을 연결하기 위한 연구 시제품이다. 새 유형의 오탐·미탐 목표 통과 여부는 `/v1/equipment-categories`와 실제 성능표를 확인한다. 기존 전달 파일과 앱 기본 모델은 자동 교체하지 않았다. APK를 만든 전달물이 아니다.

## 실행

ZIP을 새 폴더에 풀고 Python3.12에서 실행한다. CPU 서버는127.0.0.1에만 바인딩한다. 제작환경CPU실행과 새로운 pip설치 검증은 구분한다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$equipmentPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.equipment_expansion_server --checkpoint $equipmentPrototype.path --sha256 $equipmentPrototype.sha256 --evaluation evaluation.json --port 8775
```

실제 스마트폰에서는 팀의 서버 주소·인증·HTTPS를 연결한다. 사용환경을 선택한 뒤 설비 사진을 입력하는 화면에 연결한다.

## 기존 C++ 어댑터 연결

기존 `HttpAiSafetyAnalyzerConfig.inspectionTask` 또는 환경변수로 유형 ID를 전달하면 서버가 해당 설비 항목으로 분석한다. 모델을 따로11개 설치하는 설정이 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ksdd2_surface"
```

| 화면 항목 | inspectionTask 값 |
|---|---|
| 시설 표면 | facility_surface |
| 벨트 손상 | belt_damage |
| 컨베이어 이물질 | conveyor_foreign |
| 케이블 | ad_cable |
| 나사 | ad_screw |
| 너트 | ad_metal_nut |
| 전선 연결부 | loco_splicing_connectors |
| 검정/갈색/흰색 브래킷 | mpdd_bracket_black / mpdd_bracket_brown / mpdd_bracket_white |
| 금속 연결부 | mpdd_connector |
| 금속판 | mpdd_metal_plate |
| 금속 튜브 | mpdd_tubes |
| 부품 표면 | ksdd2_surface |

화면에서 선택한 유형을 런타임 config에 전달한다. 전역 환경변수는 개발 시 기본항목을 고르는 예시다. 다른 설비의 사진을 같은 유형으로 보내면 성능표와 같은 조건이 아니다.

## HTTP 계약

- GET `/health`: 모델SHA·모델수1·공통분석부수1.
- GET `/v1/equipment-categories`: 유형 ID·표시이름·학습여부·평가단계.
- GET `/v1/inference-policy`: 기존 벨트와 새 설비의 분석정책SHA.
- POST `/v1/analyze-hazard`: image·inspectionTask·userMemo·promptVersion.
- POST `/v1/compare-action`: beforeImage·afterImage·inspectionTask·actionNote·promptVersion.
- POST `/v1/summarize`: inspectionContext·promptVersion.

새 UI는아래명시적형식도사용할수있다. 기존C++별칭방식과같은모델출력이다.

```json
{
  "inspectionTask": "equipment_surface",
  "equipmentCategory": "ad_cable",
  "image": "<JPEG또는PNG base64>",
  "userMemo": "케이블 외관 점검",
  "promptVersion": "equipment-prototype-v1"
}
```

equipment_surface의유형누락·알수없는유형·별칭/유형충돌은422다. 요청하지않은다른설비를정상으로저장하지않는다. 입력은JPEG/PNG8MB이하·12MP이하이며원본을전달한다. 새설비는256RGB로분석하고벨트는기존원본구역정책을유지한다.

## 역할별 연결 작업

1. 역할1: 설비종류선택·사진입력·후보heatmap·점검자의채택/수정/거절 화면.
2. 역할2: 설비ID·종류·사진·점검의견·AI결과·사람최종확인·modelSha256·inferencePolicySha256·점검/조치이력 저장.
3. 역할3: 유형별데이터·학습·오탐/미탐/위치평가와새모델 전달. 20%넘는유형은보강상태를유지.
4. 역할4: 위화면/DB/API를연결하고조치전후·보고서·앱패키징을완성.

위험등급·조치문장·요약은규칙템플릿이며학습된LLM이나학습된위험도판정으로표현하지않는다. 새외관분석head는정상특징복원을실제학습했다. heatmap은거친후보확인자료이고확정고장위치가아니다. 내부고장은not_assessed,현장성능은미검증,사람최종확인은필수다.

[첫 실제 성능](../ai-training/reports/EQUIPMENT_EXPANSION_RESULTS_KO.md), [새 자료 출처](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md), [기존 자료 사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
