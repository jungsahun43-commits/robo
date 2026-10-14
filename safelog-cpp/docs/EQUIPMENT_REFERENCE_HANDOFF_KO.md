# 역할4 · 케이블·너트 정상 참조 연결

## 전달받는 기능

사진 오탐·미탐을 줄인 연구 연결용 버전이다. 신경망 공통 분석부는 하나이며, 정상 케이블·너트의 특징 버퍼를 같은 모델 파일에 추가했다.

| 유형 | 사진 오탐 | 사진 미탐 | 공개 평가 사진 |
|---|---:|---:|---|
| 케이블 | 10.34% (6/58) | 7.61% (7/92) | 150장 |
| 너트 | 13.64% (3/22) | 11.83% (11/93) | 115장 |
| 금속판 | 3.85% | 18.31% | 이전 결과 유지 |
| 부품 표면 | 12.64% | 10.91% | 이전 결과 유지 |

신규11유형 중4개가 사진20% 이내이며 그중3개가15% 이내다. 나머지7개는 보강/평가 부족 상태다. 같은 공개 평가 사진을 반복 사용해 후보를 선택한 개발 결과이며, 새로운 산업현장 검증 결과가 아니다. 위치 기능은 별도 성능으로 표시한다.

## 1. 실행

새 ZIP을 새 폴더에 푼다. Python3.12에서 기존 시제품과 같은 의존성을 사용한다. 이번 확인에 새 pip 설치 검사는 포함되지 않았다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$referencePrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.equipment_reference_budget_server --checkpoint $referencePrototype.path --sha256 $referencePrototype.sha256 --evaluation evaluation.json --port 8775
```

모델 파일에는 정상 특징90,624개가 포함되어 이전 금속판 버전보다 크다. 이 서버 모듈과 ZIP의 model.json을 함께 사용한다. 이전 서버로 새 파일을 읽지 않는다. 모델 파일과 평가 JSON의 SHA가 일치해야 실행된다.

서버는127.0.0.1에 연결된다. 스마트폰 실연결 주소·HTTPS·로그인·인증은 기존 팀 앱 구조에서 연결한다. 이번 전달물은 APK가 아니다.

## 2. C++ 요청

기존 C++ 어댑터를 그대로 사용한다. 유형별로 SAFELOG_AI_INSPECTION_TASK를 지정한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ad_cable"
```

너트는ad_metal_nut, 금속판은mpdd_metal_plate, 부품 표면은ksdd2_surface다. 나머지 유형 ID는 [전체 틀 가이드](EQUIPMENT_EXPANSION_HANDOFF_KO.md)를 참고한다. 기존3항목과 신규11유형을 한 파일에서 처리한다.

새 UI는inspectionTask=equipment_surface와equipmentCategory=ad_cable 또는ad_metal_nut를 명시해도 된다. 사진은 기존처럼 원본 파일 bytes를 base64로 보낸다.

## 3. 화면과 DB에서 구분할 값

| 응답 필드 | 역할4 화면·역할2 저장 내용 |
|---|---|
| candidateDetected / classResults | 해당 부품의 사진 외관 이상 후보 |
| photoScoreSource=normal_reference | 사진 판별이 정상 특징 비교에서 나온 결과 |
| referenceHeatmapPngBase64 | 사진 점수의 참고 시각화. 확정 손상 영역으로 표시하지 않음 |
| heatmapPngBase64 / heatmapSource=original_reconstruction | 기존 위치 표시. 이번에 위치 성능을 개선한 출력이 아님 |
| modelSha256 / inferencePolicySha256 | 모델 파일과 적용 경고 정책을 함께 기록 |
| normalCALAlpha / normalCALRank | 케이블.05/rank44, 너트.10/rank41 |
| benchmarkStage | 해당 유형의 사진 목표 상태. 앱 전체 정확도로 표시하지 않음 |
| mechanicalFailureState=not_assessed | 사진만으로 내부 작동 고장을 확정하지 않음 |
| humanConfirmationRequired | 점검자의 채택·수정·거절 기록 필요 |

normalCALAlpha는 정상 보정 사진에서 경고 기준을 정하는 비율이며 실제 오탐률을 보장하는 수치가 아니다. confidence와raw score도 고장 확률로 표시하지 않는다.

케이블의 손상pixel 미탐76.03%, 너트87.50%는 이전과 같다. 사진 판별15% 통과를 위치 미탐15% 통과로 표시하면 안 된다. 참조 heatmap은 위치 정답으로 평가한 분할 결과가 아니다.

## 4. 확인 순서

1. GET /health에서ready, 모델1개·공통 분석부1개, model.json의SHA를 확인한다.
2. GET /v1/equipment-categories에서11유형과 목표 상태를 확인한다.
3. GET /v1/inference-policy의referencePhoto에서 케이블·너트의 정책SHA와alpha/rank를 확인한다. 기존metalPlate·legacyBelt 정책도 유지된다.
4. POST /v1/analyze-hazard로 사진을 넣고 두 heatmap을 구분해서 연결한다.
5. POST /v1/compare-action에 같은 사진을 넣으면 조치 완료로 표시되지 않아야 한다.
6. POST /v1/summarize의 요약을 보고서 초안에 연결하고 점검자 최종 확인을 받는다. 등급·조치·요약은 규칙 템플릿이다.

이번 역할4 전달물의 실제 검사 기록은 [GitHub release JSON](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/ai-training/reports/equipment-reference-budget-v2-release.json)에 남긴다. 기록의 ZIP SHA와 전달받은 파일을 대조한다. 새 모델 연결 전에는 이전 전달 ZIP도 보관한다.

- [비교 결과·실패 후보·평가 한계](../ai-training/reports/EQUIPMENT_REFERENCE_RESULTS_KO.md)
- [자료 출처·연구 사용 조건](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
