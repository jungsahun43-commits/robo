# 역할4 · 원본 세부 정보를 보존한 나사 AI 연결

## 현재 측정 결과

같은 공개 평가160장에서 나사 사진 오탐2/41=4.88%, 미탐15/119=12.61%를 측정했다. 기존 오탐19.51%·미탐85.71%에서 개선했다. 신규11유형 중5개가 사진20% 이내이며 그중4개가15% 이내다. 케이블·너트·금속판·부품 표면의 이전 결과를 유지했다.

**전체 나사 사진의 합산 수치다. 나사산 옆면 손상은 미탐13/23=56.52%가 남아 있다.** 모든 손상 종류가15% 이내라는 뜻으로 표시하면 안 된다. 알려진 공개 평가자료로 반복 개발한 연구 시제품이며 새로운 현장 검증은 남아 있다.

## CPU 서버 실행

새 ZIP을 새 폴더에 푼다. Python3.12와 기존 의존성에 이번 원본 처리용OpenCV가 추가된다. 아래는 현재 검사 환경의 버전이며 새 pip 설치 자체를 이번 검증에서 실행한 것은 아니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$nativePrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.native_component_server --checkpoint $nativePrototype.path --sha256 $nativePrototype.sha256 --evaluation evaluation.json --port 8775
```

이번 서버와model.json을 함께 사용한다. 이전 서버는 새 모델 구조를 읽을 수 없다. 서버는127.0.0.1에서 실행되며 스마트폰 주소·HTTPS·인증은 기존 팀 앱 구조에 연결한다. 이번 전달물은 APK가 아니다.

## C++ 연결

기존 C++ 어댑터에서 유형ID는그대로다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ad_screw"
```

나사 하나가 선명하게 나온 원본 사진을 보낸다. 서버가사진만으로배경과주요축을찾아방향을맞추고768×256비율을유지해비교한다. 작은256×256정사각형으로미리줄여보내지않는다. 판별사진과원래위치출력의보존을위해같은공통신경망을두번사용한다. 모델1개·공통분석부1개이며나사응답의backbonePassesPerPhoto=2다.

어두운부분을분리하거나방향을정하기어려우면422를반환한다. 화면은다시촬영/수동검수로이어가야한다. 이경우를정상또는고장미검출로저장하지않는다.

## 화면·DB 필드

| 필드 | 의미 |
|---|---|
| candidateDetected / classResults | 사진의 나사 외관 이상 후보 |
| heatmapPngBase64 | 기존 위치 출력. 이번에도 동일하며 위치 정확도는 낮음 |
| referenceHeatmapPngBase64 | 원본 좌표로 되돌린 정상 특징 차이 참고 이미지 |
| referenceValidRegionMaskPngBase64 | 참고 이미지가 비교한 영역. 바깥은 정상으로 판단한 영역이 아님 |
| nativeViewImagePngBase64 / nativeReferenceHeatmapPngBase64 | 방향 보정 사진과 같은 좌표의 차이 이미지 |
| nativeViewGeometry | 원본과 보정 사진 사이의 입력 기반 변환 |
| defectGroupBenchmark / allDefectGroupsWithin20 | 손상 종류별 한계. 나사산 옆면 미탐56.52%, 전체 종류20% 통과false |
| benchmarkStageScope | pooled_photo_category_on_known_TEST. 전체 현장·모든 결함의 보장이 아님 |
| modelSha256 / inferencePolicySha256 | 모델과 나사 경고 정책을 함께 저장 |
| normalCALAlpha / normalCALRank | 정상CAL24장 기준.15/rank22. 실제 오탐률 보장치가 아님 |

referenceHeatmapIsEvaluatedLocalization=false이며정상특징차이이미지는정답으로검증한손상분할결과가아니다. 기존위치출력의나사pixel미탐97.20%는그대로다. 사진미탐12.61%와구분한다. confidence/raw score도실제고장확률로표시하지않는다.

최종판정은점검자가한다. 내부기계고장은mechanicalFailureState=not_assessed이고위험등급·조치·요약은기존규칙템플릿이다. 새볼스크루정상/손상자료는준비했지만이번모델에학습한자료가아니다.

GET /v1/inference-policy의nativeComponents.ad_screw와GET /v1/equipment-categories에서정책·합산사진상태·손상종류별한계를확인한다. 나머지요청은기존계약을유지한다.

- [실제 비교 결과](../ai-training/reports/NATIVE_SCREW_RESULTS_KO.md)
- [기존 전체 연결 안내](EQUIPMENT_REFERENCE_HANDOFF_KO.md)
- [기존 학습자료 출처](../ai-training/reports/EQUIPMENT_EXPANSION_ATTRIBUTION_KO.md)
- [이번 ZIP 연결 검사 기록](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/ai-training/reports/equipment-native-screw-budget-v7-release.json)
