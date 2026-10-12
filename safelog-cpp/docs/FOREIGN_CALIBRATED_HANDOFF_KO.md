# 역할4·역할2: 이물질 보정 기준 모델 v5

## 역할4가 연결할 부분

v5 ZIP에는 기존 벨트 균열 모델과 새 이물질 보정 기준 모델이 들어 있다. 이전 이물질 모델은 v3/v4 ZIP으로 보존한다.

ZIP을 압축 해제하고 기존 Python3.12·torch/torchvision/numpy/Pillow/fastapi/uvicorn 환경에서 PowerShell을 실행한다:

```powershell
$bundleModels = Get-Content models.json -Raw -Encoding utf8 | ConvertFrom-Json
$selectedVariant = $bundleModels.conveyor_foreign_calibrated_v3
python -m safelog_ai.foreign_calibrated_server --checkpoint $selectedVariant.path --sha256 $selectedVariant.sha256 --port 8772
```

`GET /health`, `POST /v1/equipment/conveyor/foreign/analyze`를 사용한다. Android USB 개발 PC 연결은 `adb reverse tcp:8772 tcp:8772`다. CPU 로컬 서버이며 스마트폰 자체에서 Python 모델을 실행하는 구성은 아니다.

요청은 이전과 같다:

```json
{
  "equipmentCategory": "conveyor_belt",
  "sceneType": "iron_ore_closeup",
  "promptVersion": "conveyor-foreign-calibrated-v2",
  "image": "<JPEG/PNG base64>"
}
```

## 기존 C++ 연결에서 바꿀 부분

- 응답 schema를 **`safelog.equipment.conveyor-foreign.v2`**로 확인한다. 기존 v1 schema만 검사하는 코드는 이 새 결과를 거절하므로 해당 분기를 추가한다.
- 판단은 서버의 `candidateDetected`를 사용한다. 클라이언트에서 `presenceScore > 0.5`를 다시 계산하지 않는다.
- `threshold`가 체크포인트에 고정된 실제 기준이다. 요청/화면에서 사진마다 기준을 임의로 변경하지 않는다.
- `presenceScore`는 보정된 확률이 아니다. 낮은 raw 점수를 정상/안전 확률로 표시하지 않는다.
- `localizationAvailable=false`: 이물질 위치를 그리지 않는다. 표면 균열은 별도8771번의 기존 모델/heatmap API를 사용한다.
- `fieldPerformanceValidated=false`, `humanConfirmationRequired=true`, `mechanicalFailureState=not_assessed`는 유지한다.

UI는 이물질 의심/이번 사진에서 미검출 → 점검자 채택·수정·거절 → 제거/점검 조치 → 재점검 사진/보고서로 연결한다. 오류 응답은 분석 실패로 남기고 수동 검토를 제공한다.

## 역할2 저장 필드

설비·원본/전송 사진 ID, `schema`, `className`, `presenceScore`, `threshold`, `candidateDetected`, `modelName`, `modelSha256`, `inputSha256`, `calibrationProtocolSha256`, `promptVersion`, 분석 시각을 저장한다.

사람 확인과 AI 원문은 각각 보존한다. 점검자 ID·채택/수정/거절·의견·확인 시각·담당자 조치·재점검 사진을 기록한다. 자른 사진의 SHA와 원본 사진 SHA를 혼동하지 않는다.

## 실제 성능을 소개할 범위

추가 학습은 실제30epoch·4,558장·4,290회 업데이트다. 그 뒤 검출 기준을 CAL320장에서 보정했으며 **보정 자체는 신경망 추가 학습0회**다.

그 다음 처음 점수를 계산한783장에서 오탐0/399·미탐0/384다. 같은 기존 TEST448장에서는 미탐0/192, 오탐2/256=0.78%다. 기존 TEST의 오탐이1→2건 늘어난 점도 함께 기록한다.

CAL은 기준을 맞춘 자료이므로 독립 검증 성능으로 소개하지 않는다. 새783장도 알려진4 TEST 촬영명의 다른 프레임이다. 독립 공장·기계 고장·스마트폰 전체 장면 성능이 아니다. 모델 적용 범위는 철광석 흐름의 잘린 벨트 근접 사진이다.

[실제 보강 실패와 기준 보정 결과](../ai-training/reports/FOREIGN_CALIBRATION_RESULTS_KO.md)를 함께 전달한다. 앱 기본 모델 자동 교체와 모바일 앱 전체 빌드는 이 전달물에 포함되지 않으며, 역할4가 기존 앱에 연결한다.
