# 역할4·역할2: 이물질 보강 모델 전달 v4

## 들어 있는 모델

| 등록 이름 | 클래스 | 역할 |
|---|---|---|
| `belt_crack` | `belt_crack` | 기존 벨트 표면 균열 후보·heatmap |
| `conveyor_foreign_v1` | `conveyor_foreign_object` | 이전1,230장 학습 모델, 비교·복구용 |
| `conveyor_foreign_expanded_v2` | `conveyor_foreign_object` | 새4,558장 학습 모델, 실제 비교 결과 포함 |

새 ZIP의 `models.json`에 각 모델의 파일 경로·SHA·실제 epoch·업데이트 수가 들어 있다. 모델을 명시적으로 선택하며 이전 파일을 덮어쓰지 않는다. 연구 참고 모델 추천은 미리 정한 VAL 비교 기준으로 기록되고, 앱 기본 모델을 자동 변경하지 않는다.

## 새 파일 실행

ZIP을 압축 해제하고, Python3.12와 `environment.json`의 torch/torchvision/numpy/Pillow/fastapi/uvicorn 환경을 사용한다. 기존 역할3의 학습 .venv도 사용할 수 있다.

압축 해제한 폴더의 PowerShell에서 새 이물질 모델을 선택한다:

```powershell
$bundleModels = Get-Content models.json -Raw -Encoding utf8 | ConvertFrom-Json
$selectedVariant = $bundleModels.conveyor_foreign_expanded_v2
python -m safelog_ai.conveyor_foreign_server --checkpoint $selectedVariant.path --sha256 $selectedVariant.sha256 --port 8772
```

이전 이물질 모델을 비교하려면 위 `selectedVariant`를 `conveyor_foreign_v1`로 바꿔 같은 서버를 재시작한다. 같은 포트에서 두 서버를 동시에 실행하지 않는다.

저장소에서는 `safelog-cpp/ai-training`의 `.venv/Scripts/python.exe`를 사용하고 체크포인트 경로 앞에 `runs/`를 붙인다. 데이터·모델은 Git에 없으므로 ZIP을 별도로 받는다.

## 앱 코드가 이어받을 내용

요청·응답 형식은 이전 [이물질 API 문서](CONVEYOR_FOREIGN_API_HANDOFF_KO.md)의 그대로다. 역할4는 새 파일 SHA와 `modelName`을 기록하도록 연결한다.

- `GET /health`
- `POST /v1/equipment/conveyor/foreign/analyze`
- 요청: `equipmentCategory=conveyor_belt`, `sceneType=iron_ore_closeup`, `promptVersion=conveyor-foreign-v1`, JPEG/PNG base64.
- 응답: 사진 유무 분류 `conveyor_foreign_object`, `candidateDetected`, `presenceScore`, 고정0.5, 모델/사진 SHA.
- 이물질 위치 정답이 없어 `localizationAvailable=false`. 이 모델에는 heatmap/이물질 경계 상자가 없다.
- `humanConfirmationRequired=true`, `mechanicalFailureState=not_assessed`, `fieldPerformanceValidated=false`.
- Android USB의 개발 PC 연결은 `adb reverse tcp:8772 tcp:8772`를 사용한다.

역할4는 점검 종류 선택 → 벨트 근접 사진/자르기 안내 → AI 제안 → 사람 확인 → 이물질 제거/조치 → 재점검/보고서를 연결한다. 역할2는 모델 버전·전송 사진·SHA·AI 원문과 사람의 채택/수정/거절을 각각 저장한다. 통신 오류는 분석 실패로 기록하고 미검출/정상으로 바꾸지 않는다.

이물질 없음과 균열 없음, 정상 가동을 구분한다. 이물질 자료는 철광석 흐름의 잘린 벨트 사진이다. 다른 운반물/카메라/공장의 성능은 별도 확인해야 한다.

## 성능 문서 읽는 순서

1. [보강 결과와 기존 미탐8장 전체](../ai-training/reports/CONVEYOR_FOREIGN_EXPANSION_RESULTS_KO.md)를 읽는다.
2. 같은 VAL320장·TEST448장의 이전/보강 오탐·미탐을 함께 본다. 기존 TEST는 이미 결과가 알려진 공개 비교 자료이며 새로운 독립 최종 시험이 아니다.
3. 추가 VAL309장은 처음 평가한 사진이지만 같은3개 촬영명이다. 새 공장 성능으로 소개하지 않는다.
4. 기존 모델을 개선했다고 소개할 때 바로잡은 오류와 새 오류를 모두 기록한다. 특정 비율이5% 미만이어도 전체 설비 고장 성능과 합치지 않는다.
5. [기존 이물질 결과](../ai-training/reports/CONVEYOR_FOREIGN_RESULTS_KO.md)와 [벨트 균열 결과](../ai-training/reports/BELTCRACK_EXPANSION_RESULTS_KO.md)는 각각 유지한다.

모델은 실제30epoch 학습이며 초기 가중치·입력 사진 보존·strict CPU/GPU 재로드·실제 HTTP 호출 기록을 함께 제공한다. 전달 ZIP은 모델 서버와 팀 연결 자료다. 모바일 앱 전체 연결/빌드는 역할4가 진행한다.
