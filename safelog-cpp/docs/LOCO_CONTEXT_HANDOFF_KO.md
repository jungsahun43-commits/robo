# 역할4 · 전선 연결부 전체 문맥 보강 전달

## 변경 내용

`loco_splicing_connectors`의 사진 미탐이66.84%→39.90%로 줄었고 오탐19.33%는 같다. 같은 공개 TEST를 재사용한 개발 수치이며 **20% 목표 미달**이다. 구성 부품의 수·배선 관계를 정확히 인식한다고 보장하지 않는다.

같은 모델 안에 정상 사진의 전체 공간 특징을 복원하는 분기를 추가했다. 기존 위치 지도·기준, KSDD2 보강과 볼스크루·다른 설비를 유지한다.

## CPU 서버 실행

새 ZIP을 새 폴더에 푼다. 체크포인트는 model.pt 하나다. 의존성은 v11 전달물과 같다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$contextPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.loco_context_server --checkpoint $contextPrototype.path --sha256 $contextPrototype.sha256 --evaluation evaluation.json --port 8775
```

검사는 기존 설치 환경에서 실행한다. 새 pip 설치 자체와 Android APK는 검증한 것이 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="loco_splicing_connectors"
```

요청 경로와 C++ 유형 ID는 같다. 스마트폰 연결 주소·HTTPS·로그인은 팀 앱 서버 구조에 맞춘다. 모델 이름·SHA와 정책 SHA는 새 값으로 보관한다.

## 화면과 DB에서 쓸 출력

| 필드 | 의미 |
|---|---|
| classResults / candidateDetected | 문맥·기존 표면 점수를 결합한 사진 이상 후보 |
| globalContextScore / localAppearanceScore | 결합 전의 원점수. 실제 고장 확률이 아님 |
| heatmapPngBase64 | 그대로 보존한 기존 위치 출력 |
| originalPixelThreshold / originalLocalizationChanged | 기존 위치 경고 기준, changed=false |
| benchmarkStage / defectGroupBenchmark | 목표 미달 및 논리/표면 결함 묶음별 미탐 |
| semanticClampWireRelationsVerified | false. 클램프 수·전선 끝점의 지도학습 정답을 사용하지 않음 |
| inferencePolicySha256 / modelSha256 | 실제 정책과 모델의 식별값 |

`GET /v1/inference-policy`의 `globalContext.loco_splicing_connectors`에서 조건을 확인한다. 사진 점수는 보정된 확률이 아니며 내부 고장은 `not_assessed`다. 작은 입력 등422 오류를 정상 결과로 저장하지 않는다. 점검자가 최종 확인한다.

- [실제 학습·결과·한계](../ai-training/reports/LOCO_CONTEXT_RESULTS_KO.md)
- [KSDD2 출력 안내](KSDD_PHOTO_HANDOFF_KO.md)
