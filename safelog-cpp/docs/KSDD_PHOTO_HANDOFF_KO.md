# 역할4 · 부품 표면 사진 미탐 보강 모델

## 적용 범위

`ksdd2_surface` 사진 판별 미탐이 10.91%→4.55%, 오탐이 12.64%→12.30%로 줄었다. 같은 공개 TEST를 재사용한 개발 수치다. 기존 위치 지도는 그대로다. 새 위치 지도 후보는 미탐이 더 커 기본 출력으로 채택하지 않았다.

## CPU 서버 실행

새 ZIP을 새 폴더에 푼다. 모델 파일은 하나다. 이전 전달 ZIP과 의존성은 같고 새 서버 모듈과 model.json을 함께 쓴다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$surfacePrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.ksdd_photo_only_server --checkpoint $surfacePrototype.path --sha256 $surfacePrototype.sha256 --evaluation evaluation.json --port 8775
```

이번 검증은 기존 설치 환경에서 실행했으며 새 pip 설치 자체나 Android APK는 검증한 것이 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ksdd2_surface"
```

C++ 유형 ID와 요청 경로는 같다. 다른 설비 유형도 이전과 같이 사용한다. 스마트폰의 주소·HTTPS·로그인 연결은 팀 앱 서버 구조에 맞춘다.

## 출력·화면·DB

| 필드 | 의미 |
|---|---|
| classResults / candidateDetected | 새 실제 마스크 학습 판별기의 사진 이상 후보 |
| heatmapPngBase64 | 기존 위치 출력. 바이트가 같은 것을 검사하며 기본 지도로 유지 |
| originalPixelThreshold / originalLocalizationChanged | 기존 위치 기준 유지, changed=false |
| supervisedAuxiliaryHeatmapPngBase64 / supervisedAuxiliaryMaskPngBase64 | 새 지도 참고 출력. 기본 지도와 구분 |
| auxiliaryLocalizationWorseFNR | true. 새 위치 미탐26.81%로 기존17.10%보다 큼 |
| inferencePolicySha256 / modelSha256 | 사진·위치 정책과 모델을 함께 보관 |
| backbonePassesPerPhoto | 2. 같은 공통 분석부를 원래 위치와 새 사진 점수에 각각 사용 |

사진 점수는 실제 고장 확률로 보정된 값이 아니다. 내부 기계 고장은 `not_assessed`이고 점검자가 최종 확인한다. 64픽셀보다 작은 입력이나 판독 불가 오류를 정상으로 저장하지 않는다.

`GET /v1/inference-policy`의 `supervisedPhoto.ksdd2_surface`와 `/v1/equipment-categories`에서 정책과 평가 범위를 확인한다. 나사산 옆면의 기존 미탐 등 다른 유형의 한계도 유지된다.

- [전체 결과·자료 분리·제외 이유](../ai-training/reports/KSDD_PHOTO_ONLY_RESULTS_KO.md)
- [이전 설비별 안내](https://github.com/jungsahun43-commits/robo/blob/feature/ai-engine/safelog-cpp/docs/INDUSTRIAL_NIGHT_HANDOFF_KO.md)
