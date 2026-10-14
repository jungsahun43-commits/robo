# 역할4 · 설비별 야간 개선 모델 연결

## 전달물과 실행

이 ZIP에는 모델 하나, Python CPU 서버와 C++ 어댑터 소스가 들어 있다. 전체 앱 또는 APK는 아니다. 이전 ZIP과 다른 새 폴더에 압축을 푼다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
$industrialPrototype = Get-Content -Raw model.json | ConvertFrom-Json
./.venv/Scripts/python.exe -m safelog_ai.industrial_batch_server --checkpoint $industrialPrototype.path --sha256 $industrialPrototype.sha256 --evaluation evaluation.json --port 8775
```

현재 검증 환경의 의존성 버전이다. 이번에 새 pip 설치 자체를 검증한 것은 아니다. `GET /health`, `/v1/equipment-categories`, `/v1/inference-policy`에서 모델 SHA와 유형·경고 정책을 확인한다.

## 앱에서 선택할 점검 항목

| 항목 | 유형 ID | 입력과 결과 |
|---|---|---|
| 새 볼스크루 표면 피팅 | ball_screw_pitting | 볼스크루 표면을 잘라낸 선명한 사진. 사진 정상/손상 후보만 제공, 위치 표시는 없음 |
| 갈색 브래킷 | mpdd_bracket_brown | 지정 부품 사진. 기존 위치 출력 + 별도 정상 차이 참고 이미지 |
| 흰색 브래킷 | mpdd_bracket_white | 원본 비율·세부 정보 보존. 기존 위치 출력 + 별도 정상 차이 참고 이미지 |
| 일반 체결나사 | ad_screw | 나사 한 개를 담은 원본 사진. 입력 방향이 모호하면 422·재촬영/수동 확인 |

기존 시설·벨트·이물질과 나머지 설비 유형 ID도 유지한다. 볼스크루와 일반 체결나사를 같은 항목으로 합치지 않는다. 전체 공장 사진에서 자동으로 설비 종류를 골라내는 모델은 아니다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8775"
$env:SAFELOG_AI_INSPECTION_TASK="ball_screw_pitting"
```

새 볼스크루 ID를 허용한 C++ 어댑터가 함께 들어 있다. 저장소의 같은 경로에 반영하거나 AI 브랜치를 합친다. 기존 HTTP API 경로와 분석·전후 비교·요약 요청은 같다. 기기 주소·HTTPS·로그인·권한은 팀 앱 서버 구조에 연결한다.

## 화면과 DB에 보관할 필드

- `modelSha256`, `inferencePolicySha256`, 원본 사진, 점검자 채택/수정 결과를 함께 보관한다.
- `benchmarkStage`는 사진 단위 합산 상태이며 `benchmarkStageScope`, `defectGroupBenchmark`의 범위·종류별 한계를 같이 표시한다.
- 볼스크루의 `requiredInput=cropped_ball_screw_surface`, `localizationAvailable=false`, `inputDomainAutoVerified=false`를 따른다. 위치 박스나 heatmap을 임의로 만들지 않는다.
- 브래킷의 `referenceHeatmapPngBase64`는 정상 차이 참고 이미지이며 `referenceHeatmapIsEvaluatedLocalization=false`다. 기존 `heatmapPngBase64`의 위치 정확도가 좋아졌다는 의미는 아니다.
- 나사산 옆면 미탐 56.52%는 남는다. 전체 나사 사진 미탐 12.61%와 구분한다.
- 판독 불가 422는 정상으로 저장하지 않고 다시 촬영하거나 수동 검수한다. 내부 기계 고장은 `mechanicalFailureState=not_assessed`다.
- confidence·sigmoid score는 실제 고장 확률로 보정된 값이 아니다. 위험 등급·조치·요약은 규칙 템플릿이고 점검자가 최종 확인한다.

볼스크루 공개 예약 평가의 오탐 0.85%·미탐 0.63%는 부품을 잘라낸 사진 기준이다. 물리적 설비·촬영 프레임 그룹을 확인하지 못해 새 현장 성능은 아직 검증하지 않았다. 기존 설비 11유형은 같은 공개 TEST를 재사용한 개발 수치다.

- [설비별 결과와 모든 후보](../ai-training/reports/INDUSTRIAL_NIGHT_RESULTS_KO.md)
- [C++ 어댑터](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp)
- [C++ 헤더](../apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.hpp)
