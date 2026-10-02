# 역할 4에게 전달: 시설 모델 연결

## 전달물

1. GitHub `feature/ai-engine`의 최신 커밋.
2. `ai-training/artifacts/safelog-trained-models.zip` 및 `.zip.sha256` 파일.
3. 현재 보강 성능은 `reports/FACILITY_OPTIMIZATION_KO.md`, 초기 기준 성능은 `reports/FACILITY_TRAINING_RESULTS_KO.md`.

2차 추가 학습은 `reports/FACILITY_FEEDBACK_KO.md`에 기록했다. 시험에서 오탐이 증가해
후보를 채택하지 않았고 기본 모델은 1차 `facility-validation-v2`다.
기존 1차 ZIP을 받은 팀원은 최신 브랜치 코드만 업데이트하면 된다. 가중치 교체는 필요 없다.
이번에 다시 만든 기본 ZIP도 동일한 가중치를 포함하며 2차 비교 결과를 추가했다.
`facility-inference-profile-round2-candidate.json` 또는 round2-candidate ZIP은 앱에 적용하지 않는다.

3차 합성 공동 보강도 시험에서 오탐이 증가해 채택하지 않았다.
`reports/FACILITY_ROUND3_KO.md`에 결과와 재현 절차를 남겼다. 기본 모델과 ZIP 가중치는 동일하다.
역할 4가 다음 현장 피드백을 연결할 때는 `FIELD_FEEDBACK_KO.md`의 항목별 정답과
현장/촬영 회차/전후 묶음을 함께 전달한다. 이 문서가 앱 DB/API 변경 구현을 의미하지는 않는다.

학습 데이터 원본과 `.venv`는 전달하지 않아도 된다. 모델 파일은 Git에서 제외되므로
브랜치 코드만 내려받아서는 시설 모델을 실행할 수 없다.

4차 실제 댐 표면 자료도 학습·평가했다. [4차 보고서](reports/FACILITY_ROUND4_KO.md)를 확인한다.
실제 학습 패치 2,009개, 추가 평가 491개이며 새 균열 분류기의 진단 조합에서는 미탐이 줄었다.
박락과 기존 검증의 교체 기준을 통과하지 못해 앱 기본 모델은 계속 v2다.
`safelog-facility-round4-research.zip`은 새 분류기 PT/ONNX와 실험 결과를 보관한 연구용 파일이다.
기본 전달 ZIP을 교체하거나 연구용 설정을 앱 기본 프로필로 복사하지 않는다.
역할 4는 기존 ZIP과 최신 브랜치 코드를 사용하면 된다. 세 API 계약의 변경은 없다.

## 균열·박락 5% 목표 실험

[통합 결과](reports/FACILITY_FIVE_PERCENT_RESULTS_KO.md)에 실제 학습 회수, 세 출처의 항목별 미탐·오탐과 시험 실행 여부를 기록한다.
큰 모델·상세 증강·640 해상도·CODEBRIM 추가·위치 감독 모델을 기존 전달 모델과 분리해 비교한다.
`runs/facility-presence-target-*`는 연구용 후보이다. 현재 기본 API/ZIP의 전체 오류율이 5% 미만임을 확인한 상태가 아니다.
검증 기준은 균열·박락 각각의 미탐률과 오탐률 모두 5% 미만이다. 현장 및 다른 다섯 시설 항목의 성능과 구분한다.
후보의 사진 점수를 기존 탐지 결과와 OR로 합치면 기존 탐지기의 오탐은 줄일 수 없다.
후보가 목표를 통과할 경우 앱에 실제 적용될 판정 규칙까지 고정하고 따로 검증한 뒤 전달한다.
CODEBRIM 추가 모델은 비상업 교육·연구 조건이므로 연구용 배포에도 [원문 조건](https://zenodo.org/records/2620293/files/license.md?download=1)을 포함한다.

## 서버 PC에서 실행

ZIP을 `safelog-cpp/ai-training` 안에 풀어 `models/facility-dacl.pt`,
`models/facility-corrosion.pt`, `models/fire-smoke.pt`가 생기는지 확인한다.
보강 전달본에는 `models/facility-dacl-optimized.pt`, `models/facility-corrosion-optimized.pt`,
`reports/facility-inference-profile.json`도 있어야 한다. 가중치와 설정을 함께 전달한다.
사진 전체 분류 보강은 `models/facility-presence.pt`다. 위치를 확정하지 않은 추가 의견이며
`detections[].box=null`, `evidence_scope=photo_presence`로 전달된다. null 박스에는 사각형을 그리지 않는다.

```powershell
cd safelog-cpp/ai-training
./setup_windows.ps1
./start_ai_server.ps1 -FacilitiesOnly
```

GPU가 없는 서버 PC는 설치할 때 `./setup_windows.ps1 -CpuOnly`를 사용한다.
시설 위주 기본 시연 조합은 시설 모델 2개 + 화재·연기 모델이다.
보강 전달본은 여기에 사진 전체 분류 모델 1개를 함께 사용한다.
보강 설정이 있으면 검증으로 선택된 가중치·해상도·항목별 탐지 기준을 자동 사용한다.
`/health`의 `facilityProfile`, `facilityThresholds`, `inferenceSizes`로 적용 여부를 확인한다.
`facilityProfile`이 `facility-validation-v2`인지 확인하고 `photoClassifiers`에서
`facility-presence`, 입력 크기 384와 항목별 기준을 확인한다.
초기 기준 모델과 비교하려면 실행 명령에 `-BaselineFacilities`를 추가한다.
PPE까지 함께 사용하려면 `-Facilities`, 보조 PPE까지 모두 쓰려면 `-AllModels`다.
같은 PC에서만 시험할 때는 `-LocalOnly`를 추가한다.

서버 PC의 `http://127.0.0.1:8080/health`에서 모델명·파일 존재·입력 크기를 확인한다.
첫 요청에서는 모델을 로드하므로 이후 요청보다 느릴 수 있다.

## 앱 연결

Windows Qt 앱은 실행 전에 다음 환경 변수를 설정한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8080"
$env:SAFELOG_AI_TIMEOUT_MS="60000"
```

Android 앱은 같은 Wi-Fi의 **서버 PC LAN IP**를 설정 화면에서 전달한다.
휴대폰에서 127.0.0.1은 휴대폰 자신이므로 사용하지 않는다.
현재 데스크톱 환경 변수 읽기만으로 Android 설정 화면이 완성되는 것은 아니다.
역할 4가 앱 설정 저장, HTTP 연결 허용 구성, 인터넷/촬영 권한, 취소/오류 흐름을 확인한다.
사진은 JPEG/PNG로 보내며 서버·C++ 어댑터 모두 8MB 제한이 있으므로 필요하면 촬영 후 줄인다.

기존 `HttpAiSafetyAnalyzer`와 세 API 계약은 계속 사용한다.
각 시설 검출은 `detections`에 클래스·모델·신뢰도·박스를 포함하고 전체 응답은
`rawJson`에 보관된다. 박스 좌표는 EXIF 회전을 적용한 입력 사진의 픽셀 기준이다.

## 화면에서 표시할 내용

| 클래스 | 사용자에게 보일 항목 |
|---|---|
| surface_crack | 표면 균열 의심 |
| concrete_spalling | 콘크리트 박리 의심 |
| rust_stain | 녹물·녹 얼룩 의심 |
| exposed_rebar | 철근 노출 의심 |
| wet_surface | 젖은 표면 의심 |
| efflorescence | 백화 의심 |
| surface_cavity | 표면 공동·파임 의심 |
| metal_corrosion | 금속 부식 의심 |

모델의 내부 키 concrete_crack은 원본 Crack와 Alligator Crack을 합친 이름이다.
API에서는 surface_crack으로 반환하며 콘크리트라는 재질을 단정하지 않는다.

위험 등급은 점검 우선순위 규칙이다. 균열 검출을 붕괴 진단으로, 젖은 표면을 배관 누수
확정으로 표시하지 않는다. AI는 결과를 제안하고 점검자가 채택/수정/거절한다.
미탐지가 안전 판정이나 자동 조치 완료가 되지 않도록 최종 확인 단계를 유지한다.
보고서 요약 API는 현재 규칙 템플릿이며 별도로 학습한 LLM은 아니다.

## 통합 확인 순서

1. `/health`에 facility-dacl, facility-corrosion, fire-smoke가 표시되는지 확인한다.
2. 사진 등록 → 시설 AI 제안 → 사람의 수정/채택 → DB 저장을 확인한다.
3. 같은 사진을 조치 전후에 넣었을 때 해결 완료로 바뀌지 않는지 확인한다.
4. 사진을 바꿔도 점검자의 최종 확인 없이 Verified로 넘어가지 않는지 확인한다.
5. 서버 연결 실패, 8MB 초과, 요청 취소 후 수동 입력이 가능한지 확인한다.
6. 시설 검출과 전후 비교의 모델명·원문·채택 이력이 보고서에 남는지 확인한다.
7. Android 실기기에서 촬영 → 전송 → 검토 → 보고서 흐름을 시험한다.

PC API 시험 결과와 데이터셋 시험 성능은 별개다. 성능 보고서를 읽고 모델이 자주
놓치는 항목은 사람이 확인하는 점검표에서 계속 다룬다.
