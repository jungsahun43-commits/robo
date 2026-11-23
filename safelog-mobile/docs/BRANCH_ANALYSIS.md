# SafeLog 원격 브랜치 분석

**2026-11-23 통합 기준:** 최신 `feature/mobile-storage-auth`의 `418024ee98b6f0ffe947098533506f777bff686c` 위에 실제 AI 연결을 통합한 별도 로컬 작업본은 `codex/mobile-db-ai`다. 현재 저장·계정·AI 연결과 검증 범위는 [README](../README.md), [DB·AI 통합 안내](DB_AI_INTEGRATION.md)를 따른다. 아래 표는 최초 모바일 이전 당시의 브랜치 이력이며 현재 원격 목록이 아니다. `feature/workflow-report`는 모든 커밋의 AI 브랜치 포함 여부와 복구 번들을 확인한 뒤 2026-11-23 원격에서 정리했다. 다른 역할의 고유 변경과 열린 PR 브랜치는 유지했다.

이 문서는 모바일 최초 커밋의 입력이 된 기존 6개 브랜치 분석 기록이다. 이후 `feature/react-native-mobile`의 `c422eb5d7f0445a0e6d3bdf55b34b2e5916fc8e9`가 추가돼 원격 브랜치는 7개로 확인됐다. 해당 커밋의 부모는 아래 main SHA와 일치한다. 2026-11-19 고정 CPU 연결 추가 내용은 [앱 README](../README.md)와 [추가 검증 기록](TEST_REPORT.md)을 따른다.

`git ls-remote --heads`와 별도 미러 `C:\dev\robo-source`를 대조했다. 원격 헤드는 아래 6개다. `C:\dev\SafeLog` 디렉터리는 존재하지만 Git 저장소로 인식되지 않았다. 모바일은 첫 커밋 전 Expo 템플릿이 스테이징되어 있었으며 별도 인덱스 변경 없이 작업했다.

| 브랜치 | HEAD SHA | main 기준 뒤/앞 | main 트리 대비 A/M/D | 최신 커밋 및 구현 |
|---|---|---:|---:|---|
| main | `53ca81fe20cf6065d460bbee972c05bacd6df023` | 0/0 | 0/0/0 | main 동기화 안내. 공통 contracts, C++ 서비스, 메모리 저장소, SQLite 초기 스키마, Qt·CLI 시작점과 평가 샘플. |
| feature/auth | `67b8b3e341314945e822ad35946ad761915fff9a` | 0/15 | 28/4/0 | 시연 프로필 공유. `AuthService`, credential port, Monocypher Argon2id, 계정 관리, SQLite 자격 증명, 인증·저장소 테스트. |
| feature/ai-engine | `3776729e08a7b11511adcae303457fbd312d9dcf` | 0/340 | 2164/19/0 | 역할 2·4 통합 안내. 학습·평가·추론 서버, Qt HTTP 어댑터, 설정과 도구, workflow-report의 모든 커밋 포함. |
| feature/capture-ai-ui | `cac44c74abc9e07a14eafb7fb0af40cf75d95c7b` | 4/2 | 8/9/1 | 읽을 수 없는 응답 Mock, 고유 ID. 촬영 컨트롤러와 3개 QML, 경량 시연 어댑터 및 전달 문서. |
| feature/storage-ai | `a7938abc722d427de8a9d77269bd382854f58187` | 0/7 | 11/4/0 | UUID 및 시연 초기 데이터. SQLite repository·codec·schema migration·테스트, SQLite amalgamation. |
| feature/workflow-report | `31231744e3057efd8ff3891882212bb699fcc43b` | 0/1 | 31/11/0 | 인증 Mock, 역할별 QML 9개, 사진 등록·조치·재조치·검토, HTML 보고서와 공유 Mock, controller 및 상태 테스트. |

`safelog-cpp/`의 주요 서브트리별 추적 파일 수다. 모든 브랜치에 database 1개, 공통 include 3개, evaluation 2개가 있다.

| 브랜치 | apps | src | tests | docs | third_party | ai-training |
|---|---:|---:|---:|---:|---:|---:|
| main | 7 | 22 | 2 | 7 | 0 | 0 |
| auth | 7 | 41 | 4 | 7 | 7 | 0 |
| AI engine | 42 | 22 | 4 | 38 | 0 | 2095 |
| capture | 14 | 23 | 2 | 7 | 0 | 0 |
| storage | 7 | 29 | 3 | 7 | 3 | 0 |
| workflow | 36 | 22 | 4 | 7 | 0 | 0 |

## 충돌 및 통합

- capture-ai-ui와 main의 공통 조상은 `4dabe0c3fc5503845a22275f7351005fb691368e`다. 나머지 feature 헤드는 main의 후손이다. capture의 main 대비 삭제 한 건은 `safelog-cpp/.gitignore`가 공통 조상 이후 main에 추가된 결과다. 모의 병합에서 main과 capture 간 텍스트 충돌은 없었다.
- capture와 workflow-report의 모의 병합은 10건 충돌했다. `apps/mobile-qt/{CMakeLists.txt,app_controller.cpp,app_controller.hpp,main.cpp,qml/Main.qml}`에서 수정 충돌, `controllers/capture/capture_controller.{cpp,hpp}`와 `qml/capture/{AiHazardReviewPage,FindingCapturePage,FindingSavedPage}.qml`에서 양쪽 추가 충돌이 발생했다. capture의 `capture_id_generator.hpp`, `capture_mock_ai_analyzer.hpp`, `src/capture/CAPTURE_AI_HANDOFF_KO.md`는 별도 구현이다. 모바일 화면은 workflow-report를 기준으로 하고 capture의 실패 응답 시연과 ID 중복 방지 요구를 반영했다.
- auth는 storage-ai를 조상으로 포함한다. SQLite 구현을 중복 이식할 이유가 없다. SQLite 포트에 credential store, 계정 데이터와 인증 테스트를 추가하는 방향이다.
- ai-engine은 workflow-report보다 339커밋 앞이며 원본 UI 외에 `apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.*`, `AI_PROVIDER_KO.md`, `tests/http_ai_safety_analyzer_tests.cpp`, 평가 도구를 추가했다. auth 및 storage-ai와 각각 모의 병합했을 때 `safelog-cpp/tests/CMakeLists.txt`가 충돌했다. Qt 통합이 필요하다면 CMake 테스트 등록과 Mock 인증·메모리 저장 주입을 먼저 정리해야 한다.
- 공통 계약의 핵심은 Profile·Inspection·Finding·Photo·ActionLog·AiAnalysis, Repository 및 AI 포트다. C++ 모듈과 SQLite 파일을 React Native에서 직접 호출하려면 네이티브 브리지가 필요하다. 모바일은 타입과 업무 규칙을 옮기고, 실제 AI는 HTTP 서버 계약을 재사용할 수 있다.

모든 헤드 15쌍을 모의 병합했다. 아래 표는 텍스트 충돌 파일 수다.

| 기준 ↓ / 상대 → | main | auth | AI engine | capture | storage | workflow |
|---|---:|---:|---:|---:|---:|---:|
| main | — | 0 | 0 | 0 | 0 | 0 |
| auth | | — | 1 | 0 | 0 | 1 |
| AI engine | | | — | 10 | 1 | 0 |
| capture | | | | — | 0 | 10 |
| storage | | | | | — | 1 |
| workflow | | | | | | — |

1건인 모든 쌍은 `tests/CMakeLists.txt`에서 충돌했다. AI engine–capture와 workflow–capture에서 충돌한 10개 경로는 동일하다. 이 검사는 브랜치 참조를 변경하지 않고 Git의 삼자 병합 결과를 계산했다.

## 주요 변경 경로 및 테스트

경로는 `safelog-cpp/` 기준이다.

- main: `include/safelog/contracts/`, `src/{ai,capture,storage,workflow,reporting}/`, `apps/demo/`, `tests/smoke_tests.cpp`, `docs/INTEGRATION_CHECKLIST.md`, `evaluation/cases.csv`.
- auth: 등록 ID 정규화, Argon2id 검증, 연속 5회 실패 시 60초 잠금, 비밀번호 변경, Manager 계정 발급을 구현한다. DB 스키마 버전 2에서 credentials를 추가한다. `src/auth/`, `src/storage/in_memory_credential_store.cpp`, `src/storage/sqlite_repository.cpp`, `database/schema.sql`, `tests/{auth_tests,sqlite_repository_tests}.cpp`.
- capture: `apps/mobile-qt/controllers/capture/`, `apps/mobile-qt/qml/capture/`, `src/capture/`; handoff 문서와 공통 테스트 트리.
- storage: `src/storage/{sqlite_repository.cpp,sqlite_codec.cpp,uuid_id_generator.cpp,schema_sql.hpp.in}`, `database/schema.sql`, `tests/sqlite_repository_tests.cpp`.
- workflow: `apps/mobile-qt/{adapters,controllers,qml}/`, `src/{reporting,workflow}/`, `apps/mobile-qt/tests/controller_tests.cpp`, `tests/{mobile_support_tests,workflow_report_tests}.cpp`.
- AI: `ai-training/{safelog_ai,scripts,configs,datasets,reports,tests}/`, `ai-training/README_KO.md` 및 `FACILITY_HANDOFF_KO.md`, Qt HTTP 어댑터 및 테스트. Python과 JS 학습·검토·회귀 테스트는 위치와 구조를 확인했다. 실험 자료 전체 및 원본 C++·Python 테스트는 실행 범위에 포함되지 않았다.

## AI 실행물과 실험 자료

- 실행 경로는 `ai-training/start_ai_server.ps1`, `safelog_ai/server.py`, 선택한 추론 프로필 및 별도 배포 가중치다. `server.py`는 base64를 디코딩하고 EXIF 방향을 보정한 RGB 사진을 YOLO 입력 BGR로 변환한다. `unified_main_server.py`는 모듈에 따라 640×640 리사이즈 또는 패딩을 적용하는 별도 구현이다.
- API는 `POST /v1/analyze-hazard` (image, userMemo, promptVersion → 범주, 1~5 등급, 탐지 배열, 설명/조치 제안), `POST /v1/compare-action` (beforeImage, afterImage, actionNote, promptVersion → likelyResolved, remainingRisks, assessment), `POST /v1/summarize` (inspectionContext, promptVersion → summary, keyRisks)와 `/health`다. 모든 분석에서 모델/프롬프트 버전과 검토 여부를 추적한다. 기존 요약 서버는 규칙 템플릿을 사용한다.
- `AI_PROVIDER_KO.md`가 가리키는 시설 기본 프로필은 `facility-validation-v2`다. 후보 설정과 학습 보고서, 데이터 출처/검토 UI는 배포 결정에 참고한다. 주요 가중치 확장자(`.pt/.pth/.onnx/.tflite/.ckpt/.safetensors`)의 파일은 AI 브랜치의 추적 경로에서 발견되지 않았으며 해당 문서는 별도 모델 ZIP을 요구한다. 학습 파일과 모델을 모바일에 복사하지 않았다. 모델 정확도 및 서버 실행은 미검증이다.

전체 변경 파일과 디렉터리는 미러에서 SHA로 재현할 수 있다.

```powershell
git --git-dir=C:/dev/robo-source for-each-ref refs/heads
git --git-dir=C:/dev/robo-source ls-tree -r --name-only <SHA>
git --git-dir=C:/dev/robo-source diff --name-status main <SHA>
git --git-dir=C:/dev/robo-source diff --name-status main...feature/capture-ai-ui
```
