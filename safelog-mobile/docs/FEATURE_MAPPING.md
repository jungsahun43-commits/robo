# Qt → React Native 기능 대응

아래 원본 경로는 모두 `safelog-cpp/` 기준이다. 9개 화면의 기준 브랜치는 `feature/workflow-report`이며, QML 경로의 공통 접두사는 `apps/mobile-qt/qml/`이다.

| Qt 화면 | 모바일 경로 | 동작과 차이 |
|---|---|---|
| `auth/LoginPage.qml` | `src/app/index.tsx` | ID·비밀번호, 오류, 시연 계정. Qt ID와 auth 브랜치의 하이픈 없는 로그인 ID를 모두 허용. |
| `home/HomePage.qml` | `src/app/home.tsx` | SafeLog 브랜드와 역할별 업무 바로가기 및 상태별 건수. 상단 계정 화면에서 로그아웃. |
| `include/safelog/contracts/types.hpp`의 `Site` | `src/app/facilities.tsx` | 시연 현장명과 주소, 볼 수 있는 점검 기록 건수. 시설물 대장 연동은 준비 중. |
| `reporting/ReportPreviewPage.qml` 진입 | `src/app/reports.tsx` | Inspector와 Manager가 접근 가능한 기록에서 보고서 미리보기를 연다. |
| `capture/FindingCapturePage.qml` | `src/app/capture.tsx` | 장소·발견 메모·권장 조치, Open 저장 후 분석. 문자열 파일 경로 입력을 촬영과 시스템 갤러리 선택으로 교체. |
| `capture/AiHazardReviewPage.qml` | `src/app/hazard.tsx` | 위험 분류·등급·신뢰도, 편집·채택·거절·수동 입력, 취소·재시도. 홈의 AI 위험 분석은 소유한 미조치 기록을 열고 보류된 제안을 다시 표시. |
| `capture/FindingSavedPage.qml` | `src/app/saved.tsx` | 기록 ID·상태·홈 이동. |
| `workflow/AssignedTasksPage.qml` | `src/app/tasks.tsx` | 역할별 목록, 배정·시작·검토·보고서 진입. 시연 담당자는 이조치. |
| `workflow/ActionSubmitPage.qml` | `src/app/action.tsx` | 수행 조치·조치 후 사진·선택 메모. 제출 후 비교 화면으로 이동하며, 해당 기록을 볼 수 있는 역할은 Mock 비교를 실행할 수 있다. |
| `workflow/AiComparisonReviewPage.qml` | `src/app/comparison.tsx` | 전후 사진, 최근 제출, Mock 비교, 판단 확인, 재조치 요청. 원 점검자만 상태 변경. |
| `reporting/ReportPreviewPage.qml` | `src/app/report.tsx` | 점검 내용과 첨부 사진, AI 제안·결정, 사람 조치 이력. Qt의 HTML + Mock 공유를 Android/iOS PDF + 시스템 공유로 교체. 웹은 HTML 다운로드. |

| Qt 서비스·계약 | 브랜치 | 모바일 구현 |
|---|---|---|
| `include/safelog/contracts/{types,ports,errors}.hpp` | main / workflow-report | `src/contracts/`의 TypeScript 모델과 포트. 사용자 오류는 서비스 예외를 화면에서 표시한다. |
| `src/auth/auth_service.cpp`, `src/auth/include/safelog/auth/demo_accounts.hpp`, `apps/mobile-qt/adapters/application_support.hpp` | auth / workflow-report | `src/features/auth/`의 세션과 시연 계정. Argon2id, 계정 관리와 운영 인증은 후속 작업. |
| `src/capture/capture_service.cpp`, `apps/mobile-qt/controllers/capture/capture_mock_ai_analyzer.hpp` | capture-ai-ui / workflow-report | `src/features/capture/photo-field.tsx`, 워크플로 서비스와 Mock 분석. Qt 시연용 사진 대신 사용자 사진을 선택. |
| `src/workflow/workflow_service.cpp` | workflow-report | `src/features/workflow/service.ts`. Open 배정, 담당자가 시작·제출, 원 점검자가 확인·추가 조치 요청. |
| `src/ai/ai_safety_service.cpp`, `apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp` | workflow-report / ai-engine | `src/features/ai/`의 Mock과 취소 가능한 UI 수명 주기. 실제 HTTP는 설계 단계. |
| `database/schema.sql`, `src/storage/sqlite_repository.cpp` | storage-ai / auth | `src/repositories/memory.ts`. 모바일 시연에서는 점검당 한 발견 기록을 만들며 DB 설계는 다건 관계를 유지한다. |
| `src/reporting/{report_service,html_renderer}.cpp` | workflow-report | `src/features/reporting/report.ts`. AI 요약 호출은 포함하지 않는다. |
