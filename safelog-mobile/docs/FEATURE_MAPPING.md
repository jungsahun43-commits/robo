# Qt → React Native 기능 대응

아래 원본 경로는 모두 `safelog-cpp/` 기준이다. 9개 화면의 기준 브랜치는 `feature/workflow-report`이며, QML 경로의 공통 접두사는 `apps/mobile-qt/qml/`이다.

| Qt 화면 | 모바일 경로 | 동작과 차이 |
|---|---|---|
| `auth/LoginPage.qml` | `src/app/index.tsx` | ID·비밀번호, 오류, 시연 계정. Qt ID와 auth 브랜치의 하이픈 없는 로그인 ID를 모두 허용. |
| `home/HomePage.qml` | `src/app/home.tsx` | SafeLog 브랜드와 역할별 업무 바로가기 및 상태별 건수. 상단 계정 화면에서 로그아웃. |
| `include/safelog/contracts/types.hpp`의 `Site` | `src/app/facilities.tsx` | 시연 현장명과 주소, 볼 수 있는 점검 기록 건수. 시설물 대장 연동은 준비 중. |
| `reporting/ReportPreviewPage.qml` 진입 | `src/app/reports.tsx` | Inspector와 Manager가 접근 가능한 기록에서 보고서 미리보기를 연다. |
| `capture/FindingCapturePage.qml` | `src/app/capture.tsx` | 고정 AI 20항목의 전체 분야와 볼스크루를 포함한 발표 후보 5개 필터를 지원한다. 기타 분야 직접 검토와 명시적 Mock도 제공한다. 분석 방식·분야와 사진·장소·메모·권장 조치를 Open 기록에 저장한다. |
| `capture/AiHazardReviewPage.qml` | `src/app/hazard.tsx` | 실제 모델 후보·모델 식별·참고 점수(확률 아님), 편집·채택·거절·수동 입력, 취소·재시도. 홈의 AI 위험 분석은 소유한 미조치 기록을 열고 보류된 제안을 다시 표시한다. |
| `capture/FindingSavedPage.qml` | `src/app/saved.tsx` | 기록 ID·상태·홈 이동. |
| `workflow/AssignedTasksPage.qml` | `src/app/tasks.tsx` | 역할별 목록, 배정·시작·검토·보고서 진입. 시연 담당자는 이조치. |
| `workflow/ActionSubmitPage.qml` | `src/app/action.tsx` | 수행 조치·조치 후 사진·선택 메모. 제출 후 비교 화면으로 이동하며 기록에 저장한 분석 방식·분야를 사용한다. |
| `workflow/AiComparisonReviewPage.qml` | `src/app/comparison.tsx` | 전후 사진, 최근 제출, 일반 모델 또는 명시적 Mock 비교, 판단 확인, 재조치 요청. PCB 실모델은 자동 비교를 제공하지 않아 직접 검토한다. 원 점검자만 상태 변경. |
| `reporting/ReportPreviewPage.qml` | `src/app/report.tsx` | 점검 내용과 첨부 사진, AI 제안·결정, 사람 조치 이력. Qt의 HTML + Mock 공유를 Android/iOS PDF + 시스템 공유로 교체. 웹은 HTML 다운로드. |

| Qt 서비스·계약 | 브랜치 | 모바일 구현 |
|---|---|---|
| `include/safelog/contracts/{types,ports,errors}.hpp` | main / workflow-report | `src/contracts/`의 TypeScript 모델과 포트. 사용자 오류는 서비스 예외를 화면에서 표시한다. |
| `src/auth/auth_service.cpp`, `src/auth/include/safelog/auth/demo_accounts.hpp`, `apps/mobile-qt/adapters/application_support.hpp` | auth / workflow-report | 기존 모바일 scrypt 로그인·계정 생성·비밀번호 변경을 유지하고 DB 프로필을 업무 배정·보고서에 연결한다. 운영 서버 인증과 다기기 공유는 별도 작업. |
| `src/capture/capture_service.cpp`, `apps/mobile-qt/controllers/capture/capture_mock_ai_analyzer.hpp` | capture-ai-ui / workflow-report | `src/features/capture/photo-field.tsx`, 워크플로 서비스와 분석 방식 선택. JPEG·PNG 사진을 사용하고 고정 모델용 형식·8MiB/분야별 해상도 제한을 확인한다. |
| `src/workflow/workflow_service.cpp` | workflow-report | `src/features/workflow/service.ts`. Open 배정, 담당자가 시작·제출, 원 점검자가 확인·추가 조치 요청. |
| `src/ai/ai_safety_service.cpp`, `apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp` | workflow-report / ai-engine | `src/features/ai/fixed-cpu.ts`의 실제 HTTP 구현, 모델 SHA·응답 검사·원문 보관, `use-analysis.ts`의 취소·범위 수명 주기. `mock.ts`는 명시적 시연용이며 실제 오류를 대체하지 않는다. |
| `database/schema.sql`, `src/storage/sqlite_repository.cpp` | storage-ai / auth | `src/repositories/sqlite/`의 네이티브 저장과 v3 마이그레이션. 사진·AI 원응답·검토·모델/입력 SHA를 보존한다. 웹은 `memory.ts`를 사용한다. |
| `src/reporting/{report_service,html_renderer}.cpp` | workflow-report | `src/features/reporting/report.ts`. AI 요약 호출은 포함하지 않는다. |

## 고정 모델 연결과 남은 검증

시설·컨베이어 3항목, 전기·배선 2항목, 볼스크루를 포함한 금속·기계 10항목은 기존 `v16-r1` CPU 서버 8775, PCB 5유형은 `pcb-multisource-v3` 서버 8774를 사용한다. [전체 항목표](INSPECTION_CATALOG.md)의 허용 범위를 유지하며 기타 분야는 AI 없이 직접 검토한다. [loopback gateway](FIXED_CPU_GATEWAY.md)가 웹/USB 개발 연결용 8780의 `/general`·`/pcb` 경로를 각각 전달한다. 모델 가중치와 기존 서버 코드는 별도 패키지로 유지한다.

2026-11-19 변경은 원본 모바일 커밋 `c422eb5d7f0445a0e6d3bdf55b34b2e5916fc8e9` 기반의 로컬 연결 코드이며 GitHub 게시·병합은 별도로 확인한다. 실제 CPU 네트워크 연결에서는 승인된 시설·컨베이어·RobustAD PCB 사진 각 2장의 기대 판정과 컨베이어 메모리 워크플로를 확인했고 Expo Web export도 생성했다. 이 검사는 브라우저 화면 조작이나 실제 현장 조치, 독립 성능 평가가 아니다. 전체 항목의 자동 HTTP 계약·워크플로·gateway 71개 테스트와 Android/iOS 실기기 실행도 구분한다. 현재 Android SDK·실기기·LAN·iOS 검증과 볼스크루·케이블을 포함한 나머지 항목의 별도 사진 리허설은 남아 있다.

2026-11-23 최신 DB 브랜치 기반의 [AI·SQLite 통합](DB_AI_INTEGRATION.md)에서는 네이티브 로컬 로그인·계정·사진·점검·실제 AI 결과 저장을 함께 사용한다. 전체 103개 자동 검사와 실제 CPU·Node SQLite·사진 복사·재개방 흐름을 확인했다. 실제 Expo 기기·네이티브 PDF 공유는 추가 리허설이 필요하다. 웹은 메모리 저장이며 운영 서버 인증과 다기기 협업 동기화는 남아 있다. 오탐·미탐 연구 평가와 현재 입력 사진의 판정, 사진 판정과 위치 정확도를 구분한다.
