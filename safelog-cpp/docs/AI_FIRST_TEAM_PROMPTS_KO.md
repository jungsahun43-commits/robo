# SafeLog AI 중심 4인 개발 프롬프트

기존 `TEAM_PROMPTS_KO.md` 대신 이 문서를 사용한다. 각 팀원에게 공통 프롬프트와 본인 역할 프롬프트를 이어서 보낸다.

## 공통 프롬프트

```text
당신은 C++20과 Qt 6으로 개발하는 학술제 프로젝트 SafeLog AI의 팀원이다.

서비스 목표:
현장 사진을 멀티모달 AI가 분석해 위험 종류, 위험 등급, 발견 문장, 개선안을 제안한다. 사람이 결과를 채택·수정·거절한다. 담당자의 조치 후에는 AI가 전후 사진을 비교하고, 점검자가 최종 확인한다. 마지막으로 AI가 처리 과정을 보고서 문장으로 요약한다.

작업 전에 반드시 읽을 파일:
- README.md
- QUICKSTART_KO.md
- docs/ARCHITECTURE.md
- docs/AI_ARCHITECTURE_KO.md
- include/safelog/contracts/types.hpp
- include/safelog/contracts/ports.hpp
- src/ai/include/safelog/ai/ai_analyzer.hpp
- src/ai/include/safelog/ai/ai_safety_service.hpp

공통 규칙:
1. C++20과 Qt 6/QML을 사용한다.
2. 담당 경로 밖의 파일은 임의로 수정하지 않는다.
3. contracts의 타입과 IRepository, IAiSafetyAnalyzer의 함수 이름을 임의로 바꾸지 않는다.
4. UI에서 SQL을 직접 실행하지 않는다.
5. UI에서 모델 API를 직접 호출하지 않는다. AI 호출은 IAiSafetyAnalyzer 구현을 통한다.
6. AI 응답은 구조화된 JSON으로 받고 필수 필드와 값 범위를 검사한다.
7. AI 분석 중에는 로딩, 취소, 시간 초과, 재시도, 실패 후 수동 입력을 처리한다.
8. AI 결과는 자동 확정하지 않는다. 사용자의 채택·수정·거절을 저장한다.
9. 모바일 앱에 API 비밀 키를 넣지 않는다.
10. 모델 서버가 없어도 MockAiSafetyAnalyzer로 자기 기능을 실행할 수 있어야 한다.
11. 법적 적합성을 AI가 보장한다고 표시하지 않는다.
12. 생성된 DB, 사진, 모델 파일, API 키, build 폴더를 커밋하지 않는다.

공통 상태:
- FindingStatus: Open, InProgress, PendingReview, Verified
- AiReviewDecision: Pending, Accepted, Edited, Rejected
- AiAnalysisType: BeforeHazard, AfterComparison, ReportSummary

시연 데이터:
- 사업장: 세이프 금속 가공공장
- 점검자: 김안전 / inspector-1
- 관리자: 박관리 / manager-1
- 담당자: 이조치 / assignee-1
- 장소: 2층 가공라인 통로
- 위험: 통로 자재 적치로 인한 넘어짐 위험

완료 전에 수행할 작업:
- 담당 모듈 빌드
- 정상 흐름 확인
- 입력 누락 또는 실패 흐름 확인
- 기존 smoke_tests 실행
- 변경 파일과 연결 방법 문서화

완료 보고 형식:
[담당 역할]
[구현 내용]
[변경 파일]
[실행 방법]
[테스트 결과]
[AI 실패 시 동작]
[남은 작업]
[다른 파트 연결 요청]

이제 이어지는 역할별 요구사항만 구현하라.
```

## 역할 1 — 현장 촬영과 AI 검토 화면

브랜치: `feature/capture-ai-ui`

```text
담당 경로:
- src/capture/**
- apps/mobile-qt/qml/capture/**
- apps/mobile-qt/controllers/capture/**

구현할 화면:
- FindingCapturePage.qml
- AiHazardReviewPage.qml
- FindingSavedPage.qml

필수 흐름:
1. 장소, 짧은 메모, 사진 한 장을 입력한다.
2. CaptureService로 finding을 생성한다.
3. AiSafetyService::analyzeFinding(findingId)를 백그라운드에서 실행한다.
4. AI가 제안한 위험 종류, 1~5 위험 등급, 발견 문장, 개선안을 표시한다.
5. 사용자가 채택, 수정, 거절 중 하나를 선택한다.
6. 채택 또는 수정 시 최종 문장을 AiSafetyService::reviewHazard로 저장한다.
7. 거절 시 사용자가 직접 작성한 원문을 유지한다.

UI 필수 상태:
- 사진 선택 전
- 분석 중
- 분석 성공
- 분석 시간 초과
- JSON 오류
- 모델 서버 연결 실패
- 수동 입력 전환
- 저장 완료

하지 않을 일:
- 실제 AI HTTP 클라이언트 구현
- SQLite 구현
- 조치 후 사진 비교
- PDF 생성

완료 기준:
- MockAiSafetyAnalyzer로 전체 화면 흐름이 작동한다.
- AI 결과를 수정한 경우 Edited로 저장된다.
- AI 실패 후에도 수동 입력으로 finding을 저장할 수 있다.
```

## 역할 2 — SQLite와 AI 이력 저장

브랜치: `feature/storage-ai`

```text
담당 경로:
- src/storage/**
- database/**
- apps/mobile-qt/adapters/storage/**

구현 목표:
1. QtSql 기반 SqliteRepository를 구현한다.
2. IRepository의 전체 함수를 구현한다.
3. sites, profiles, inspections, findings, photos, action_logs, ai_analyses를 영구 저장한다.
4. AiAnalysis의 모델명, 프롬프트 버전, 신뢰도, 원본 JSON, 검토 결정, 검토자를 손실 없이 저장한다.
5. 앱 재실행 후 AI 분석과 사용자 결정까지 복원한다.
6. 준비된 쿼리와 bind value를 사용한다.
7. 사진은 앱 데이터 폴더에 복사하고 DB에는 경로만 저장한다.
8. 시연용 사업장과 사용자 세 명을 중복 없이 생성한다.

필수 테스트:
- 전체 객체 저장 후 DB를 닫고 다시 열어 동일하게 조회
- analysesForSubject가 유형별 AI 이력을 반환
- 신뢰도와 위험 등급 범위 제약 확인
- 잘못된 외래키와 없는 사진 처리

하지 않을 일:
- AI 모델 호출
- QML 화면
- 보고서 작성

완료 기준:
- InMemoryRepository를 SqliteRepository로 바꾸는 것 외에는 서비스 코드 변경이 없다.
- 앱 재실행 후 finding, 사진, 상태, AI 결과, 사용자 검토 결정이 모두 유지된다.
```

## 역할 3 — 실제 AI 엔진과 평가

브랜치: `feature/ai-engine`

```text
담당 경로:
- src/ai/**
- apps/mobile-qt/adapters/ai/**
- evaluation/**

현재 MockAiSafetyAnalyzer와 AiSafetyService가 있으므로 이를 삭제하지 말고 실제 HttpAiSafetyAnalyzer를 추가한다.

권장 실행 구조:
- Qt Android 앱은 같은 Wi-Fi의 노트북 AI 서버에 HTTP 요청한다.
- 앱 설정 SAFELOG_AI_BASE_URL로 서버 주소를 받는다.
- 클라우드 모델을 쓸 경우 API 키는 노트북 프록시에만 둔다.

구현 목표:
1. IAiSafetyAnalyzer를 구현하는 HttpAiSafetyAnalyzer를 만든다.
2. analyzeHazard는 이미지와 사용자 메모를 전송하고 HazardSuggestion을 반환한다.
3. compareBeforeAfter는 전후 이미지와 조치 내용을 전송하고 ActionAssessment를 반환한다.
4. summarize는 점검 문맥을 보내고 ReportSummary를 반환한다.
5. AI 호출은 UI 스레드를 막지 않게 백그라운드 작업에서 실행할 수 있도록 한다.
6. 연결 시간 초과, HTTP 오류, 잘못된 JSON, 필수 필드 누락을 구분한다.
7. 위험 등급 1~5와 신뢰도 0~1을 검증한다.
8. 모델명과 프롬프트 버전을 모든 결과에 기록한다.
9. 이미지와 프롬프트에 대해 개인정보를 포함하지 않는 시연 규칙을 문서화한다.

프롬프트 출력 규칙:
- 설명 문장 없이 JSON 객체 하나만 반환하도록 요청한다.
- 사진만으로 확정할 수 없는 내용은 추측하지 말고 확인 필요 항목으로 반환한다.
- 최종 법적 적합성 판단을 생성하지 않는다.

평가:
- evaluation/cases.csv를 읽어 최소 20개 사례를 측정한다.
- 위험 종류 일치율, 위험 등급 평균 오차, 평균 지연시간, JSON 실패율을 계산한다.
- 수치를 임의로 만들지 않는다.

하지 않을 일:
- QML 화면 제작
- SQLite 구현
- finding 상태 자동 완료

완료 기준:
- 서버 연결 시 실제 이미지 분석 결과를 구조체로 받는다.
- 서버가 없을 때 MockAiSafetyAnalyzer로 즉시 전환할 수 있다.
- 같은 입력과 프롬프트 버전의 평가 결과를 CSV로 남길 수 있다.
```

## 역할 4 — 조치 비교, 보고서, 앱 홈

브랜치: `feature/workflow-report`

```text
담당 경로:
- src/workflow/**
- src/reporting/**
- apps/mobile-qt/qml/workflow/**
- apps/mobile-qt/qml/reporting/**
- apps/mobile-qt/qml/Main.qml
- apps/mobile-qt/controllers/workflow/**
- apps/mobile-qt/controllers/reporting/**

기존 WorkflowService, ReportService, HtmlRenderer를 활용한다.

구현할 화면:
- HomePage.qml
- AssignedTasksPage.qml
- ActionSubmitPage.qml
- AiComparisonReviewPage.qml
- ReportPreviewPage.qml

필수 흐름:
1. 관리자가 Open finding의 담당자를 지정한다.
2. 담당자가 조치를 시작한다.
3. 담당자가 조치 내용과 조치 후 사진을 제출한다.
4. PendingReview가 되면 AiSafetyService::compareAction을 호출한다.
5. 전후 비교 결과, 잔여 위험, 신뢰도를 점검자에게 보여준다.
6. 점검자가 AI 의견을 채택 또는 거절한 뒤 WorkflowService::verify를 실행한다.
7. AiSafetyService::summarizeInspection으로 보고서 요약을 생성한다.
8. ReportService와 HtmlRenderer로 전후 사진, AI 이력, 사람의 판단을 포함한 보고서를 만든다.

필수 화면 상태:
- AI 비교 중
- AI가 개선 가능성이 높다고 판단
- AI가 잔여 위험을 제안
- AI 비교 실패 후 사람이 직접 확인
- 최종 확인 완료

보고서 필수 항목:
- AI 모델명과 프롬프트 버전
- 위험 종류와 위험 등급
- AI 제안 신뢰도
- 사용자의 채택·수정·거절
- 조치 전후 사진
- AI 전후 비교 결과
- 최종 사람 확인

하지 않을 일:
- 실제 AI HTTP 클라이언트 구현
- SQLite 구현
- AI가 자동으로 Verified 상태를 만들게 하지 않음

완료 기준:
- MockAiSafetyAnalyzer로 조치 제출부터 보고서 생성까지 동작한다.
- AI 비교 실패 상태에서도 점검자가 직접 확인할 수 있다.
- 보고서에 AI 결과와 사람의 최종 결정이 구분되어 표시된다.
```

## 최종 통합 프롬프트

```text
SafeLog AI의 네 브랜치를 다음 순서로 통합한다.
1. feature/storage-ai
2. feature/ai-engine
3. feature/capture-ai-ui
4. feature/workflow-report

같은 IRepository를 CaptureService, WorkflowService, AiSafetyService, ReportService에 주입한다. 같은 IAiSafetyAnalyzer 구현을 모든 AI 기능이 사용하게 한다. 모델 서버가 설정되면 HttpAiSafetyAnalyzer, 없으면 MockAiSafetyAnalyzer를 사용한다.

전체 시나리오:
사진 등록 → AI 위험 분석 → 사람 수정/채택 → 담당자 조치 → AI 전후 비교 → 사람 최종 확인 → AI 요약 → 보고서 생성 → 앱 재실행 후 전체 이력 조회.

AI 장애 시 수동 흐름이 유지되는지도 반드시 확인한다. 충돌 해결을 위해 기능 로직을 Main.qml이나 AppController 한 파일에 몰아넣지 않는다.
```
