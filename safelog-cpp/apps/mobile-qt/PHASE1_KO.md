# SafeLog AI 1단계 앱 통합

## 상태 및 실행

작업 브랜치: `feature/workflow-report`. `origin/main`을 fast-forward 병합했다.
공통 계약과 src 모듈은 변경하지 않았다. commit/push, Android APK 빌드 및 최종 Workflow/보고서 작업은 수행하지 않았다.

기본 빌드와 기존 smoke test, Qt 비의존 앱 지원 테스트는 통과했다. 이 작업 환경에는 Qt 6가 없어 Qt 구성은
`Qt6Config.cmake` 누락으로 중단되었다. Homebrew 설치도 Intel macOS 바이너리 미지원으로
대규모 소스 빌드가 필요해 중단했다. 따라서 새 Qt 코드의 컴파일, 컨트롤러 테스트와 실제 QML 화면 실행은 **아직 검증되지 않았다**.

Qt 6.5 이상(Quick, Quick Controls 2, Quick Dialogs, Concurrent, Test)을 설치한 환경에서:

```sh
cmake -S . -B build-qt -DSAFELOG_BUILD_QT=ON -DCMAKE_PREFIX_PATH="/path/to/Qt/6.x/macos"
cmake --build build-qt
ctest --test-dir build-qt --output-on-failure
```

macOS 실행 파일: `build-qt/apps/mobile-qt/safelog_mobile.app/Contents/MacOS/safelog_mobile`.
이 문서의 명령은 `safelog-cpp`에서 실행한다.

## 분석한 기존 구조

- `src/capture`: CaptureService, NewFindingInput, 사진 등록 및 점검 생성.
- `src/storage`: InMemoryRepository, LocalPhotoStore, SystemClock, SequentialIdGenerator.
- `src/ai`: IAiSafetyAnalyzer, MockAiSafetyAnalyzer, AiSafetyService 및 AI 검토 이력.
- `src/workflow`: WorkflowService(배정, 시작, 제출, 확인).
- `src/reporting`: ReportService, ReportData, HtmlRenderer.
- `apps/mobile-qt`: 기존 단일 화면/자동 최종 확인 데모를 로그인 기반 Shell로 교체.
- `tests`: 기존 safelog_smoke_tests 유지.
- 현재 브랜치에는 SQLiteRepository, 로그인 서비스 및 역할 1 QML 화면이 없다.

## 화면과 계정

`ApplicationWindow → StackView → LoginPage → HomePage → FindingCapturePage → AiHazardReviewPage(Loading/성공/실패/수동) → FindingSavedPage → HomePage`.
미구현 메뉴는 PlaceholderPage로 연결한다. Main.qml은 화면 이동과 역할 가드만 처리한다.
사진 입력과 저장은 CaptureController, 로그인은 AuthController, 분석은 AiController가 담당한다.

Mock 사용자와 공통 개발 비밀번호 `demo1234`:

| 사용자 | ID | 역할 | 홈 메뉴 |
|---|---|---|---|
| 김안전 | inspector-1 | Inspector | 점검 등록, AI 검토, 최종 확인, 완료 보고서 |
| 박관리 | manager-1 | Manager | 담당자 지정, 전체 진행 상황, 검토 상태, 보고서 |
| 이조치 | assignee-1 | Assignee | 배정된 작업, 조치 제출, 진행 중 작업 |

`AuthController::session`은 QVariantMap이며 `userId/userName/role/siteId/siteName`을 제공한다.
기본 사업장은 `site-1 / 세이프 금속 가공공장`이다. 로그인 중/실패 메시지를 표시하며
로그아웃 시 session, 현재 draft, AI 결과 및 진행 중 요청의 UI 연결을 초기화한다.
AuthController는 Qt 비의존 MockAuthAdapter를 호출한다. 실제 인증 어댑터로 교체할 TODO를 남겼다.

대시보드는 현재 프로세스에서 생성한 점검을 집계한다. Inspector는 본인 점검,
Manager는 전체, Assignee는 자신에게 배정된 기록을 조회한다. 완료된 점검은 모든 finding이
Verified인 점검 수다. 1단계에서는 상태를 Open으로 유지하므로 나머지 집계는 0이다.

## 역할 1 연결과 데이터

임시 역할 1 화면은 `qml/capture/{FindingCapturePage,AiHazardReviewPage,FindingSavedPage}.qml`,
연결부는 `controllers/capture/capture_controller.*`에 있다. 역할 1 구현을 합칠 때 이 경계를 교체한다.

사진은 FileDialog 또는 로컬 경로/file URL로 입력한다. 시연용 도형 PNG도 생성할 수 있다.
사진 형식을 확인한 뒤 기존 CaptureService와 LocalPhotoStore로 가져온다.
기본 장소는 `2층 가공라인 통로`, 메모는 `통로에 자재가 적치되어 있음`이다.

`CaptureController::draft`의 `photoPath/memo/location/inspectionId/findingId/userId/action`을 사용한다.
photoPath는 가져온 사진의 로컬 파일 경로이며 문자열은 C++ 경계에서 UTF-8 std::string으로 변환한다.
공통 `NewFindingInput`을 재사용하며, 새 도메인 데이터 타입을 만들지 않았다.

사진 등록 시 finding은 먼저 Open 상태로 메모리 저장소에 저장된다. 뒤로 가거나 AI를 취소해도
이 기록은 남는다. 동일 화면의 재시도는 같은 finding을 사용한다. 현재 홈의 AI 검토 메뉴는
이전 세션의 미검토 목록을 복구하지 않으며, 목록/재개 기능은 후속 단계에 연결한다.

## AiController API 및 실행 모델

- `analyzeHazard(QString photoPath, QString memo)`
- `compareBeforeAfter(QString beforePhotoPath, QString afterPhotoPath, QString memo)`
- `manualFallback()`, `reset()`
- Q_PROPERTY: `state`, `loading`, `error`, `result`, `demoFailure`
- 시작: `analysisStarted`, `comparisonStarted`
- 성공: `analysisSucceeded`, `comparisonSucceeded`; 구조화된 데이터는 `result`로 조회.
- 오류: `analysisFailed(QString state, QString message)`, `comparisonFailed(QString state, QString message)`
- 상태 변경: `stateChanged`, `loadingChanged`

`AppController`가 공유 IRepository와 서비스를 조립한다. 기본 analyzer는 기존
MockAiSafetyAnalyzer이며 AppController 생성자의 두 번째 인수로 다른
`shared_ptr<IAiSafetyAnalyzer>`를 주입할 수 있다. 실제 Provider는 이번 단계에서 만들지 않았다.

QtConcurrent worker에서 IAiSafetyAnalyzer를 호출한다. worker는 저장소에 접근하지 않는다.
GUI 스레드에서 결과를 수신한 CaptureController가 IRepository로 Pending 분석 이력을 저장하고,
명시적 사람 검토는 기존 AiSafetyService::reviewHazard로 처리한다.
모델 원본 JSON/모델명/프롬프트 버전/신뢰도를 보존한다. Mock의 JSON에는 설명 필드가 없으므로
화면에는 JSON에서 추출한 값 대신 HazardSuggestion의 구조화된 설명/권장 조치를 사용한다.

Provider는 전송/파싱 오류를 구분하려면 앱 어댑터의 AiRequestError(`Timeout`, `InvalidJson`,
`ConnectionError`)로 전달한다. 일반 예외는 Failure가 된다. typed 필드, 위험 등급,
유한한 신뢰도 범위 및 원본 JSON 객체를 검사한다.

기본 UI 제한 시간은 30초다. 취소/로그아웃/시간 초과 후 늦은 응답은 요청 세대 번호로 무시한다.
동기식 IAiSafetyAnalyzer 계약은 실행 중인 호출을 강제로 취소할 수 없으므로 실제 Provider에도
유한한 네트워크 timeout을 설정해야 한다. Provider 호출은 mutex로 직렬화한다.

## AI 성공, 실패와 사람 판단

상태: Idle, Loading, Success, Failure, Timeout, InvalidJson, ConnectionError, ManualFallback.
사진 화면의 개발용 선택기로 오류를 시연할 수 있다. 개발용 Timeout은 즉시 오류를 반환하며,
실제 30초 deadline 동작은 별도로 구현했다.

성공 시 위험 종류/등급/설명/권장 조치/신뢰도/모델명을 표시한다.
“AI 분석 결과는 참고 제안입니다. 최종 안전 판단과 확인은 사람이 수행합니다.”를 항상 표시한다.
원문 그대로 저장하면 Accepted, 설명/조치를 바꾸면 Edited, 거절하면 Rejected로 기록한다.
거절은 최초 입력을 유지한다. 수동 입력은 명시적으로 작성한 설명과 조치를 저장한다.

실패 → 다시 시도(동일 finding), 정상 Mock으로 다시 시도, 또는 수동 입력 → 내용 저장 → 저장 완료.
분석 중에도 취소 후 수동 입력이 가능하다. 사람 검토는 최종 Workflow 확인과 구분하며
어떤 AI 경로도 Verified로 바꾸지 않는다.

## 생성/수정 파일

생성:

- `controllers/ai/ai_controller.hpp`, `.cpp`
- `controllers/auth/auth_controller.hpp`, `.cpp`
- `controllers/capture/capture_controller.hpp`, `.cpp`
- `qml/auth/LoginPage.qml`
- `qml/home/HomePage.qml`
- `qml/capture/FindingCapturePage.qml`, `AiHazardReviewPage.qml`, `FindingSavedPage.qml`
- `qml/common/ErrorPanel.qml`, `LoadingOverlay.qml`, `PhotoPreview.qml`, `PlaceholderPage.qml`
- `tests/controller_tests.cpp`
- `adapters/application_support.hpp`: MockAuthAdapter, 권한 검사, AI 값 검증, 수동 저장, RequestEpoch.
- `PHASE1_KO.md`

수정:

- `CMakeLists.txt`: 모든 새 C++/QML 및 Qt 테스트 등록, Concurrent 링크.
- `main.cpp`: 앱/로그인/AI/사진 컨트롤러를 QML에 노출.
- `app_controller.hpp`, `.cpp`: Composition Root와 대시보드 집계.
- `qml/Main.qml`: StackView와 역할별 Navigation.

위 경로는 모두 `apps/mobile-qt` 기준이다. 추가로 프로젝트 루트 기준 `tests/mobile_support_tests.cpp`를 생성하고
`tests/CMakeLists.txt`를 수정하여 Qt 없이도 앱 지원 테스트를 실행하도록 등록했다.

## 검증 및 남은 작업

실행 완료: 기본 configure/build 성공, CTest **2/2 통과**, git diff --check 통과.

- 기존 `safelog_smoke_tests` 유지 및 통과.
- 신규 `safelog_mobile_support_tests`의 8개 시나리오 통과:
  1. Mock 로그인 성공/잘못된 비밀번호/없는 계정 및 사업장.
  2. Inspector 등록 권한과 점검 소유자만 검토·수동 저장 가능.
  3. 기존 Mock AI 결과 및 사람의 채택·수정·거절 이력, Open 유지.
  4. 위험 등급/신뢰도 범위, NaN/Infinity, 필수 필드 누락 거절.
  5. 실제 Mock 입력 파일 오류, 전후 비교 파일 오류 후 수동 저장 및 로그.
  6. AI 성공 후 수동 전환 시 AI 거절 이력과 수동 내용 저장.
  7. 실패 후 같은 finding으로 분석 재시도.
  8. 취소/로그아웃/timeout/수동 전환 시 무효화된 요청의 worker 응답 무시, 새 요청 수락.

순수 C++ 테스트는 실제 Qt 컨트롤러가 호출하는 `application_support.hpp`와 기존 서비스를 사용한다.
마지막 테스트는 promise로 worker 반환 순서를 제어하여 지연 응답의 무효화를 검증한다.
QTimer의 시간 경과나 Qt signal 전달 자체를 검증한 것은 아니다.

Qt 부재로 실행하지 못한 부분: Qt target 컴파일, `safelog_mobile_tests`, 실제 QML 화면 실행,
로그인/분석 signal, JSON 파싱 오류 UI, 파일 선택/이미지 디코딩,
QTimer deadline, QtConcurrent watcher 수명과 화면 전환 동작.
컨트롤러 테스트 코드는 이러한 경로를 별도로 등록해 두었으며 Qt 환경에서 실행해야 한다.
사용자 지시에 따라 Qt Homebrew 설치/소스 빌드는 재시도하지 않는다.

2단계:

- Qt가 설치된 환경에서 모바일 target/컨트롤러 테스트 및 실제 화면 동작 검증.
- 역할 1 카메라/갤러리와 Android content URI/권한/취소 어댑터 연결.
- 역할 2 SQLite/실제 인증 연결, 사진·기록 영속화 및 트랜잭션 복구.
  기존 CaptureService는 사진 복사 전에 finding을 저장하므로 디스크 오류에 대한 원자성은 후속 통합 필요.
- 역할 3 실제 analyzer 주입 및 전송 timeout/취소 연결.
- 미검토 위험 목록, 담당자 지정·조치 제출·전후 비교 검토·사람 최종 확인을 WorkflowService로 연결.
- ReportService/HtmlRenderer 기반 보고서, PDF/공유 및 Android APK 검증.
