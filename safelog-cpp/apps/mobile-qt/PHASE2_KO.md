# SafeLog AI 역할 4 — 2단계 통합 결과

## 담당 역할 / 브랜치

역할 4 / Workflow + Reporting + Qt App Integration.
기존 저장소 `robo/safelog-cpp`, 브랜치 `feature/workflow-report`에서 1단계 코드를 유지·확장했다.
최종 커밋은 `git log -1 --oneline`으로 확인한다. GitHub 브랜치:
https://github.com/jungsahun43-commits/robo/tree/feature/workflow-report

## 구현한 기능과 상태 전환

- Manager만 담당자를 배정할 수 있다. 배정은 Open 상태를 유지한다.
- 배정된 Assignee만 Open → InProgress로 조치를 시작하고, 조치 내용·사진을 제출하여 PendingReview로 전환한다.
- 전후 비교 결과는 Pending AI 이력으로 저장한다. AI는 상태를 Verified로 바꾸지 않는다.
- 원 점검자인 Inspector만 전후 사진 확인 체크와 최종 판단을 입력한 뒤 Verified로 전환한다.
- AI 비교 실패/취소 후에도 직접 확인할 수 있다. AI가 없을 때 Accepted를 선택하는 것은 차단한다.
- 추가 조치 요청은 원 점검자가 사유를 입력하여 PendingReview → InProgress로 되돌리는 명시적 경로다.
  재제출 이후 최신 조치 사진/메모를 비교하며, 이전 조치 회차의 AI 결과로 최종 확인하는 것은 차단한다.
- 모든 상태 변경은 WorkflowService를 통한다. 기존 ValidationError/TransitionError/NotFoundError를 사용한다.
- 보고서는 ReportService/ReportData/HtmlRenderer를 사용한다. QML에서 HTML을 만들지 않는다.
- 출력 HTML에는 사진을 base64로 포함한다. 다른 기기에 전달해도 앱 내부 사진 경로가 필요하지 않다.
- 공유는 SharingAdapter 인터페이스와 MockSharingAdapter로 연결했다. 실제 Android 공유창 호출은 아직 없다.

## 전체 화면 이동 / 클릭 시연

```text
LoginPage → HomePage → FindingCapturePage
→ AI Loading → AiHazardReviewPage → FindingSavedPage → HomePage
→ Manager 로그인 → AssignedTasksPage → 담당자 배정
→ Assignee 로그인 → AssignedTasksPage → 조치 시작 → ActionSubmitPage
→ AI Comparison Loading → AiComparisonReviewPage
→ Inspector 로그인 → AssignedTasksPage → AI 비교 / 직접 확인 → 최종 확인
→ Verified → ReportPreviewPage → HTML 생성 → Mock 공유
```

상단 뒤로/홈/로그아웃 제공. 홈에는 Development / Mock 계정 전환 버튼이 있다.
기존 로그아웃 및 Mock 로그인 함수를 그대로 호출하므로 역할 변경도 같은 권한 검사를 적용한다.
앱을 종료하지 않고 계정을 바꾸면 같은 InMemoryRepository 기록을 계속 볼 수 있다.

시연 순서:

1. `inspector-1 / demo1234` 로그인 → 점검 등록 → Mock 시연 이미지 → 등록 후 AI 분석 → 채택.
2. 홈 → Mock `manager-1` 로그인 → 담당자 지정 → 이조치에게 배정.
3. 홈 → Mock `assignee-1` 로그인 → 배정된 작업 → 조치 시작 → Mock 조치 후 사진 → 조치 제출.
4. 비교 결과 확인 후 홈 → Mock `inspector-1` 로그인 → 최종 확인 → 해당 기록의 전후 비교.
5. AI 의견 채택/거절 또는 직접 확인 선택 → 사람이 확인한 내용 입력 → 전후 사진 확인 체크 → 최종 확인.
6. 보고서 보기 → HTML 생성 완료 → Mock 공유 결과 성공/취소/실패 시연.

오류 시연은 사진 입력/조치 제출 화면의 AI 응답 선택기를 사용한다.
공유 취소는 오류 문구 없이 별도의 ShareCancelled 상태로 표시한다.

## 로그인 / Session / 역할별 Home

1단계 AuthController → MockAuthAdapter와 `userId/userName/role/siteId/siteName` Session을 유지했다.
계정: 김안전(inspector-1), 박관리(manager-1), 이조치(assignee-1), 비밀번호 `demo1234`.

- Inspector: 점검 등록, AI 위험 검토, 최종 확인, 완료 보고서.
- Manager: 담당자 지정, 전체 진행 상황, 검토 상태, 보고서.
- Assignee: 배정된 작업, 조치 제출, 진행 중 작업.

홈 메뉴는 공통 작업 목록으로 연결하되 표시할 기록과 버튼을 역할에 따라 제한한다.
Inspector는 본인 점검, Manager는 전체 점검, Assignee는 본인에게 배정된 기록만 본다.
최종 확인/보고서 함수에도 서버 측에 해당하는 C++ 권한 검사를 적용했다.
로그아웃은 선택한 작업, AI 요청, 보고서 경로 및 Session을 초기화한다.

## 역할별 연결 위치

- 역할 1: 기존 `controllers/capture/`, `qml/capture/` 유지. Open finding의 AI 재검토를 위한
  `resumeFinding()`과 전후 Mock 도형 사진을 구분하는 `demoPhoto(bool after = false)`를 추가했다.
- 역할 2: `AppController`의 InMemoryRepository와 AuthController/MockAuthAdapter.
  현재 브랜치에는 SQLiteRepository/실제 로그인 구현이 없어 대체 구현을 만들지 않았다.
- 역할 3: `AppController(QObject*, shared_ptr<IAiSafetyAnalyzer>)` 생성자 주입.
  기본값은 기존 MockAiSafetyAnalyzer. 실제 Http 구현이 생기면 동일 인터페이스로 주입한다.
- 역할 4: `controllers/workflow/`, `controllers/reporting/`, `qml/workflow/`, `qml/reporting/`.

같은 Repository를 CaptureService, WorkflowService, AiSafetyService, ReportService에 주입한다.
QML은 SQL/HTTP/HTML 생성이나 상태 enum 쓰기를 하지 않는다.

## AiController API / 데이터 / Signal

기존 `analyzeHazard(photoPath, memo)`와 `compareBeforeAfter(beforePhotoPath, afterPhotoPath, memo)` 유지.
전후 비교의 findingId/inspectionId는 WorkflowController의 selected map이 관리한다.
Map은 `findingId/inspectionId/beforePhotoPath/afterPhotoPath/actionNote` 등을 제공한다.
사진은 가져온 로컬 파일 경로, 메모는 QString → UTF-8 std::string으로 전달한다.

- 시작: analysisStarted / comparisonStarted.
- 성공: analysisSucceeded / comparisonSucceeded.
- 실패: analysisFailed(state, message) / comparisonFailed(state, message).
- 상태: Idle, Loading, Success, Failure, Timeout, InvalidJson, ConnectionError, ManualFallback.

기존 QtConcurrent 호출, 30초 UI deadline, RequestEpoch 무효화를 유지한다.
비교 성공 시 WorkflowController가 GUI 스레드에서 AI 메타데이터와 Pending 이력을 저장한다.
원본 JSON은 그대로 유지한다. Mock JSON에 없는 구조화된 설명/개선 의견은 AI 제안 로그로 함께 저장한다.
순수 동기 Provider 인터페이스의 실행 자체는 강제 중단할 수 없으므로 실제 Provider에는 유한한 네트워크 timeout이 필요하다.

## WorkflowService / 사람의 직접 확인

WorkflowController가 assign/beginWork/submitAction/verify/requestChanges를 호출한다.
Qt 비의존 `workflow_support.hpp`의 finalizeReview는 소유자, 상태, 확인 체크, 최종 메모,
전후 사진, AI 분석 유형/대상/회차/검토 여부를 검사한다.
AI 검토 결과를 저장한 다음 WorkflowService::verify를 호출한다.

AI 실패 → 다시 비교 또는 AI 없이 직접 확인 → 사진/조치 검토 → 확인 체크와 최종 메모 → Verified.
성공한 AI를 거절하고 사람이 직접 확인하는 경로도 지원한다.

## ReportService / 보고서 생성 / 공유

ReportingController → ReportService::build → ReportData → HtmlRenderer::writeFile.
사업장/점검자/점검 일시/장소/위험 등급/개선 의견/담당자/조치 내역/사진/AI 이력/
모델명/프롬프트 버전/confidence/사람 검토 결정/최종 판단/상태를 표시한다.
ReportData에 participants를 추가하여 담당자 이름을 표시한다. 공통 contracts 파일은 변경하지 않았다.

Qt 화면에서는 구조화된 데이터로 미리보기를 표시한다. HTML 생성은 Repository에서 분리한
ReportData 스냅샷으로 백그라운드 실행한다. 이전 요청의 완료는 로그아웃/재생성 후 무시한다.
출력 위치: Qt AppDataLocation 아래 `reports/safelog-<uuid>.html`.
AI 제안 영역과 사람의 기록 영역을 색상과 제목으로 구분한다.
미완료 기록은 실제 상태를 그대로 표시하며 완료된 것처럼 표시하지 않는다.
사진 누락, 관계 데이터 누락, 파일 쓰기 실패를 오류로 처리한다. PDF는 구현하지 않았다.

SharingAdapter::share(path, mimeType) → ShareSucceeded / ShareCancelled / ShareFailed.
현재 MockSharingAdapter는 실제 외부 전송을 하지 않으며 UI에 Mock임을 표시한다.
Android에서는 같은 인터페이스에 FileProvider의 content URI, 읽기 권한 및 ACTION_SEND를
구현하는 어댑터를 주입해야 한다. 기존 Android 패키지/Manifest/공유 어댑터는 저장소에 없다.

## 생성 및 수정 파일

2단계 생성:

- `apps/mobile-qt/controllers/workflow/workflow_controller.hpp`, `.cpp`
- `apps/mobile-qt/controllers/reporting/reporting_controller.hpp`, `.cpp`
- `apps/mobile-qt/adapters/workflow_support.hpp`, `sharing_adapter.hpp`
- `apps/mobile-qt/qml/workflow/AssignedTasksPage.qml`, `ActionSubmitPage.qml`, `AiComparisonReviewPage.qml`
- `apps/mobile-qt/qml/reporting/ReportPreviewPage.qml`
- `tests/workflow_report_tests.cpp`
- `apps/mobile-qt/PHASE2_KO.md`

2단계 수정:

- `src/workflow/workflow_service.cpp` 및 해당 공개 헤더: 역할 검사/최종 확인 검증/추가 조치 요청.
- `src/reporting/{report_service,html_renderer}.cpp` 및 해당 공개 헤더: 담당자/시간순 이력/독립 HTML/출력 오류.
- `apps/mobile-qt/app_controller.*`, `main.cpp`, `CMakeLists.txt`: 기존 Composition Root와 QML 등록 확장.
- 기존 `controllers/ai/`, `controllers/capture/`: 구조화 비교 결과, AI 제안 로그, 재검토, 조치 후 Mock 사진.
- 기존 `qml/Main.qml`, `qml/home/HomePage.qml`: 전체 Navigation과 Mock 역할 전환.
- 기존 Qt `tests/controller_tests.cpp`: 로그인부터 HTML·Mock 공유까지 통합 테스트 추가.
- `tests/CMakeLists.txt`: 새 순수 C++ 테스트 등록.

커밋에는 아직 커밋하지 않았던 1단계 파일도 포함한다. 1단계 원본 테스트는 삭제하지 않았다.
역할 1/2/3의 `src/capture`, `src/storage`, `src/ai` 내부와 공통 contracts는 변경하지 않았다.

## 실행 / 빌드 / 테스트 결과

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

- 기본 configure/build 성공.
- 기존 smoke + 1단계 앱 지원 + 2단계 Workflow/Report 테스트: **3/3 통과**.
- 신규 2단계 7개 시나리오: 권한/잘못된 전환/누락 입력, 전체 Mock 및 HTML,
  비교 실패 후 수동 확인, AI 거절, 추가 조치/최신 사진/이전 분석 차단,
  보고서 누락/쓰기 실패/HTML 이스케이프, 공유 세 상태와 목록 권한.
- HTML 파일 생성, 메타데이터, 사진 base64 포함을 실제 C++ 테스트에서 확인했다.
- Qt 컨트롤러 테스트 및 Qt/QML 화면 실행은 Qt 6 미설치로 실행하지 않았다.
  소스 빌드나 Qt 설치는 재시도하지 않았다.

Qt 설치된 별도 환경에서:

```sh
cmake -S . -B build-qt -DSAFELOG_BUILD_QT=ON -DCMAKE_PREFIX_PATH="/path/to/Qt/6.x/macos"
cmake --build build-qt
ctest --test-dir build-qt --output-on-failure
./build-qt/apps/mobile-qt/safelog_mobile.app/Contents/MacOS/safelog_mobile
```

## Android 환경 확인 / APK 빌드 방법

현재 확인한 설치:

- SDK: `/Users/choeminjun/Library/Android/sdk`, platform `android-36`.
- Build Tools: `35.0.0`, `36.1.0`.
- NDK: `28.2.13676358`.
- Android Studio 내장 JDK: `/Applications/Android Studio.app/Contents/jbr/Contents/Home`, OpenJDK **21.0.8**.
  `/usr/libexec/java_home`에는 등록되지 않았으나 내장 java 실행으로 확인했다.
- **Qt 6 desktop host kit / Qt Android kit 없음**. APK는 생성하지 않았다.
- SDK의 `cmdline-tools` 디렉터리는 현재 확인되지 않았다.

Qt 버전별 요구사항이 다르다. 아래는 공식 Qt 6.11 문서를 기준으로 한 예시이며
현재 컴퓨터에 설치하거나 설정을 변경한 것은 아니다:

- JDK 21, SDK platform 36, Build Tools 36.0.0, NDK 27.2.12479018.
- 동일 Qt 버전의 macOS host kit와 Android arm64-v8a kit(Quick/Quick Controls/Concurrent 포함).
- 현재 NDK 28.2는 위 Qt 6.11 문서 기준 버전과 다르므로 무조건 재사용하지 않는다.

출처: [Qt Android prerequisites](https://doc.qt.io/qt-6/android-configure-dev-environment.html),
[Qt Android supported configurations](https://doc.qt.io/qt-6/android.html),
[Qt Android CMake build](https://doc.qt.io/qt-6/android-building-projects-from-commandline.html).

준비된 Qt 6.11 환경에서 `safelog-cpp`를 작업 디렉터리로 사용:

```sh
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
"$HOME/Qt/6.11.2/android_arm64_v8a/bin/qt-cmake" \
  -S . -B build-android -GNinja \
  -DSAFELOG_BUILD_QT=ON -DSAFELOG_BUILD_TESTS=OFF \
  -DQT_HOST_PATH="$HOME/Qt/6.11.2/macos" \
  -DANDROID_SDK_ROOT="$HOME/Library/Android/sdk" \
  -DANDROID_NDK_ROOT="$HOME/Library/Android/sdk/ndk/27.2.12479018"
cmake --build build-android --target apk
```

Qt 설치 버전/경로에 맞춰 바꿔야 한다. Android APK/실기기 권한/공유는 **미검증**이다.

## 아직 구현되지 않은 부분 / 다른 역할 연결 요청

- 역할 1: 실제 카메라/Android content URI 가져오기와 권한 처리. 현재 로컬 파일·Mock PNG 입력.
- 역할 2: SQLite/실제 인증/트랜잭션. 현재 데이터는 앱 실행 중에만 유지하며 사진/보고서 파일만 로컬에 남는다.
  IRepository의 전체 목록 API가 없어 현재 실행 중 생성한 점검 ID 목록을 앱에서 관리한다.
- 역할 3: 실제 HTTP Provider 및 네트워크 timeout/취소. 생성자 주입 경계 준비 완료.
- Android 공유: 네이티브 FileProvider/ACTION_SEND 구현 필요. 현재 Mock 결과 시연만 가능.
- PDF 출력, 실제 장치에서의 QML 레이아웃·뒤로가기·앱 수명주기 검증.
- 기존 IRepository는 트랜잭션 계약이 없으므로 여러 저장 호출 중 저장소 장애가 발생하면
  부분 기록이 남을 수 있다. SQLite 연결 시 역할 2와 원자성 정책을 합의해야 한다.

구현 완료와 실행 검증을 구분한다: C++ 서비스/지원 로직은 실행 검증했지만,
Qt 화면을 클릭하여 끝까지 진행한 실기기 시연 성공을 주장하지 않는다.
