# SafeLog 팀 개발 시작 방법

이 문서는 기본 실행 방법을 설명한다. 최신 AI 중심 역할과 프롬프트는 `docs/AI_FIRST_TEAM_PROMPTS_KO.md`를 사용한다.

## 1. 팀장이 한 번만 할 일

1. `SafeLog-cpp-starter.zip`을 압축 해제한다.
2. 압축을 푼 `safelog-cpp` 폴더를 GitHub 저장소에 올린다.
3. 네 명 모두 그 저장소를 clone한다.
4. 아래 브랜치를 만든다.

```text
feature/capture-ai-ui
feature/storage-ai
feature/ai-engine
feature/workflow-report
```

5. 공통 계약 파일 세 개는 첫 회의 이후 함부로 수정하지 않는다.

```text
include/safelog/contracts/types.hpp
include/safelog/contracts/ports.hpp
include/safelog/contracts/errors.hpp
```

## 2. 모든 팀원의 개발 환경

필수 프로그램:

- Git
- CMake 3.24 이상
- C++20 컴파일러
  - Windows: Visual Studio 2022 Community의 `Desktop development with C++`
  - macOS: Xcode Command Line Tools
  - Linux: GCC 또는 Clang

모바일 화면 담당 및 최종 통합 PC에는 추가로 설치한다.

- Qt 6.5 이상
- Qt Quick
- Qt Quick Controls 2
- Android로 빌드한다면 Qt 설치 관리자의 Android 구성 요소, Android SDK, NDK, JDK

## 3. 현재 코드 실행

프로젝트 최상위 폴더에서 실행한다.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows Visual Studio 환경에서는 실행 파일이 다음 중 한 곳에 만들어질 수 있다.

```text
build/apps/demo/Debug/safelog_demo.exe
build/apps/demo/safelog_demo.exe
```

데모를 실행하면 다음 작업을 자동으로 수행한다.

```text
점검 생성 → 발견 사항 등록 → 담당자 지정 → 조치 등록
→ 점검자 확인 → HTML 보고서 생성
```

생성된 보고서는 실행 위치의 `safelog-demo-data/inspection-report.html`에서 확인한다.

## 4. 각 팀원이 실제로 수정할 부분

### A 담당자: 현장 점검 등록

브랜치: `feature/capture`

주 작업 폴더:

```text
src/capture/
```

이미 만들어진 `CaptureService`를 사용해 다음 기능을 완성한다.

- 새 점검 생성
- 장소, 발견 내용, 조치 의견 입력
- 여러 장의 조치 전 사진 등록
- 작성 중인 점검 수정
- 입력 누락 메시지

다른 기능이 필요하면 `IRepository`를 호출한다. 저장소 구현 파일을 직접 수정하지 않는다.

### B 담당자: 저장소와 사진

브랜치: `feature/storage`

주 작업 폴더:

```text
src/storage/
database/
```

현재 `InMemoryRepository`는 앱을 종료하면 데이터가 사라진다. 이를 참고해서 `SqliteRepository`를 구현한다.

- `database/schema.sql`로 SQLite 테이블 생성
- `IRepository`의 모든 함수 구현
- 사진을 앱 데이터 폴더로 복사
- 앱 재실행 후 데이터 복원
- 시연용 사업장과 계정 데이터 삽입

완성 후 통합 코드에서 `InMemoryRepository`만 `SqliteRepository`로 교체할 수 있어야 한다.

### C 담당자: 담당자 지정과 조치 처리

브랜치: `feature/workflow`

주 작업 폴더:

```text
src/workflow/
```

이미 만들어진 `WorkflowService`를 확장한다.

- 담당자 지정
- 조치 시작
- 조치 내용과 조치 후 사진 제출
- 점검자 확인 완료
- 확인 반려 및 반려 사유
- 기한 초과 목록
- 잘못된 상태 전환 테스트

모든 상태 변경은 반드시 `WorkflowService` 안에서 처리한다.

### D 담당자: 보고서

브랜치: `feature/reporting`

주 작업 폴더:

```text
src/reporting/
```

현재 HTML 보고서 생성 코드를 기준으로 다음 기능을 만든다.

- 보고서 미리보기
- 조치 전후 사진 표시
- 조치 이력 표시
- 누락 항목 검사
- Qt의 `QPdfWriter`를 이용한 PDF 생성
- Android 공유 메뉴 호출

보고서는 데이터베이스를 직접 읽지 않고 `ReportService`가 제공하는 `ReportData`만 사용한다.

## 5. Qt 화면 작업 방법

현재 화면 시작점은 다음 위치에 있다.

```text
apps/mobile-qt/qml/Main.qml
apps/mobile-qt/app_controller.hpp
apps/mobile-qt/app_controller.cpp
```

Qt가 설치된 PC에서 다음과 같이 켠다.

```bash
cmake -S . -B build-qt -DSAFELOG_BUILD_QT=ON -DCMAKE_PREFIX_PATH="Qt 설치 경로"
cmake --build build-qt
```

각 담당자가 자신의 QML 화면을 추가하되, C++ 기능 호출은 `AppController` 또는 기능별 Controller를 통한다. QML에서 데이터베이스 SQL을 직접 실행하지 않는다.

추천 화면 분담:

```text
qml/capture/NewInspectionPage.qml        A 담당
qml/capture/FindingFormPage.qml          A 담당
qml/workflow/AssignedTasksPage.qml       C 담당
qml/workflow/ActionDetailPage.qml        C 담당
qml/reporting/ReportPreviewPage.qml       D 담당
qml/common/                               통합 시 공동 관리
```

## 6. 매일 작업을 합치는 방법

각 팀원은 작업을 작은 단위로 커밋하고 자기 브랜치에 push한다.

```bash
git add 담당폴더
git commit -m "feat: 발견 사항 입력 검증 추가"
git push origin feature/capture
```

담당 폴더 밖의 파일을 수정했다면 Pull Request 설명에 이유를 적는다.

## 7. 최종 병합 순서

다음 순서로 `main`에 병합한다.

1. `feature/storage`
2. `feature/capture`
3. `feature/workflow`
4. `feature/reporting`
5. `apps/mobile-qt`에서 최종 연결

각 단계가 끝날 때마다 빌드와 테스트를 실행한다.

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

## 8. 학술제 완성 기준

다음 시나리오가 한 대의 Android 휴대폰에서 끊기지 않고 실행되면 1차 완성이다.

1. 점검자가 사업장을 선택한다.
2. 현장 사진과 발견 내용을 등록한다.
3. 관리자가 조치 담당자를 지정한다.
4. 담당자가 조치 후 사진과 내용을 등록한다.
5. 점검자가 전후 사진을 보고 확인한다.
6. 앱이 PDF 보고서를 생성하고 공유한다.

AI 문장 정리와 통계 화면은 이 흐름이 완성된 뒤 추가한다.
