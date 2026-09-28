# SafeLog 팀원 전달용 개발 프롬프트 — 이전 비 AI 버전

> AI 중심 대회용 개발에는 이 문서를 사용하지 말고 `AI_FIRST_TEAM_PROMPTS_KO.md`를 사용한다.

## 전달 방법

각 팀원에게 다음 세 가지를 보낸다.

1. GitHub 저장소 주소: `https://github.com/jungsahun43-commits/robo.git`
2. 담당 브랜치 이름
3. 아래의 `공통 프롬프트`와 본인의 `역할별 프롬프트`

팀원은 저장소를 clone하고 `safelog-cpp` 폴더를 개발 도구에서 연다.

```bash
git clone https://github.com/jungsahun43-commits/robo.git
cd robo/safelog-cpp
```

AI 코딩 도구에는 공통 프롬프트를 먼저 붙이고, 한 줄을 띄운 다음 역할별 프롬프트 전체를 이어 붙인다.

---

# 모든 팀원이 공통으로 붙일 프롬프트

```text
당신은 4명이 병렬로 개발하는 C++20/Qt 6 프로젝트 ‘SafeLog’의 담당 개발자다.

프로젝트 목적:
제조업 현장의 안전점검자가 발견 사항과 조치 전 사진을 등록하고, 관리자가 담당자를 배정하며, 담당자가 조치 내용과 조치 후 사진을 제출하면, 최초 점검자가 이를 확인하고 전 과정을 보고서로 생성하는 학술제용 Android 앱을 만든다.

작업을 시작하기 전에 반드시 다음 파일을 읽어라.
- README.md
- QUICKSTART_KO.md
- docs/ARCHITECTURE.md
- docs/TEAM_ASSIGNMENT_EASY_KO.md
- include/safelog/contracts/types.hpp
- include/safelog/contracts/ports.hpp
- include/safelog/contracts/errors.hpp

기술 규칙:
1. 표준 C++ 코드는 C++20을 사용한다.
2. 모바일 UI는 Qt 6/QML을 사용한다.
3. 공통 상태값은 Open, InProgress, PendingReview, Verified 네 개만 사용한다.
4. 화면에서 SQL을 직접 실행하지 않는다.
5. 화면에서 상태를 임의로 바꾸지 않고 CaptureService 또는 WorkflowService를 호출한다.
6. 보고서는 IRepository를 직접 조작하지 않고 ReportService가 만든 ReportData를 사용한다.
7. include/safelog/contracts 아래의 공통 타입과 인터페이스 이름을 임의로 변경하지 않는다.
8. 다른 담당자의 기능 폴더를 수정하지 않는다. 공통 파일 수정이 반드시 필요하면 직접 변경하지 말고 마지막 작업 보고에 변경 제안으로 기록한다.
9. 아직 다른 파트가 완성되지 않았으면 InMemoryRepository와 가상 데이터를 이용해 자기 파트를 독립 실행 가능하게 만든다.
10. 실제 법적 적합성이나 AI 위험 판정을 보장하는 문구를 앱에 넣지 않는다.

시연용 고정 데이터:
- 사업장: 세이프 금속 가공공장
- 점검자: 김안전 / inspector-1
- 관리자: 박관리 / manager-1
- 조치 담당자: 이조치 / assignee-1
- 장소: 2층 가공라인 통로
- 발견 내용: 통로에 자재가 적치되어 이동 중 걸려 넘어질 위험이 있음
- 조치 의견: 자재를 지정 보관구역으로 이동하고 통로 폭을 확보할 것
- 조치 결과: 자재 이동 및 통로 정리 완료

공통 개발 순서:
1. 현재 브랜치와 파일 구조를 확인한다.
2. 기존 코드를 먼저 빌드하거나 최소한 담당 모듈의 공개 API를 확인한다.
3. 담당 기능을 작은 단위로 구현한다.
4. 정상 흐름과 오류 흐름을 각각 확인한다.
5. 기존 스모크 테스트가 깨지지 않게 한다.
6. 담당 기능의 테스트 또는 독립 확인 방법을 추가한다.

금지 사항:
- 기존 핵심 코드를 전부 새 프레임워크로 다시 작성하지 않는다.
- 계약 타입을 복사하여 별도의 유사 타입을 만들지 않는다.
- 비밀 키나 개인 경로를 코드에 넣지 않는다.
- build 폴더, IDE 설정, 생성된 DB와 보고서를 Git에 커밋하지 않는다.
- 다른 담당자의 미완성 기능을 대신 대규모로 구현하지 않는다.

완료 후 반드시 다음 형식으로 보고하라.
[담당 파트]
[구현한 기능]
[변경한 파일]
[실행 또는 테스트 방법]
[확인한 정상 흐름]
[확인한 오류 흐름]
[아직 남은 작업]
[다른 파트에 필요한 연결 요청]

이제 아래에 이어지는 역할별 요구사항만 구현하라.
```

---

# 역할 1 프롬프트 — 현장 점검 등록

브랜치: `feature/capture`

```text
내 담당은 현장 점검 등록 파트다.

수정 허용 범위:
- src/capture/**
- apps/mobile-qt/qml/capture/**
- 담당 파트 전용 테스트 파일

현재 제공된 기반:
- CaptureService::startInspection
- CaptureService::addFinding
- NewFindingInput
- InMemoryRepository
- IPhotoStore

구현 목표:
1. apps/mobile-qt/qml/capture 폴더를 만들고 아래 화면을 구현한다.
   - NewInspectionPage.qml
   - FindingFormPage.qml
   - FindingSavedPage.qml
2. 사업장, 점검자, 장소, 발견 내용, 조치 의견, 조치 전 사진 경로를 입력받는다.
3. 장소·발견 내용·조치 의견은 공백 문자열을 허용하지 않는다.
4. 조치 전 사진이 없으면 저장하지 않고 사용자에게 메시지를 보여준다.
5. 1차 구현에서는 실제 카메라 대신 Qt 파일 선택기로 JPG 또는 PNG 한 장을 고르게 한다.
6. 저장 성공 시 finding ID와 Open 상태를 결과 화면에 표시한다.
7. 저장 중 버튼 중복 클릭을 막고 성공 또는 실패 메시지를 표시한다.
8. C++ 서비스와 QML 사이에 필요한 연결 클래스가 있다면 capture 파트 내부에 CaptureController를 추가한다.

공개 연결 규칙:
- 다른 파트가 사용할 결과는 inspectionId와 findingId다.
- 상태값을 화면 코드에서 직접 변경하지 않는다.
- SQLite나 보고서 코드는 작성하지 않는다.

필수 오류 확인:
- 빈 장소
- 빈 발견 내용
- 빈 조치 의견
- 사진 미선택
- 존재하지 않는 사업장 ID

완료 기준:
- InMemoryRepository만으로 점검 한 건과 발견 사항 한 건을 생성할 수 있다.
- 저장 성공 후 생성된 ID를 확인할 수 있다.
- 필수 입력이 빠지면 프로그램이 종료되지 않고 화면에 오류가 나온다.
- 기존 smoke_tests가 계속 통과한다.

추가 기능은 필수 기능이 끝난 경우에만 구현한다.
- 조치 전 사진 여러 장
- 실제 카메라 촬영
- 작성 중 임시저장
```

---

# 역할 2 프롬프트 — SQLite와 사진 저장

브랜치: `feature/storage`

```text
내 담당은 로컬 데이터 영구 저장 파트다.

수정 허용 범위:
- src/storage/**
- database/**
- apps/mobile-qt/adapters/storage/**
- 저장 파트 전용 테스트 파일

현재 제공된 기반:
- IRepository 인터페이스
- InMemoryRepository 참고 구현
- LocalPhotoStore
- database/schema.sql

구현 목표:
1. Qt 앱에서 사용할 SqliteRepository를 구현한다.
2. 표준 C++ 코어가 Qt에 종속되지 않게 QtSql 기반 구현은 apps/mobile-qt/adapters/storage 아래에 둔다.
3. SqliteRepository는 IRepository의 모든 순수 가상 함수를 정확히 override한다.
4. 앱 데이터 디렉터리에 safelog.db를 생성한다.
5. 최초 실행 시 database/schema.sql과 동일한 테이블을 만든다.
6. sites, profiles, inspections, findings, photos, action_logs를 저장하고 조회한다.
7. TimePoint는 ISO 8601 UTC 문자열로 저장하고 다시 TimePoint로 복원한다.
8. SQL 값은 문자열 연결이 아니라 prepared query와 bind value를 사용한다.
9. 최초 실행 시 시연용 사업장과 사용자 세 명을 중복 없이 삽입한다.
10. 사진은 DB BLOB으로 넣지 않고 앱 데이터 폴더로 복사한 뒤 storagePath만 저장한다.
11. 저장 실패 시 예외 또는 명확한 오류 결과를 반환하고 DB 오류 내용을 로그로 남긴다.

필수 조회 시나리오:
- ID로 사업장과 사용자 조회
- inspectionId로 발견 사항 목록 조회
- findingId로 사진과 처리 이력 조회
- assigneeId로 담당 업무 목록 조회

필수 오류 확인:
- 잘못된 외래키
- 같은 기본키 중복 저장
- DB 파일을 열 수 없는 경우
- 존재하지 않는 ID 조회
- 없는 사진 파일 복사

독립 테스트:
1. 임시 폴더에 테스트 DB를 만든다.
2. 모든 객체를 한 건씩 저장한다.
3. repository 객체와 DB 연결을 닫는다.
4. 새 repository로 같은 DB를 연다.
5. 저장된 finding, 사진 경로, action log가 그대로 조회되는지 확인한다.

공개 연결 규칙:
- IRepository 함수 시그니처는 변경하지 않는다.
- 앱 통합 시 InMemoryRepository 생성 한 줄을 SqliteRepository로 바꿀 수 있어야 한다.
- QML 화면과 상태 전환 로직은 작성하지 않는다.

완료 기준:
- 앱 재실행 이후에도 기록이 유지된다.
- database/schema.sql이 SQLite에서 오류 없이 실행된다.
- 기존 InMemoryRepository 및 smoke_tests도 계속 작동한다.

추가 기능은 필수 기능이 끝난 경우에만 구현한다.
- DB 내보내기와 백업
- 마이그레이션 버전 테이블
- 데이터 암호화
```

---

# 역할 3 프롬프트 — 담당자 배정과 조치 처리

브랜치: `feature/workflow`

```text
내 담당은 담당자 배정과 조치 처리 파트다.

수정 허용 범위:
- src/workflow/**
- apps/mobile-qt/qml/workflow/**
- 담당 파트 전용 테스트 파일

현재 제공된 기반:
- WorkflowService::assign
- WorkflowService::beginWork
- WorkflowService::submitAction
- WorkflowService::verify
- findingsAssignedTo

구현 목표:
1. apps/mobile-qt/qml/workflow 폴더를 만들고 아래 화면을 구현한다.
   - FindingListPage.qml
   - ActionFormPage.qml
   - ReviewPage.qml
2. 목록 화면에서 상태 코드 대신 다음 한글 문구를 보여준다.
   - open: 조치 필요
   - in_progress: 조치 중
   - pending_review: 확인 대기
   - verified: 확인 완료
3. 관리자는 Open finding에 이조치 담당자를 지정할 수 있다.
4. 담당자는 자신에게 지정된 finding만 조치 시작할 수 있다.
5. 담당자는 조치 내용과 조치 후 JPG/PNG 한 장을 입력해야 제출할 수 있다.
6. 점검자는 조치 전후 사진과 처리 내용을 비교한 뒤 확인 완료할 수 있다.
7. 각 작업 후 최신 finding을 다시 조회하여 화면을 갱신한다.
8. 필요한 경우 workflow 파트 내부에 WorkflowController를 추가한다.
9. 서비스에 반려 기능을 추가한다.
   - pending_review에서만 반려 가능
   - 반려 사유 필수
   - 상태는 in_progress로 돌아감
   - action log에 rejected와 사유 기록

필수 상태 테스트:
- Open → InProgress 성공
- InProgress → PendingReview 성공
- PendingReview → Verified 성공
- Open에서 바로 Verified 실패
- 다른 담당자의 submitAction 실패
- 조치 내용 또는 조치 후 사진이 없으면 실패
- 최초 점검자가 아닌 사용자의 verify 실패
- 반려 사유가 없으면 실패

공개 연결 규칙:
- DB를 직접 수정하지 않고 IRepository를 사용한다.
- 점검 생성이나 보고서 렌더링은 작성하지 않는다.
- 상태 전환 규칙은 QML이 아닌 WorkflowService에 둔다.

완료 기준:
- InMemoryRepository로 전체 상태 흐름과 반려 흐름을 테스트할 수 있다.
- UI 버튼은 현재 상태에서 가능한 작업만 활성화된다.
- 전후 사진 경로와 처리 이력이 화면에 표시된다.
- 기존 smoke_tests가 계속 통과한다.

추가 기능은 필수 기능이 끝난 경우에만 구현한다.
- 기한 초과 표시
- 상태별 필터
- 담당자별 건수
```

---

# 역할 4 프롬프트 — 보고서와 앱 홈

브랜치: `feature/reporting`

```text
내 담당은 보고서 생성과 앱 공통 홈 화면 파트다.

수정 허용 범위:
- src/reporting/**
- apps/mobile-qt/qml/Main.qml
- apps/mobile-qt/qml/reporting/**
- apps/mobile-qt/adapters/reporting/**
- 보고서 파트 전용 테스트 파일

현재 제공된 기반:
- ReportService::build
- ReportData
- HtmlRenderer::render
- HtmlRenderer::writeFile
- 초기 Main.qml

구현 목표:
1. 홈 화면에 다음 메뉴를 만든다.
   - 새 점검
   - 조치할 항목
   - 확인 대기
   - 완료 보고서
2. apps/mobile-qt/qml/reporting 아래 ReportPreviewPage.qml을 만든다.
3. ReportData를 이용해 다음 항목을 미리보기로 표시한다.
   - 사업장명과 주소
   - 점검자와 점검 일시
   - 발견 장소와 발견 내용
   - 조치 의견
   - 조치 전후 사진
   - 현재 상태
   - 시간순 처리 이력
4. HTML 보고서가 UTF-8 한글을 정상적으로 표시하게 한다.
5. 로컬 사진 경로를 HTML에서 실제 표시 가능한 file URL로 변환한다.
6. verified 상태의 finding만 최종 완료 보고서로 표시하도록 선택 옵션을 제공한다.
7. Qt 기반 PdfReportWriter를 apps/mobile-qt/adapters/reporting 아래 구현한다.
8. QPdfWriter와 QTextDocument를 사용해 앱 문서 폴더에 PDF를 생성한다.
9. PDF 실패 시에도 HTML 보고서를 저장하고 사용자에게 파일 경로를 보여준다.

필수 보고서 검증:
- 한글이 깨지지 않음
- 조치 전 사진 표시
- 조치 후 사진 표시
- 발견 및 조치 문구 표시
- action log 시간순 정렬
- finding이 하나도 없으면 생성 거부와 안내 메시지

공개 연결 규칙:
- 저장소를 직접 수정하지 않는다.
- WorkflowService의 상태 전환 코드를 수정하지 않는다.
- ReportService가 만든 ReportData만 렌더러에 전달한다.
- 다른 파트 화면의 내부 구현을 대신하지 않는다. 홈 버튼의 목적지만 신호로 노출한다.

완료 기준:
- 가상 inspection ID로 미리보기 화면을 열 수 있다.
- 전후 사진과 한글이 들어간 HTML 파일을 생성할 수 있다.
- Qt가 준비된 환경에서는 PDF 파일을 생성할 수 있다.
- PDF 구현이 지연되어도 HTML 보고서는 반드시 완성한다.
- 기존 smoke_tests가 계속 통과한다.

추가 기능은 필수 기능이 끝난 경우에만 구현한다.
- Android 공유 Intent
- 회사 로고와 서명 영역
- 월간 보고서 통계
```

---

# 팀장용 최종 통합 프롬프트

네 브랜치가 각각 Pull Request 준비 상태가 된 뒤 팀장 또는 네 명이 함께 사용한다.

```text
SafeLog의 네 기능 브랜치를 main에 통합하려고 한다.

병합 순서:
1. feature/storage
2. feature/capture
3. feature/workflow
4. feature/reporting

통합 목표:
- Qt 앱 시작 시 SqliteRepository를 한 번 생성한다.
- 같은 IRepository 인스턴스를 CaptureService, WorkflowService, ReportService에 주입한다.
- 각 기능별 Controller를 한 곳에서 생성하고 QML에 노출한다.
- Main.qml에서 새 점검, 조치 목록, 확인 대기, 보고서 화면으로 이동한다.
- 기능 모듈끼리는 직접 참조하지 않는다.
- 계약 타입과 상태 문자열을 바꾸지 않는다.

필수 통합 시나리오:
1. 김안전이 점검과 발견 사항을 생성한다.
2. 조치 전 사진이 저장된다.
3. 박관리가 이조치를 담당자로 지정한다.
4. 이조치가 조치를 시작하고 조치 후 사진과 내용을 제출한다.
5. 김안전이 전후 사진을 보고 확인 완료한다.
6. verified 상태와 전체 처리 이력이 DB에 남는다.
7. 앱을 재실행해도 기록이 유지된다.
8. 같은 기록으로 HTML 보고서를 생성한다.
9. Qt 환경에서 PDF 생성을 시도한다.

통합 과정에서 오류가 발생하면 오류를 소유한 모듈을 식별하고 해당 모듈 안에서 최소 수정한다. 한 파일에 모든 로직을 옮겨서 해결하지 않는다.

완료 후 다음을 보고하라.
- 병합한 브랜치와 순서
- 충돌 파일과 해결 내용
- 전체 빌드 결과
- 테스트 결과
- Android 또는 데스크톱 시연 결과
- 남은 기능과 발표 시 우회 방법
```
