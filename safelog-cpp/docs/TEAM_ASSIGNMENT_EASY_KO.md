# SafeLog 4인 균형 분담표 — 이전 비 AI MVP 버전

> AI 중심 대회용 분담은 `AI_FIRST_TEAM_PROMPTS_KO.md`를 기준으로 한다.

이 분담표는 네 명이 서로 다른 브랜치에서 작업하고 마지막에 연결하는 방식이다. 한 사람이 서버, 인증, AI, 배포까지 떠맡지 않도록 1차 범위를 줄였다.

## 0. 전원이 지켜야 할 범위

### 1차 발표에 반드시 들어갈 기능

```text
가상 사업장 선택
→ 새 점검 생성
→ 발견 내용과 조치 전 사진 1장 등록
→ 담당자 지정
→ 조치 내용과 조치 후 사진 1장 등록
→ 점검자 확인
→ 보고서 미리보기 및 파일 생성
```

### 1차 범위에서 제외할 기능

- 회원가입 및 실제 로그인
- Supabase, AWS 같은 원격 서버
- 푸시 알림
- AI 위험 판정
- GPS 자동 기록
- 전자서명
- 여러 사업장 동기화

처음에는 고정된 가상 사용자 세 명을 사용한다.

```text
김안전: 점검자
박관리: 관리자
이조치: 조치 담당자
```

## 역할 1 — 현장 점검 화면 담당

### 맡을 범위

- 새 점검 화면
- 발견 장소, 발견 내용, 조치 의견 입력
- 조치 전 사진 1장 선택
- 필수 입력 검사
- `CaptureService` 호출

### 수정할 위치

```text
src/capture/
apps/mobile-qt/qml/capture/
```

### 만들어야 할 화면

```text
NewInspectionPage.qml
FindingFormPage.qml
FindingSavedPage.qml
```

### 화면 입력값

```text
siteId
inspectorId
location
description
actionOpinion
beforePhotoPath
```

### 사용할 C++ 함수

```cpp
auto inspection = captureService.startInspection(siteId, inspectorId);

auto bundle = captureService.addFinding({
    inspection.id,
    inspectorId,
    location,
    description,
    actionOpinion,
    beforePhotoPath
});
```

### 완료 기준

- 빈 항목이 있으면 저장되지 않는다.
- 사진이 없으면 안내 메시지가 나온다.
- 정상 입력하면 finding ID가 화면에 표시된다.
- 저장된 finding ID를 역할 3 화면으로 넘길 수 있다.

### 하지 않을 일

- SQLite 쿼리 작성
- 상태값 변경
- PDF 생성
- 실제 카메라 연동

사진은 1차 개발에서 파일 선택기로 받는다. 실제 카메라는 시간이 남으면 추가한다.

### 예상 난도

중간. QML 폼과 C++ 함수 호출만 연결하면 된다.

---

## 역할 2 — 로컬 데이터 저장 담당

### 맡을 범위

- SQLite 연결
- 테이블 최초 생성
- `IRepository` 구현
- 시연용 사업장과 사용자 저장
- 앱 종료 후 재실행해도 기록 유지

### 수정할 위치

```text
src/storage/
database/schema.sql
```

### 새로 만들 파일

```text
src/storage/include/safelog/storage/sqlite_repository.hpp
src/storage/sqlite_repository.cpp
src/storage/database_bootstrap.cpp
```

### 구현할 함수

```cpp
saveSite
saveProfile
saveInspection
saveFinding
savePhoto
saveActionLog
findSite
findProfile
findInspection
findFinding
photosForFinding
logsForFinding
findingsForInspection
findingsAssignedTo
```

함수 이름과 반환형은 `IRepository`에 이미 정해져 있으므로 변경하지 않는다.

### 쉬운 구현 순서

1. `database/schema.sql`을 앱 시작 시 한 번 실행한다.
2. 먼저 `saveSite`, `findSite` 두 함수만 만든다.
3. 같은 패턴으로 나머지 객체를 추가한다.
4. 날짜는 ISO 8601 문자열로 저장한다.
5. 사진 파일 자체는 DB에 넣지 않고 경로만 저장한다.

### 완료 기준

- 앱 실행 시 DB 파일이 자동 생성된다.
- 가상 사업장과 사용자 세 명이 저장된다.
- finding 한 건을 저장하고 앱을 재실행해도 조회된다.
- 기존 `InMemoryRepository` 대신 `SqliteRepository`를 넣어도 다른 코드가 바뀌지 않는다.

### 하지 않을 일

- 로그인 서버
- 네트워크 동기화
- 화면 디자인
- 상태 전환 규칙 작성

### 예상 난도

중간. 화면 작업은 없으며 같은 SQL 저장·조회 패턴을 반복한다.

---

## 역할 3 — 조치 처리 화면 담당

### 맡을 범위

- 미조치 발견 사항 표시
- 담당자 지정
- 조치 시작
- 조치 내용과 조치 후 사진 1장 등록
- 점검자의 확인 완료

### 수정할 위치

```text
src/workflow/
apps/mobile-qt/qml/workflow/
```

### 만들어야 할 화면

```text
FindingListPage.qml
ActionFormPage.qml
ReviewPage.qml
```

### 사용할 C++ 함수

```cpp
workflowService.assign(findingId, managerId, assigneeId);
workflowService.beginWork(findingId, assigneeId);
workflowService.submitAction(findingId, assigneeId, note, afterPhotoPath);
workflowService.verify(findingId, inspectorId, reviewNote);
```

### 화면 상태 표시

| 코드 | 화면 문구 |
|---|---|
| `open` | 조치 필요 |
| `in_progress` | 조치 중 |
| `pending_review` | 확인 대기 |
| `verified` | 확인 완료 |

### 완료 기준

- 버튼을 누르면 정해진 순서대로만 상태가 바뀐다.
- 담당자가 아니면 조치를 제출할 수 없다.
- 조치 내용이나 사진이 없으면 제출되지 않는다.
- 전후 사진을 함께 볼 수 있다.

### 하지 않을 일

- DB 직접 수정
- 점검 최초 등록
- 보고서 파일 생성
- 알림 기능

### 예상 난도

중간. 상태 전환 함수가 이미 구현되어 있어 목록과 버튼을 연결하는 작업이 중심이다.

---

## 역할 4 — 보고서와 공통 화면 담당

### 맡을 범위

- 홈 화면과 메뉴 이동
- 보고서 미리보기
- HTML 보고서 저장
- 가능하면 PDF 저장
- 최종 발표 시연 데이터 준비

### 수정할 위치

```text
src/reporting/
apps/mobile-qt/qml/Main.qml
apps/mobile-qt/qml/reporting/
```

### 만들어야 할 화면

```text
HomePage.qml
ReportPreviewPage.qml
```

### 사용할 C++ 함수

```cpp
ReportService reportService(repository);
HtmlRenderer renderer;

auto data = reportService.build(inspectionId);
renderer.writeFile(data, outputPath);
```

### 보고서 필수 항목

- 사업장명과 주소
- 점검자와 점검 일시
- 발견 장소와 내용
- 조치 의견
- 조치 전 사진
- 조치 내용
- 조치 후 사진
- 현재 상태와 처리 이력

### 완료 기준

- 확인 완료된 점검을 선택할 수 있다.
- 미리보기에서 한글과 전후 사진이 보인다.
- HTML 파일을 생성할 수 있다.
- PDF가 늦어지면 HTML 저장까지 먼저 완료한다.

### 하지 않을 일

- SQLite 구현
- 상태 전환 로직 수정
- AI 문장 생성
- 전체 팀원의 오류를 대신 수정

### 예상 난도

중간. HTML 생성기는 이미 있으므로 화면 연결과 출력 확인이 중심이다.

---

## 공통 연결 담당은 따로 두지 않는다

한 사람에게 통합을 모두 맡기면 그 사람이 가장 힘들어진다. 마지막 연결은 네 명이 함께 아래 순서로 진행한다.

| 순서 | 담당자가 직접 연결할 내용 |
|---|---|
| 1 | 역할 2가 `SqliteRepository`를 앱에 연결한다. |
| 2 | 역할 1이 점검 등록 화면을 연결하고 직접 오류를 고친다. |
| 3 | 역할 3이 상태 변경 화면을 연결하고 직접 오류를 고친다. |
| 4 | 역할 4가 보고서 화면과 홈 메뉴를 연결한다. |
| 5 | 네 명이 전체 시나리오를 한 번씩 실행한다. |

## 3주 일정 예시

### 1주 차

- 전원 프로젝트 빌드
- 역할 1: 입력 폼 완성
- 역할 2: SQLite 연결과 2개 테이블부터 구현
- 역할 3: 목록 및 상태 버튼 완성
- 역할 4: 홈과 보고서 미리보기 완성

### 2주 차

- 역할 1: 사진 선택과 저장 연결
- 역할 2: 전체 repository 함수 완성
- 역할 3: 조치 후 사진과 확인 화면 연결
- 역할 4: HTML 저장과 PDF 실험

### 3주 차

- 월요일: storage 병합
- 화요일: capture 병합
- 수요일: workflow 병합
- 목요일: reporting 병합
- 금요일: Android 시연과 발표 영상 촬영

## 팀원이 매일 보고할 형식

```text
[담당 파트]
오늘 완료: 
현재 실행되는 것: 
막힌 부분: 
다른 파트에 필요한 요청: 
내일 할 일: 
```

## 발표 전 최종 기준

다음 한 가지 시나리오만 완벽하게 동작시키는 것을 우선한다.

```text
2층 가공라인 통로에 자재 적치 발견
→ 사진과 개선 의견 등록
→ 이조치 담당자 지정
→ 자재 이동 후 사진 등록
→ 김안전 점검자 확인
→ 전후 사진이 포함된 보고서 생성
```

여러 기능을 조금씩 만드는 것보다 이 흐름 한 건을 오류 없이 시연하는 것이 먼저다.
