# SafeLog AI 역할 1 구현 및 연결 보고

검증일: 2026-11-09
브랜치: `feature/capture-ai-ui`
확인 환경: Windows, Qt 6.8.3, MinGW 13.1.0 64-bit, C++20, Debug
로컬 프로젝트: `C:\dev\robo\safelog-cpp`

[담당 역할]

역할 1 — 현장 사진 등록과 AI 위험 제안 검토 화면.

이번 구현과 확인 범위는 Windows 데스크톱에서 사진 파일을 선택하여 실행하는 Mock 기반 흐름이다.

[구현 내용]

- 장소, 현장 메모, 사진 한 장 입력 및 사진 미리보기.
- 입력 누락, 읽을 수 없는 사진, 이미지 해석 실패 검사.
- CaptureService로 사진과 원래 메모를 먼저 저장한 뒤 AI 분석 시작.
- 백그라운드 QThread에서 AiSafetyService::analyzeFinding 실행.
- 로딩, 결과 수신 취소, 약 30초 시간 초과, JSON 오류, 분석 실패, 재시도 표시.
- JSON 필수 필드와 자료형, 위험 등급 1~5 정수, 유한한 신뢰도 0~1 검증.
- JSON의 위험 종류·등급·신뢰도가 AiAnalysis 메타데이터와 일치하는지 검사.
- 채택·수정·거절을 AiSafetyService::reviewHazard로 저장.
- 수정된 문장은 Edited로 저장하고, 거절 시 원래 발견 문장과 개선안을 유지.
- AI 실패 또는 취소 후 수동 입력으로 같은 finding을 갱신.
- 수동 입력 저장 시 변경 전 내용과 finding_manual_saved 이력 기록.
- 서비스가 반환한 저장 결과로 완료 화면 표시.
- 중복 저장 및 완료 후 재분석 방지.
- Mock에 명시적인 실행 옵션을 전달하여 실패·JSON 오류·지연을 재현.

[변경 파일]

아래 경로의 프로젝트 위치는 각 팀원의 체크아웃 위치에 맞게 바꾼다.

역할 1 담당 파일:

- `C:\dev\robo\safelog-cpp\src\capture\capture_service.cpp`
- `C:\dev\robo\safelog-cpp\src\capture\include\safelog\capture\capture_service.hpp`
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\controllers\capture\capture_controller.hpp`
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\controllers\capture\capture_controller.cpp`
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\controllers\capture\capture_mock_ai_analyzer.hpp`
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\qml\capture\FindingCapturePage.qml`
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\qml\capture\AiHazardReviewPage.qml`
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\qml\capture\FindingSavedPage.qml`

화면과 서비스를 실행 앱에 연결하기 위해 수정한 공용 파일 — 병합 시 다른 팀원과 함께 확인:

- `C:\dev\robo\safelog-cpp\apps\mobile-qt\CMakeLists.txt`: QML, 컨트롤러 소스, AI 라이브러리 연결.
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\main.cpp`: CaptureController 생성 및 QML 컨텍스트 등록.
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\app_controller.hpp`: AI 의존성 및 서비스 접근자.
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\app_controller.cpp`: AiSafetyService 생성자 연결.
- `C:\dev\robo\safelog-cpp\apps\mobile-qt\qml\Main.qml`: 화면 전환과 검토·수동 저장 완료 신호 연결.

[실행 방법]

1. Qt Creator에서 프로젝트 루트 CMakeLists.txt를 연다.
   `C:\dev\robo\safelog-cpp\CMakeLists.txt`
2. Desktop Qt 6.8.3 MinGW 64-bit, Debug 키트를 사용한다.
3. CMake 설정 SAFELOG_BUILD_QT와 SAFELOG_BUILD_TESTS를 ON으로 설정한다.
4. 빌드하고 실행 대상을 safelog_mobile로 선택한다.
5. 정상 실행 시 Command line arguments를 비운다.
6. A · 새 점검/사진 등록 → 장소·메모·사진 입력 → 저장 후 AI 분석.
7. AI 검토 화면에서 채택, 수정 후 저장, 또는 거절을 선택한다.
8. 완료 화면에서 저장 방식, 기록 ID, 최종 발견 문장, 최종 개선안을 확인한다.

Mock 실패 재현은 앱을 종료한 뒤 Run Settings의 Command line arguments에 다음 옵션 중 하나만 넣고 다시 실행한다.

| 실행 옵션 | 동작 |
|---|---|
| --capture-ai-fail | 모델 서버 연결 실패 메시지를 내는 예외 재현. 실제 네트워크 요청은 하지 않음. |
| --capture-ai-bad-json | 잘못된 JSON 응답 재현. |
| --capture-ai-timeout | Mock 작업을 32초 지연하여 UI의 약 30초 시간 초과 확인. |
| --capture-ai-slow | Mock 작업을 10초 지연하여 로딩 및 결과 수신 취소 확인. |

옵션은 Mock 객체 생성 시 읽는다. 실행 중 옵션을 바꿔도 적용되지 않으며, 실패 옵션을 유지한 재시도는 다시 실패한다. 정상 모드 복귀 시 옵션을 지우고 앱을 재시작한다.

[테스트 결과]

사용자가 앱에서 확인했다고 보고한 항목:

- Qt Creator 빌드 성공.
- 정상 분석 후 채택 완료 화면.
- 수정 후 저장 완료 화면.
- 거절 후 원문 유지.
- 연결 실패 재현 후 수동 저장.
- JSON 오류 후 수동 저장.
- 시간 초과 및 취소 후 수동 저장.
- 테스트 옵션 제거 후 정상 흐름 복귀.

별도 임시 복사본에서 자동 검증한 항목:

- 수동 입력 화면 연결 및 완료 화면 전환.
- 공백 입력 차단.
- 서비스 저장 실패 시 편집 내용 유지 및 완료 신호 미발생.
- 같은 finding ID와 사진 유지.
- 변경 전 내용 이력 기록.
- 중복 저장 및 완료 후 재분석 차단.
- Accepted, Edited, Rejected의 저장소 상태 및 완료 화면.
- 다른 점검자의 수동 변경 거부.
- 검토 확정된 기록의 수동 덮어쓰기 거부.
- JSON 오류·취소·시간 초과 후 수동 저장.
- 수동 저장이 AI 분석을 자동으로 채택하거나 거절하지 않음.

현재 프로젝트 빌드 산출물의 기존 테스트 직접 실행 결과:

- `C:\dev\robo\safelog-cpp\build\Desktop_Qt_6_8_3_MinGW_64_bit_Debug\tests\safelog_smoke_tests.exe`
- 종료 코드: 0, 통과.
- 테스트마다 별도의 임시 폴더를 사용하여 이전 실행 사진과의 충돌을 피했다.
- git diff --check: 통과.

기존 smoke_tests는 통합 서비스 흐름을 확인한다. 모든 화면 동작을 검사하는 테스트는 아니다. 임시 검증 프로그램은 프로젝트의 테스트 파일로 추가하지 않았다.

[AI 실패 시 동작]

사진과 원래 메모는 AI 분석 전에 저장된다. 실패 후 재시도는 같은 finding ID를 사용한다.

수동 입력에서는 최종 발견 문장과 개선안을 모두 입력해야 한다. 수동 저장은 CaptureService::saveManualFinding으로 기존 finding을 갱신하며, 화면 표시용 값 Manual을 사용한다. AiReviewDecision enum에는 Manual을 추가하지 않는다.

취소와 시간 초과는 늦게 도착한 결과를 화면에서 무시하는 방식이다. 백그라운드 분석 호출 자체는 종료될 때까지 계속되며, 그동안 수동 입력 작성은 가능하지만 저장·재시도·화면 이동은 제한된다. 작업이 끝나면 수동 저장할 수 있다.

유효하지 않거나 무시된 분석 레코드가 서비스 저장소에 Pending으로 남을 수 있다. 사용자의 검토 없이 Accepted 또는 Rejected로 변경하지 않는다.

[남은 작업]

- 역할 2와 SQLite 영구 저장 및 사진 저장 구현 연결.
- 역할 3과 실제 IAiSafetyAnalyzer 구현 연결.
- 실제 모델 호출의 네트워크 제한 시간과 취소 방식 확인.
- 무시되거나 잘못된 AI 분석 레코드의 보관·재분석 정책 합의.
- Android 등 실제 모바일 기기에서 사진 선택·카메라·권한·파일 경로 검증.
- 공용 파일 변경 내용을 다른 팀원의 변경과 병합.
- 문서와 소스 변경을 확인한 뒤 커밋 및 GitHub 공유.

현재 AppController는 InMemoryRepository를 사용한다. 프로그램 재시작 후 기록을 복원하는 영구 저장은 구현 범위에 포함되지 않는다. 현재 Mock 응답은 고정된 예시이며 실제 사진 내용을 판단한 결과가 아니다.

[다른 파트 연결 요청]

역할 2 — 저장소:

- CaptureController와 AiSafetyService가 같은 저장소의 finding·photo·AI 분석·이력을 공유해야 한다.
- 수동 저장에 추가된 CaptureService::saveManualFinding(findingId, actorId, description, actionOpinion)을 확인한다.
- UI에서 SQL을 실행하지 않는다.
- SQLite 연결의 스레드 사용 조건을 확인한다. 현재 AI 서비스 호출은 백그라운드 스레드에서 이루어진다.
- 검토 결정, finding 갱신, 이력 저장 도중 실패했을 때의 일관성 및 복구 방법을 함께 검토한다.

역할 3 — AI:

- IAiSafetyAnalyzer의 함수 이름과 공통 계약 타입은 유지한다.
- HazardSuggestion의 rawJson에 아래 필드를 제공한다.

| JSON 필드 | 요구 값 |
|---|---|
| hazardCategory | 공백이 아닌 문자열 |
| riskLevel | 1~5의 정수 |
| detectedHazards | 문자열 배열 |
| suggestedDescription | 공백이 아닌 문자열 |
| suggestedAction | 공백이 아닌 문자열 |
| confidence | 유한한 숫자, 0~1 |

- rawJson의 hazardCategory, riskLevel, confidence는 반환 객체의 category, riskLevel, confidence와 일치해야 한다.
- 기존 Mock의 부족한 JSON 필드는 역할 1의 CaptureMockAiAnalyzer가 보완하고 있다. 실제 AI 구현에서도 동일한 필드가 필요하다.
- 실제 모델 호출에는 제한 시간을 설정하고 예외를 UI에 전달할 수 있어야 한다.
- 모바일 앱에 API 비밀 키를 넣지 않는다.

역할 3·4 및 공용 화면 담당:

- 검토 저장에는 findingId가 아닌 AI 분석 ID를 AiSafetyService::reviewHazard에 전달한다.
- 채택·수정·거절은 AI 검토 결정이다. 조치 완료 검증 상태 Verified와 구분한다.
- 이번 화면에서 기록과 검토를 저장해도 FindingStatus는 Open으로 유지한다.
- 담당자 배정, 조치 후 사진 비교, 최종 검증, 보고서 생성은 후속 흐름에서 처리한다.

Git 공유 전 확인:

- build, .qtcreator, 생성된 DB·사진·모델, API 키를 커밋하지 않는다.
- 현재 보이는 build와 .qtcreator는 Git의 로컬 제외 목록에 추가한다.
- 이 보고서는 `C:\dev\robo\safelog-cpp\src\capture\CAPTURE_AI_HANDOFF_KO.md`에 복사하여 역할 1 문서로 포함할 수 있다.
