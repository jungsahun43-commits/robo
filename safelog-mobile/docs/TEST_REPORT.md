# SafeLog Mobile 검증

## SafeLog AI 홈 개편

Android 15 API 35 Expo Go에서 Inspector, Manager, Assignee 홈과 각 역할의 메뉴·건수를 확인했다. 한 시연 기록으로 촬영·등록, 보류된 AI 제안 다시 열기, 분석 실패와 재시도, 제안 채택, 관리자 배정, 조치 시작과 진행 작업 목록, 조치 사진 제출, 원 점검자 전후 비교 및 최종 확인, 보고서 미리보기와 PDF 공유 창까지 실행했다. 공유 대상을 선택하지 않고 닫았다.

Expo lint, TypeScript, 기존 워크플로 테스트 3개가 통과했다. 이 검증은 시연 계정, Mock AI, 메모리 저장 환경에서 수행했다.

## 초기 이전 검증

환경은 Windows 11, Android Studio Medium Phone 에뮬레이터(Android 15, API 35), Expo Go, Expo SDK 58이다. PowerShell 실행 정책이 `npx.ps1`을 차단해 `npx.cmd`를 사용했다.

## 자동 검사

| 검사 | 결과 |
|---|---|
| `npx.cmd expo lint` | 통과. 기존 템플릿의 브라우저 테마 훅과 모바일 AI 화면의 React Hooks 린트 오류를 수정했다. |
| `npx.cmd tsc --noEmit` | 통과. Metro가 Expo Router 경로 타입을 재생성한 뒤 실행했다. 기존 웹 탭의 ref 타입 불일치도 수정했다. |
| `npx.cmd --yes tsx --test tests/workflow.test.mjs` | 3개 통과. 잘못된 로그인과 입력, 역할·소유 권한, Open 배정·시작·제출·재조치·최종 확인, AI 분리·실패·취소를 검증했다. |
| `npx.cmd expo install --check` | SDK 의존성 검사 통과. |
| `npx.cmd expo export --platform ios --output-dir .expo/ios-check` | iOS JavaScript 번들 생성 통과. |
| `git diff --check` | 통과. Git의 LF→CRLF 안내가 출력됐다. |

## Android 실제 확인

- Metro `/status` 응답 `packager-status:running`. Expo Go 초기 진입에서는 로딩이 계속됐고, Android 호스트 주소 `exp://10.0.2.2:8081`로 다시 열자 로그인 화면이 표시됐다.
- 시연 Inspector로 로그인해 대시보드와 역할을 확인하고, 새 점검 화면으로 이동했다.
- 사진 없이 등록 버튼을 눌러 필수 사진 오류 메시지를 확인했다.
- 시스템 카메라 권한 창, 촬영, 카메라의 완료 버튼을 거쳐 점검 화면에 첨부 사진이 표시되는 것을 확인했다.
- Android 갤러리 선택 취소 시 기존 첨부가 유지됐다. 별도 실행에서 에뮬레이터에 등록한 테스트 PNG를 갤러리에서 선택해 점검 화면의 사진 미리보기까지 확인했다.

- 장소·발견 메모·조치 의견을 입력해 Open 기록을 만들었다. 자동 Mock 위험 분석, 실패 응답, 정상 응답 재시도, 제안 채택을 확인했다.
- 진행 보고서 미리보기에서 점검 내용, 사진, AI 제안, 사람 조치 이력을 확인했다.
- 로그아웃 후 Manager로 로그인해 담당자를 배정하고 Open 유지와 담당자 이름 표시를 확인했다.
- Assignee로 로그인해 해당 작업만 보이는 목록에서 조치를 시작하고, 내용과 촬영 사진을 제출해 PendingReview 화면으로 이동했다.
- 원 Inspector로 로그인해 전후 사진 Mock 비교와 최종 확인 입력을 거쳐 Verified 표시 및 완료 보고서 진입 버튼을 확인했다. 추가 조치 요청과 이전 AI 제안 재사용 방지는 서비스 자동 검사에서 확인했다.

- 처음에는 Expo Go가 Print의 호스트 캐시 PDF 공유를 거부했다. Android에서 앱 캐시에 공유용 PDF를 쓰도록 수정하고 새 Metro 번들에서 재시험했다. Android 공유 창에 `safelog-finding-….pdf`가 표시됐고, 공유 대상을 선택하지 않고 닫았다.

## 검증 범위

iOS 및 웹 실행, 별도 네이티브 빌드, 운영 인증, 영구 저장, 실제 AI 추론과 모델 성능은 미검증이다. 패키지 설치 도중 npm이 중간 등급 취약점 3건을 보고했다. 의존성 강제 갱신은 적용하지 않았다.
# 2026-11-19 · 고정 CPU AI 연결 추가 검증

기준 커밋은 `c422eb5d7f0445a0e6d3bdf55b34b2e5916fc8e9`이며, 별도 로컬 `codex/react-native-ai` 작업본에서 검증했다. 아래 기존 최초 검증 기록은 당시 결과로 보존한다.

- 잠금 파일 기준 새 의존성 설치 성공. Expo lint, TypeScript 검사, 웹 export 성공.
- 워크플로 7개, 고정 CPU HTTP 계약 23개, loopback gateway 9개로 총 39개 테스트 통과.
- 기존 고정 CPU 서버 두 개와 새 모바일 HTTP 분석기·gateway를 연결해 승인된 시연 사진 6장(시설·컨베이어·PCB의 정상/이상 각 1장)의 기대 후보 판정과 입력 SHA가 모두 일치했다.
- 같은 모바일 메모리 WorkflowService에서 컨베이어 등록·AI 검토·담당자 배정·조치 제출·실제 전후 비교·사람 최종 확인을 통과했다. 서로 다른 촬영 기록의 예시이므로 현장 조치 증명이나 새 성능 벤치마크가 아니다.
- 고정 TEST 읽기, GPU 작업, 가중치 변경은 0이다. 이번에 시작한 모델 서버와 gateway는 검증 후 종료했다.
- Expo 웹 번들은 생성됐으나 실제 브라우저 조작, Android/iOS 실기기, 발표 PC 리허설은 아직 검증하지 않았다. 금속판·나사의 별도 정상/손상 사진 리허설도 남아 있다.
- 메모리 저장과 공개 시연 계정은 유지한다. 운영 인증·영속 저장 완료로 표시하지 않는다.

작업 공간의 원형 기록: `outputs/safelog-rn-live-20261119/result.json`, `process-receipt.json`, `safelog-rn-tests-20261119-current-tsx.log`, `safelog-rn-lint-20261119-final-verified.log`, `safelog-rn-typecheck-20261119-final-verified.log`, `safelog-rn-web-export-20261119-verified.log`. 초기 실행기 호환 오류와 제한 환경의 출력 폴더 오류도 별도 로그로 보존했다.

## 2026-11-19 · 전체 항목과 볼스크루 포함 시연 목록

앞선 39개 검사·6장 실제 연결 기록을 보존하고 전체 항목 확장을 별도로 검증했다.

- 고정 AI 20항목은 실제 서버의 일반 3개·장비 12개·PCB 5유형과 일치한다. 장비 12종의 사진 평가 집계를 선정된 전달본 `evaluation.json`과 대조해 차이가 없었다.
- 워크플로 9개, 고정 CPU HTTP 계약 53개, gateway 9개로 **71개 모두 통과**했다. 모든 분야의 요청·응답 범위, PCB 유형별 SHA 확인·자동 비교 차단, 직접 검토 분야의 AI 요청 차단과 볼스크루 포함 시연 순위를 검사했다. 이 테스트는 모형 응답을 사용한다.
- 전체 항목 최종 Expo lint는 경고·오류 없이 통과했고 TypeScript·Expo Web export도 통과했다. 실제 브라우저 화면 조작이나 Android/iOS 실행을 검증한 것은 아니다.
- 최초 전체 테스트는 제한 환경에서 loopback 연결 `EACCES`로 실패했다. 해당 로그는 보존하고 로컬 loopback 통신을 허용한 별도 실행에서 71개 통과를 확인했다. 실패 원형을 통과로 수정하지 않았다.
- 볼스크루·시설 균열·컨베이어 이물질·RobustAD PCB·PCB4는 기존 집계의 `max(오탐률, 미탐률)` 순서다. 정상·손상 각각 20장 이상인 항목만 비교했다. 시설 균열의 일부 출처 수치를 시설 전체에 대입하지 않는다.
- 실제 사진 연결 검증은 앞선 시설·컨베이어·RobustAD PCB 6장 범위다. 볼스크루·PCB4의 별도 시연 사진 모바일 리허설은 남아 있다. 새 TEST 추론·학습·GPU·가중치 변경·추가 실모델 실행은 하지 않았다.

새 로컬 근거: `outputs/safelog-rn-all-fields-tests-20261119-loopback.log`(71개 통과), `safelog-rn-all-fields-lint-20261119-final.log`, `safelog-rn-all-fields-typecheck-20261119-final.log`, `safelog-rn-all-fields-web-export-20261119.log`. 처음 제한 환경 실패는 `safelog-rn-all-fields-tests-20261119.log`에 보존한다. 전체 항목표와 입력 범위는 [INSPECTION_CATALOG.md](INSPECTION_CATALOG.md)에 있다.

## 2026-11-20 · 발표 5번째를 케이블로 변경

- 사용자의 선택에 따라 `demoTasks`를 볼스크루·시설 균열·컨베이어 이물질·RobustAD PCB·케이블로 고정했다. 숫자 기준 순위 `researchTopTasks`는 따로 보존하고 PCB4도 전체 항목에 유지했다. 화면은 “발표 선정 5개”로 표시한다.
- 변경 후 71개 테스트, Expo lint(오류·경고 0), TypeScript, Expo Web export가 통과했다. 순위와 실제 발표 선택이 서로 다름을 검사한다.
- 모델·판정 기준·기존 평가 집계는 바꾸지 않았다. 1·2·3번째 분야는 유지하고 [케이블 우선 개선 순서](ERROR_REDUCTION_PLAN.md)를 준비했다. 신규 학습·실모델 추론·TEST 열람은 0이며 실제 기기와 케이블 사진 리허설은 미검증이다.

로컬 근거: `outputs/safelog-rn-cable-demo-tests-20261119.log`, `safelog-rn-cable-demo-lint-20261119.log`, `safelog-rn-cable-demo-typecheck-20261119.log`, `safelog-rn-cable-demo-web-export-20261120.log`. 앞선 날짜의 기록은 변경하지 않았다.

## 2026-11-23 · 최신 로그인·SQLite와 실제 AI 통합

기준은 `feature/mobile-storage-auth`의 `418024ee98b6f0ffe947098533506f777bff686c`이며 별도 로컬 `codex/mobile-db-ai`에서 통합했다. 위 날짜별 기록은 당시 검증 범위로 보존한다.

| 검사 | 결과와 범위 |
|---|---|
| 잠금 파일 기준 설치 | `npm ci --ignore-scripts --no-audit --no-fund` 성공. 기존 잠금 파일과 의존성 버전 유지. |
| 전체 자동 검사 | **103개 통과, 실패·건너뜀 0개**. HTTP 계약, 워크플로, 계정, SQLite 업그레이드·재시작·롤백 포함. |
| TypeScript / Expo lint | 둘 다 통과. lint 오류·경고 0개. Expo 타입 선언을 포함해 별도 타입 우회 옵션 없이 검사. |
| Expo export | Android·iOS Hermes 및 웹 번들 생성 통과. 네이티브 앱 설치·기기 실행 성공을 뜻하지 않는다. |
| 기존 DB 보존 | v1·v2 문자열을 유지하고 v3 추가. 자격 증명, 기존 사진·로그 ID, AI JSON·검토 결정 보존과 중단 시 롤백 통과. |
| 실제 CPU → SQLite | 아래 9장 원응답·SHA·점수 저장, 디스크 사진 사본과 DB 재개방 후 값 검증 통과. |
| 배정 → 조치 → 최종 확인 | 새 SQLite 담당자 계정으로 배정 후 재개방, 조치 제출, 실제 전후 비교, 사람 Verified 확인, 보고서 이름·사진 복원 통과. |

실제 모델 호출은 시설·컨베이어·RobustAD PCB의 기존 시연 6장과 볼스크루 FIT 2장, 케이블 정상 참조 FIT 1장이다. 고정 2개 모델의 SHA를 확인하고 현재 모바일 HTTP 분석기·gateway·WorkflowService·Node SQLite 드라이버 및 실제 디스크 복사를 연결했다. PCB의 없는 confidence는 SQL NULL로 보관하며, 원응답 문자열을 표시 값으로 재구성하지 않는다. 이 검증은 새로운 정확도 벤치마크나 모든 20항목의 실사진 통과를 뜻하지 않는다. 학습·GPU·고정 TEST 열람·모델 기준 변경은 0이다.

최초 103개 검사 중 기존 임의 사진 ID의 재저장에서 충돌 1건을 발견했다. 사진·로그의 자연키로 갱신하도록 수정한 뒤 관련 13개와 전체 103개 검사가 통과했다. 최초 실패 로그는 그대로 보존했다. 실제 모델 리허설의 최초 실행은 복원 객체의 속성 순서를 JSON 문자열로 비교한 검사 코드 때문에 중단됐다. 최초 DB·원응답·실패 기록은 보존하고, 9장 추론을 반복하지 않은 DB 사본에서 값·원응답·사진을 검증한 뒤 아직 실행하지 않은 전후 비교를 이어갔다.

근거 폴더 `outputs/safelog-mobile-db-ai-2026-11-23/`: `tests-initial.log`, `db-tests-after-legacy-key-fix.log`, `tests-final.log`, `typecheck-final.log`, `lint-final.log`, `export-all.log`, `live/result.json`(최초 원형), `live-continuation/result.json`(후속 통과), `runtime-model-pins-budget.json`, `runtime-host-final.json`. 이 작업에서 시작한 모델 서버와 gateway는 종료했다.

**남은 실제 기기 확인:** Android/iOS에서 사진 저장 → 앱 종료·재실행 → 로그인 → AI 원응답·검토 결정·사진 복원 → PDF 공유. 이번 Node 검사는 Expo 네이티브 SQLite·파일 드라이버나 실제 기기 화면을 검증하지 않는다. 웹은 계속 메모리 저장이며 새로고침하면 초기화된다. 실행·인계 범위는 [DB·AI 통합 안내](DB_AI_INTEGRATION.md)를 따른다.
