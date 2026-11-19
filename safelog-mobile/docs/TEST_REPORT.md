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
