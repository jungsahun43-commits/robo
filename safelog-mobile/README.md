# SafeLog Mobile

현장 발견사항을 사진으로 등록하고, 담당자 조치와 점검자 최종 확인을 기록하는 Expo SDK 58 앱입니다. Android와 iOS를 대상으로 하며, 웹에서는 보고서를 HTML로 내려받을 수 있습니다.

## Windows에서 실행

Android Studio에서 에뮬레이터를 실행한 뒤 PowerShell에서 다음 명령을 실행합니다.

```powershell
npm.cmd ci
$env:ANDROID_HOME = 'C:\Android\Sdk'
npx.cmd expo start --android --go
```

## 시연

| 역할 | 사용자 ID |
|---|---|
| Inspector · 김안전 | `inspector-1` |
| Manager · 박관리 | `manager-1` |
| Assignee · 이조치 | `assignee-1` |

공통 비밀번호는 `demo1234`입니다. Inspector의 **안전점검 등록**에서 시연 현장의 사진과 발견사항을 입력하고 **AI 위험 분석**에서 위험 제안을 검토합니다. 오른쪽 위 **내 계정**에서 로그아웃하고, Manager로 로그인해 **개선 조치 관리**에서 담당자를 배정합니다. Assignee의 **배정된 작업**에서 조치를 시작하고 **진행 중 작업**에서 사진과 내용을 제출합니다. 원 Inspector가 전후 사진을 확인한 뒤 **점검 보고서**에서 미리보기와 공유를 열 수 있습니다. 기록은 앱의 메모리에 저장되며 AI 응답은 시연용입니다.

```powershell
npx.cmd expo lint
npx.cmd tsc --noEmit
npx.cmd --yes tsx --test tests/workflow.test.mjs
```

- [원본 브랜치 분석](docs/BRANCH_ANALYSIS.md)
- [기능 대응표](docs/FEATURE_MAPPING.md)
- [아키텍처](docs/ARCHITECTURE.md)
- [이전 계획](docs/MIGRATION_PLAN.md)
- [테스트 결과](docs/TEST_REPORT.md)
