# SafeLog Mobile

현장 발견사항을 사진으로 등록하고, 담당자 조치와 점검자 최종 확인을 기록하는 Expo SDK 58 앱입니다. Android와 iOS를 대상으로 하며, 웹에서는 보고서를 HTML로 내려받을 수 있습니다.

## Windows에서 실행

Android Studio에서 에뮬레이터를 실행한 뒤 PowerShell에서 다음 명령을 실행합니다.

```powershell
npm.cmd ci
$env:ANDROID_HOME = 'C:\Android\Sdk'
npx.cmd expo start --android --go
```

## 사진 자동 판정 연결

Windows에서 Ollama가 실행 중이면 다음 명령으로 개발용 분류 서버를 띄울 수 있습니다. 처음 실행할 때 모델을 내려받습니다.

```powershell
ollama pull qwen3-vl:4b
node scripts/photo-classifier.mjs
```

다른 터미널에서 Android 에뮬레이터용 주소를 지정해 Expo를 시작합니다.

```powershell
$env:EXPO_PUBLIC_SAFELOG_CLASSIFIER_URL = 'http://10.0.2.2:8780/classify'
npx.cmd expo start --android --go
```

USB로 연결한 Android 기기에서는 `adb reverse tcp:8780 tcp:8780`을 실행하고 주소를 `http://127.0.0.1:8780/classify`로 바꿉니다. 이 서버는 범용 비전 모델로 사진의 주요 대상을 분류합니다. 출력 점수는 모델 자체 추정값이며 현장 정확도로 검증되지 않았습니다. 불분명한 사진은 미판정으로 처리합니다.

`.env.local`의 `EXPO_PUBLIC_SAFELOG_CLASSIFIER_URL`에 기기에서 접근할 수 있는 사진 분류 API 주소를 넣고 Expo 개발 서버를 다시 시작합니다. 설정값은 앱 번들에 포함되므로 주소에 비밀 키를 넣지 마세요.

앱은 사진 등록 후 AI 검토 화면에서 API에 `POST`로 JSON `{ imageBase64, mimeType }`을 보냅니다. 서버는 사진을 보고 `{ category, model, confidence, assessment, action }`을 반환해야 합니다. `category`는 `ball_screw_pitting`, `facility_crack`, `conveyor_foreign`, `robustad_pcb`, `cable_exterior` 중 하나이고, `confidence`는 0~1입니다. 판정할 수 없으면 서버는 카테고리를 비워 반환할 수 있습니다. 유형 지정이 필요한 기존 통합 추론 서버를 사용하려면 앞단에서 사진 유형을 판별하고 해당 모델에 라우팅하는 API가 필요합니다.

판정된 분야는 예비보고서 상단의 점검 대상 및 점검표 해당 여부에 반영됩니다. 판정 실패·거절 시 점검 대상은 미판정입니다. 서버 주소가 비어 있으면 시연용 Mock을 사용할 수 있습니다.

## 시연

| 역할 | 사용자 ID |
|---|---|
| Inspector · 김안전 | `inspector-1` |
| Manager · 박관리 | `manager-1` |
| Assignee · 이조치 | `assignee-1` |

공통 비밀번호는 `demo1234`입니다. Inspector의 **안전점검 등록**에서 시연 현장의 사진과 발견사항을 입력하고 **AI 위험 분석**에서 위험 제안을 검토합니다. 오른쪽 위 **내 계정**에서 로그아웃하고, Manager로 로그인해 **개선 조치 관리**에서 담당자를 배정합니다. Assignee의 **배정된 작업**에서 조치를 시작하고 **진행 중 작업**에서 사진과 내용을 제출합니다. 원 Inspector가 전후 사진을 확인한 뒤 **점검 보고서**에서 미리보기와 공유를 열 수 있습니다. 기록은 앱의 메모리에 저장됩니다. 사진 분류 서버를 설정하지 않은 경우 사진 분석과 전후 비교에 시연용 Mock을 사용합니다.

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
