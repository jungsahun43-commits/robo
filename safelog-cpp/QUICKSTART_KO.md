# SafeLog AI 팀 개발 시작 방법

상세 작업 지시는 `docs/AI_FIRST_TEAM_PROMPTS_KO.md`를 사용한다.

## 1. 저장소 받기

```bash
git clone https://github.com/jungsahun43-commits/robo.git
cd robo/safelog-cpp
git fetch origin
```

각 팀원은 담당 브랜치 하나만 선택한다.

| 역할 | 브랜치 |
|---|---|
| 사진 촬영·AI 검토 UI | `feature/capture-ai-ui` |
| SQLite·AI 이력 저장 | `feature/storage-ai` |
| 실제 AI 엔진·평가 | `feature/ai-engine` |
| 조치 흐름·보고서 | `feature/workflow-report` |

예시:

```bash
git switch --track origin/feature/ai-engine
```

## 2. 작업 전 읽을 문서

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/AI_ARCHITECTURE_KO.md`
- `docs/AI_FIRST_TEAM_PROMPTS_KO.md`
- `docs/INTEGRATION_CHECKLIST.md`

공통 계약 파일은 팀 합의 없이 변경하지 않는다.

- `include/safelog/contracts/types.hpp`
- `include/safelog/contracts/ports.hpp`
- `include/safelog/contracts/errors.hpp`

## 3. 개발 환경

- Git
- CMake 3.24 이상
- C++20 컴파일러
- 화면 담당 및 최종 통합 PC: Qt 6.5 이상, Qt Quick, Qt Quick Controls 2
- Android 빌드 PC: Qt Android 구성 요소, Android SDK, NDK, JDK

## 4. 빌드와 테스트

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Qt 화면:

```bash
cmake -S . -B build-qt -DSAFELOG_BUILD_QT=ON -DCMAKE_PREFIX_PATH="Qt 설치 경로"
cmake --build build-qt
```

## 5. 매일 작업 방법

```bash
git branch --show-current
git status
git add 담당폴더
git commit -m "feat: 구현 내용"
git push origin HEAD
```

완료 후 자신의 브랜치에서 `main`으로 Pull Request를 만든다. 담당 범위 밖 파일을 수정했다면 PR 설명에 이유와 영향을 적는다.

## 6. 권장 병합 순서

1. `feature/storage-ai`
2. `feature/ai-engine`
3. `feature/capture-ai-ui`
4. `feature/workflow-report`
5. Qt/Android 최종 연결과 전체 시나리오 확인

각 병합 직후 빌드와 스모크 테스트를 다시 실행한다.

## 7. 1차 완성 기준

사진과 메모 등록 → AI 위험 제안 → 사람 검토 → 담당자 지정 → 조치 후 사진 제출 → AI 전후 비교 → 점검자 최종 확인 → 보고서 생성 흐름이 한 대의 Android 휴대폰에서 동작해야 한다.

AI 서버가 실패해도 수동 입력과 최종 확인 흐름은 계속 사용할 수 있어야 한다.
