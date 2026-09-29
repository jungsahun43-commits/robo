# SafeLog AI

현장 사진을 AI가 분석해 위험 요소와 개선안을 제안하고, 안전관리자의 검토·조치·최종 확인 이력을 보고서로 남기는 C++20·Qt 프로젝트입니다.

AI 결과는 참고 제안이며 최종 판단은 점검자가 수행합니다. 모델 서버가 없어도 Mock AI로 전체 시연 흐름을 실행할 수 있습니다.

## 시작 위치

- 프로젝트 코드: [`safelog-cpp/`](safelog-cpp/)
- 빠른 시작: [`safelog-cpp/QUICKSTART_KO.md`](safelog-cpp/QUICKSTART_KO.md)
- 4인 역할별 프롬프트: [`safelog-cpp/docs/AI_FIRST_TEAM_PROMPTS_KO.md`](safelog-cpp/docs/AI_FIRST_TEAM_PROMPTS_KO.md)
- AI 설계: [`safelog-cpp/docs/AI_ARCHITECTURE_KO.md`](safelog-cpp/docs/AI_ARCHITECTURE_KO.md)

## 역할별 브랜치

| 역할 | 브랜치 |
|---|---|
| 사진 촬영·AI 검토 UI | `feature/capture-ai-ui` |
| SQLite·AI 이력 저장 | `feature/storage-ai` |
| 실제 AI 엔진·평가 | `feature/ai-engine` |
| 조치 흐름·보고서 | `feature/workflow-report` |

상세 빌드 방법과 협업 규칙은 빠른 시작 문서를 확인하세요.
