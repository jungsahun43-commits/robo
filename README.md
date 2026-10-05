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

## AI 실험 최신 결과 · 2026-10-05

같은 초기 모델·자료·표본 순서로 입력 640과 960을 각각 6epoch 추가 학습했습니다.
공개 자료 VAL에서 균열·박락의 출처별 미탐률·오탐률 12개 중 최대값은
기존 모델 22.28%, 640 대조군 22.53%, 960 보강군 22.28%입니다.
작은 결함의 누락도 960에서 기존 모델 대비 줄지 않아 연구 후보 기준과 5% 미만 목표에 미달했습니다.
공장 현장 정확도는 아직 측정하지 않았으며 앱 기본 `facility-validation-v2`는 유지합니다.

- [비교 결과·그림·실행 기록](safelog-cpp/ai-training/reports/FACILITY_RESOLUTION_STUDY_RESULTS_KO.md)
- [고정한 실험 조건·재현 순서](safelog-cpp/ai-training/reports/FACILITY_RESOLUTION_STUDY_PLAN_KO.md)
- [완료 가중치 재로딩·38개 코드 테스트 검증](safelog-cpp/ai-training/reports/facility-resolution-study-verification.json)
- [전체 실험 진행·결과](safelog-cpp/ai-training/reports/FACILITY_FIVE_PERCENT_RESULTS_KO.md)

학습 자료·가중치는 로컬에 보관하며 GitHub에는 코드와 집계 결과를 올립니다.
