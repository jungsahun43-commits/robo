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

## AI 실험 최신 결과 · 2026-10-07

보존 손실 가중치0·1을 각6epoch 비교한 뒤, 가중치4 후보만6epoch 추가 학습하고 검증했습니다.
완료된 대조군을 재사용해 이번 후속 대조군 재학습은0epoch입니다. 두 비교에서 신규 학습은 총18epoch입니다.
원본 / 대조0 / 보존1 / 보존4 순서로 최대 source-VAL 미탐·오탐은 22.28% / 23.06% / 22.53% / 22.28%입니다.
DACL 철근 노출 AP는 0.7010 / 0.6486 / 0.6715 / 0.6758, 작은 손상 항목·사진 사례의 FN 합계는 71 / 67 / 67 / 72건입니다.
AP는 순위 판별 지표이며 오류율이 아닙니다. 가중치4의 기존 항목 보존 기준은 **미달**, 연구 후보 기준은 **미달**입니다.
12개 균열·박락 FNR/FPR 각각5% 미만 목표는 미달이며 기본 앱 모델을 교체하지 않았습니다.
후속 소스74개·실제 테스트28개·보호 파일58,934개·교사 고정·실제 추출과 업데이트를 검증했습니다.
앞선 비교 테스트34개와 보고서 보완 테스트2개도 별도로 통과했습니다.
누적 완료 기록은 모델 실행212epoch와 이전 중단 기록2epoch를 합한214epoch입니다.
이전 공개 VAL 결과를 보고 선택한 후속 조건이므로 독립 공장 현장 성능으로 해석하지 않습니다.

- [보존 강도 후속 결과·네 모델 비교](safelog-cpp/ai-training/reports/FACILITY_RETENTION_STRENGTH_STUDY_RESULTS_KO.md)
- [가중치4 고정 조건·재현 순서](safelog-cpp/ai-training/reports/FACILITY_RETENTION_STRENGTH_STUDY_PLAN_KO.md)
- [재사용 대조군·신규6epoch·원본 보존 검증](safelog-cpp/ai-training/reports/facility-retention-strength-study-verification.json)

## 앞선 가중치0·1 성능 보존 비교 · 2026-10-07

기존 다른5개 손상 항목의 성능을 보존하도록 고정 원본 교사의 확률을 참고하는 학습을
증류 없는 대조군과 각6epoch·총12epoch 실제 비교했습니다. 사진 순서·정답·기존 손실·
교사 실행 횟수는 동일하고 추가 보존 손실의 가중치0·1만 달랐습니다.
최대 source-VAL 미탐·오탐은 원본22.28% / 대조23.06% / 보존22.53%입니다.
DACL 철근 노출 AP는 0.7010 /0.6486 /0.6715로 부분 회복됐습니다. AP는 순위 판별 지표이며 오류율이 아닙니다.
철근 노출 및 CODEBRIM 백화 AP 하락 폭이 원본 대비0.02를 넘어 성능 보존 기준은 미달했습니다.
작은 손상 FN 합계는71 /67 /67건이고 기존 연구 후보 기준·5% 목표도 미달입니다.
소스61개·실제 테스트34개·원본/보호 파일58,890개·교사 고정·실제 업데이트를 검증했습니다.
원본 학습 기록의 생략된 계획 해시는 검증된 프로토콜에서 메모리 복사본에 연결했으며 보고서 보완 테스트2개가 통과했습니다.
누적 완료 기록은 모델 실행206epoch와 이전 중단 기록2epoch를 합한208epoch입니다.
공장 현장 정확도는 측정하지 않았으며 앱 기본 `facility-validation-v2`를 유지합니다.

- [성능 보존 비교 결과·그림·실행 비용](safelog-cpp/ai-training/reports/FACILITY_RETENTION_STUDY_RESULTS_KO.md)
- [고정 조건·재현 순서](safelog-cpp/ai-training/reports/FACILITY_RETENTION_STUDY_PLAN_KO.md)
- [완료 학습·교사 고정·입력 보존 검증](safelog-cpp/ai-training/reports/facility-retention-study-verification.json)
- [전체 실험 진행·결과](safelog-cpp/ai-training/reports/FACILITY_FIVE_PERCENT_RESULTS_KO.md)

## 앞선 박락 음성 표본 비교 · 2026-10-06

원래 박락 음성이며 관련 표면 손상 태그를 가진 DACL TRAIN 사진666장의 추출 빈도를 높이는
조건을 대조군과 각6epoch·총12epoch 실제 학습해 비교했습니다. 두 군은 모든 추출 위치에서
출처·full/crop·균열/박락 정답을 보존했고, 원본 사진·마스크·손실 가중치를 유지했습니다.
공개 VAL의 균열·박락×3출처×미탐률/오탐률12개 중 최대값은 초기22.28%, 대조22.84%, 보강23.32%입니다.
작은 손상 항목·사진 사례의 FN 합계는 각각71·67·68건입니다. 보강군은 연구 후보 기준과5% 목표에 미달했습니다.
다른 알려진 항목 AP도 초기 모델 대비 악화가 있어 새 모델을 승격하지 않았습니다.
코드49개·실제 테스트28개·원본 파일58,874개·실제 추출과 업데이트를 검증했습니다.
누적 완료 기록은 모델 실행194epoch와 이전 중단 기록2epoch를 합한196epoch입니다.
공장 현장 정확도는 측정하지 않았으며 앱 기본 `facility-validation-v2`는 유지합니다.

- [박락 음성 태그 비교 결과·그림·실행 기록](safelog-cpp/ai-training/reports/FACILITY_SUBTYPE_STUDY_RESULTS_KO.md)
- [고정한 실험 조건·재현 순서](safelog-cpp/ai-training/reports/FACILITY_SUBTYPE_STUDY_PLAN_KO.md)
- [완료 가중치·28개 코드 테스트·자료 무결성 검증](safelog-cpp/ai-training/reports/facility-subtype-study-verification.json)
- [직전 원본 ROI 비교](safelog-cpp/ai-training/reports/FACILITY_NATIVE_ROI_STUDY_RESULTS_KO.md)
- [직전 640·960 해상도 비교](safelog-cpp/ai-training/reports/FACILITY_RESOLUTION_STUDY_RESULTS_KO.md)
- [전체 실험 진행·결과](safelog-cpp/ai-training/reports/FACILITY_FIVE_PERCENT_RESULTS_KO.md)

학습 자료·가중치는 로컬에 보관하며 GitHub에는 코드와 집계 결과를 올립니다.

## 학습 전 준비: 박락 TRAIN 감사 · 2026-10-06

기존 검수 사진 130장의 박락 오답 54장과 마스크를 대조했습니다.
전체 DACL TRAIN 6,225장의 저장 점수에서는 박락 음성·관련 손상 태그 집단의
오탐 비율이 26.43%, 해당 태그가 없는 음성 집단은 8.58%였습니다. 이는 학습 사진의 기술 집계입니다.
그 집단 666장을 제한적으로 강조하는 다음 표본 추출 방식을 준비하고 모의 실행했습니다.
이 준비 단계에서는 새 학습·추론·라벨 수정 없이 조건만 확인했습니다. 이후 실제 학습 결과는 위 최신 비교에 기록했습니다.

- [실측 감사·다음 실험 준비·화면 사용 안내](safelog-cpp/ai-training/reports/FACILITY_SPALLING_TRAIN_AUDIT_KO.md)
- [기존 표본 순서 재현·구성 보존 모의 실행](safelog-cpp/ai-training/reports/facility-spalling-sampler-dry-run.json)
