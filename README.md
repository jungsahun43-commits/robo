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

기존 모델에 고정된 ImageNet ConvNeXt-Tiny 특징을 추가하고, 원본0773에서 신규 후보6epoch를 실제 학습·검증했습니다.
원래 모델324개 state와7종 사진·19종 보조·80×80 지도 계약을 유지하고, 새 연결층3개를0으로 초기화한 뒤 학습했습니다.
ConvNeXt 특징 추출기는 동결했습니다. 원래 모델과 새 연결층은 학습하므로 ConvNeXt 전체를 재학습한 결과는 아닙니다.
완료된 저학습률 대조군6epoch를 재사용해 이번 신규 학습은6epoch·대조군 재학습은0epoch입니다.
원본 / 기존 저학습률 대조 / 새 특징 보강 순서의 최대 source-VAL 미탐·오탐은 **22.28% / 22.02% / 22.22%**입니다.
DACL 철근 AP는 **0.7010 / 0.7129 / 0.7099**, 작은 손상 항목·사진 사례 FN 합계는 **71 / 70 / 71건**입니다.
알려진 다른5항목의 원본 대비 AP 하락 한도 검사는 **통과**, 철근 회복 등을 합한 기존 유지 후보 기준은 **미달**, 연구 후보 기준은 **미달**, 엄격한5% 목표는 **미달**입니다.
AP는 오류율과 다른 순위 지표입니다. 최대 오류는 두 항목·세 자료·FNR/FPR12개 비율의 최대이며 전체 사진 오답률이 아닙니다.
추가 데이터 후보의 정답 범위와 접근 조건을 검토했지만 이번 후보에 새로운 시설 사진·정답을 추가하지 않았습니다.
총 파라미터31,085,624개 중 고정 encoder27,820,128개, 새 학습 연결층21,345개입니다. 표본·epoch는 같지만 연산량·모델 크기는 같지 않습니다.
TRAIN 단일 사진의 GPU float32 forward 예비 측정 평균은 원래 모델3.17ms / 새 모델11.90ms입니다. 입력 전송·전처리·Android 비용은 제외한 제한된 측정입니다.
소스112개·실제 집중 테스트25개·원본 보호 입력59,026개·표본 순서·교사·BN·고정 encoder·새 연결층·offline CPU 재로드 검증이 통과했습니다.
누적 완료 기록은 모델 실행230epoch와 이전 중단 기록2epoch를 합한232epoch입니다.
반복 공개 VAL과 한 seed의 연구 결과이며 독립 공장 현장 성능으로 해석하지 않습니다. 기본 앱 모델은 교체하지 않았습니다.

- [새 특징 비교 결과·그림](safelog-cpp/ai-training/reports/FACILITY_SEMANTIC_STUDY_RESULTS_KO.md)
- [데이터 후보·모델 변경 계획](safelog-cpp/ai-training/reports/FACILITY_DATA_AND_FEATURES_PLAN_KO.md)
- [새 모델·encoder·원본 보존 검증](safelog-cpp/ai-training/reports/facility-semantic-study-verification.json)

## 앞선 head 학습률 비교 · 2026-10-07

head 초기 학습률을0.00025에서0.0001로 낮춘 후보를 원본0773부터 새로6epoch 학습하고 전체 공개 VAL을 검증했습니다.
완료된 BN 고정 대조군6epoch를 재사용했으며, 이번 후속의 신규 학습은6epoch·대조군 재학습은0epoch입니다.
두 군 모두 원본 BN 통계·가중치4/T2 보존 손실·640 입력·원래 정답·추출 순서·backbone 학습률을 유지했습니다.
원본 / 기존 head LR0.00025 / 새 head LR0.0001 순서의 최대 source-VAL 미탐·오탐은 **22.28% / 22.41% / 22.02%**입니다.
DACL 노출 철근 AP는 **0.7010 / 0.7075 / 0.7129**, 작은 손상 항목·사진 사례 FN 합계는 **71 / 70 / 70건**입니다.
AP는 오류율과 다른 순위 지표입니다. 성능 보존 기준은 **미달**, 연구 후보 기준은 **미달**, 엄격한5% 목표는 **미달**입니다.
알려진 다른5항목의 모든 출처별 AP 하락은 원본 대비0.02 한도 안에 있습니다.
기존 철근 회복 조건도 그대로 적용했으며 결과에 맞춰 기준을 완화하지 않았습니다. 기본 앱 모델은 교체하지 않았습니다.
실제6개 epoch 시작 optimizer LR과 최종 cosine 최저 LR0.000005를 검증했습니다. 대조군 곡선은 기존 설정·소스에서 계산한 값이며 실제 관측 기록으로 표시하지 않습니다.
BN47개 층·버퍼141개와 교사 모델은 보존했고, affine94개 및 backbone·head 가중치는 계속 학습했습니다.
소스97개·집중 테스트24개·보호 입력58,995개·추출 순서·실제 업데이트·CPU 재로드 검증이 통과했습니다.
누적 완료 기록은 모델 실행224epoch와 이전 중단 기록2epoch의 합226epoch입니다.
이전 결과를 보고 선택한 반복 공개 VAL 탐색이며, 독립 공장 현장 성능이나 전체 사진 오답률을 나타내지 않습니다.

- [head 학습률 비교 결과·그림](safelog-cpp/ai-training/reports/FACILITY_HEAD_LR_STUDY_RESULTS_KO.md)
- [학습 전 고정 조건](safelog-cpp/ai-training/reports/FACILITY_HEAD_LR_STUDY_PLAN_KO.md)
- [실제 학습률·완료 모델·입력 보존 검증](safelog-cpp/ai-training/reports/facility-head-lr-study-verification.json)

## 앞선 BN 통계 비교 · 2026-10-07

보존 손실 가중치0·1·4 비교에 이어 정규화 통계 고정 후보6epoch를 추가 학습하고 전체 공개 VAL을 검증했습니다.
완료된 일반 BN 가중치4 모델을 재사용해 대조군 재학습은0epoch입니다. 이번 작업의 신규 학습은 총24epoch입니다.
원본 / 일반 BN 가중치4 / 고정 BN 가중치4 순서의 최대 source-VAL 미탐·오탐은 22.28% / 22.28% / 22.41%입니다.
DACL 철근 노출 AP는 0.7010 / 0.6758 / 0.7075, 작은 손상 항목·사진 사례 FN 합계는 71 / 72 / 70건입니다.
AP는 오류율과 다른 순위 지표입니다. 고정 BN의 성능 보존 기준은 **미달**, 연구 후보 기준은 **미달**입니다.
철근 노출 AP는 회복했지만 DACL 공동·패임 AP 하락0.020277이 미리 정한 한도0.02를 넘었습니다. 작은 손상 FN 합계70건은 재사용 대조72건보다2건 줄었으나 원본71건보다1건 줄어 기존 연구 기준에 미달했습니다.
12개 균열·박락 FNR/FPR 각각5% 미만 목표는 미달이며 기본 앱 모델을 교체하지 않았습니다.
BN47개 층·버퍼141개는 원본 해시를 유지했고, affine94개와 backbone·head 가중치는 계속 학습했습니다.
후속 소스86개·실제 테스트32개·보호 파일58,965개·실제 추출과 업데이트를 검증했습니다.
앞선 비교 테스트34개·보고서 보완2개·강도 비교28개도 별도로 통과했습니다.
누적 완료 기록은 모델 실행218epoch와 이전 중단 기록2epoch의 합220epoch입니다.
반복 공개 VAL 결과에 따른 후속 실험이며 공장 현장 성능이나 정규화 통계의 인과 효과를 입증하지 않습니다.

- [정규화 통계 비교 결과·그림](safelog-cpp/ai-training/reports/FACILITY_BATCHNORM_STUDY_RESULTS_KO.md)
- [고정 조건·재현 절차](safelog-cpp/ai-training/reports/FACILITY_BATCHNORM_STUDY_PLAN_KO.md)
- [버퍼 불변·계수 학습·실제6epoch 검증](safelog-cpp/ai-training/reports/facility-batchnorm-study-verification.json)

## 앞선 가중치4 후속 비교 · 2026-10-07

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
