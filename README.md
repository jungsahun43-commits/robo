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

## 균열·박락 손실 변경 재학습 · 2026-10-10

표본을 조금 더 자주 보여 주던 이전 방식에서 다음 단계로 진행했습니다. **원본 사진 추출 순서와 정답은 유지**하고 균열·박락의 사진 손실을 비대칭 방식으로 바꿔 **실제6epoch·22.26분** GPU 학습했습니다.
양성gamma0·음성gamma4·확률 이동0.05를 사전 고정했습니다. 쉬운 음성 기여를 줄이고, 미확인 정답은 손실·gradient에서 제외합니다. 다른5개 항목·위치·19종 보조·교사 증류 계산은 유지합니다.

앞서 판단 보류한 TRAIN 오류 후보12장을 원래 출처 사진으로 다시 확인했습니다. 3장은 두 항목 판단이 명확해졌고9장은 한 항목 이상 판단 불가가 남았습니다. AI 진단은 사람·전문가의 정답이 아니며 학습 정답이나 표본 선별에 사용하지 않았습니다.

| 지표 | 기존 저학습률 대조군 | 비대칭 사진 손실 후보 |
|---|---:|---:|
| 검증 항목 중 최대 오탐·미탐률 | 22.0207% | 22.4138% |
| 작은 DACL 손상 미탐 사례 합계 | 70건 | 71건 |

연구 후보 기준 **미달**, 다른 항목 보존 기준 **미달**, 모든 개별 비율5% 미만 목표 **미달**입니다.
같은 공개 VAL1,745장을 반복 사용한 탐색 결과입니다. 독립 공장 현장 정확도는 측정하지 않았습니다.

집중 테스트13개, 소스143개, 입력72,634개 보존과 실제 모델 strict CPU 재로드가 통과했습니다.
실제 optimizer 업데이트10,679회. 일회성 사전 업데이트1회는 epoch로 세지 않았습니다.
누적 완료 기록은 **모델 학습254epoch + 이전 중단2epoch = 256epoch**이며 재사용 대조군을 다시 세지 않았습니다.
앱 기본 모델을 유지하고 sourceTEST 추론은 수행하지 않았습니다.

- [변경 방법·실제 비교 결과](safelog-cpp/ai-training/reports/FACILITY_PRIMARY_ASYMMETRIC_STUDY_RESULTS_KO.md)
- [학습 경과·선택 모델·작은 손상 비교](safelog-cpp/ai-training/reports/facility-primary-asymmetric-study-comparison.png)
- [실제 추출·업데이트·보존 검증](safelog-cpp/ai-training/reports/facility-primary-asymmetric-study-verification.json)

## AI 보조 사진 검토·재학습 결과 · 2026-10-10

산업현장 앱이 사진으로 다룰 수 있는 **콘크리트 균열·박락** 기준으로 TRAIN 사진200장을 AI가 검토하고 **6epoch·23.97분** 실제 GPU 재학습했습니다.
두 항목 모두 명확하고 원본 주석과 일치하는31장(DACL13·Dam11·CODEBRIM7)에 epoch당2회씩 추가 노출했습니다.
불일치24장과 한 항목 이상 판단 불가145장은 추가 노출에서 제외했습니다. 기존 사진·위치·19종 정답을 유지하며 AI 판단을 사람 검수나 새 정답으로 취급하지 않습니다.

| 지표 | 기존 저학습률 대조군 | AI 검토 일치 사진 추가 노출 |
|---|---:|---:|
| 검증 항목 중 최대 오탐·미탐률 | 22.0207% | 22.0207% |
| 작은 DACL 손상 미탐 사례 합계 | 70건 | 69건 |

연구 후보 기준 **미달**, 기존 항목 보존 기준 **미달**, 모든 개별 비율5% 미만 목표 **미달**입니다.
같은 공개 VAL1,745장을 반복 사용하는 탐색 결과이며 독립 공장 현장 정확도는 측정하지 않았습니다.
원래 데이터는 공개 교량·댐·콘크리트 표면 자료이며 실제 산업체 출처로 설명하지 않습니다.

집중 테스트12개, 동결 소스137개, 입력 파일72,600개 보존 검사와 실제 모델 strict CPU 재로드를 통과했습니다.
실제 optimizer 업데이트10,679회이며 일회성 사전 업데이트1회는 epoch에 포함하지 않습니다.
누적 완료 기록은 **모델 학습248epoch + 이전 중단2epoch = 250epoch**입니다. 과거 대조군6epoch를 다시 세지 않았습니다.
앱 기본 모델과 sourceTEST는 유지합니다. 실제 사람 검수 경로는 아래에서 별도로 제공합니다.

- [검토 기준·실제 학습 비교 결과](safelog-cpp/ai-training/reports/FACILITY_AI_AGREEMENT_STUDY_RESULTS_KO.md)
- [검토 선별·검증 비교 그림](safelog-cpp/ai-training/reports/facility-ai-agreement-study-comparison.png)
- [실제 업데이트·추출 순서·자료 보존 증거](safelog-cpp/ai-training/reports/facility-ai-agreement-study-verification.json)

## 사람 검수로 다음 학습 준비 · 2026-10-09

[이 PC의 검수 화면](http://127.0.0.1:8770/)에서 균열·박락을 각각 있음/없음/판단 불가로 선택합니다. 오류 의심120장과 비교80장, 총200장을 준비했고 자동 저장·복원·제출이 가능합니다.

이 사람 검수 경로는 실제 입력을 기다리는 준비 단계이며 사람 정답을 사용한 새 학습은0epoch입니다. 위 AI 보조 선별 재학습6epoch와 구분합니다. 집중 테스트9개와 별도 UI 테스트가 통과했고 테스트 입력은 학습에서 제외됩니다. 실제 제출된 판단만 별도 정답 자료로 만들며 원본·검증·시험 정답을 유지합니다.

- [검수 방법·학습 연결·서버 다시 시작](safelog-cpp/ai-training/reports/FACILITY_HUMAN_REVIEW_START_KO.md)
- [검수 준비·테스트 확인 기록](safelog-cpp/ai-training/reports/facility-human-review-ready.json)

## 앞선 원본19종 위치 보조 학습 결과 · 2026-10-09

기존 DACL 학습 사진 **6,225장**의 원본19종 주석으로 손상 위치 보조 학습을 추가하고, **6epoch·약18.22분** 실제 GPU 학습했습니다.
이전에는19종을 사진 전체 태그로만 학습했고, 이번에는 각 손상의 위치도 알려 줍니다. 새 독립 사진은0장입니다.
원래7종 사진·위치 정답, 사진 순서와 증강·학습률·교사 증류를 유지했습니다. 좌표가 부적합한785개 항목·사진 위치 채널은 미확인으로 두고 원래 사진 태그를 보존했습니다.

| 지표 | 기존 저학습률 대조군 | 새19종 위치 보조 모델 |
|---|---:|---:|
| 균열·박락 최대 미탐률/오탐률 | 22.0207% | 22.2222% |
| 작은 DACL 손상 미탐 사례 합계 | 70건 | 70건 |
| DACL 노출 철근 AP | 0.7129188 | 0.7160148 |
| DACL 공동 AP | 0.6030326 | 0.6024160 |

최대 오류 변화는 **0.2015퍼센트포인트 악화**입니다. 연구 후보 기준은 **미달**, 기존 항목 유지 후보 기준은 **미달**, 각 항목5% 미만의 검증 목표는 **미달**입니다.
최대값은 균열·박락×3자료×미탐·오탐의12개 비율 중 최대입니다. 앱 전체 오류율과 다르며, AP는 별도의 순위 지표입니다.
세 검증 자료(DACL710·Dam424·CODEBRIM611)에서 같은 조건으로 평가했습니다. 반복 공개 VAL과 한 seed의 탐색 결과이며 새로운 산업체 현장 정확도를 뜻하지 않습니다.

집중 테스트25개, 동결 소스132개, 원본과 신규 보호 파일72,549개 보존 검사가 통과했습니다.
실제 optimizer 업데이트10,679회, AMP 건너뜀7회입니다. 사전 GPU 점검의 임시 업데이트1회는 학습 epoch로 세지 않았습니다.
학습용328개 state 체크포인트와 별도324개 state 추론 파일을 저장했고, 재로드 후 기존 공개7종 출력과 앞선 세 출력이 정확히 일치했습니다.
누적 기록은 **완료 모델 학습242epoch + 이전 중단 기록2epoch = 244epoch**입니다. 재사용 대조군6epoch는 다시 합산하지 않았습니다.
source-TEST 추론과 앱 기본 모델 교체는 수행하지 않았습니다.

- [실제 비교 결과·실행 비용](safelog-cpp/ai-training/reports/FACILITY_DENSE_AUXILIARY_STUDY_RESULTS_KO.md)
- [기존 모델과 새 위치 보조 모델 비교 그림](safelog-cpp/ai-training/reports/facility-dense-auxiliary-study-comparison.png)
- [정답 준비와 미확인 처리](safelog-cpp/ai-training/reports/FACILITY_DENSE_AUXILIARY_DATA_KO.md)
- [실제 학습·출력 호환·자료 보존 검증](safelog-cpp/ai-training/reports/facility-dense-auxiliary-study-verification.json)

## 앞선 RC 자료·양성 위치 보강 결과 · 2026-10-08

새 RC2119 시설 사진 200장과 저자가 표시한 균열·박락 위치를 추가해 후보 모델을 **6epoch, 약16.43분** 실제 학습했습니다.
표시가 없는 항목과 배경은 미확인으로 유지했습니다. 새로운 음성 정답이나 배경 음성 픽셀을 만들지 않았습니다.
기존 저학습률 대조군의 완료된6epoch를 재사용했으며 대조군을 다시 학습하지 않았습니다.

| 지표 | 기존 저학습률 대조군 | RC 자료·양성 위치 보강 |
|---|---:|---:|
| 균열·박락의 최대 미탐률/오탐률 | 22.0207% | 22.2222% |
| 작은 DACL 손상의 미탐 사례 합계 | 70건 | 70건 |
| DACL 노출 철근 AP | 0.7129188 | 0.7131089 |

최대 오류는 **0.2015퍼센트포인트 악화**됐고 작은 손상 미탐은 줄지 않았습니다. 연구 후보 기준과 각 항목5% 미만 목표는 미달이며 앱 기본 모델은 유지합니다.
최대 오류는 균열·박락×3자료×미탐률/오탐률의12개 비율 중 최대입니다. AP는 순위 지표이며 전체 사진 오답률과 다릅니다.

최초 별도 평가에서 CODEBRIM 연결 정보가 빠진 문제는 학습 후 평가 도구로 보완했습니다. DACL710·Dam424·CODEBRIM611개, 총1,745개 검증 사진·패치가 최종 비교에 포함됩니다.
원본 학습 기록·동결 소스123개·완료 가중치를 보존했습니다. 실제 학습 중 저장된 세 자료의 확률과 최종 평가 캐시가 정확히 같고, 선택된6epoch의 평가값도 일치합니다.
집중 코드 테스트26개와 평가 연결 보완 테스트2개는 별도 검사입니다. 처음 실패한 보고 단계는 과거 기록으로 보존했고 최종 전체 비교 보고서와 기술 검증은 완료됐습니다.
누적 기록은 **완료된 모델 학습236epoch + 이전 중단 실행 기록2epoch = 238epoch**입니다. 사전 점검용 임시 업데이트2회는 학습 epoch에 포함하지 않았습니다.
반복 공개 검증 자료의 연구 결과이며 독립 산업체 현장의 정확도를 측정한 결과는 아닙니다.

- [RC 보강 결과·실행 비용](safelog-cpp/ai-training/reports/FACILITY_RC_POSITIVE_STUDY_RESULTS_KO.md)
- [기존 모델과 RC 보강 비교 그림](safelog-cpp/ai-training/reports/facility-rc-positive-study-comparison.png)
- [전체 검증 평가를 보완한 이유와 확인 내용](safelog-cpp/ai-training/reports/FACILITY_RC_EVALUATION_MAINTENANCE_KO.md)
- [실제 학습·입력 보존 기술 검증](safelog-cpp/ai-training/reports/facility-rc-positive-study-verification.json)
- [평가 확률·선택 epoch 일치 확인](safelog-cpp/ai-training/reports/facility-rc-post-evaluation-review.json)

## 앞선 ConvNeXt 특징 보강 비교 · 2026-10-07

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
