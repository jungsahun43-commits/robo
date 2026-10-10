# SafeLog 산업안전 AI 학습

이 폴더는 역할 3의 데이터 다운로드, 검증, YOLO 파인튜닝, 평가, ONNX 변환과
C++ 앱용 로컬 API 서버를 재현하기 위한 코드다. 원본 데이터와 학습 결과는 Git에
올리지 않는다.

## ConvNeXt 특징 추출기 전체 학습 · 2026-10-10

새 시설 모델을 **실제6epoch·58.40분** 학습했습니다. ImageNet ConvNeXt-Tiny 특징 추출기 전체와 사진7종·손상 위치·19종 보조 head를 학습합니다. 기존 고정 ConvNeXt 특징 보정층 실험과 다른 새 모델입니다.

| 지표 | 기존 저학습률 대조군 | ConvNeXt 전체 학습 |
|---|---:|---:|
| 공개 검증 최대 오탐·미탐률 | 22.0207% | 22.2798% |
| 작은 DACL 손상 누락 사례 | 70건 | 73건 |
| DACL 철근 노출 AP | 0.7129 | 0.7819 |
| DACL 공동 AP | 0.6030 | 0.6975 |

연구 후보 기준 **미달**, 기존 항목 보존 기준 **통과**, 각 개별 비율5% 미만 **미달**입니다. AP는 순위 지표이며 정확도 백분율이 아닙니다.

보존 통과는 추가 학습 전 원본 대비 다른 항목 AP 하락 제한과 철근 노출 회복을 보는 별도 기준입니다. 기존 저학습률 대조군 대비 CODEBRIM 녹 AP는 **0.8940 → 0.8739**로 낮아졌으며, 연구 후보 기준의 AP 제한도 넘었습니다. 박락과 작은 손상 누락 악화 때문에 전체 오류 개선 모델로 채택하지 않았습니다.

같은 RTX4070SUPER에서640픽셀·FP32·batch1 추론 중앙값은 **4.23ms → 15.61ms**였습니다. 예열 후 모델 연산만 측정했으며 앱 전체 응답 시간이 아닙니다.

새 모델은 학습 파라미터28,259,976개·state tensor199개입니다. 원본 추출 순서와 정답·손실·교사는 유지하고, 학생의 구조·초기값·정규화가 바뀌는 구성 비교입니다. 실제 업데이트10,678회, 테스트9개, 입력72,662개 보존과 새 graph strict CPU 재로딩을 확인했습니다.

누적 완료 기록은 **모델 학습260epoch + 이전 중단2epoch = 262epoch**입니다. 대조군 재사용을 새 학습으로 세지 않습니다. 같은 공개 VAL1,745장의 반복 탐색이며 독립 산업현장 성능과 앱 전체 오류율을 측정한 것은 아닙니다. sourceTEST 추론과 앱 기본 모델 교체는 수행하지 않았습니다.

- [학습 방법·실제 비교 결과](reports/FACILITY_CONVNEXT_STUDY_RESULTS_KO.md)
- [모델 전달·항목별 AP·추론 시간·사용 명령](reports/FACILITY_CONVNEXT_HANDOFF_KO.md)
- [회차별 곡선과 작은 손상 비교](reports/facility-convnext-study-comparison.png)
- [실제 업데이트·보존·재로딩 검증](reports/facility-convnext-study-verification.json)

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
당시 누적 완료 기록은 **모델 학습254epoch + 이전 중단2epoch = 256epoch**이며 재사용 대조군을 다시 세지 않았습니다.
앱 기본 모델을 유지하고 sourceTEST 추론은 수행하지 않았습니다.

- [변경 방법·실제 비교 결과](reports/FACILITY_PRIMARY_ASYMMETRIC_STUDY_RESULTS_KO.md)
- [학습 경과·선택 모델·작은 손상 비교](reports/facility-primary-asymmetric-study-comparison.png)
- [실제 추출·업데이트·보존 검증](reports/facility-primary-asymmetric-study-verification.json)

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

- [검토 기준·실제 학습 비교 결과](reports/FACILITY_AI_AGREEMENT_STUDY_RESULTS_KO.md)
- [검토 선별·검증 비교 그림](reports/facility-ai-agreement-study-comparison.png)
- [실제 업데이트·추출 순서·자료 보존 증거](reports/facility-ai-agreement-study-verification.json)

## 직접 검수200장 시작

[이 PC의 검수 화면](http://127.0.0.1:8770/)에서 균열·박락을 각각 있음/없음/판단 불가로 선택합니다. 오류 의심120장과 비교80장, 총200장을 준비했고 자동 저장·복원·제출이 가능합니다.

이 사람 검수 경로는 실제 입력을 기다리는 준비 단계이며 사람 정답을 사용한 새 학습은0epoch입니다. 위 AI 보조 선별 재학습6epoch와 구분합니다. 집중 테스트9개와 별도 UI 테스트가 통과했고 테스트 입력은 학습에서 제외됩니다. 실제 제출된 판단만 별도 정답 자료로 만들며 원본·검증·시험 정답을 유지합니다.

- [검수 방법·학습 연결·서버 다시 시작](reports/FACILITY_HUMAN_REVIEW_START_KO.md)
- [검수 준비·테스트 확인 기록](reports/facility-human-review-ready.json)

## 항목별 5% 미만 목표 실험

현재 **균열·박락 각각의 미탐률과 오탐률 모두 5% 미만 목표는 미달**이다.
시험 정답은 반복 학습/임계값 선택에 쓰지 않는다. 검증 결과가 목표를 통과한 경우에만 설정을 고정해 시험한다.
공개 자료의 검증 점수는 새로운 산업체 현장의 성능 보장이 아니다.

### 앞선 원본19종 위치 보조 학습 결과 · 2026-10-09

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

- [실제 비교 결과·실행 비용](reports/FACILITY_DENSE_AUXILIARY_STUDY_RESULTS_KO.md)
- [기존 모델과 새 위치 보조 모델 비교 그림](reports/facility-dense-auxiliary-study-comparison.png)
- [정답 준비와 미확인 처리](reports/FACILITY_DENSE_AUXILIARY_DATA_KO.md)
- [실제 학습·출력 호환·자료 보존 검증](reports/facility-dense-auxiliary-study-verification.json)

### 앞선 RC 자료·양성 위치 보강 결과 · 2026-10-08

새 RC2119 시설 사진200장과 저자가 표시한 균열·박락 위치를 추가해 후보를 **6epoch, 약16.43분** 실제 학습했습니다.
표시가 없는 손상과 배경은 미확인으로 유지했습니다. 새 음성 사진 정답·배경 음성 픽셀·19종 보조 정답은0개입니다.
기존 저학습률 대조군의 완료된6epoch를 재사용했으며 이번 신규 모델 학습은6epoch입니다.

| 지표 | 기존 저학습률 대조군 | RC 자료·양성 위치 보강 |
|---|---:|---:|
| 균열·박락의 최대 미탐률/오탐률 | 22.0207% | 22.2222% |
| 작은 DACL 손상의 미탐 사례 합계 | 70건 | 70건 |
| DACL 노출 철근 AP | 0.7129188 | 0.7131089 |

최대 오류는 **0.2015퍼센트포인트 악화**됐고 작은 손상 미탐은 줄지 않았습니다. 기존 연구 후보 기준과 항목별5% 미만 목표는 미달입니다.
최대값은 균열·박락×DACL·Dam·CODEBRIM×미탐률/오탐률의12개 비율 중 최대이며 전체 사진 오답률이 아닙니다. AP는 별도의 순위 지표입니다.
새 자료와 양성 위치 손실을 함께 바꾼 실험이므로 데이터만의 효과를 분리해서 측정한 결과는 아닙니다.

최초 별도 평가의 CODEBRIM 연결 누락은 학습 후 도구로 보완했습니다. 최종 비교는 DACL710·Dam424·CODEBRIM611개, 총1,745개 검증 사진·패치에 기반합니다.
동결 소스123개·원본 학습 기록·완료 가중치를 보존했고, 실제 학습 중 저장된 확률과 세 자료의 최종 캐시가 정확히 같습니다. 선택된6epoch의 평가값도 일치합니다.
집중 코드 테스트26개와 평가 연결 보완 테스트2개는 별도 검사입니다. 처음 실패한 보고 단계의 과거 기록은 유지했고 전체 비교 보고서와 기술 검증은 완료됐습니다.
누적 기록은 **완료된 모델 학습236epoch + 이전 중단 실행 기록2epoch = 238epoch**입니다. 사전 점검용 임시 업데이트2회는 학습 epoch에 포함하지 않습니다.
source-TEST는 사용하지 않았으며 기본 앱 모델은 교체하지 않았습니다. 반복 공개 VAL과 한 seed의 결과를 독립 산업체 현장 성능으로 해석하지 않습니다.

- [RC 보강 결과·실행 비용](reports/FACILITY_RC_POSITIVE_STUDY_RESULTS_KO.md)
- [RC 보강 비교 그림](reports/facility-rc-positive-study-comparison.png)
- [전체 검증 평가 보완과 확인 내용](reports/FACILITY_RC_EVALUATION_MAINTENANCE_KO.md)
- [실제 학습·입력 보존 검증](reports/facility-rc-positive-study-verification.json)
- [평가 확률·선택 epoch 일치 확인](reports/facility-rc-post-evaluation-review.json)

### 앞선 ConvNeXt 특징 보강 결과 · 2026-10-07

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

- [최신 ConvNeXt 특징 보강 결과](reports/FACILITY_SEMANTIC_STUDY_RESULTS_KO.md)
- [추가 데이터·새 특징 계획](reports/FACILITY_DATA_AND_FEATURES_PLAN_KO.md)
- [실제 모델·교사·encoder·원본 보존 검증](reports/facility-semantic-study-verification.json)

### 앞선 head 학습률 비교 결과

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

- [최신 head 학습률 비교 결과](reports/FACILITY_HEAD_LR_STUDY_RESULTS_KO.md)
- [고정 조건·재현 명령](reports/FACILITY_HEAD_LR_STUDY_PLAN_KO.md)
- [실제 LR·교사·BN·완료 모델 검증](reports/facility-head-lr-study-verification.json)

### 앞선 BN 통계 비교 결과

보존 손실 가중치0·1·4 비교에 이어 정규화 통계 고정 후보6epoch를 추가 학습하고 전체 공개 VAL을 검증했습니다.
완료된 일반 BN 가중치4 모델을 재사용해 대조군 재학습은0epoch입니다. 이번 작업의 신규 학습은 총24epoch입니다.
원본 / 일반 BN 가중치4 / 고정 BN 가중치4 순서의 최대 source-VAL 미탐·오탐은 22.28% / 22.28% / 22.41%입니다.
DACL 철근 노출 AP는 0.7010 / 0.6758 / 0.7075, 작은 손상 항목·사진 사례 FN 합계는 71 / 72 / 70건입니다.
AP는 오류율과 다른 순위 지표입니다. 고정 BN의 성능 보존 기준은 **미달**, 연구 후보 기준은 **미달**입니다.
철근 노출 AP는 회복했지만 DACL 공동·패임 AP 하락0.020277이 고정한 한도0.02를 넘었다. 작은 손상 FN70건은 대조72건보다2건 줄었으나 원본71건보다1건 줄어 기존 연구 기준에는 미달이다.
12개 균열·박락 FNR/FPR 각각5% 미만 목표는 미달이며 기본 앱 모델을 교체하지 않았습니다.
BN47개 층·버퍼141개는 원본 해시를 유지했고, affine94개와 backbone·head 가중치는 계속 학습했습니다.
후속 소스86개·실제 테스트32개·보호 파일58,965개·실제 추출과 업데이트를 검증했습니다.
앞선 비교 테스트34개·보고서 보완2개·강도 비교28개도 별도로 통과했습니다.
누적 완료 기록은 모델 실행218epoch와 이전 중단 기록2epoch의 합220epoch입니다.
반복 공개 VAL 결과에 따른 후속 실험이며 공장 현장 성능이나 정규화 통계의 인과 효과를 입증하지 않습니다.

- [최신 BN 통계 비교 결과](reports/FACILITY_BATCHNORM_STUDY_RESULTS_KO.md)
- [고정 조건·재현 명령](reports/FACILITY_BATCHNORM_STUDY_PLAN_KO.md)
- [원본 통계·학습 계수·완료 모델 검증](reports/facility-batchnorm-study-verification.json)

### 앞선 가중치4 비교 결과

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

- [최신 가중치4 결과](reports/FACILITY_RETENTION_STRENGTH_STUDY_RESULTS_KO.md)
- [고정 조건·재현 명령](reports/FACILITY_RETENTION_STRENGTH_STUDY_PLAN_KO.md)
- [완료 모델·재사용 대조군·입력 보존 검증](reports/facility-retention-strength-study-verification.json)

앞선 완료 작업은 [기존 항목 성능 보존 비교](reports/FACILITY_RETENTION_STUDY_RESULTS_KO.md)다.
원본0773·같은640 TRAIN·같은 사진 추출 순서·같은 교사 실행에서 각6epoch·총12epoch 실제 학습했다.
알려진 다른5항목의 교사 확률을 보존하는 손실 가중치0·1만 다르다. 교사 확률은 새로운 정답이 아니다.
최대 source-VAL 미탐·오탐은 초기22.28% /대조23.06% /보존22.53%, 작은 손상 FN 합계71 /67 /67이다.
DACL 철근 노출 AP는0.7010 /0.6486 /0.6715로 일부 회복됐다. AP는 오류율과 다른 순위 지표다.
철근 노출·CODEBRIM 백화 AP 하락이0.02를 넘어 보존 기준은 미달이며 연구 후보 기준·5% 목표도 미달이다.
실제 테스트34개·소스61개·보호 파일58,890개와 교사 고정·CPU 재로딩·실제 업데이트 검증이 통과했다.
생략된 계획 해시를 검증된 프로토콜에서 메모리 복사본에 연결하는 보고서 보완 테스트2개도 통과했다.
누적 완료 학습은 모델 실행206epoch와 이전 중단 완료2epoch의 합208epoch다. 기본 앱 모델은 유지한다.

- [보존 손실 조건·재현 명령](reports/FACILITY_RETENTION_STUDY_PLAN_KO.md)
- [원본 입력·교사·완료 모델 검증](reports/facility-retention-study-verification.json)

앞선 작업은 [박락 음성 태그 추출 대조 학습](reports/FACILITY_SUBTYPE_STUDY_RESULTS_KO.md)이다.
같은0773 모델·원래640 TRAIN·손실 가중치에서 각6epoch·총12epoch를 실제 추가 학습했다.
최대 source-VAL 미탐·오탐은 초기22.28% / 대조22.84% / 보강23.32%, 작은 손상 FN 합계는71 /67 /68이다.
다른 알려진 항목 AP 악화도 확인되어 연구 후보 기준·5% 목표 모두 미달이며 앱 모델을 승격하지 않는다.
원래 표본 구성을 모든 위치에서 유지하면서 관련 태그 음성 노출3,539→4,743회, 교체1,204곳을 검증했다.
실제 테스트28개·소스49개·원본 파일58,874개와 가중치 CPU 재로딩 검증이 통과했다.
누적 완료 학습은 모델 실행194epoch와 이전 중단 완료2epoch의 합196epoch이다.

- [고정 조건·재현 명령](reports/FACILITY_SUBTYPE_STUDY_PLAN_KO.md)
- [실제 기록·원본 보존·소스 검증](reports/facility-subtype-study-verification.json)

학습 전 준비는 [박락 TRAIN 오답 감사](reports/FACILITY_SPALLING_TRAIN_AUDIT_KO.md)다.
130장 중 박락 FP/FN54장과 원본 주석·기존80 마스크를 대조하고, DACL6225장의 저장 점수를
원래 태그로 집계했다. 박락 음성·관련4태그 집단666장의 TRAIN 오탐 비율은26.43%였다.
한 가지 제한적 표본 추출 보강을 준비했으며 모의85,488위치 중1,204곳만 같은 층의 음성으로 대체했다.
모든 위치의 출처·full/crop·균열/박락 정답을 보존했다. 감사14개·태그 집계6개·표본 추출6개 테스트가 통과했다.
당시 준비 단계의 새 GPU학습·추론·라벨 수정은0건이었다. 실제 후속 학습의 결과는 위 비교와 구분한다.

- [표본 추출 모의 실행·수치 안정성 보완](reports/facility-spalling-sampler-dry-run.json)
- 현재 PC의 검수 화면: `runs/facility-spalling-train-audit/SPALLING-AUDIT.html`. 자료·화면은 로컬 보관이다.

앞선 원본 ROI 비교에서는 기존 TRAIN 영역 5,928개를 같은 원본에서 두 방식으로 준비하고,
각각 6epoch, 총 12epoch를 실제 추가 학습했다. 공개 자료 VAL의 균열·박락 최대 미탐·오탐은
초기 모델 22.28% / 먼저 축소한 대조군 23.77% / 원본에서 바로 자른 보강군 23.58%다.
작은 결함 FN은 각각 71 / 68 / 73건으로 보강군의 개선 효과를 확인하지 못했다.
연구 후보 기준과 엄격한 5% 목표 모두 미달이며 앱 기본 프로필은 유지한다.
새 독립 사진·전문가 정답·정밀 위치 정답을 추가한 실험은 아니다.

- 항목별 오류·그림·실제 비용: [원본 ROI 실측 결과](reports/FACILITY_NATIVE_ROI_STUDY_RESULTS_KO.md)
- 학습 전 고정 조건·재현 순서: [원본 ROI 대조 계획](reports/FACILITY_NATIVE_ROI_STUDY_PLAN_KO.md)
- 30개 실제 코드 테스트·원본 파일·PNG·완료 가중치 확인: [완료 검증](reports/facility-native-roi-study-verification.json). 정확도 측정과 별도다.
- 보고서 재현은 `python scripts/report_facility_native_roi_results.py`를 사용한다. 정수값으로 저장된 float 개수의 보고 형식만 보정하며, [추가 테스트 4개](reports/facility-native-roi-report-adapter-tests.json)와 보정 경위를 결과에 기록했다. 기존 학습 전 소스·정답·후보 기준은 보존했다.
- 직전 입력 640·960 비교: [해상도 실측 결과](reports/FACILITY_RESOLUTION_STUDY_RESULTS_KO.md)

다음 우선순위는 기존 TRAIN 검수 대상의 박락 양성·음성 오답을 원본 폴리곤,
80격자 지도와 배경 질감의 관계에 따라 분류하는 감사다. 검수 의견과 원본 정답을 구분하고,
실제 근거를 확인한 뒤 다음 보강 조건을 하나 정한다.

별도 검수 화면 작업에서는 최근 ROI 모델의 TRAIN 사진 23건을 AI가 확인하고 항목 의견 31개를
사람의 검수 의견과 분리했다. AI 참고 관찰·원인 후보·질문은 검수 화면에서 기본 숨김인
읽기 전용 자료로 볼 수 있다. 전체 130건의 입력 크기도 확인했으며, 전문가 판정·
라벨 수정·새 학습은 0건이다. [AI 관찰 집계](reports/FACILITY_ROI_AI_REVIEW_KO.md)와
[실행 안내](FACILITY_REVIEW_FEEDBACK_KO.md#6-ai-보조-관찰을-연결한-화면)를 참고한다.

앞선 pooling 작업에서는 좁은 피크와 넓은 증거를 비교하는 사진 pooling 계수 7개를 추가하고,
기존 구조와 각각 6epoch, 총 12epoch를 실제 추가 학습했다.
최대 검증 미탐·오탐은 기존 최고 후보 22.28% / 대조군 22.84% / 새 pooling 22.84%로
개선되지 않았다. 작은 손상 FN 합계는 기존 71건 / 대조군 71건 / 후보 70건으로,
한 건 감소했지만 사전 선언한 후보 기준은 통과하지 못했다. 앱 기본 모델은 유지한다.

- 최신 pooling 비교·항목별 오류: [실측 결과](reports/FACILITY_POOL_CONTEXT_RESULTS_KO.md)
- 학습 전에 고정한 조건·재현 명령: [pooling 대조 계획](reports/FACILITY_POOL_CONTEXT_PLAN_KO.md)
- 실제 코드 테스트 155개·저장 모델 재로딩·6epoch 표본 순서 비교: [기술 검증](reports/facility-pool-context-technical-verification.json). 정확도 측정과 별도다.
- 손상 정의·비슷한 표면·촬영 정보의 질문 목록: [전문가 검수 준비](reports/FACILITY_CONTEXT_LABEL_REVIEW_KO.md). 검수 대상 준비이며 전문가 확정·원본 라벨 수정은 0건이다.
- 검수자·항목별 의견 JSON 저장/불러오기와 오류 원인·충돌 집계: [검수 피드백 사용 안내](FACILITY_REVIEW_FEEDBACK_KO.md). 130건의 새 로컬 화면을 만들었으며 실제 전문가 판정·라벨 수정·추가 학습은 0건이다.

앞선 세부 특징 작업에서는 작은 손상을 위한 stride 4 세부 특징 분기를 구현하고,
기존 구조 대조군과 각각 8epoch, 총 16epoch를 실제 추가 학습했다.
최대 검증 미탐·오탐은 기존 최고 후보 22.28% / 대조군 22.53% / 새 구조 22.80%로
개선되지 않았다. 작은 손상 미탐은 기존 71건 / 대조군 67건 / 새 구조 69건으로,
새 분기의 효과를 확인하지 못했다. 연구 후보 기준과 엄격한 5% 기준 모두 미달이며
앱 기본 프로필은 `facility-validation-v2`를 유지한다. 공장 현장 성능은 미측정이다.

- stride 4 구조 비교·항목별 오류·다음 우선순위: [세부 특징 구조 실측 결과](reports/FACILITY_DETAIL_ARCHITECTURE_RESULTS_KO.md)
- 재현 명령·학습 전에 고정한 조건: [세부 특징 구조 대조 계획](reports/FACILITY_DETAIL_ARCHITECTURE_PLAN_KO.md)
- 실제 코드 테스트 143개·저장 모델 재로딩·8epoch 표본 순서 비교: [세부 특징 기술 검증](reports/facility-detail-architecture-technical-verification.json). 정확도 측정과 별도다.

앞선 구분 학습에서는 정답의 범위를 정리하고, 같은 출처의 손상 양성·음성을 구분하는
순위 손실을 구현해 대조군과 각각 6epoch, 총 12epoch를 실제 추가 학습했다.
최대 검증 미탐·오탐은 기존 후보 22.28% / 대조군 23.06% / 보강군 22.28%로,
기존 후보보다 개선되지 않았다. 작은 손상은 균열 29→28/93, 박락 42→41/105로
미탐 사진이 각각 한 장 줄었지만 일반적인 성능 향상의 증거로 보기는 부족하다.
연구 후보 기준은 미달이며 앱 기본 모델은 `facility-validation-v2`를 유지한다.

- 구분 학습 비교·오차 범위: [손상 구분 학습 결과](reports/FACILITY_TARGET_DISCRIMINATION_RESULTS_KO.md)
- 정답으로 확인한 범위와 미확인 항목: [라벨 범위](reports/FACILITY_LABEL_SCOPE_KO.md)
- 재현 명령·고정 조건: [대조 실험 계획](reports/FACILITY_TARGET_DISCRIMINATION_PLAN_KO.md)
- 코드 테스트 131개 및 실제 저장 모델 재로딩 확인: [기술 검증](reports/facility-target-discrimination-technical-verification.json). 현장 정확도 측정과 별도다.

- 조건과 분리: [FACILITY_FIVE_PERCENT_PLAN_KO.md](reports/FACILITY_FIVE_PERCENT_PLAN_KO.md)
- 실제 학습 회수와 항목별 오류: [통합 결과](reports/FACILITY_FIVE_PERCENT_RESULTS_KO.md). `python scripts/report_facility_target.py`로 현재 기록을 다시 모은다.
- 새 실제 자료 감사: [CODEBRIM](reports/facility-target-codebrim-data-audit.json)
- 4GB ZIP 위치 호환 처리·CRC 확인: [extraction](reports/facility-target-codebrim-extraction.json)
- 더 큰 사진 모델/상세 증강/해상도 보강 결과: `reports/facility-presence-target-*-target-validation.json`
- 위치 감독 준비: [spatial data](reports/facility-target-spatial-data-audit.json). 추가한 조각은 새 독립 현장 사진이 아니다.
- S2DS 추가: [사용 자료·제외 이유](reports/facility-target-s2ds-screened-data-audit.json). 저자 TRAIN만 사용하며 원본 장면 ID가 없어 독립 현장 성능으로 주장하지 않는다.
- 추가 반복 학습: [어려운 TRAIN 사례 보강 계획·실행 명령](reports/FACILITY_HARD_TRAINING_PLAN_KO.md). 기존 최고 모델이 어려워하는 학습 사진의 추출 비중을 제한해서 높이며, 같은 세 자료의 검증 기준을 유지한다.
- 확인 대기 진단: [자동 판단 비율·조건부 오류](reports/facility-presence-target-spatial-review-diagnostic_KO.md). 검증 자료에서 불확실한 사진을 보류하는 비교이며, 기존 전체 사진의 미탐·오탐 기준을 통과한 결과도 앱에 적용된 결과도 아니다.
- 세부 태그 보강: [원본19종 보조 학습](reports/FACILITY_AUXILIARY_PLAN_KO.md). DACL TRAIN 전체 사진만 원래 태그로 함께 학습한다. 앱 추론은 기존7항목을 유지한다.
- 이번 추가24epoch 결과: [이전 후보와 비교·한계·적용 상태](reports/FACILITY_CONTINUED_TRAINING_RESULTS_KO.md). 최대 검증 오류는 낮아졌지만 일부 자료·항목은 나빠졌고5% 목표는 미달이다.
- 오류 검수와 작은 손상 비교: [실행 방법](FACILITY_REVIEW_AND_SMALL_REGION_KO.md), [같은 조건의 비교 계획](reports/FACILITY_EFFICIENT_DEVELOPMENT_PLAN_KO.md), [추가12epoch 실측 결과](reports/FACILITY_SMALL_REGION_RESULTS_KO.md). 보강군 최대 검증 오류23.15%는 대조군22.28%보다 높아 채택을 보류했다. 검수 의견은 자동 정답 변경으로 쓰지 않는다.
- 새 콘크리트 사진 보강: [준비·비교 명령](reports/FACILITY_BUILDING_SUPPLEMENT_PLAN_KO.md), [ConViD192장 감사](reports/facility-convid-data-audit.json), [추가12epoch 실측 결과](reports/FACILITY_BUILDING_SUPPLEMENT_RESULTS_KO.md). 기존 후보22.28% /대조군22.80% /보강군22.85%로 최대 검증 미탐·오탐 개선을 확인하지 못해 앱에 적용하지 않았다. 사진마다 저자 폴더의 손상 하나만 양성이고 다른6종은 미확인이다. 공장 현장 평가 자료가 아니며 박락 정의의 전문가 검증도 완료되지 않았다.
- 원본 TRAIN 검수: [23장 AI 육안 확인](reports/FACILITY_TRAIN_VISUAL_REVIEW_KO.md). 전문가 검수나 정답 수정이 아니며,140개 검수 대상 중117개는 아직 사진을 확인하지 않았다.
- 새 자료 취득 상태: [공식 출처·라이선스·접근 조사](reports/INDUSTRIAL_SOURCE_RESEARCH_KO.md), [기존 산업체 자료의 재사용 한계](reports/INDUSTRIAL_REUSE_AUDIT_KO.md). PECCD는 공식 아카이브 SHA가 맞지만 숫자 클래스 대응이 미확인되어 학습0장이다.
- 새 모델의 실제 파일 확인: [CPU 재로딩·7개 출력·코드 버전 확인](reports/facility-building-technical-verification.json). 정확도 측정과 별도이다. 실행 당시 코드는 commit `b842cc27cbcd9349fb337603d3e51a55fe55af5e`에 보존했고 이후 바뀐 trainer 설명 문구와 구분한다.

이 실험들은 `runs/facility-presence-target-*`에 분리한다. 기본 서버 프로필은 기존 채택 버전
`facility-validation-v2`이며, 새 실험 가중치를 기본 모델에 덮어쓰지 않는다.
CODEBRIM은 교육·비상업 연구 전용 조건이다. 팀원도 [원문 조건](https://zenodo.org/records/2620293/files/license.md?download=1)을 확인한다.
원본과 변환 데이터는 저장소/모델 ZIP에 포함하지 않는다.

S2DS 파이프라인(코어 자료와 `facility-presence-target-spatial` 학습이 먼저 준비되어 있어야 한다):

```powershell
python scripts/download_s2ds.py
python scripts/extract_s2ds.py
python scripts/prepare_s2ds.py
python scripts/audit_s2ds_crops.py
python scripts/train_facility_spatial.py --name facility-presence-target-s2ds --seed 48 --initial runs/facility-presence-target-spatial/best.pt --supplement-spatial data/s2ds-spatial-training/crop-screened-train.json --backbone-lr .00004 --head-lr .00025
python scripts/evaluate_facility_target.py select --name facility-presence-target-s2ds
python scripts/report_facility_target.py
python scripts/plot_facility_target.py
```

기존 압축 해제 폴더와 학습 체크포인트는 보존한다. 이미 끝낸 단계는 반복할 필요 없다.
S2DS 정답 색상은 저자의 RGB 변환표를 따른다. 출처·동결 snapshot SHA·[이용 조건](https://github.com/ben-z-original/s2ds)은 `datasets/s2ds_source.json`에 기록한다.

## 시설 성능 보강 전달본

보강 모델 ZIP과 최신 코드를 함께 사용하면 `./start_ai_server.ps1 -FacilitiesOnly`가
검증으로 선택한 시설 모델·입력 해상도·항목별 탐지 기준을 자동 적용한다.
설정은 `reports/facility-inference-profile.json`, 성능과 한계는 `reports/FACILITY_OPTIMIZATION_KO.md`다.
가중치 SHA256이 설정과 일치하지 않으면 분석 API가 503을 반환한다.
초기 시설 기준 모델과 비교하려면 `-BaselineFacilities`를 추가한다.
이 보강은 기존 데이터로 수행한 실험이며 새로운 시설 현장 데이터가 추가된 것은 아니다.
사진 전체 분류 모델은 검출이 놓친 항목을 추가 의견으로 제안하며 위치 박스를 만들지 않는다.
`box=null`, `evidence_scope=photo_presence`를 구분해 표시한다. 분류 보강이 박스 검출 AP를 높인 것은 아니다.

## 2차 피드백 결과와 현재 적용 모델

추가 학습과 비교 결과는 [2차 피드백 보고서](reports/FACILITY_FEEDBACK_KO.md)에 있다.
시험 975장에서 항목-사진 쌍의 미탐은 732→731, 오탐은 309→317이었다.
2차 후보는 교체 기준을 통과하지 못해 기본 설정은 **1차 `facility-validation-v2`**를 유지했다.
기존 1차 모델 ZIP을 가진 팀원은 코드만 업데이트해도 된다. 모델 가중치는 바뀌지 않았다.
`facility-inference-profile-round2-candidate.json`과 round2-candidate ZIP은 실험 기록이다.
이를 앱의 기본 설정으로 복사하지 않는다. `/health.photoClassifiers`에서 실제 활성 모델을 확인한다.

피드백은 검증 사진의 오류 예시 확인 → 학습 분할만 재학습 → 검증에서 설정 고정 →
시험 비교 → 교체 판단 순서로 기록한다. 오탐과 미탐을 함께 비교하고 악화된 후보는 채택하지 않는다.
같은 시험 분할을 반복 확인한 결과는 독립적인 현장 검증이 아니다.
다음 외부 평가는 현장/촬영 회차를 구분한 정상·손상 사진과 사람이 확인한 정답으로 구성한다.
시설 모델 연결과 팀원 전달 순서는 [역할 4 안내](FACILITY_HANDOFF_KO.md)를 따른다.

## 3차 합성 공동 보강

[공식 synthcavity 자료](https://doi.org/10.60776/9D6E4M)의 렌더 사진 5,000장을 추가해 학습했다.
동일 장면의 두 변형을 포함한 2,500개 생성 장면이며 실제 현장 사진이 아니다.
정답이 있는 공동 항목만 학습하고 나머지 여섯 항목은 음성으로 간주하지 않았다.
시험에서 공동 미탐은 45.5%→43.7%, 오탐은 7.1%→7.4%였다.
오탐 증가로 3차 후보도 채택하지 않아 기본 설정은 **facility-validation-v2**다.
이 비교는 반복 확인한 기존 시험 분할의 항목 존재 평가이며 현장 안전 정확도가 아니다.

[3차 결과·재현 절차](reports/FACILITY_ROUND3_KO.md)와
[현장 사진 피드백 수집 안내](FIELD_FEEDBACK_KO.md)를 참고한다.
`facility-inference-profile-round3-candidate.json`은 실험 기록이며 기본 설정으로 복사하지 않는다.
기존 1차 모델 ZIP을 가진 팀원은 코드만 업데이트하면 된다.

## 4차 실제 댐 표면 보강

[DamSegment](https://data.mendeley.com/datasets/z5z6gtt5t4/1)의 공식 세 파일을 검증해
실제 패치 2,009개를 추가 학습하고 491개를 별도 보류 평가했다. 균열·박락 정답만 사용했고
미확인 다섯 항목은 음성으로 학습하지 않았다. [4차 결과](reports/FACILITY_ROUND4_KO.md)를 확인한다.
새 균열 분류기를 진단 목적으로 결합한 보류 패치 평가에서는 미탐 52.4%→21.7%, 오탐 5.5%로 동일했다.
박락과 기존 검증의 교체 기준은 통과하지 못해 기본 모델은 **facility-validation-v2**를 유지한다.
이는 한 댐의 패치 평가이며 다른 현장에서의 성능을 보장하지 않는다. 연구용 새 가중치를 기본 ZIP과 섞지 않는다.

## 1. 설치

Windows에 Python 3.11 또는 3.12를 설치하고 PowerShell에서 실행한다.

```powershell
cd safelog-cpp/ai-training
./setup_windows.ps1
./.venv/Scripts/Activate.ps1
```

설치 스크립트는 NVIDIA GPU가 있으면 CUDA 12.6용 PyTorch를 설치하고, 없으면 CPU용을
설치한다. CPU용을 명시하려면 `./setup_windows.ps1 -CpuOnly`를 쓴다. 실제 검증된 패키지
버전은 `requirements-tested.txt`에 있으며 `python scripts/check_environment.py`로 CUDA
사용 여부를 확인한다. 앱은 C++/Qt, 모델 학습과 추론 서버는 Python으로 개발한다.

## 2. 공식 데이터 다운로드와 검사

```powershell
python scripts/download_dataset.py construction-ppe
python scripts/audit_yolo_dataset.py data/construction-ppe/data.yaml
python scripts/download_dataset.py chvg
python scripts/prepare_chvg.py
python scripts/audit_yolo_dataset.py data/chvg-yolo/data.yaml
python scripts/download_dataset.py indoor-fire-smoke
python scripts/prepare_fire_smoke.py
python scripts/audit_yolo_dataset.py data/indoor-fire-smoke/data.yaml
python scripts/download_dataset.py sh17
python scripts/prepare_sh17.py
python scripts/audit_yolo_dataset.py data/sh17/data.yaml
python scripts/prepare_sh17_training_copy.py
```

Construction-PPE 1,416장, CHVG 1,699장, 실내 화재·연기 5,000장과 SH17 8,099장을
각각 사용한다. 출처와 라이선스는 `datasets/sources.json`에 고정했다. 다운로드 결과는
`data/`에 저장되며 Git에서 제외된다. 대회 제출 또는 모델 공개 전에는 각 라이선스의
상업 이용 및 재배포 조건을 다시 확인한다.

## 3. 모델별 학습

```powershell
python scripts/train.py --data data/construction-ppe/data.yaml --epochs 100 --batch 16 --device 0 --name ppe-baseline
python scripts/train.py --data data/indoor-fire-smoke/data.yaml --epochs 100 --batch 16 --device 0 --name fire-smoke
python scripts/train.py --data data/chvg-yolo/data.yaml --epochs 100 --batch 16 --device 0 --name chvg-ppe
python scripts/train.py --data data/sh17-1280/data.yaml --model yolo11s.pt --epochs 100 --batch 16 --device 0 --name sh17-ppe --cache disk
python scripts/train.py --data data/construction-ppe/data.yaml --model runs/sh17-ppe/weights/best.pt --epochs 100 --batch 16 --device 0 --name ppe-sh17-transfer
```

라벨 뜻이 다른 데이터셋을 억지로 한 파일로 합치지 않는다. Construction-PPE는 미착용
판단, 화재·연기는 화재 위험 판단, CHVG와 SH17은 PPE 부품 탐지와 외부 검증에 쓴다.
각 학습 결과는 `runs/<이름>/weights/best.pt`에 저장된다. 조기 종료가 작동하므로 실제
학습 횟수는 100회보다 적을 수 있다.

추가 PPE 전이 실험은 SH17 학습 결과로 Construction-PPE를 다시 학습한다. 원본 PPE
모델과 전이 모델은 검증 분할의 네 미착용 클래스 평균 AP50으로 선택하며, 독립 test
분할은 모델 선택에 쓰지 않는다. `reports/selected-models.json`에 선택 근거를 저장하고
서버 실행 스크립트가 선택된 모델을 사용한다.

SH17은 8,099장의 고해상도 이미지와 75,994개 객체를 포함해 저장 공간과 시간이 많이
필요하다. 공식 분할은 train 6,479장과 val 1,620장이며 별도 test가 없다. 따라서 SH17
수치는 공식 val 기준이라고 명시한다. SH17은 더 큰 `yolo11s`를 사용하고 RTX 4070
SUPER 12GB에서 배치 16으로 학습한다. 메모리가 부족한 PC에서는 배치를 8로 낮춘다.

## 4. 평가와 ONNX 변환

```powershell
python scripts/evaluate.py runs/ppe-baseline/weights/best.pt data/construction-ppe/data.yaml
python scripts/export_onnx.py runs/ppe-baseline/weights/best.pt
```

평가 결과는 콘솔과 `runs/evaluation-<학습명>.json`에 저장된다. 모든 모델 평가 후
`python scripts/summarize_results.py`를 실행하면 비교표가 만들어진다. 발표에는
precision, recall, mAP50, mAP50-95와 테스트 이미지 수를 함께 기록한다.

## 5. C++ 앱 연결 서버

```powershell
$env:SAFELOG_MODEL_PATH="runs/ppe-baseline/weights/best.pt"
$env:SAFELOG_FIRE_MODEL_PATH="runs/fire-smoke/weights/best.pt"
# 선택: 추가 PPE 모델도 함께 추론할 때만 지정한다.
# $env:SAFELOG_AUX_MODEL_PATH="runs/chvg-ppe/weights/best.pt"
python -m uvicorn safelog_ai.server:app --host 0.0.0.0 --port 8080
```

앱 실행 전 서버 PC의 IP를 설정한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8080"
```

서버는 C++ 앱이 요구하는 세 API를 제공한다.

- `POST /v1/analyze-hazard`
- `POST /v1/compare-action`
- `POST /v1/summarize`

서버는 지정된 모델을 차례로 실행하고 탐지 결과를 하나의 위험 판정으로 합친다. 운영
기본 조합은 미착용을 직접 학습한 Construction-PPE 모델과 화재·연기 모델이다. CHVG와
SH17은 클래스 의미가 다르므로 부품 탐지 근거를 추가하는 보조 모델로 선택한다.

한 번에 여섯 모델을 시작하려면 `./start_ai_server.ps1 -AllModels`를 실행한다. 기본
`./start_ai_server.ps1`은 PPE+화재 모델을 사용한다. 모델 ZIP을 `ai-training/`에 풀면
`models/`의 학습된 파일을 우선 사용한다. 원본 데이터셋 없이도 서버를 실행할 수 있다.

PPE 후보는 `ppe-baseline`과 SH17 가중치에서 전이 학습한 `ppe-sh17-transfer`다.
`reports/selected-models.json`이 있으면 서버 실행 스크립트가 선택된 후보를 사용한다.
선택 기준은 Construction-PPE val에서 미착용 네 클래스의 AP50 평균이며 test 지표는
선택에 사용하지 않는다. 두 후보와 모든 클래스의 평가 결과는
`reports/TRAINING_RESULTS_KO.md`에 기록된다. `no_boots`의 val 정답이 4개뿐이므로
선택 점수만으로 현장 성능을 확정하지 않는다.

Windows 앱을 같은 PC에서 실행할 때는 `SAFELOG_AI_BASE_URL=http://127.0.0.1:8080`을
설정한다. Android 휴대폰에서는 같은 Wi-Fi에 연결한 서버 PC의 실제 IP를 사용한다.
`127.0.0.1`은 휴대폰 자체를 가리킨다. 앱 설정 저장 또는 APK의 서버 주소 전달은
역할 4의 앱 시작 설정과 연결한다.

## 시설 위주의 추가 모델

시설 표면 손상과 금속 부식을 별도 데이터셋·모델로 학습한다.
출처와 라이선스는 `datasets/facility_sources.json`에 기록했다.
데이터 준비부터 학습·시험 평가·ONNX 변환까지 재현하려면 다음을 실행한다.

```powershell
./run_facility_training.ps1
```

이미 다운로드했다면 `-SkipDownload`, 자료 준비만 하려면 `-PrepareOnly`를 사용한다.
학습 중단 후 같은 명령을 다시 실행하면 last.pt에서 이어 학습한다.
GPU 학습은 순차 진행한다. 시설 학습은 입력 960px·YOLO11s·배치 8을 사용한다.
실제 학습 모델이 없는 초기 상태에서는 시설 서버를 실행할 수 없다.

완료된 모델 ZIP을 ai-training에 풀고 다음 명령으로 시설 모델 2개와 화재·연기를 시작한다.

```powershell
./start_ai_server.ps1 -FacilitiesOnly
```

`-Facilities`는 PPE+화재+시설 4개, `-AllModels`는 보조 PPE까지 6개를 실행한다.
`/health`에 모델 목록과 입력 크기가 표시된다. 시설 모델은 960px로 추론하며 기존 모델은 640px다.
시설 성능·고정 신뢰도 0.25에서의 미탐·오탐 비중·변환 규칙·자료 한계는
`reports/FACILITY_TRAINING_RESULTS_KO.md`에서 확인한다.

공개 교량 사진의 손상 폴리곤과 금속 부식 마스크를 영역 박스로 변환했다.
백화/젖은 표면을 배관 누수로, 철근 노출을 전선 노출로 해석하지 않는다.
위험 등급은 점검 우선순위 규칙이며 구조 진단 모델의 판단이 아니다.
모든 제안과 조치 후 상태는 점검자의 최종 확인을 거친다.

PPE/화재/시설 탐지는 실제 파인튜닝한 YOLO 모델이다. 위험도·조치 문장과 `/v1/summarize`는
현재 규칙과 템플릿으로 생성한다. 문장을 생성하는 별도의 언어 모델을 학습한 것은 아니다.
`detections`에는 탐지 모델, 클래스, 신뢰도와 좌표가 포함돼 앱의 원본 AI 분석 기록에
함께 저장된다. PPE 부품이 탐지되지 않았다는 사실만으로 미착용을 확정하지 않는다.

## 6. 다시 학습하거나 팀에 전달하기

데이터셋 준비 후 `./run_training_suite.ps1`은 학습·평가·ONNX 변환을 순서대로 실행한다.
기존 `last.pt`가 있으면 이어 학습하며 완료된 학습은 반복하지 않는다. 학습명을 새로
지정하면 별도의 실험을 시작한다. SH17 사본은 이미지의 최대 변을 1280으로 줄이고
정규화된 라벨은 유지한다. 이 과정에서 모든 이미지의 VOC 클래스와 YOLO 번호를 대조한다.
기존 PPE/화재 모델 입력 크기는 640이고 시설 모델은 960이다.

```powershell
python scripts/train.py --data data/sh17-1280/data.yaml --name sh17-ppe --resume --device 0
python scripts/evaluate.py runs/sh17-ppe/weights/best.pt data/sh17-1280/data.yaml --device 0 --split val
python scripts/summarize_results.py
python scripts/build_training_report.py
python scripts/package_models.py
```

시설 학습·시험·ONNX·시설 보고서까지 준비되면 ZIP에 시설 모델을 자동으로 포함한다.
시설 모델 포함을 반드시 검사하려면 `python scripts/package_models.py --require-facilities`를 사용한다.
기존 PPE/화재 학습만 완료한 상태에서는 기존 5개 모델로 ZIP을 만든다.

GitHub에는 코드·출처·결과 보고서가 올라가고, 데이터·가중치는 제외된다. 별도로 생성된
`artifacts/safelog-trained-models.zip`을 팀원에게 전달한다. 팀원은 동일 브랜치를 받고
설치 스크립트를 실행한 뒤 ZIP을 `ai-training/`에 풀어 서버를 시작한다. ZIP에는 PT와
ONNX, 모델별 SHA256과 평가 지표가 들어 있다.

검사는 `python -m unittest discover -s tests -v`로 실행한다. 직접 설치했다면 먼저
`pip install -r requirements-dev.txt`로 테스트 의존성을 설치한다.

## 재현성과 발표 시 기록할 내용

- 데이터셋 이름, 버전, 라이선스, 이미지 수와 분할 수
- 학습 모델(`yolo11n.pt` / SH17·전이·시설은 `yolo11s.pt`), 입력 크기(기존 640 / 시설 960), seed 42, 실제 epoch
- 학습에 쓰지 않은 test 분할의 precision, recall, mAP50, mAP50-95
- 잘 되는 클래스와 표본 부족으로 약한 클래스
- AI 결과는 제안이며 안전관리자가 최종 확인한다는 앱 흐름

## 클래스 주의사항

Construction-PPE에는 `no_helmet`, `no_gloves`, `no_boots`, `no_goggle`가 있지만
`no_vest`는 없다. 따라서 안전조끼 미착용 판단을 이 데이터만으로 학습했다고 발표하면
안 된다. `no_vest`는 별도 데이터 또는 팀 자체 라벨을 추가한 뒤 활성화한다.
