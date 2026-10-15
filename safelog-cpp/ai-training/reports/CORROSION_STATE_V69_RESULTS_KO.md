# v69 실제 4상태 학습 결과 — 채택 제외

## 실제 수행

`c8a2bb0`에 코드·조건을 먼저 push하고 v68의 실제 FIT **276사진**과 내부 검증 **60사진**으로 새 분기만 **30epoch×64 = 실제 1,920 optimizer** 업데이트했다. 같은 사진을 반복 뽑은 7,680 encoder 입력을 새 자연 사진 7,680장으로 세지 않는다.

새 분기는 원래 polygon 외관 0 Good/최소 부식·1 Fair·2 Poor·3 Severe를 구분한다. 원래 v16·승인 3작업·연구 v62와 공통 encoder는 고정했다. 입력의 비율을 유지한 512 letterbox와 ignore padding을 사용했으며 추가 dense feature 파일을 만들지 않았다. FIT/검증 uint8 입력·정답 336MiB를 RAM에 두고 특징은 batch마다 online으로 계산했다.

내부 검증 손실은 초기 **2.16480697 → 1.22721627**, **epoch6 / 384회 업데이트** 가중치를 선택했다. 마지막 epoch30 손실 1.39434248은 선택하지 않았다. 마지막 실제 가중치·각 epoch head/optimizer·update 장부도 보존했다.

CAL60에서 고정 softmax temperature1.0/argmax 정책과 지표를 저장한 뒤 원TEST44를 한 번 평가했다. TEST에서 임계값이나 구조를 다시 고르지 않았다.

## 원래 embedded 좌표의 픽셀 지표

| 원래 상태 | IoU | recall |
|---|---:|---:|
| Good / 최소 부식 포함 | 85.19% | 88.38% |
| Fair | 34.59% | 66.54% |
| Poor | 33.95% | 60.04% |
| Severe | **0.00%** | **0.00%** |

macroIoU **38.43%**, macroDice **48.52%**, 전체 픽셀 정확도 **83.65%**다. TEST에서 Severe를 포함한 사진은 12개이며 실제 Severe 1,894,856픽셀의 올바른 Severe 예측은 0개였다. 이것을 전체 사진 단위 미탐률이나 장비 전체 안전 성능으로 바꾸지 않는다.

`Fair 이상` 대 `Good/최소 부식`으로 집계한 픽셀 오탐은 **11.62%**, 미탐은 **19.05%**다. 이 이분 집계만 골라 채택하지 않는다. Good은 미미한 녹을 포함하므로 이 수치는 모든 녹 대 무부식 오차도 아니다. 다른 자료인 Ostrava의 이진 부식 지표와 직접 같은 과제처럼 비교하지 않는다.

**macroIoU≥40%와 각 Fair/Poor/Severe recall≥25% 조건에 미달해 후보 전체를 제외했다.** 조건을 낮추거나 Severe를 사후 삭제하지 않는다. 새 API·CPU/GPU 패키지·C++·ZIP 검사와 새 ZIP은 모두 0개다. 조건부 fresh CPU/GPU 44장 검사는 source gate 실패로 실행하지 않았다. 기존 앱과 역할4 승인 전달물을 유지한다.

## 보존과 근거

실제 학습 runtime 오류 0개다. 부모 v62의 모든 tensor/buffer/metadata가 정확히 같고, 이는 v16·승인 3작업 보존도 포함한다. 디스크 delta 복원 전체 상태도 정확히 같았다. 기존 Python **775파일 SHA**, 보호 **330,430파일 stat**, 실제 update 장부 1..1920을 확인했다. 보호 원본 전체 SHA를 모두 재계산한 것은 아니다.

연구 실패 delta: `runs/industrial-corrosion-state-v69/trained/final-delta.pt`

- 파일 SHA256: `6989f555dffe17718236f996a6044aa77b9214b3a788a9f51d8ade3660419060`
- 결합 tensor 상태 SHA256: `092420b1672881082a2fe8953873b162fb59588c7ca8ff8d7fabc25c344a2c69`

파일 SHA와 결합 상태 SHA는 서로 다르며 새 앱 API modelSha로 배포하지 않았다. 원래 v16이 필요한 연구 차이본이고 단독 앱 모델이 아니다.

모델 특징 입력은 FIT normalization276, 학습7,680, epoch0 포함 VAL1,860, CAL60, TEST44로 합계9,920이다. 일반 forward와 fine 경로 low/high의 첫 FIT4 비교는 추가4입력이며 새 사진이 아니다. 원TEST의 고정 1,000 proxy-group bootstrap도 같은 예측의 불확실성 참고값이며 추가 모델 학습·독립 물리 부품 보정이 아니다.

## 다음 진단

학습을 바로 반복하지 않고 v70에서 FIT 해상도/클래스 노출과 내부 VAL만 검사한다. 선택 epoch6과 마지막 epoch30을 같은 VAL60 특징에서 비교하며 새 CAL/TEST 점수는 만들지 않는다. 마지막 가중치를 승격하지 않는다. 이 진단이 희소 상태의 손실 기여 부족을 뒷받침할 때만 FIT로 고정한 한 가지 가중 손실 후보를 별도 사전 등록한다. 원인으로 확정하거나 TEST에 맞는 여러 가중치/샘플링을 탐색하지 않는다.

자료는 Bianchi·Hebdon Virginia Tech v2, 데이터 DOI **10.7294/16624663.v2**, 논문 **10.1061/(ASCE)CP.1943-5487.0001045**, 공식 데이터 CC0다. 기존 InsPLAD CC BY-NC3.0와 Ostrava CC BY4.0 등 모든 부모 조건을 유지한다. 공개 TEST 개발 결과·날짜 문구·미확인 촬영/부품 그룹으로 인해 독립 현장 성능이라고 주장하지 않는다.
