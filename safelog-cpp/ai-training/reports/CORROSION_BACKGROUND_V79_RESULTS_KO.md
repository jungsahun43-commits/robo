# v79: 같은 사진 안의 양성·배경 sparse ROI 학습, 전체 후보 제외

## 실제 학습

`f194051`에 코드·조건을 사전 일반 push했다. 공통 encoder와 v16·승인 세 head·원래 v62 분기는 고정했다. v62에서 복제한 새 100,593파라미터 분기만 FIT15장으로 30번 순회했다. 배치4/4/4/3, epoch당4회로 **실제 optimizer120회**, FIT 사진 노출450회다. 새 자연사진450장을 확보하거나 1,920회 학습한 것으로 세지 않는다.

512 전체 비율 유지 입력과 제공 native ROI를 사용했다. 주석 양성·알려진 음성의 BCE 평균을 .5/.5로 두고 unknown/padding은 손실과 분모에서 제외했다. 정규화는 새 분기에 복제한 v62 FIT 값 그대로다. 초기 VAL1.5067120→0.3431603의 epoch14/56업데이트 가중치를 선택했다. 모든 epoch·optimizer·장부를 보존했고 학습 runtime 오류는0이다.

## 고정 CAL 후 source TEST 한 번

CAL6장의 제공 ROI에서 사전19cut 중 .60을 저장한 뒤 예약 TEST7장을 한 번 평가했다. 고정 v62 비교 기준은 원cut .20이다.

| native 제공 ROI 안의 지표 | v62 | v79 |
|---|---:|---:|
| 주석 음성 FP | 8.5214% | 11.2750% |
| 주석 양성 FN | 60.6995% | 35.0000% |
| 제공 ROI 안에서만 계산한 IoU | 36.2165% | 58.4181% |

균형 정확도는11.4730%p 개선됐지만 FN35%가20%를 넘었다. 사진별 최대 FP60%, FN82.31%로 사전50% 제한도 위반했다. macro 사진 FN35.7653%도25% 제한에 미달했다. **전체 후보를 제외**했고 낮은 cut을 다시 고르거나 좋은 사진만 채택하지 않았다.

이 수치는 사진 전체의 오탐·미탐 또는 전체 mask의 IoU가 아니다. sparse ROI 밖 unknown에는 FP 정답이 없다. TEST7개 연결 그룹의 bootstrap 참고95% 구간은 FP1.34~28.01%, FN14.31~56.48%로 넓고, 물리부품/현장 독립성 조정값도 아니다. 107/115의 유사한 판무늬와 출처 설명·실제 ROI 크기 차이도 남아 있다. 전체 사진20% 앱 gate는 항상 false다.

source gate가 실패해 **새 Ostrava retention TEST 입력0, fresh CPU/GPU0, API/C++/ZIP/APK검사0**이다. selection의 retention false는 새 유지 검사를 수행해 실패했다는 뜻이 아니라 조건부 검사가 실행되지 않았다는 뜻이다. 기존 승인 앱/overlay 및 source v62 기준을 유지했다.

## 검증과 보존

- 부모 전체 tensor/buffer/metadata·새 정규화 초기값 보존, delta 디스크 복원 exact를 확인했다.
- 기존 Python800파일 SHA, 보호330,860파일 stat, 장부1..120 및 FIT450노출을 확인했다. 전체 보호 원본 SHA를 모두 재계산한 것은 아니다.
- 첫 FIT4개 비교 low/high bitwise exact, 추가 일반/probe encoder8입력을 별도로 기록했다. 새 loss unit3개가 통과했다.
- 실패 delta5,638,581bytes, 파일SHA `e7c070e6ae002cbbe0e68589f3df2aa951a571f8b062dad60cfbf67f2d375a41`, 결합 tensor상태SHA `d238ed7eece6ec58d98e278875b7231f94fe66f3682373d1f251d70fee1e4dc7`다. 단독 앱 모델 또는 servedAPI SHA가 아니다.
- 이번 optimizer120, 추적 일부 연구 합59,080은 실패 포함 부분합이며 전체 역사 학습 총량이 아니다. 기존 브래킷/튜브 및 추가12유형 중8개만20% 이내인 상태는 그대로다.

다음 v80은 FIT15의 GT 좌표·크기만으로 전체512 축소와 입력 기반 native512/stride384 grid의 기하 지원을 비교한다. 새 RGB/encoder/점수/VAL_CAL_TEST/학습 없이 하나의 입력 가설을 준비하며 성능 개선 증거로 부르지 않는다. 동일 loss 가중치나 TEST cut을 재탐색하지 않는다.

Rios/Netto/Roehl, MendeleyDataV1 DOI10.17632/kcyn4nhv2c.1, CC BY4.0. 원저자·출처·버전·좌표 표현 변경과 Ostrava/InsPLAD/VT 등 모든 부모 이용조건을 유지한다.
