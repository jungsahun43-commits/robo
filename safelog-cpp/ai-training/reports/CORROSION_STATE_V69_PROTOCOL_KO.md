# v69 금속 외관 4상태 분기의 사전 고정 조건

## 목적과 정답 범위

v68을 통과한 원저자 JSON의 embedded 이미지와 수동 polygon 주석으로 **Good/최소 부식, Fair, Poor, Severe** 네 가지 사진 외관 상태를 학습한다. 원래 v66 native PNG 계약의 실패는 보존한다. 이 실험은 상태별 위치 연구이며 전체 사진 정상/고장, 전기·기계 건강 또는 교량 전체 안전 진단 정답을 만들지 않는다. 다른 자료의 이진 부식 mask에 없는 상태를 임의로 붙이지 않는다.

## 한 후보

- 기존 v16·승인 3작업과 연구 v62를 모두 복원·고정한다. 동일한 공통 encoder의 fine24/low40/high960만 사용한다.
- 새 `corrosion_state_head` 한 개가 v62의 projection/fusion을 복사하며 마지막 출력만 4채널·0 초기화다. **새 분기 100,692개 파라미터만** 학습한다. 기존 분기·normalization·buffer·metadata·기본 위치 기준은 그대로다.
- 전체 내장 RGB의 비율을 유지해 512 캔버스에 넣고, RGB padding은 (124,116,104), 정답 padding은 ignore 255다. 정답은 nearest resize하며 GT로 crop을 고르지 않는다.
- 새 normalization은 FIT만의 FP16 경계 특징을 online으로 읽어 float64 합/제곱합으로 구한다. 표준편차 하한 .001, 표준화 범위 [-8,8]이다. 기존 v62 normalization은 바꾸지 않는다.
- 손실은 사진별로 존재하는 상태의 **클래스 평균 categorical CE + soft Dice**를 평균한다. ignore/padding 픽셀은 두 항 모두에서 제외한다. Good을 모든 녹이 없는 이진 음성으로 쓰지 않는다.
- seed 6969, batch4, AdamW lr .0003/weight decay .01, cosine30/min .000001, gradient clip5, FP32/TF32 off/threads4, **30epoch×64 = 실제 1,920 optimizer 계획**이다. FIT 그룹을 균등하게 뽑은 고정 replacement draw를 사용한다.

## 선택과 평가

내부 검증 60항목의 사진별 손실을 연결 그룹별로 평균해 epoch0과 각 epoch를 비교하고, 가장 먼저 나온 엄격한 최솟값을 선택한다. epoch0이면 CAL/TEST 새 점수를 계산하지 않고 제외한다.

선택 epoch가 0보다 크면 CAL60에 **softmax temperature1.0/고정 argmax**를 적용하고 그 정책과 CAL 지표를 저장한다. 이 네 상태 위치 과제에는 전체 정상 사진 CAL이 없으므로 사진용 85% quantile을 적용하거나 TEST에서 임계값을 찾지 않는다. 저장 후 원TEST44를 한 번 평가한다. 확률 지도에서 padding을 잘라 원래 embedded 크기로 bilinear 복원하고, 원래 polygon 정답과 비교한다.

원좌표 4×4 confusion, 상태별 IoU/Dice/recall, macroIoU를 보고한다. `Fair 이상` 대 `Good/최소 부식`의 픽셀 FP/FN도 보고하지만 이를 모든 녹 대 무부식 또는 전체 사진 FP/FN으로 설명하지 않는다.

소스 연구 조건은 선택 epoch>0, TEST 각 상태 사진≥5/연결 그룹≥3, macroIoU≥.40, Fair 이상 픽셀 FP/FN 각각≤20%, Fair/Poor/Severe 각 recall≥.25 및 부모 exact 보존이다. 고정 1,000회 proxy-group bootstrap은 불확실성 참고값이며 독립 물리 부품 보정을 했다는 뜻이 아니다. TEST 점수에 맞춰 조건·손실·구조·임계값을 다시 바꾸지 않는다.

통과 시에만 TEST44의 원래 내장 이미지를 CPU와 새 GPU에서 다시 읽어 512 확률의 atol/rtol .002 및 512 argmax 차이 비율 .001을 검사한다. 이는 native pixel 완전 일치·Android 성능·패키지 연결 검사가 아니다. 소스가 통과해도 전체 정상 사진이 없으므로 앱 사진 gate 통과나 기존 alias 교체, 새 C++/ZIP 전달을 자동 수행하지 않는다.

## 저장·보존

FIT/검증의 uint8 입력·정답만 512MiB 이하 RAM에 보관하고 encoder 특징은 매 batch online 계산한다. 추가 dense feature 파일이나 435MB 부모 복제를 만들지 않는다. CAL/TEST는 허용 단계에만 online으로 읽는다. 장치 비교용 TEST 확률은 256MiB 이하 RAM에만 둔다.

epoch별 head·optimizer·실제 update 장부·오류 시 부분 checkpoint를 보존한다. 원래 v16에 묶인 작은 delta에 기존 승인/연구 분기와 새 분기를 저장하고, 부모 tensor/buffer/metadata 및 디스크 복원 exact를 검증한다. 모든 원본/실패/캐시/ZIP은 보존하며 디스크 reserve 1GiB를 유지한다.
