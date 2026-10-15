# v29 기어 공간 특징 MIL 학습 조건

2026-10-15. 이전 FIT32개 조사와 별개로 전체 특징·gradient·학습·VAL/CAL/TEST 점수 전에 고정한 후보 하나다. v28 자료 분리와 기어 표면만의 범위를 그대로 둔다. 원래 전체 부품 실패와 TEST3,109개도 보존한다.

## 자료와 특징

FIT5,466/VAL1,440/CAL1,116/기어 TEST2,179패치, 연결 묶음8/3/3/4개다. 각 패치는 원래128×128 RGB를 bicubic256 정사각형으로 확대하고 ImageNet 정규화한다. 기존 공통 encoder 하나의 fine24@64×64를 FP16으로 보존한다. 라벨·GT에 따른 crop이나 별도 encoder는 없다.

전체10,201패치의 fine tensor는2,005,598,208bytes다. 저장 시작 전에 그 크기와3GiB 추가 여유를 요구한다. 파일별 ID·원본 이미지SHA·128→256 좌표·encoder체크포인트SHA·manifestSHA·protocolSHA를 묶는다. 매 입력에서 pooled fine72차원을 기존 고정 descriptor와 `atol=.002, rtol=.002`로 비교하고, FIT 첫batch의 low/high는 일반forward와 bitwise 동일함을 확인한다. 불일치 시 기록·중단하고 허용오차를 바꾸지 않는다.

정규화는 FIT 전체 patch/pixel에서 채널별 합·제곱합을 FP64로 구한 모평균·모표준편차다. 분산 하한0/표준편차 하한1e-3, FP32buffer,[-8,8] clamp를 쓴다. VAL/CAL/TEST는 FIT 통계만 사용한다.

## 후보·손실·학습

새 head는 Conv24→32(3×3,padding1)/GroupNorm4/GELU/Conv32→1(1×1)이다. 마지막층 weight/bias는0으로 초기화한다. 상위 `ceil(.10×공간크기)` logit의 평균을 사진 조각 logit으로 쓴다. 실제 저자 OK/nOK 패치 라벨의 BCE만 학습한다. class라벨을 pixel mask 정답으로 바꾸지 않는다.

기존 모든encoder·head·가중치·buffer·metadata는 고정하고 새 spatial head만 학습한다. batch32에서 OK/nOK16개씩, 라벨별 묶음 패치 수의 역수 가중치로 추출한다. cached fine 특징만 좌우·상하 각각.5 확률로 뒤집으며 실제RGB 증강/새 encoder 추출과 구분한다.

seed2929, AdamW lr1e-4/wd1e-4, cosine30epoch/min1e-6, clip5, FP32/TF32off. **30epoch×64=1,920 optimizer회**를 계획한다. 실제step장부·매epoch optimizer/scheduler/RNG를 저장하며 오류·중간 가중치와 재실행된 부분epoch의 실제step도 보존한다.

## 선택과 적용

VAL에서 라벨·묶음별 평균 BCE를 구한 뒤 두 라벨을 균등 평균한다. epoch0과 매epoch 중 가장 이른 엄격한 최솟값을 선택하며 epoch0은 배포하지 않는다. 선택 후 CAL라벨로 고정 minimax 오탐·미탐 임계값을 저장한다. 동률은 합계/0.5까지 거리/낮은값 순, strict `>`다. CAL저장 이후에만 TEST 새 특징·점수를 계산한다.

기어 TEST 두 라벨 각각20개 이상·연결그룹3개 이상, 선택epoch>0, 오탐·미탐 모두20% 이하를 요구한다. 통과 후보만 새 패키지CPU/GPU·실제C++·기존alias·기본위치PNG보존을 검사한 뒤 연구 전달물에 반영한다. 기존 브래킷/튜브의 전이·교체 실험이 아니며 이 단계에서 그 지표를 개선했다고 설명하지 않는다.

새 공간 출력은 weak patch class에 따른 학습 출력이다. 주석으로 검증된 결함 위치가 아니므로 기존 기본 위치 지도나 pixel기준을 바꾸지 않는다. 결과는128패치의 표면 마모 허용이며 전체 기어박스·장비 작동 건강·내부고장·임의 현장 전체사진 성능이 아니다. 같은 공개TEST를 이전 개발에서 본 이력과 적은 독립 부품·촬영 그룹 한계를 유지한다.

Re-GBC version1 CC BY4.0, [원배포본](https://www.kaggle.com/datasets/jhnnsbr/gearbox-components-remanufacturing-inspection), [저자 논문](https://arxiv.org/abs/2511.15440), DOI10.1109/ETFA65518.2025.11205777을 인용하고 기존 다른 자료의 이용조건은 그대로 둔다.
