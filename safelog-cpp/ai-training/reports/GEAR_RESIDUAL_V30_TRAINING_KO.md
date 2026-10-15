# v30 고정 전체 특징 판별부와 제한된 공간 보정

## 근거와 초기값

같은 VAL에서 v28 전체 특징 판별부의 BCE .587897이 v29 공간 단독 판별부의 .653942보다 낮았다. 전체 특징을 유지하면서 학습되는 공간 보정을 더하는 후보 하나를 등록한다. TEST 임계값·혼합 비율·구조 조합을 탐색하지 않는다.

현재 앱용 accepted v16의 모든 기존 가중치·buffer·metadata를 보존한다. v28 연구 체크포인트의 global 선형 판별부와 FIT 정규화는 고정한다. v29 연구 체크포인트의 convolution/GN와 FIT fine 정규화로 새 공간 판별부를 초기화하고 마지막1×1층 weight/bias는0으로 다시 초기화한다. 두 초기값은 이전에 채택하지 않은 연구본이며 앱 반영과 구분한다. 초기 새 VAL logit이 고정 global logit과 bitwise 같고 VAL BCE도 원래 v28과1e-6 이내인지 확인한 뒤 학습한다.

## 고정 구조와 학습

한 공통 encoder의 global3072 descriptor와 fine24 공간 특징을 사용한다. 최종 logit은 `global_logit + .5*tanh(local_MIL_logit)`이다. local은 Conv24→32(3×3)/GN4/GELU/1×1→1과 상위10% logit 평균이다. 보정 절댓값.5는 학습 전에 고정하며 결과에 맞춰 바꾸지 않는다.

새 spatial head만 학습하고 기존 encoder·전체 판별부·두 FIT 정규화는 고정한다. v28 분리 FIT5,466/VAL1,440/CAL1,116/기어 TEST2,179를 그대로 둔다. 기존 v29 fine와 v26에서 온 global descriptor 캐시를 ID·SHA로 결합하므로 새 RGB/encoder 특징 추출은0이다.

seed3030, batch32에서 OK/nOK16개씩, 라벨별 묶음 균등 효과의 가중 추출, cachedfine H/Vflip.5다. pooled descriptor는 공간 순서 통계라 그대로 사용한다. 실제RGB 증강이나 새 자연 사진으로 세지 않는다. actual patch OK/nOK의 BCE만 학습하고 pixel mask나 정답 위치를 만들지 않는다.

AdamW lr1e-4/wd1e-4, cosine30epoch/min1e-6, clip5, FP32/TF32off, **30epoch×64=1,920 optimizer회**를 계획한다. 매epoch optimizer/scheduler/RNG와 실제step을 남기며 재실행된 부분epoch의 실제step도 세고 실패를 보존한다.

## 선택·채택

VAL은 라벨·묶음별 평균 BCE를 균등 평균한다. 초기epoch0과 매epoch 중 가장 이른 엄격한 최솟값을 선택한다. epoch0은 이전 v28 초기값을 유지한 것이므로 새 학습 개선/배포로 세지 않는다.

선택 이후 CAL 라벨의 고정 minimax FPR/FNR 임계값을 저장하고 strict `>`를 적용한다. TEST 점수는 CAL 저장 후에만 계산한다. gear TEST라벨 각각20개 이상·연결그룹3개 이상, 선택epoch>0, 오탐·미탐 모두20% 이하, 기존 상태와 고정global 상태 완전 보존을 요구한다. 통과 후보만 새 패키지CPU/GPU·실제C++·기존alias·기본위치PNG보존 검사 후 연구 전달물에 반영한다.

범위는128×128 기어 표면 마모 허용이다. 전체 부품·장비 작동 건강·내부고장·임의 현장 전체사진·검증된 위치 성능으로 확대하지 않는다. 기존 브래킷·튜브를 교체하지 않는다. 기존 전체부품 실패, 적은 TEST묶음4개와 반복 공개TEST 개발·부모 과거근접노출의 한계를 유지하며 독립 현장 검증을 주장하지 않는다.

Re-GBC version1 CC BY4.0, [원배포본](https://www.kaggle.com/datasets/jhnnsbr/gearbox-components-remanufacturing-inspection)과 DOI10.1109/ETFA65518.2025.11205777을 인용하고 기존 다른 자료 이용조건도 유지한다.
