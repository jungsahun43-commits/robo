# v33 사전 고정: 다른 부품 사이의 마모 라벨 공통 특징

## 질문과 범위

v32의 VAL만의 진단에서 부품별 정상/결함 구성 차이가 크고, 선택 기준만 바꾸면 큰 개선이 생긴다는 근거가 부족했다. 이번에는 실제 원저자 OK/nOK가 같은 **서로 다른 FIT 부품 묶음**의 특징을 가까이 두는 보조 손실과 그에 맞는 배치를 한 후보로 사용한다. 두 요소를 함께 바꾸므로 손실 하나의 독립적인 인과 효과로 해석하지 않는다. 원저자 논문의 문제의식을 참고한 별도 변형이며 원논문을 재현했다고 주장하지 않는다.

범위는 Re-GBC v1의 128×128 기어 표면 패치 마모 허용이다. 전체 설비 작동, 내부 고장, 임의 현장 전체사진, 결함 위치의 정답을 만들지 않는다. 원저자 OK에는 허용되는 흔적도 있을 수 있다. 모든 라벨과 제외자료는 유지한다.

## 자료·입력·저장

- 기존 v28 manifest를 유지한다. FIT5,466/VAL1,440/CAL1,116/원래 TEST2,179패치, 연결 묶음8/3/3/4개이며 독립 부품 수와 패치 수를 구분한다.
- 기존 v29의 원RGB128→bicubic256/ImageNet 정규화→같은 encoder stage3 fine24@64×64 FP16 캐시를 ID/RGB SHA/순서/좌표/encoder/manifest/protocol SHA로 확인해 사용한다. 새 RGB 특징·새 캐시는 만들지 않는다. 이전 FIT32의 정밀도 경계 허용은 atol .02/rtol .01이고, 학습 전 초기 VAL BCE 차이가 v28 대비 .005를 넘으면 중단한다.
- 현 디스크 약3.9GB에서 캐시를 재사용하고 작은 progress/best/부모 결합 delta만 저장한다. 시작 시1GiB 이상을 남기고 기존 원본·전체 가중치·실패·ZIP은 삭제하지 않는다. delta는 부모 파일/전체 state SHA와 추가/결합 state SHA를 검사하고 직렬화 후 모든 tensor/buffer/metadata가 정확히 같게 복원되는지 확인한다. 부모가 필요한 연구본이며 단독 앱 전달물이 아니다.

## 하나의 확정 구조·학습 조건

accepted v16의 기존 모든 분기·가중치·buffer·metadata와 일반 forward를 유지한다. v28의 기어 선형 분류부 및 FIT 정규화를 초기값으로 가져오고 stage3 뒤에 새 zero-output Conv24→4/GELU/Conv4→24 잔차 모듈을 둔다. 이후 원래 같은 후반 encoder 객체를 **한 번** 실행한다. 새 encoder/복제 backbone/hook/return_layers 변경이 없다.

동일한 후반 특징의 FP16 경계 및 mean/popstd/max3,072차원 descriptor를 v28 FIT 통계로 표준화해[-8,8]로 제한한다. 이 특징을 선형 분류부와 보조 손실에 공유한다. 보조 손실 내부에서는 각 특징을 L2 정규화하고 cosine/temperature .2를 사용한다. 같은 라벨·다른 묶음만 positive이며 같은 묶음의 쌍은 분자/분모 모두에서 제외한다. 묶음 ID는 배치/손실에만 쓰며 추론 입력·가짜 embedding으로 넣지 않는다.

학습 대상은 새 adapter220개와 기어 head3,073개, 총3,293개 파라미터다. 원래 backbone·기존 모든 분기·v28 통계는 고정/eval이다. 목적함수는 `BCE + 1.0*sum(head.weight²) + .1*cross_group_loss(T=.2)`이다. pixel mask나 위치 손실은 없다.

배치는 각 라벨마다2패치 이상인 서로 다른4묶음을 균일 비복원 선택한 뒤 묶음마다 서로 다른2패치를 선택해 총16개다. 한 배치에 사진 중복이 없고 배치 사이 재추출은 새 사진 수로 세지 않는다. 사진/feature 증강은 없다.

seed3333, AdamW adapter1e-4/head1e-5, weight decay1e-4, gradient clip5, 30epoch×64step=**계획1,920회**다. 각 LR은 `.01+.99*.5*(1+cos(pi*epoch/30))`를 곱하고 FP32/TF32off를 유지한다. 마지막 optimizer/RNG와 모든 실제 step 장부를 남겨 부분 epoch 재실행도 실제 횟수에 포함한다. 구체 값은 새 source gradient 전에 코드와 protocol JSON에 고정하고 commit/push한다.

새 모델/배치 합성 unit3개와 기존 cross-group loss unit3개를 확인한다. 실제 source gradient·가중치 갱신은 사전 등록 뒤에 실행한다.

## VAL·CAL·TEST·채택

초기와 완료 epoch의 실제 label/group macro BCE가 가장 작은 가장 이른 가중치를 선택한다. 보조 손실이나 TEST를 선택 기준으로 쓰지 않는다. **epoch0이면 학습 후보를 제외하고 새 CAL/TEST 점수도 계산하지 않는다.** 기존 선택/마지막 가중치의 반복 TEST 평가를 피한다.

선택 epoch>0일 때만 정상 CAL의 정렬 점수 `ceil(.85*n)-1`를 cut으로 고정하고 strict `>`로 판정한다. 저장한 checkpoint SHA/cut을 확인한 뒤 고정 TEST를 한 번 평가한다. 이는 이전 연구의 labeled minimax CAL과 다른 고정 정책이므로 수치 차이를 전부 학습 개선으로 설명하지 않는다. TEST를 보고 cut·계수·구조·선택 조건을 바꾸지 않는다.

채택은 선택epoch>0, TEST 각 라벨20개 이상·연결 묶음3개 이상, 오탐/미탐 모두20% 이하, 원래 state/메타데이터 보존이다. 통과한 후보만 새 패키지 CPU/GPU/실제 C++/기존 alias·위치 PNG 보존을 검사한다. 이전 검사 수를 이번 검사로 세지 않는다. 현재 브래킷·튜브나 위치 지도에는 이 연구를 바로 반영하지 않는다.

## 근거·한계·이용 조건

같은 공개 TEST를 사용한 개발 이력, 개발 분리 재편 이력, TEST4묶음의 작은 부품 수, 과거 부모 근접 노출/촬영 독립성 미검증의 한계가 남는다. 학습 효과나 현장 성능 개선을 미리 보장하지 않는다. near 사진의 다른 라벨은 자동으로 정답 오류로 바꾸지 않는다.

Re-GBC 공개 v1 CC BY4.0, 원저자 배포본과 DOI10.1109/ETFA65518.2025.11205777을 인용하고 변경과 버전을 표시한다. 기존 KSDD/NEU 등 다른 출처의 이용 조건도 각각 유지한다. [원저자 논문](https://arxiv.org/abs/2511.15440), [공개 배포본](https://www.kaggle.com/datasets/jhnnsbr/gearbox-components-remanufacturing-inspection).
