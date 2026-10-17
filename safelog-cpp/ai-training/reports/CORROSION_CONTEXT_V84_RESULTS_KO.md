# v84 FIT 문맥 좌표·수치 조사

`4b04f57`에 코드·정확한 조건을 먼저 일반 push했다. 고정 FIT는 Rios/Ostrava 원ID 정렬 첫3장씩이며, 입력 크기 기반 grid의 처음/가운데/마지막으로18개 tile을 고정했다. 정답이나 TEST 내용으로 crop을 고르지 않았다.

원래 v62의 전체512 확률에서 letterbox padding을 제거하고 native 확률로 bilinear 복원한 뒤 `clampP(1e-6,1-1e-6)→logit/8→clamp[-1,1]`로 문맥 특징을 만들었다. native tile crop/0 문맥 padding/128 bilinear 변환 순서를 검사했다. 이 특징은 모델 응답이며 새 자연 정답이 아니다.

- Rios 고정 FIT3장의 원RGB 재해독·새 encoder3입력/기존 head3보기를 실행했다.
- Ostrava 고정 FIT3장은 기존 FIT whole 캐시를 SHA/ID/raw/mask/encoder/protocol로 검증해 기존 head3보기만 계산했다.
- 18개 native crop roundtrip exact, 문맥 padding0,128특징 유한값·범위 검사를 통과했다.
- 부모 전체 tensor/buffer 전후 SHA `61677d5f1d038006cbcfc83894af03a90b3105055ca94d5eef3c40567dc7332f`와 묶은 파일/캐시 SHA가 그대로다.
- 새 학습 분기·gradient·optimizer·VAL/CAL/TEST·정확도 점수·시각 열람은0이다. 좌표 unit3개도 통과했다. 성능 개선의 증거가 아니다.

다음 v85는 이 실제 전체보기 특징을 기존 native 분기에 전달하는 한 후보를 별도 사전등록한다. 원래88입력 convolution을 그대로 계산하고 zero초기화한 문맥 convolution을 더하는 구현을 우선 검토해 초기 출력의 수치 exact를 검사한다. 기존 v83의 seed/샘플링/초기 원가중치/손실/VAL/CAL/guard를 유지하고 문맥 입력 하나만 추가할지 구체화한다. 아직 실제 v85 학습은0이다.

원래 ROI/역할/예약 TEST·unknown과 Rios/Ostrava CC BY4.0, InsPLAD CC BY-NC3.0/VT CC0 및 모든 부모 조건을 보존한다. whole/native는 한 encoder의 두 입력 보기이며 한 forward로 세지 않는다. source7사진을 전체사진 앱·현장·0미탐·깊이·장비 건강 검증으로 확대하지 않는다.
