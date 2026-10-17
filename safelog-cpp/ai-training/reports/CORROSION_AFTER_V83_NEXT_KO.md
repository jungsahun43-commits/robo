# v83 이후 조건부 후속 메모

현재 v83은 사전 등록한450update 학습을 실행 중이다. 이 문서는 다음 실험의 정식 프로토콜이나 결과가 아니다. v83이 끝나기 전에 다른 GPU 작업이나 추가 모델을 시작하지 않는다. source/retention/device 결과를 실제 산출물과 장부로 확인하고, 완료 연구를 재실행하거나 TEST에 맞춰 조건을 바꾸지 않는다.

## v83이 두 gate를 통과한 경우

조건부 새 원본 CPU/GPU 검사까지 실제 완료됐는지 확인한다. 연구 입출력 범위와 source7사진·부분 ROI 정답·공개 반복 개발의 한계를 유지한다. 앱 alias/경고/기본지도를 자동 교체하지 않는다. 연구 전달이 필요하면 기존 승인 세 head/모든 alias/원래 PNG 보존·실제 C++ 연결과 제한된 입력 계약을 먼저 별도 코드/JSON에 고정한다.

## v83이 제외되거나 epoch0인 경우

정해진 판단을 유지하고 좋은 source 지표나 마지막 epoch만 승격하지 않는다. 기존 v82 FIT 근거는 **같은 원래 분기에서도 whole512→native 입력으로 응답이 달라진다**는 것이다. v83 teacher KL만으로 이 차이를 보완했는지 FIT/VAL 성분을 읽되 새 TEST 점수로 계수·cut을 다시 고르지 않는다.

독립적으로 가능한 다음 후보 준비는 **실제 전체 RGB 보기의 문맥 응답을 native tile 좌표에 전달할 수 있는지 확인하는 FIT 한정 기하·수치 조사**다. 현재 새 조사 코드/정식 정책/모델/특징/gradient/optimizer는0이다. 가짜 source/date/category embedding을 추가하거나 원래 정답을 바꾸는 방향이 아니다. 원v62의 같은 공통 encoder·기존 rust head가 원RGB 전체512 보기에서 만든 확률을 이미지 특징으로만 검토하고 검증된 새 GT로 세지 않는다.

먼저 조사 조건을 코드·JSON에 고정·사전 push한다. Rios/Ostrava 각각 원ID 정렬 첫3FIT, 각 사진의 입력 크기 기반 grid에서 처음/가운데/마지막의 서로 다른 tile을 고정하는 한 가지 규칙을 검토한다. GT나 TEST 내용으로 crop을 고르지 않는다. whole512 letterbox의 padding 제거→원본 좌표 확률 복원→tile crop→고정128 feature 좌표 변환 순서를 선언하고 unknown/RGB padding과 혼동하지 않는다. 확률의 logit/clamp/스케일이 필요하면 먼저 정확히 고정한다. 모델 학습이나 성능 비교 없이 좌표 roundtrip/유한값/기존 출력·state exact만 검사한다. 기존 Ostrava FIT whole cache는 ID/raw/geometry/encoder/protocol/SHA 검증 후 재사용하고 전역 RGB/중복/분리 감사는 반복하지 않는다.

향후 실제 새 입력 후보를 쓴다면 기존 fusion88 입력을 보존하고 추가 문맥 채널 가중치0으로 원래 초기 출력 exact를 먼저 검증하는 한 구조를 별도로 검토할 수 있다. 이 문장은 구조 확정이나 학습 승인 조건을 대신하지 않는다. 하나의 초기값/학습대상/샘플링/손실/VAL/CAL/기존 gate/장치/delta/저장량을 실제 코드·JSON에 새로 고정한 뒤만 gradient를 실행한다. 여러 문맥·크기·구조·계수를 동시에 돌리지 않는다.

두 입력 보기는 같은 encoder를 여러 번 호출하는 것이며 한 forward로 세지 않는다. teacher 응답은 입력 특징/보존 목표이고 실제 자연 주석이 아니다. source7사진으로 wholephoto 앱20%·현장 독립성·0미탐·장비 건강·부식 깊이를 선언하지 않는다. 현재 source/retention 기준과 기존 v16/승인 세 head/원v62 tensor-buffer-meta/기본 위치 및 RuntimeRealMaskMain 계약을 보존한다.

원본·실패·모든 epoch·로그·캐시·보고서·ZIP을 보존하고 새 dense feature 디스크 파일/부모 전체 복제/불필요 전체 확장 없이 최소1GiB 여유를 유지한다. Rios/Ostrava CC BY4.0, VT CC0/InsPLAD CC BY-NC3.0 및 모든 부모 자료의 출처·저자·버전·변경 조건은 그대로다.
