# v80 FIT 기하 조사 완료, 다음은 native 입력 한 후보의 프로토콜

## 조사 결과

`0fd08a6`에 조사 코드·조건을 먼저 push한 뒤 FIT15의 기존 ROI·원본 크기로만 계산했다. native/기존512 GT SHA는 v78과 정확히 같았다. 새 RGB 해독·encoder 특징·모델 점수·gradient·optimizer·VAL/CAL/TEST 계산·사진 시각 열람은 모두0이다.

입력 크기만으로 정한 native512/stride384 grid는 FIT15에서200tiles다. 가로·세로 시작점은0부터384 간격, 끝이 맞지 않으면 `축길이-512`를 마지막으로 추가한다. 축이512보다 작으면0에서 시작하고 오른쪽·아래를 padding한다. 입력 grid를 GT로 고르지 않는다.

| 기하량 | 알려진 음성 | 알려진 양성 |
|---|---:|---:|
| 기존 전체512의 stride4 label bin | 1,071 | 1,041 |
| native grid의 stride4 label bin 노출(중첩 포함) | 9,157 | 8,877 |
| 고유 native 주석 pixel | 60,400 | 59,600 |
| grid에서 주석 pixel 노출(중첩 포함) | 116,920 | 112,040 |

양성 지지가 있는 tile129, 음성 지지가 있는 tile151이며 모든 알려진 native pixel이 최소 한 tile에 포함된다. 중첩 노출은 새 정답이나 새 자연사진이 아니다. stride4 bin은 실제 encoder 반응·수용영역·정확도 측정이 아니며 개선을 보장하지 않는다. 새로운 모델 프로토콜/초기값/optimizer는 아직 없다.

좌표 unit3개는 마지막 경계·가느다란 축의 전체 coverage, 중첩에서 native GT 재구성, unknown/padding 보존을 확인했다. 기존 v79는 전체 제외 상태다. TEST cut/loss 가중치 재탐색을 하지 않는다.

## v81에서 먼저 구체화할 한 방향

아래는 **미확정 제안**이다. 실제 입력·학습조건을 새 코드 JSON에 확정하고 commit/push한 뒤에만 새 특징/gradient를 실행한다. 여러 tile 크기/stride/구조를 병렬 탐색하지 않는다.

1. v78 manifest·역할·reserved TEST·원 ROI/unknown을 그대로 둔다. 새로운 자연사진이나 물리 독립 자료를 확보한 것이 아니다. 누락83/시각화 사본과 설명문5×5 차이는 보존한다. 기존 완결 지문/graph/분리를 다시 돌리지 않는다.
2. 입력 하나는 **원본 RGB 무축소 native512/stride384**이다. 부족한 오른쪽/아래 RGB만(124,116,104)로 padding, target은255. 추론과 CAL/TEST는 GT와 무관한 전체 grid를 전부 실행하고, native 위치에서 중첩 확률을 균등 평균한다. GT로 TEST crop을 선택하거나 틀린 부분만 추가 입력하지 않는다.
3. 한 공통 encoder와 별도 새 namespace 분기 하나를 쓴다. 초기값은 원래 v62 등 하나를 명시하고 실패 v79를 슬쩍 승격하지 않는다. v16/승인 세 head/v62 모든 tensor/buffer/meta/기준/기본 위치·forward_surface/forward_real_mask 계약은 exact 보존한다. 가짜 source/date/category 입력과 별도 encoder는 쓰지 않는다.
4. sparse 주석의 양성·음성 손실 예산을 정확히 정의한다. 검토 가능한 방식은 사진별 native 고유 양성/음성 분모를 유지하고, tile에 중복 포함된 pixel의 가중치를 `1/coverage`로 두어 같은 정답을 반복 가산하지 않는 방식이다. unknown/padding은 양쪽 분모/gradient에서 제외한다. 전체 mask가 없으므로 Dice 또는 unknown을 배경으로 채우지 않는다.
5. 모든 tile의 학습 graph를 GPU에 쌓지 않는다. 사진당 microbatch를 순차 처리해 gradient를 모은 뒤 하나의 update를 하는 방식 등을 한 가지로 고정한다. FIT의 주석 없는 tile은 손실0이지만 CAL/TEST grid는 GT로 생략하지 않는다. 회차/실제update수·원사진/반복tile노출·microbatch 수를 따로 기록한다. 동일 클래스/사진 샘플링, 학습률/손실/선택도 먼저 확정한다.
6. 초기와 각 epoch의 같은 objective VAL을 비교해 earliest minimum을 고른다. epoch0 선택은 새 학습 개선으로 승격하지 않고 CAL/TEST 새 점수 없이 제외한다. 정확한 normalization/FP16 경계/좌표 원상 복원/부모 및 delta 검증을 고정한다.
7. CAL 정책 저장 후 source TEST7을 한 번만 평가한다. source의 실제 음성/양성 ROI FP/FN만 계산하고 unknown coverage는 FP로 부르지 않는다. source point 기준은 v79의 각 class≥5사진/group≥3/native pixel≥375, pooled FP/FN≤20%, macro FP/FN≤25%, 최악 사진 FP/FN≤50%, 비교 기준 대비 균형 정확도5%p 개선을 유지한다. 새 입력의 기여와 학습의 기여를 구분하기 위해 고정 v62의 **같은 native grid** 비교 여부·비교 cut도 사전에 정한다. 기존 whole512 v62/v79 결과와 혼동하지 않는다.
8. source gate 통과 때만 Ostrava retention을 검사한다. 기존 v62의 원래 whole512/cut.20 기준과 새 native grid/cut의 좌표를 명시한다. 서로 다른 입력 보기라면 사진당 encoder 호출 수를 정확히 구분한다. 기존 fullmask pixel FP/FN≤20%, IoU≥50%, FP/FN 증가각5%p/IoU감소5%p 이내 조건을 완화하지 않는다. source 실패면 새 retention TEST를 낭비하지 않는다.
9. 두 gate를 통과한 경우만 새 원본 CPU/GPU 오차 검사를 한다. source7사진은 wholephoto20% 앱 통과 표본이 아니다. source 통과만으로 alias·경고·기본지도·앱ZIP을 자동 교체하지 않는다. 별도 연구 전달이 필요하면 CPU/GPU·실제 C++·모든 기존 alias·기본 PNG 보존과 제한된 입출력 계약을 먼저 고정한다.
10. bbox/분류를 mask로 바꾸지 않고 원 ROI 표현을 유지한다. source7TEST·촬영 그룹 미검증·유사 체크무늬 판·반복 공개 TEST 개발 한계를 숨기지 않는다. 실제 현장 독립성/전체 장비 건강/정밀 부식 깊이/0미탐 보장은 없다.

추가 dense feature 디스크 파일/435MB 부모 복제/전체 ZIP 확장/기존 원본·실패·로그 삭제 없이 bounded RAM·online/streaming과 parent-bound 작은 delta를 쓴다. 시작 전 여유를 확인하고 ≥1GiB reserve를 유지한다. 모든 오류·부분 가중치·실제 optimizer 장부를 보존한다.

Rios/Netto/Roehl MendeleyDataV1 DOI10.17632/kcyn4nhv2c.1 CC BY4.0/저자·출처·버전·변경, Ostrava CC BY4.0, VT CC0, InsPLAD CC BY-NC3.0 및 모든 부모 자료 조건은 유지한다.
