# v71 비교 조건: FIT의 상태별 손실 예산을 맞추기

v70의 FIT·내부 VAL 진단을 근거로 **손실의 상태별 가중치 하나만** 바꾼다. v69의 마지막 head를 승격하지 않고, 고정한 새 후보를 평가한다.

## 고정한 한 가지 변경

원래 사진별 present-class CE+Dice의 각 상태 항에 `1/(4*a_k)`를 곱한다. `a_k`는 FIT 그룹 균등 샘플링에서 `상태 존재 / 존재 상태 수`의 기대값이다. 고정된 가중치는 Good/Fair/Poor/Severe 순서로 **0.6987341772151905, 0.7156439066551432, 1.0739299610894926, 4.160804020100503**이다. 사진 안에서 가중치를 다시 정규화하지 않는다. 실제 gradient가 같아진다는 뜻은 아니다.

같은 목적을 학습과 VAL 선택에 사용한다. 새 가중 VAL 값은 이전 비가중 VAL 값과 직접 비교하지 않는다. 가중치·손실 계수·샘플러·구조를 여러 개 돌려 TEST에서 고르지 않는다.

## 비교를 위해 잠근 조건

- v69의 **epoch0 head**와 FIT 전용 normalization을 그대로 초기값으로 읽는다. 선택 epoch6 또는 마지막 epoch30으로 초기값을 고르지 않는다.
- 입력·정답의 uint8 바이트 SHA와 1,920×4 draw 파일 SHA가 v69와 정확히 같아야 한다.
- 구조·100,692 학습 파라미터·seed6969·AdamW/lr/cosine·batch4·30×64회는 같다. normalization 특징 계산을 반복하지 않는다.
- 동일한 공통 encoder와 기존 v16·승인 3작업·연구 v62 tensor/buffer/metadata를 고정하고 새 분기만 학습한다. v69 실패 가중치와 로그도 별도로 보존한다.
- ignore/padding·원좌표 복원·원래 TEST44·CAL 저장 후 TEST 순서를 유지한다. Good/최소 부식을 모든 녹이 전혀 없는 정답으로 바꾸지 않는다.

source 연구 기준은 v69와 같은 macroIoU≥40%, Fair 이상 픽셀 FP/FN 각각≤20%, 각 Fair/Poor/Severe recall≥25%, 상태별 TEST 사진≥5/그룹≥3 및 선택 epoch>0이다. 조건을 완화하지 않는다. source가 통과할 때만 별도 fresh CPU/GPU 비교를 실행한다. 전체 정상 사진 GT가 없으므로 새 앱 사진 gate 통과나 자동 alias/ZIP 교체를 주장하지 않는다.

추가 dense feature 캐시 없이 RAM 입력과 online 특징을 사용하며 원본·이전 실패·epoch별 checkpoint를 보존하고 디스크 reserve 1GiB를 유지한다.
