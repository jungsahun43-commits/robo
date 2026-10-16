# v81 사전 조건: 원본 해상도 한 후보

v78의 FIT15/VAL6/CAL6/예약 TEST7과 제공 subpixel ROI 좌표를 유지한다. v79 실패 가중치를 쓰지 않고 원래 v62 가중치·정규화를 복제한 새 `corrosion_native_head` 100,593파라미터만 학습한다. 한 공통 encoder와 v16/승인 세 head/원래 v62 전체를 고정한다. 코드·JSON을 일반 commit/push한 뒤에만 실제 특징·gradient를 실행한다.

## 입력과 손실

원본 RGB를 축소하지 않고 512×512, stride384의 입력 크기 기반 격자로 순회한다. 마지막 경계를 포함하고 작은 축의 오른쪽/아래 RGB만 (124,116,104), 정답은 unknown255로 padding한다. 추론·CAL·TEST는 주석 위치에 관계없이 모든 tile을 실행하고 중첩 확률을 native 좌표에서 균등 평균한다.

사진의 알려진 음성/양성 고유 픽셀 수를 N0/N1, 입력 격자 포함 횟수를 C(x)로 둔다. 각 tile의 알려진 픽셀 BCE를 `0.5/(N_class*C(x))`로 가중해 더한다. unknown/padding은 손실·분모·gradient에서 제외한다. 같은 GT의 중첩 노출을 추가 정답으로 세지 않는다. 전체 mask가 없으므로 Dice를 적용하지 않는다.

## 실제 업데이트와 선택

30회 순회, 매회 저장한 순서로 15사진을 한 번씩 사용한다. 사진 안에서 microbatch 최대4 tile씩 순차 backward한 뒤 clip5와 optimizer를 한 번 실행한다. 총 **450 optimizer**, 사진 노출450회이며 tile·microbatch 횟수는 별도로 남긴다. FIT 주석 없는 tile은 손실0이라 건너뛸 수 있지만 VAL/CAL/TEST의 전체 grid는 GT로 생략하지 않는다.

AdamW lr1e-4/wd0.01, cosine30/min1e-6, seed8181, FP32 encoder→FP16 feature 경계→FP32 head, TF32 off다. v62 FIT 정규화는 다시 맞추지 않는다. 초기 epoch0과 매 epoch의 같은 coverage 가중 VAL 목적을 비교해 가장 이른 최소를 선택한다. epoch0이면 새 CAL/TEST 점수 없이 제외한다.

## CAL과 검증 기준

선택 epoch>0일 때만 CAL6 전체 grid의 제공 ROI에서 고정19cut(0.05 간격)의 균형 오류가 최소인 가장 작은 cut을 저장한다. 그 뒤 TEST7을 한 번 평가한다. 비교 기준은 원래 v62의 **같은 native grid, 원cut0.20**이다. v79의 whole512 기준 수치와 구분하며 비교에는 입력과 고정 임계값 정책의 차이가 포함된다.

각 class 지원 사진≥5/group≥3/native pixel≥375, pooled FP/FN≤20%, macro FP/FN≤25%, 최악 사진 FP/FN≤50%, 같은 grid 기준 대비 균형 정확도5%p 이상 개선을 모두 요구한다. source 통과 때만 원래 Ostrava15 TEST를 새 native grid/newcut 대 기존 v62 whole512/cut0.20으로 검사한다. Ostrava 2048 사진당 native25개+whole512 비교1개로 두 입력 보기이며 한 encoder 호출로 세지 않는다. FP/FN≤20%, IoU≥50%, 각 FP/FN 증가≤5%p/IoU 감소≤5%p를 유지한다.

두 단계 통과 때만 새 원본 CPU/GPU에서 native 확률 atol/rtol0.002와 cut 불일치≤0.1%를 검사한다. 이 연구는 API/C++/ZIP/APK 검사가 아니며 앱 alias·기본 지도·경고·승인 ZIP을 자동 교체하지 않는다.

## 보존과 범위

새 dense feature 파일/전체 부모 복제/전체 ZIP 확장/기존 파일 삭제 없이 native RGB·GT·coverage를 제한된 RAM에 둔다. 새 디스크 예산256MiB와 최소1GiB 여유를 검사한다. 모든 epoch head/optimizer·실제 장부·오류 부분 가중치를 보존한다. 작은 parent-bound delta의 부모/추가/결합 SHA와 디스크 복원 exact를 확인한다. 단독 앱 모델이 아니다.

TEST7의 sparse ROI 밖에는 FP 정답이 없다. 전체 mask IoU·전체사진 FP/FN20% 앱 통과·현장 독립성·미탐0 보장·장비 건강·부식 깊이로 확대하지 않는다. 비슷한 체크무늬 판의 촬영 상관, 물리 그룹 미검증, 누락83, 설명문5×5와 실제 ROI 차이 및 공개 TEST 반복 개발 한계가 남아 있다.

Rios/Netto/Roehl (2023), MendeleyDataV1 DOI10.17632/kcyn4nhv2c.1 CC BY4.0와 저자/출처/버전/좌표 표현 변경을 표시한다. Ostrava CC BY4.0, VT CC0, InsPLAD CC BY-NC3.0 및 모든 부모 자료 조건을 유지한다. 정확한 파일·코드 SHA와 기계 실행 조건은 `industrial-corrosion-native-v81-protocol.json`에 고정했다.
