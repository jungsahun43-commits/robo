# v85 사전 조건: 실제 전체보기 문맥 입력 하나 추가

v82 FIT 입력·분기 비교와 v84 고정 FIT6장/18tile 좌표 검사를 근거로 한 후보를 고정했다. v83의 epoch1 선택/주석 FN25.1356% 실패는 그대로 보존한다. 계수·임계값을 바꾸거나 실패 모델의 좋은 지표만 반영하지 않는다.

## 바뀌는 입력과 구조

원v62의 같은 공통 encoder·기존 rust head가 전체 RGB512 보기에서 만든 확률을 새 native 분기의 **입력 특징**으로 준다. letterbox padding 제거→native 확률 복원→clampP[1e-6,1-1e-6]→logit/8→clamp[-1,1]→native tile crop/문맥0padding→128 bilinear 순서를 고정한다. source/date/category ID를 입력하지 않고, 모델 응답을 검증된 자연 정답으로 바꾸지 않는다.

새 corrosion_context_head는 실패v81 선택 분기의100,593파라미터와 정규화를 exact 복제하고, 문맥1→64의3×3 bias없는 convolution576개만0으로 추가한다. 원88채널 convolution은 그대로 계산한 뒤 문맥 경로를 더한다. 총101,169파라미터다. 기존 kernel reduction을 유지해 초기 출력 exact를 검사한다. constructor의 추가 초기화는 CPU fork_rng로 원래 RNG 흐름을 보존한다. 별도 encoder가 아닌 한 encoder의 whole/native 두 보기다.

실제 gradient 전에 Rios/Ostrava 고정 첫3FIT씩, 크기 기반 처음/가운데/마지막18tile에서 같은 FP16 특징을 원래 RustMaskHead 계산과 새 문맥 계산에 공유해 `torch.equal`, 최대차0을 요구한다. 새 정확도 평가가 아니다. 합성 unit3개는 초기 출력 exact/RNG 보존, 문맥 경로의 학습 가능성과 생산자 gradient 차단, 입력 범위/좌표 거절을 확인했다.

## 그대로 유지하는 v83 조건

- v61/v78 자료·정답·역할·reserved TEST·중복 제외를 유지한다. 새 자연사진/주석/분리/전역 감사0이다.
- v83 paired-orders.npz를 바이트 그대로 복사한다. 30epoch×15사진 pair=450updates, Rios450+Ostrava450=900노출이다. 원본은 Rios15/Ostrava60이다.
- 손실은0.5 Rios sparse class 균형 BCE +0.5 Ostrava class 균형 teacher KL, 고유 native class 분모/1coverage/unknown 제외다. teacher는 원v62 whole512 보존 목표이고 새 GT가 아니다. Dice/새 supervised mask 혼합/계수 변경0이다.
- seed8383, AdamW1e-4/wd0.01, cosine30/min1e-6, clip5, microbatch4 순차 누적 후 pair당1update를 유지한다. 초기 원가중치는 v83과 같은 **실패v81**이며 v83 선택/마지막 가중치를 쓰지 않는다.
- VAL은 같은0.5 Rios6 BCE 평균+0.5 Ostrava15 teacher KL 평균, 초기포함 earliest minimum이다. epoch0은 새 CAL/TEST 없이 제외한다.
- 선택 epoch>0일 때만 Rios CAL6 고정19cut 저장 후 TEST7을 한 번 평가한다. class별5사진/3그룹/375픽셀, pooled FP/FN≤20%, macro≤25%, 최악≤50%, 원v62 같은 grid/cut0.20 대비 균형 정확도≥5%p를 유지한다.
- source 통과 때만 원Ostrava15 TEST: FP/FN≤20%, IoU≥50%, FP/FN 증가≤5%p/IoU 감소≤5%p를 모두 요구한다. 두 gate 통과 때만 fresh CPU/GPU 허용오차 검사다. 조건·TEST cut 변경/좋은 source만 채택0이다.

## 실제 입력 수와 보존

Ostrava FIT/VAL의75개 원래 whole512 응답은 기존 캐시에서 한 번 얻어 teacher와 문맥에 공유한다. 추가 encoder 추출0이다. Rios FIT15/VAL6은 전체보기1회씩 추출해 native RGB/GT/coverage/문맥을 합해 RAM≤512MiB에 둔다. CAL/TEST 및 새 CPU/GPU 검사는 허용 단계에 원RGB로 전체보기1+모든 native tiles를 실행한다. Ostrava 유지 비교의 parent whole 응답은 문맥 producer와 공유하므로 중복 추출하지 않는다. 사진당Nnative+1whole이며 한 forward로 세지 않는다.

부모 v16/승인 세 head/원v62 모든 tensor/buffer/meta·점수·기본 위치와 기존 forward_surface/RuntimeRealMaskMain.forward_real_mask를 exact 보존한다. 작은 originalv16-bound delta의 추가 namespace는 corrosion_context_head이며 파일/부모/추가/결합 SHA와 직렬화 복원 exact를 검사한다. 단독 앱 모델이 아니다. 기존 v83의 검증·선택·gate runner는 이 새 프로세스 안에서 명시적 v85 함수로 연결하며 기존 파일을 수정하거나 완료된 v83을 재실행하지 않는다.

모든 epoch/optimizer/RNG/실제 장부/오류 부분 가중치를 보존하고 새 저장≤256MiB·최소1GiB 여유를 유지한다. 새 dense feature 디스크 파일/부모 전체 복제/전체 ZIP 확장/기존 삭제0이다. source나 두 연구 gate 통과도 앱 alias·경고·기본 지도·ZIP 자동 교체가 아니며 별도 실제 전달 검증이 필요하다.

Rios ROI 밖 unknown에는 FP 정답이 없고 source7사진으로 전체사진20% 앱 통과·독립현장·0미탐·부식 깊이·장비 건강을 주장하지 않는다. 공개 반복 TEST 개발·물리 촬영 그룹 미검증·누락83/설명5×5 차이와 모든 원자료 한계를 유지한다. Rios/Ostrava CC BY4.0, InsPLAD CC BY-NC3.0/VT CC0 및 모든 부모 저자·출처·버전·변경 조건을 보존한다. 정확한 코드·파일 SHA는 industrial-corrosion-context-v85-protocol.json에 있다.
