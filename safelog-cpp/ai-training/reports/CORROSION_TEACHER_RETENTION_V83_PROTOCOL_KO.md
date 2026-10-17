# v83 사전 조건: 전체 보기 응답 보존 한 후보

v82의 FIT 전용 비교에서 원v62도 whole512→native 입력으로 Ostrava BCE가0.05226→0.25518로 달라졌고, v81 native는0.33220이었다. Rios FIT에서는 v81 native가0.06715로 낮았다. 이 FIT 근거로 **원래 전체512 응답을 native 분기에 보존하는 증류 한 후보**를 고정한다. 미관측 성능 원인의 인과 증명이 아니다.

새 `corrosion_retained_head` 100,593파라미터를 명시적으로 **실패한 v81 선택 분기**에서 초기화한다. 실패 모델을 앱에 승격하는 것이 아니다. 부모는 여전히 v16+승인 세 head+원v62이며 모두 tensor/buffer/meta exact 고정한다. 정규화는 원v62와 같은 초기값 그대로다. 한 공통 encoder, 기존 surface/real_mask 계약, 별도 namespace·메서드를 유지한다.

## 고정 학습

- 입력은 v81과 동일한 native512/stride384이며 원래 ROI/역할/TEST/unknown을 바꾸지 않는다.
- 한 update에 Rios FIT 한 사진과 Ostrava FIT 한 사진을 짝짓는다. Rios15를 epoch마다 한 번, Ostrava60은 고정 순열을15장씩 순회해 사진당7~8회 사용한다. 30epoch, 실제450updates, Rios450+Ostrava450=900사진 노출이며 새900사진이 아니다.
- 손실은 `0.5 × Rios 알려진 class 균형 BCE + 0.5 × Ostrava 원v62 whole512 teacher 대 새native student의 class 균형 Bernoulli KL`이다. 둘 다 고유 native class 분모와1/coverage로 중첩 정답의 반복 가산을 막는다. fullnegative는 실제 있는 class만 쓴다. unknown/padding은0가중치다.
- Ostrava 원주석은 알려진 class별 KL 예산을 정한다. teacher 확률은 기존 모델 응답 보존용 soft 목표이며 검증된 자연 정답이 아니다. 원v62 FIT/VAL의 기존 whole512 특징 캐시만 재사용하고 teacher 입력을 다시 추출하지 않는다. VT나 새로운 mask BCE/Dice를 추가하지 않는다.
- 사진 안 microbatch4를 순차 backward하고 두 사진이 끝난 뒤 한 번 clip5/update한다. AdamW1e-4/wd0.01, cosine30/min1e-6, seed8383, FP32 encoder→FP16 feature 경계→FP32 head, TF32 off다. 여러 계수·구조·초기값을 탐색하지 않는다.

## 선택과 유지 조건

초기와 매 epoch의 `0.5×Rios VAL6 BCE 평균 + 0.5×Ostrava VAL15 teacher KL 평균`의 가장 이른 최소를 선택한다. 모든 VAL grid를 실행한다. epoch0이면 새 CAL/TEST 없이 제외한다.

선택 epoch>0일 때만 Rios CAL6의 고정19cut 중 균형 오류 최소·작은 cut을 저장하고 원TEST7을 한 번 평가한다. source의 class별 최소5사진/3그룹/375픽셀, pooled FP/FN≤20%, macro≤25%, 최악≤50%, 원v62 같은 native grid/cut0.20 대비 균형 정확도≥5%p를 모두 유지한다.

source 통과 때만 원Ostrava15 TEST를 새 nativegrid/newcut 대 원v62 whole512/cut0.20으로 검사한다. FP/FN≤20%, IoU≥50%, FP/FN 증가 각각≤5%p 및 IoU 감소≤5%p를 모두 요구한다. 사진당 native25+whole512비교1의 두 보기다. source만 골라 반영하거나 TEST 조건·cut을 바꾸지 않는다. 두 gate 통과 때만 fresh CPU/GPU 허용오차를 검사하며 API/C++/ZIP/APK·앱 alias·경고·기본 지도 교체는 별도 전달 검증 전 실행하지 않는다.

모든 epoch/RNG/optimizer/실제 장부·실패를 보존한다. 작은 originalv16-bound delta의 추가/결합/부모 SHA와 디스크 복원 exact를 검증한다. 새 디스크≤256MiB 및 최소1GiB 여유, teacher75개512확률은 RAM≤75MiB, Ostrava native RGB/GT/teacher는 사진 한 장씩 처리한다. 새 dense feature 디스크 파일/부모 전체 복제/전체 확장/기존 삭제0이다.

합성 unit에서 큰 fullmask count×uint16 coverage의 overflow를 발견해 float64 분모로 고쳤고 원형/실패 기록을 보존했다. 실제 source gradient 이전이며 학습 runtime 오류로 세지 않는다. unit3개가 통과했다.

공개 TEST 반복 개발, 사진7장·ROI밖 unknown·물리 촬영 그룹 미검증·누락83·설명5×5 불일치의 한계가 있다. 전체사진20% 앱 gate·독립현장·깊이·장비 건강·0미탐 보장이 아니다. Rios/Ostrava CC BY4.0, InsPLAD CC BY-NC3.0/VT CC0 및 모든 부모 조건·저자/출처/버전/변경 표시를 유지한다. 정확한 SHA와 기계 조건은 `industrial-corrosion-teacher-retention-v83-protocol.json`에 있다.
