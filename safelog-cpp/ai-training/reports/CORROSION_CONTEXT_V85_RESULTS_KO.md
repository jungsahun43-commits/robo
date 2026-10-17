# v85: 전체 RGB 문맥 입력 한 후보의 결과

`e0e995e`에 단일 코드·정확한 조건을 사전 일반 push했다. v84는 고정 FIT6장/18tile의 좌표·유한값·부모 state 보존을 확인한 조사이며 성능 검증이 아니다. 새 encoder3/기존 캐시 parent head3/새 parent head3보기와18문맥 필드만 계산했고 gradient/optimizer/VAL/CAL/TEST0이었다.

새 corrosion_context_head101169파라미터는 실패v81 원분기100593 exact+0초기화 문맥 convolution576이다. 원88입력 convolution을 그대로 계산한 뒤 문맥 경로를 더한다. 고정 FIT6장/18tile의 초기 old/new logit이 torch.equal, 최대차0을 실제 확인한 뒤 gradient를 시작했다. 기존 v83과 같은 초기 원가중치/seed8383/paired-orders 바이트/손실/VAL/CAL/guards를 유지하고 실제 전체보기 응답 특징 하나만 추가했다.

문맥은 원v62 전체RGB512 확률의 padding 제거→native 복원→clampP/logit÷8/clamp[-1,1]→tile/0padding→128 특징이다. 원 RGB가 제공하는 모델 응답이며 새 자연 GT·가짜 source/date/category ID가 아니다. producer는 고정·detach했고 한 common encoder를 whole/native 두 보기로 호출했다.

실제30epoch/**450optimizer**, Rios450+Ostrava450=900사진 노출, Rios tile5,340/Ostrava tile11,250/microbatch4,650다. 새900사진이 아니다. 목표는0.5 sparse BCE+0.5 원래 whole응답의 class균형 KL, 고유 native class 분모/1coverage/unknown 제외로 고정했다. teacher 응답을 전문가 주석으로 세지 않는다.

초기 VAL0.31237138→0.29594964, epoch4/60업데이트를 선택했다. 초기포함 같은 두 VAL 목적의 earliest minimum이다. 모든 epoch/optimizer/RNG/장부를 보존했다.

CAL6 고정19cut 중0.35를 저장한 뒤 TEST7을 한 번 평가했다. 기준은 원v62 같은 native grid/cut0.20이다.

| 제공 native ROI 안 지표 | v62 | v85 |
|---|---:|---:|
| FP | 14.8929% | 11.2679% |
| FN | 65.6995% | 16.9879% |
| 주석 ROI 안에서만 IoU | 29.8571% | 74.6111% |

macro FP/FN=11.2679%/17.2004%, 최악 FP/FN=56.6750%/49.0500%, 균형 정확도 차이26.1683%p, source gate=False.

source가 통과하지 않아 새 Ostrava retention TEST0이다. 미실행을 수행 후 실패로 설명하지 않는다.

최종 연구 gate=False. class별 최소5사진/3group/375pixel, pooled FP/FN≤20%/macro≤25%/최악≤50%/기준 대비 균형 정확도≥5%p 및 Ostrava FP/FN≤20%/IoU≥50%/각 FP/FN 증가≤5%p/IoU 감소≤5%p를 그대로 적용했다. 좋은 source만 반영하거나 TEST cut/조건을 변경하지 않았다.

조건부 fresh CPU/GPU: 0회(두 gate 동시 통과 조건 미달).

Ostrava FIT/VAL75 whole응답은 기존 캐시에서 teacher/문맥에 공유해 새 encoder0이다. Rios FIT15/VAL6은 whole1회씩 준비했고 CAL/TEST/새 장치는 사진당 whole1+모든 native tiles를 실행했다. Ostrava 유지 기준은 이미 계산한 whole 문맥 응답과 공유해 추가 중복 encoder0이다. 전체 실측 입력은 verification의 단계별 로그에 있으며 두 보기를 한 forward로 세지 않는다.

부모 전체 tensor/buffer/meta·정규화와 디스크delta 복원 exact, 기존 Python818파일 SHA/보호331,210파일 stat/실제1..450장부·각 사진 반복수를 검증했다. 모든 원본 전체 SHA 재계산 주장은 아니다. 학습 runtime 오류0, 새합성unit3개 통과다. 새 API/C++/ZIP/APK0이며 기존 승인 모델·alias·경고·기본 위치는 유지했다.

연구 delta5,641,398bytes, 파일SHA `d04194627df3eb1a30ea0c57990a3ef5103f883f3507e49559dc85c1b15f0c3b`, 결합 tensor상태SHA `6c0cdaa450fd5e65b927e7da9addab0d088f0103034923cff465f5abb163a829`다. originalv16 parent가 필요하며 단독 앱 모델/servedAPI SHA가 아니다. source나 연구 gate 통과를 자동 앱교체로 연결하지 않는다.

이번450 포함 추적 일부 연구 optimizer60,430은 실패 포함 부분합이며 전체 역사 총량이 아니다. source7사진의 sparse ROI 밖에는 FP 정답이 없고 ROI IoU를 전체 mask IoU·전체사진20% 앱 통과·독립현장·0미탐·부식 깊이·장비 건강으로 확대하지 않는다. 공개 TEST 반복 개발, 물리 촬영 그룹 미검증, 유사 판무늬, 누락83/설명5×5 차이와 기존 브래킷/튜브 한계를 유지한다.

Rios/Netto/Roehl MendeleyV1 CC BY4.0 DOI10.17632/kcyn4nhv2c.1, OstravaFric ZenodoV2 CC BY4.0 DOI10.5281/zenodo.11235637, InsPLAD CC BY-NC3.0/VT CC0 및 모든 부모 저자·출처·버전·변경 조건을 유지한다.
