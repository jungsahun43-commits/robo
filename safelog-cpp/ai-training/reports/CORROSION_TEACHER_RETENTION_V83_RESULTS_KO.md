# v83: 원래 전체보기 응답 보존 학습 결과

`d72242d`에 코드·조건을 사전 일반 push한 뒤 한 후보만 학습했다. v82 FIT만의2×2 비교가 입력·가중치 변화에 따른 출력 차이를 보여 원v62 whole512 응답을 native 분기에 보존하는 증류를 정했다. 이 FIT 진단은 새 VAL/CAL/TEST 점수·gradient·optimizer0이며 TEST 실패 원인의 인과 증명이 아니다.

새 corrosion_retained_head100593파라미터만 실패v81 선택 분기에서 명시적으로 초기화했다. 실패 모델을 승격한 것이 아니다. 기존 v16/승인 세 head/원v62와 정규화는 exact 고정했다. 한 모델·공통 encoder·기존 surface/real_mask 계약을 유지했다.

Rios 실제 sparse BCE와 Ostrava 원v62 whole512 soft 응답에 대한 class 균형 KL을0.5/0.5로 썼다. 고유 native class 분모·1/coverage·unknown255 제외를 유지했다. teacher는 보존 정규화 목표이며 검증된 새 자연 주석이 아니다. 실제 **450optimizer**, Rios450+Ostrava450=900사진 노출, Rios tile5,340/Ostrava tile11,250/microbatch4,650이다. 원본은 기존 Rios15/Ostrava60을 재사용했다. teacher75개 head보기는 기존 캐시에서 얻어 새 encoder 추출0이고 새75사진도 아니다.

초기 두 목적 VAL=0.31237138→0.29712676, epoch1/15업데이트를 선택했다. 매번 같은 Rios6/Ostrava15를 사용했다. 서로 다른 목적의 v81 VAL 숫자와 직접 개선 비교하지 않는다. 모든 epoch/optimizer/RNG·실제 장부를 보존했다.

CAL6 고정19cut 중0.75를 저장한 뒤 예약 TEST7을 한 번 평가했다. 비교 기준은 원v62 같은 native grid/cut0.20이다.

| 제공 native ROI 안의 지표 | 원v62 같은 grid | v83 |
|---|---:|---:|
| 음성 FP | 14.8929% | 5.4071% |
| 양성 FN | 65.6995% | 25.1356% |
| ROI 안에서만 IoU | 29.8571% | 71.0266% |

macro FP/FN=5.4071%/25.2983%, 최악 FP/FN=28.0750%/69.2500%, 균형 정확도 차이25.0248%p다. source gate=False.

class별 사진≥5/group≥3/native pixel≥375, pooled FP/FN≤20%, macro≤25%, 최악≤50%, 기준 대비 균형 정확도≥5%p를 그대로 적용했다. 개별 사진·좋은 지표만 고르거나 TEST cut/기준을 바꾸지 않았다.

source가 통과하지 않아 새 Ostrava retention TEST0이다. 미실행을 유지 검사 수행 후 실패로 설명하지 않는다.

최종 결합 연구 gate=False. 기존 승인 앱·alias·경고·기본 위치·원v62를 유지했다. 새 API/C++/ZIP/APK 검사는0이다. source 점수만 골라 반영하지 않고 source 통과만으로 앱을 자동 교체하지 않는다.

조건부 fresh CPU/GPU 검사: 0회(두 gate 동시 통과 조건 미달).

## 검증과 보존

부모 전체 tensor/buffer/meta·복제 정규화와 초기 분기 복제 exact, delta 디스크 복원 exact를 확인했다. 기존 Python812파일 SHA·보호331,090파일 stat와 실제 장부1..450/각 Rios30회/Ostrava각7~8회를 검증했다. 보호 원본 전체 SHA를 모두 재계산한 것은 아니다. 학습 runtime 오류는0이다. 합성 unit3개 통과 전 uint16 곱 overflow 원형과 수정 기록은 별도로 보존했고 실제 학습 오류로 세지 않았다.

연구 delta5,638,709bytes, 파일SHA `2527fb4120fa3edbd56d3ccea618658656bcb493a056b9ab0bb8148f9561b238`, 결합 tensor상태SHA `49030aa4826b05d43a4e9f9867fa08f2156414b5f2eae6de7c69104d5b03fa99`. parent가 필요한 차이본이며 단독 앱 모델/servedAPI SHA가 아니다. 새 저장96,333,809bytes/256MiB 이하, 원본·실패·모든 epoch·로그·캐시·ZIP 삭제0이다.

이번450을 포함한 추적 일부 연구 optimizer59,980은 실패 포함 부분합으로 전체 역사 총량이 아니다. TEST7 sparse ROI 밖에는 FP 정답이 없고 ROI IoU를 전체 mask IoU로 부르지 않는다. 7사진을 wholephoto20% 앱 통과·독립현장·0미탐·부식 깊이·전체장비 건강으로 확대하지 않는다. 공개 TEST 반복 개발, 물리 촬영 그룹 미검증, 유사 판무늬, 누락83/설명5×5 차이의 한계와 기존 브래킷/튜브 지표는 그대로다.

Rios/Netto/Roehl2023 MendeleyV1 CC BY4.0 DOI10.17632/kcyn4nhv2c.1, OstravaFric ZenodoV2 CC BY4.0 DOI10.5281/zenodo.11235637, InsPLAD CC BY-NC3.0/VT CC0와 모든 부모 이용조건·저자/출처/버전/변경 표시를 유지한다.
