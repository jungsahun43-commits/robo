# v81: native 입력 한 후보의 완료 결과

`06041a9`에 코드·단일 프로토콜을 사전 일반 push한 뒤 실제 학습했다. v78의 FIT15/VAL6/CAL6/예약 TEST7과 제공 ROI는 그대로다. 원본 RGB는 축소하지 않은 native512/stride384, 추론은 주석과 무관한 전체 grid 및 native 중첩 확률 평균이다. 원래 v62 가중치·정규화를 복제한 새 100,593파라미터 분기만 학습했고 v16/승인 세 head/원 v62 전체를 고정했다.

사진마다 알려진 양성·음성의 고유 native 분모와 1/coverage 가중치로 손실 예산을 각각0.5로 뒀다. unknown/padding은 분모·gradient에서 제외했고 Dice를 쓰지 않았다. 30회×15사진으로 **실제 optimizer450회**, 사진 노출450회, FIT encoder tile입력5,340회/microbatch1,500회다. 새 자연사진450장이 아니다.

초기 VAL1.85532593→0.31717791, epoch10/150업데이트 가중치를 선택했다. 초기·매 epoch에 같은 coverage 가중 목적과 고정 VAL 전체 grid를 사용했다. 모든 epoch/optimizer/RNG와 장부를 보존했다.

CAL6의 고정19cut에서 0.60를 저장한 뒤 TEST7을 한 번 평가했다. 비교 기준은 **원래 v62, 같은 native grid, 고정cut0.20**이다. 과거 v79의 whole512 수치와 구분한다. 비교에는 임계값 정책 차이도 있어 학습 단독 효과의 인과 증명이 아니다.

| 제공 native ROI 안의 지표 | v62 같은 grid | v81 |
|---|---:|---:|
| 음성 FP | 14.8929% | 6.9786% |
| 양성 FN | 65.6995% | 18.0871% |
| 주석 ROI 안에서만 IoU | 29.8571% | 76.5730% |

균형 정확도 차이는 27.7634%p다. macro 사진 FP/FN은 6.9786%/18.2173%, 최악 사진 FP/FN은 28.5750%/49.3000%다. 사전 source point gate 결과는 **True**다.

기준은 각 class≥5사진/group≥3/native pixel≥375, pooled FP/FN≤20%, macro≤25%, 최악 사진≤50%, 기준 대비 균형 정확도≥5%p다. 좋은 사진·지표만 반영하거나 TEST cut/조건을 바꾸지 않았다.

| 사진 | FP | FN |
|---|---:|---:|
| rios/134 | 28.5750% | 0.0250% |
| rios/144 | 9.7000% | 0.0000% |
| rios/19 | 0.4500% | 49.3000% |
| rios/225 | 0.0000% | 35.0000% |
| rios/53 | 0.0000% | 14.9167% |
| rios/59 | 5.6500% | 22.7114% |
| rios/65 | 4.4750% | 5.5682% |

proxy 연결그룹 bootstrap 참고95% FPR/FNR 구간은 [[0.014714285714285714, 0.06369718309859156], [0.15017857142857144, 0.32161143448146734]]이다. 실제 물리부품/현장 독립성 조정값이 아니다.

source 통과에 따라 Ostrava15 원TEST retention을 검사했다. 기존 whole512/cut0.20 FP/FN/IoU=9.5801%/3.7154%/70.8030%, 새 nativegrid/newcut=12.2393%/6.0911%/64.3302%다. 유지 gate=False. 2048 원본당 native25개+whole512비교1개로 두 입력 보기이며 한 encoder 호출이 아니다.

조건부 fresh CPU/GPU 검사는0이다. 두 gate를 모두 통과하지 않아 새 장치·전달 검사를 실행하지 않았다.

최종 연구 gate=False. Ostrava IoU 감소6.4727%p가 사전5%p 제한을 넘어 **전체 후보를 제외**했다. source 수치만 골라 반영하거나 retention 조건을 완화하지 않았다. 기존 승인 앱/overlay·원 v62 위치 기준은 유지했다. 새 API/C++/ZIP/APK 검사는 모두0이다. source 연구 통과도 앱 alias·경고·기본 지도·ZIP 자동 교체로 연결하지 않는다.

## 검증과 한계

부모 전체 tensor/buffer/metadata 및 복제 정규화 보존, delta 디스크 복원 exact를 확인했다. 기존 Python806파일 SHA/보호330,972파일 stat, 장부1..450과 각 epoch15사진을 검증했다. 전체 보호 원본 SHA를 모두 재계산한 것은 아니다. 첫 FIT4 tile low/high 일반 forward bitwise exact와 추가 비교 encoder8입력을 구분했다. 새 합성 unit3개가 통과했고 학습 runtime 오류는0이다.

연구 delta 5,638,581bytes, 파일SHA `f5938468e7acb4b5dfbb7025b0c986c8e5a2cfc90bfe3c0f5ec6431fa983cf10`, 결합 tensor상태SHA `91180ead165861bb4856e8e251843c766c2ca6890cf3067d623cb7e07724afc5`다. parent가 필요한 차이본이며 단독 앱 모델이나 servedAPI SHA가 아니다. 새 run 디스크 사용은 94,412,981bytes, 원본/실패/로그/캐시/ZIP 삭제0이다.

제공 ROI 밖 unknown에는 FP 정답이 없다. 표의 IoU는 전체 mask IoU가 아니며 TEST7을 전체사진20% 앱 통과·현장 독립 성능·0미탐 보장·부식 깊이·기계/전기 건강으로 확대하지 않는다. 누락83, 원설명5×5와 실제 ROI 차이, 비슷한 체크무늬 판 및 물리 촬영 그룹 미검증, 공개 TEST 반복 개발 한계는 남아 있다.

이번450을 포함한 추적 일부 연구 optimizer 합59,530은 실패 포함 부분합이며 전체 역사 학습 총량이 아니다. 현재 브래킷/튜브 및 기존 추가12유형 중8개만20% 이내인 상태는 그대로다. 이 native 입력 우선 연구를 완료했고 같은 TEST에 맞춘 추가 구조/stride/loss/cut 탐색을 하지 않는다. 후속 가설에는 독립 FIT 근거나 검수된 자연 주석이 필요하다.

Rios/Netto/Roehl (2023), MendeleyDataV1 DOI10.17632/kcyn4nhv2c.1 CC BY4.0와 저자/출처/버전/좌표 표현 변경을 표시한다. Ostrava CC BY4.0, VT CC0, InsPLAD CC BY-NC3.0 및 모든 부모 자료 조건을 유지한다.
