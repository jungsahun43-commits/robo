# v23 실제 주석·상세 영역 학습 결과

2026-10-15. e0a06f2에 조건·코드를,18641c7에 실제 분리·중복 감사와 manifest SHA를 먼저 commit/push한 뒤 새 특징과 학습을 실행했다.

## 실제 학습 범위

**실제 optimizer 1920회**,30epoch. 검증 선택 epoch **3**, 선택 가중치의 예정 갱신192회.
VAL loss 2.546012 → 1.959180. v23의 두 집단·원본 사진 가중 VAL이므로 이전 실험의 loss와 직접 비교하지 않는다.
실제 배포 mask 원본 FIT35장/VAL10장, 영역312/88개다. 반복 추출이나 crop을 새 실제 사진 수로 세지 않았다. 새 합성은0개다.
같은 사진의 결함 영역과 주석상 결함이 없는 영역을 구분하도록 학습했다. source 정상 전체 사진은0장이며 주석상 음성을 부품 전체의 정상 작동으로 바꾸지 않았다. target 정상은 gradient 학습·VAL 선택에 넣지 않았고 기존 CAL/TEST만 사용했다.
기존 encoder와 모든 부모 분기·tensor·buffer·메타데이터·high projection은 고정했다. 새로운 fine projection/fusion만 갱신했고 잔차 입력은 모든 source/target에서0이다. 별도 encoder·가짜 category embedding을 추가하지 않았다.

## 고정 CAL 이후 공개 TEST

| 유형 | 방법 | 사진 오탐 | 사진 미탐 | 채택 조건 |
|---|---|---|---|---|
| mpdd_bracket_black | 현재v16 | 12.50% (4/32) | 63.83% (30/47) | None |
| mpdd_bracket_black | detail_only | 9.38% (3/32) | 89.36% (42/47) | False |
| mpdd_bracket_black | max_detail_parent | 15.62% (5/32) | 65.96% (31/47) | False |
| mpdd_tubes | 현재v16 | 3.12% (1/32) | 68.12% (47/69) | None |
| mpdd_tubes | detail_only | 9.38% (3/32) | 84.06% (58/69) | False |
| mpdd_tubes | max_detail_parent | 9.38% (3/32) | 62.32% (43/69) | False |

선택 방법: `{}`.

사진마다 같은9개 영역의 점수 최댓값으로 평가했고 원본 사진을 한 번씩 집계했다. TEST 정답 위치로 crop을 고르거나 TEST에 맞게 임계값·채택 조건을 바꾸지 않았다.
통과 후보가 있어도 새 패키지 CPU/GPU·실제 C++·모든 alias·원래 위치 PNG를 검사하기 전까지 기존v16-r1을 유지한다. 이 연구 결과 기록 시 새 패키지 API/C++/ZIP은0이다. 새 상세 지도를 target 기본 위치로 승격하지 않았다.

## 보존과 자료 요건 중단

기존 Python627파일 SHA 검사, 보호88378파일 stat 동일, 메타데이터 변경 재해시0개다. 보호 원본 전체 SHA를 모두 다시 계산한 것은 아니다. 실제 optimizer 장부·새 캐시 SHA·부모 전체 보존과 고정 high projection을 검증했다. source-only unit2개와 같은 회차의 geometry/mask/dispatch unit4개가 통과했다.
앞선v22는 중복 제외 후 브래킷 정상 FIT29장으로 사전 최소30장을 충족하지 못해 features/점수/optimizer0에서 중단했다. 조건을 낮추지 않았고 자료·실패 기록을 보존했다. v23은 target 정상 전체 사진 학습을 제거하고 같은 실제 source 사진 안의 주석상 양성/음성을 구분하는 별도 학습 설계다.
35FIT/10VAL의 파일 계열 예약, 흰 표시 Scratch6장 제외와 v22의5746 TEST 전체·8326전체/영역 참조 대조를 그대로 상속했다. 원래 photo 그룹의 근접 제외를 되돌리지 않았다.

## 한계

공개 TEST를 반복 개발에 사용한 연구 결과이며 독립 현장 성능이 아니다. source VAL은 다른 파일 계열10장이며 물리 부품·촬영 세션 독립성이 검증되지 않았다. 팀이45장 모든 mask 픽셀이나 실제 기계·전기 고장 진실을 검수한 것은 아니다. 과거 부모 모델의 근접 노출, KSDD1 결함18장, 미검수 전선 관계 라벨의 한계도 유지한다.
NEU 원 이미지 제공자와 abin24 주석 배포자를 인용하고, 명시된 학술 연구 전용/상업 금지 조건을 유지한다. 존재하지 않는 SPDX를 붙이지 않으며 원자료·가중치·실패·출력·보고서를 보존한다.
