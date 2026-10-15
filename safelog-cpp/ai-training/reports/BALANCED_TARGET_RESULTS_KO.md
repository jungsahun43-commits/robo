# v21 작은 영역 학습 비중 보강 결과

2026-10-15. 조건·코드를de35fee에 먼저 commit/push한 뒤 실제 학습했다.

## 실제 학습

실제 optimizer **1920회**,30epoch. 합성 VAL 선택 epoch **21**, 선택 가중치의 예정 갱신1344회.
새 loss의 VAL 1.979888 → 1.558172. 이전 v20과 loss 정의가 달라 loss 숫자 자체를 서로 비교하지 않는다.

기존 v20의 세밀한 판별부 구조와 VAL 초기값을 유지하고 픽셀 BCE의 양성·배경 평균을 사진마다 동등하게 반영했다. 정상 사진은 배경 평균만 쓴다. 새 fine projection·fusion만 학습했고 기존 모든 부모 가중치·buffer·메타데이터와 high projection은 고정했다.

FIT609/VAL162는 원래 정상·절차적 변형 자료를 그대로 썼다. 새 실제 자연 고장 라벨은0개다. 기존 coarse와 v20 fine 캐시6개를 SHA로 검증해 재사용했으며 새 encoder 특징 추출은0회다. TEST 점수는 새 CAL을 저장한 뒤 계산했다.

## FIT 학습 신호 진단

| 출력 logit gradient | 양성 절댓값 합 | 배경 절댓값 합 | 양성/배경 |
|---|---|---|---|
| previous_global_weighted | 0.042480 | 0.242271 | 0.175340 |
| per_photo_balanced | 0.238943 | 0.241512 | 0.989361 |

고정 FIT8입력의 학습 전 진단이며 모델 가중치 갱신0회다. 양성 신호가 커지는 것은 설계한 loss의 작동 확인이고 자연 결함 성능 개선의 증거가 아니다.

## 고정 CAL 이후 공개 TEST

| 유형 | 방법 | 오탐 | 미탐 | 조건 통과 |
|---|---|---|---|---|
| mpdd_bracket_black | 현재 v16 | 12.50% (4/32) | 63.83% (30/47) | None |
| mpdd_bracket_black | fine_only | 18.75% (6/32) | 72.34% (34/47) | False |
| mpdd_bracket_black | max_fine_parent | 21.88% (7/32) | 70.21% (33/47) | False |
| mpdd_tubes | 현재 v16 | 3.12% (1/32) | 68.12% (47/69) | None |
| mpdd_tubes | fine_only | 0.00% (0/32) | 94.20% (65/69) | False |
| mpdd_tubes | max_fine_parent | 3.12% (1/32) | 69.57% (48/69) | False |

선택 방법: `{}`.

통과 후보가 있어도 새 ZIP의 실제 CPU/GPU·C++ 연동과 원래 위치 PNG 보존을 검사한 뒤에만 반영한다. 이 결과 기록 시 새 API/C++/ZIP 검사는 각각0이며 현재 v16-r1을 유지한다.

## 보존·한계

기존 Python 615파일 SHA, 보호 88230파일 stat 동일, 메타데이터 변경 재해시0개를 확인했다. 보호 자료 전체 SHA를 다시 계산한 것은 아니다. 모든 부모 tensor/buffer/metadata·고정 high projection·실제 optimizer 장부·새 loss unit3개를 검증했다.

원래 target 위치 지도와 pixel 기준을 유지하고 합성 지도를 자연 결함의 기본 지도로 승격하지 않았다. 공개 TEST 반복 개발로 얻은 수치이며 독립 현장 성능이 아니다. 합성 VAL, 자연 손상 실재성 미검증, 물리 촬영 그룹과 부모의 과거 근접 노출, KSDD1 TEST 결함18장 및 v14 사람 검수 대기의 한계를 그대로 남긴다.

KSDD1 CC BY-NC-SA4.0·비상업 연구/인용/동일조건·DOI10.1007/s10845-019-01476-x를 유지한다. 원본·이전 가중치·실패 후보·보고서를 모두 보존했다.
