# v20 세밀한 특징 학습 결과

2026-10-15. 단일 후보 조건과 코드를0c8c709에 먼저 commit/push한 뒤 실제 학습·평가했다.

## 실제 학습

30epoch / 실제 optimizer **1920회**. 합성 VAL에서 고른 epoch는 **20**(선택 가중치에 반영된 예정 갱신1280회)다.
VAL loss 1.402781 → 1.348272. 새 fine projection 변경=True, 처음0인 fine 입력 가중치 L1=38.431324.

같은 공통 encoder의 stage3 fine24와 기존 low/high·설비 복원 잔차를 사용했다. 새 fine projection·fusion만 갱신했고 기존 backbone/분기/high projection/부모 메타데이터는 그대로다. FIT609/VAL162는 v19의 고정 정상·절차적 변형 자료이며 자연 고장 정답을 새로 추가한 것은 아니다.

## 고정 CAL 이후 공개 TEST

| 유형 | 방법 | 오탐 | 미탐 | 조건 통과 |
|---|---|---|---|---|
| mpdd_bracket_black | 현재 v16 | 12.50% (4/32) | 63.83% (30/47) | None |
| mpdd_bracket_black | fine_only | 15.62% (5/32) | 68.09% (32/47) | False |
| mpdd_bracket_black | max_fine_parent | 15.62% (5/32) | 78.72% (37/47) | False |
| mpdd_tubes | 현재 v16 | 3.12% (1/32) | 68.12% (47/69) | None |
| mpdd_tubes | fine_only | 12.50% (4/32) | 92.75% (64/69) | False |
| mpdd_tubes | max_fine_parent | 9.38% (3/32) | 65.22% (45/69) | False |

선택 방법: `{}`.

선택된 후보가 있어도 새 패키지 CPU/GPU·실제 C++와 원래 위치 PNG 보존을 검사하기 전까지 기존 v16-r1 전달물을 유지한다. 합성 위치 지도를 자연 결함의 기본 위치 정답으로 승격하지 않았다.

## 검증과 한계

기존 Python 611파일 SHA 검사, 보호 88194파일 stat 동일·메타데이터 변경 0개 재해시. 보호 원본 전체 SHA를 모두 다시 읽었다는 의미는 아니다. 실제 optimizer 장부·새/기존 캐시 SHA·모든 부모 tensor/buffer/metadata·고정 high projection을 확인했다. 새 unit3개 통과. 이 단계의 새 API 검사0, C++ 작업0, ZIP0이다.

캐시별 사진 수, fresh FP16 low/high/복원 잔차 최대 차이와 bit 일치는 verification JSON의 fine_cache_receipts에 남겼다. 허용치는 학습 전에 atol5e-4/rtol1e-3으로 고정했고 기존 캐시를 재생성하지 않았다.

첫 sandbox 백그라운드 실행은 stdout/stderr0byte, operation lock 없이 종료됐고 실제 학습0회였다. 종료된 PID를 확인한 뒤 같은 코드·조건을 host에서 실행했다. 원인은 확정하지 않았으며 원래 로그·관찰 기록을 보존했다.

공개 TEST 반복 개발 결과이며 독립 현장 성능이 아니다. 합성 VAL은 실제 자연 결함 VAL을 대신하지 않는다. 물리 촬영 그룹 미검증과 부모의 과거 근접 자료 노출, KSDD1 결함18장 표본 한계, v14 관계 라벨 사람 검수 대기는 해소되지 않았다. 같은 TEST에 임계값을 맞추거나 통과 조건을 완화하지 않았다.

KSDD1 CC BY-NC-SA4.0 비상업 연구/인용/동일조건 및 DOI10.1007/s10845-019-01476-x를 유지한다. 원자료·가중치·기존 실패 후보·ZIP은 모두 보존한다.
