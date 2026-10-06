# 기존 손상 항목 성능 보존 비교 계획

작성: 2026-10-07. 아래 조건은 새 학습과 검증 전에 고정한다.

## 실험 이유

직전 박락 음성 표본 비교에서 추가 학습을 한 두 조건 모두 DACL 철근 노출 AP가 원본 모델보다 약 0.05 낮았다. 표본 보강 조건만의 효과라고 단정할 수 없으며, 공유 특징을 다시 학습하는 과정에서 기존 항목의 판별 능력이 바뀌는지 확인할 필요가 있다.

## 이번에 바꾸는 한 가지

원본 `0773` 모델을 고정된 교사로 사용한다. 학생 모델과 교사에 **같은 증강 TRAIN 사진**을 넣고, 기존 7개 사진 항목 중 균열·박락을 제외한 5개 항목의 확률 분포가 지나치게 달라지지 않도록 Bernoulli KL 손실을 추가한다.

- 대조군: 증류 손실 가중치 0.
- 성능 보존군: 증류 손실 가중치 1, 온도 2, 온도 제곱으로 보정.
- 두 조건 모두 매 TRAIN 배치에서 교사를 한 번 실행한다.
- 원본 정답이 알려진 5개 항목에만 손실을 적용한다. 알 수 없는 항목에는 적용하지 않는다.
- 교사 확률은 기존 모델의 판단을 보존하기 위한 보조 신호다. 새로운 정답이나 현장 안전 판정으로 사용하지 않는다.
- 교사는 평가 모드와 무경사 상태를 유지한다. 학습 전후 324개 상태 텐서의 해시를 비교한다.

## 공통 조건

원본 640 입력, 7개 사진 정답, 원본 80×80 위치 정답, 19개 보조 태그, 원본 학습 자료와 손실 가중치를 유지한다. 두 학생 모두 원본 `0773` 체크포인트에서 시작한다. 직전 보강 표본은 사용하지 않고, 준비된 원본 대조군의 6개 추출 배열을 두 조건에 똑같이 사용한다.

각 조건 6 epoch, 배치 8, epoch당 14,248회 추출, seed 56, 원본 학습률과 스케줄러를 사용한다. 사진 순서·자료 출처·원본/잘라낸 사진·7개 정답의 노출량이 두 조건에서 동일한지 실제 로그로 검증한다. AMP가 건너뛴 업데이트는 완료 업데이트와 따로 센다.

## 선택 및 평가

체크포인트는 기존 규칙대로 3개 source-VAL 자료의 균열·박락 FNR/FPR 중 최대값과 합을 기준으로 선택한다. 검증 결과를 보고 증류 가중치나 온도를 바꾸지 않는다. TEST 추론은 하지 않는다.

성능 보존 후보 조건은 다음과 같다.

1. 정답이 알려진 각 출처의 기존 5개 항목 AP가 원본보다 0.02를 초과하여 낮아지지 않는다.
2. DACL 철근 노출 AP가 이번 대조군보다 0.02 이상 회복된다.
3. 균열·박락의 최대 FNR/FPR이 원본 및 이번 대조군보다 2%p를 초과하여 악화되지 않는다.

이 조건과 별도로 기존 연구 후보 기준, 작은 손상 누락 수, **12개 FNR/FPR 각각 5% 미만** 기준을 그대로 보고한다. 성능 보존 조건 통과는 앱 교체나 5% 목표 달성을 뜻하지 않는다. 데이터는 주로 교량·댐 공개 자료이며, 공장·실제 산업 현장 성능을 입증하지 않는다.

## 결과물

프로토콜, 실제 6+6 epoch 학습 기록, 고정 교사와 입력 보존 증거, source-VAL 재추론 결과, 항목별 AP와 누락·오탐 비교표 및 그림을 남긴다. 이전 49개 고정 소스와 실험 기록은 유지하며, 앱의 기본 추론 프로필은 바꾸지 않는다.

## 재현 순서

아래 명령은 `safelog-cpp/ai-training`에서 기존 원본 자료·체크포인트·표본 추출 기록을 갖춘 상태로 실행한다. 기존 실행 폴더나 증거 파일이 있으면 덮어쓰지 않도록 중단한다. 이미 완료한 학습을 다시 실행하는 명령이 아니다.

```powershell
.venv/Scripts/python.exe scripts/facility_retention_study.py --declare
.venv/Scripts/python.exe scripts/test_facility_retention.py
.venv/Scripts/python.exe scripts/preflight_facility_retention.py
# 고정 소스와 프로토콜을 커밋한 다음
.venv/Scripts/python.exe scripts/verify_facility_retention.py --snapshot-before-training
.venv/Scripts/python.exe scripts/run_facility_retention_pair.py
.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-retention-control --grids 1
.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-retention-control --aggregate-only
.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-retention-distill --grids 1
.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-retention-distill --aggregate-only
.venv/Scripts/python.exe scripts/verify_facility_retention.py --test-results runs/facility-retention-test-results.json
.venv/Scripts/python.exe scripts/report_facility_retention_results.py
.venv/Scripts/python.exe -m scripts.plot_facility_retention
```
