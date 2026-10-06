# 기존 항목 성능 보존 강도 후속 비교 계획

작성: 2026-10-07. 앞선 가중치0·1 비교를 완료한 뒤, 다음 후보 **하나**를 학습 전에 고정한다.

## 이유와 변경점

가중치1 보존 학습은 DACL 철근 노출 AP를 대조0.6486에서0.6715로 회복했으나, 원본0.7010 대비 하락0.0296으로 허용 범위0.02를 넘었다. CODEBRIM 백화 AP 하락0.0237도 허용 범위를 넘었다. 균열·박락 최대 미탐·오탐은 대조23.06%에서22.53%로 낮아졌으나 원본22.28% 및5% 목표에는 미달했다.

후속 후보는 **기존 보존 손실의 가중치만4로 높인다.** 온도2, 원본 teacher, 알려진 다른5항목 마스크, 사진·픽셀·19종 보조 손실, 입력640, 배치8, seed56, optimizer·학습률·스케줄러, 원본 사진 순서와 증강 조건은 유지한다. 기존 모델의 확률은 보조 신호이며 새로운 정답으로 삼지 않는다.

## 신규 학습량과 비교 대상

- 새 가중치4 학생은 원본0773에서 다시 시작해6epoch를 학습한다. 가중치1 결과를 이어 학습하지 않는다.
- 앞서 완료한 가중치0 대조군6epoch를 재사용한다. 동일 자료·손실·교사 실행·사진 순서·학습 설정은 기록과 파일 해시로 확인한다.
- 이번에 추가하는 학습량은 **6epoch**다. 재사용한 대조군과 가중치1 모델은 신규 학습량에 다시 더하지 않는다.
- 보고서는 원본·가중치0·가중치1·가중치4를 함께 비교한다. 같은 source-VAL 결과를 보고 정한 후속 후보이므로 반복 검증 탐색이라는 한계를 명시한다.

## 검증과 판정

학생·교사의 동일 증강 TRAIN 입력, 실제7항목 정답 노출과 추출 순서, 교사 파라미터·버퍼·경사 상태, 원본 파일 보존을 검사한다. GPU 사전 검사는 새 가중치4 학생의8장짜리 배치1개에만 수행한다. 별도 사전 검사 업데이트는6epoch 학습량에 더하지 않는다.

기존 체크포인트 선택 규칙과 성능 보존 기준은 유지한다: 각 알려진 다른5항목의 출처별 AP 하락이 원본 대비0.02 이하, DACL 철근 노출 AP가 재사용 대조군보다0.02 이상 회복, 균열·박락 최대 오류 악화가 원본 및 대조군 대비2%p 이하. 기존 연구 후보 기준·작은 손상 누락·12개 미탐/오탐 각각5% 미만 기준은 별도 보고한다. TEST 추론과 앱 자동 교체는 수행하지 않는다.

원래61개 소스·학습 기록은 유지하고 보고서 연결 보완 소스2개와 신규 실험 소스를 함께 고정한다. 현재 자료의 공개 VAL 평가이며, 공장 현장 정확도나 정밀 위치·구조 안전을 입증하는 실험이 아니다.

## 재현 순서

기존 원본 자료와 완료 대조군 기록이 있는 `safelog-cpp/ai-training`에서 실행한다. 기존 산출물은 덮어쓰지 않는다.

```powershell
.venv/Scripts/python.exe scripts/test_facility_retention_strength.py
.venv/Scripts/python.exe scripts/facility_retention_strength_study.py --declare
.venv/Scripts/python.exe scripts/preflight_facility_retention_strength.py
# 고정 소스와 프로토콜 커밋 뒤
.venv/Scripts/python.exe scripts/verify_facility_retention_strength.py --snapshot-before-training
.venv/Scripts/python.exe scripts/run_facility_retention_strength.py
.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-retention-strong --grids 1
.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-retention-strong --aggregate-only
.venv/Scripts/python.exe scripts/verify_facility_retention_strength.py --test-results runs/facility-retention-strength-test-results.json
.venv/Scripts/python.exe scripts/report_facility_retention_strength.py
.venv/Scripts/python.exe scripts/plot_facility_retention_strength.py
```
