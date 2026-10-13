# 오탐 개선을 유지하면서 미탐 방향을 보정하는 비교

이전 normal ROI 실험과 같은 데이터·draw 배열·seed·초기 가중치·optimizer·손실·4epoch512업데이트를 사용한다. 실제 Adam 가중치 변화가 학습용 양성pixel 손실을 일차 근사로 늘리는 방향을 제거한다. 실제 검증 recall이 보장되는 알고리즘이 아니므로, 알려진353장과 정상CAL3으로 다시 비교한다.

## 완료된 실행 기록

1. 최초study_unified_belt_recall_step.py는 첫 optimizer 호출에서 수치 조건 실패. checkpoint·완료journal0회이고 제외 기록을 보존했다. 이 버전으로 학습을 다시 시작하지 않는다.
2. belt_recall_step_v2.py는FP32 쓰기 오차 경계의 보정 여유와 정확한 파라미터 복원 fallback을 사용한다. optimizer의 raw moments는 유지한다. 모든 유효 변화의 signed dot를 다시 검사한다.
3. tests/test_belt_recall_step_v2.py의5개 테스트는 유해 방향 제거·직교 방향 유지·실제Adam momentum 변화·known양성만의손실·고차원FP32오차를 검증한다.
4. study_unified_belt_recall_step_v2.py --mode freeze: 기존 control draw 배열 exact 일치, 소스·입력·사전조건을 고정한다. 먼저commit한 뒤 train을 실행한다.
5. --mode train: 실제4epoch512회.64업데이트마다 파라미터·optimizer·scheduler·cursor·CPU/CUDA RNG를 저장한다. Windows PID는 읽기 전용process_exists로 확인한다. 완료된final은 재학습하지 않는다.
6. --mode eval: 같은 알려진 손상VAL353, 정상CAL3. 보관7개와BeltCrack예약TEST는 모델 점수를 계산하지 않는다.
7. --mode verify: journal의signed dot·보정/복원 횟수·실제완료 업데이트·progress·비벨트tensor·기존입력·소스·보고서 보존을 검사한다.
8. report_unified_belt_recall_step.py: 비교·선택·한계를 기록한다. 연구 선택 조건이 통과한 경우에만 별도CPU API·C++ 연결 전달물을 검증한다. 앱 기본 모델은 자동 교체하지 않는다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_recall_step_v2.py
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode freeze
# 소스와 protocol commit 후
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_recall_step_v2.py --mode verify
./.venv/Scripts/python.exe scripts/report_unified_belt_recall_step.py
```

완료된prepare·protocol·eval·report를 다시 실행해 기존 결과를 덮어쓰지 않는다. 이후 비교는 새run과사전조건을 사용한다. 원자료 normal ROI는 사진 전체 정상 정답이 아니고, CAL3 수치로 전체5% 미만을 선언하지 않는다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_RECALL_STEP_RESULTS_KO.md), [기존 정상 자료 범위](../ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md), [아이디어 참고](https://arxiv.org/abs/2001.06782).
