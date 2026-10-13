# 정상 벨트 구역 · 신규 자료 검수와 초기 학습

## 자료와 정답 범위

Gofaones Workspace의 belt defects v1·v2를 Roboflow 로그인 후 COCO ZIP으로 내려받았다. 원본은 data/belt-normal-roboflow-v1 및 data/belt-normal-rois-v1에 보관하며 Git에 올리지 않는다. CC BY4.0이고 실제 출처·라이선스와 변경 내역은 결과 보고서에 기록한다.

prepare_belt_normal_rois.py는 두 버전의 실제 라벨을 검사하고, 명시적 normal belt bbox의 중앙 구역만 고른다. 마모 라벨 부모, 다른 라벨과 겹치는 구역, 그룹 중복, 이전 검증 자료와의 유사성을 검사한다. 사진 전체 정상과 dense mask 정답은 만들지 않는다. 본 연구 준비 경로는 제작 환경의 Downloads와 고정된 data 경로를 사용한다.

## 실제 흐름

1. tests/test_belt_normal_roi.py: 정상 구역 범위·금지 겹침·frame gap·holdout·증강 그룹 확인.
2. prepare_belt_normal_rois.py: 원본 ZIP은 보존하고 FIT14·CAL3·보관7 정상 구역과 source SHA를 생성한다.
3. tests/test_belt_normal_learning.py: 구역 부재 분류 손실과 single-class 출력 조건 확인.
4. study_unified_belt_normal.py --mode freeze: TRAIN draws, 실제4epoch512업데이트, source·입력 SHA, 비교 조건과 적용 금지를 고정한다.
5. 소스와 protocol을 먼저 commit한 뒤 --mode train으로 학습한다. 가중치·optimizer·scheduler·cursor·CPU/CUDA RNG를64업데이트마다 저장한다. Windows PID는 읽기 전용process_exists로 확인한다. 완료된final은 다시 학습하지 않는다.
6. --mode eval: 같은 알려진 손상VAL353장과 새 정상CAL3개를 비교한다. 보관7개와 예약BeltCrack TEST 모델 점수는 계산하지 않는다.
7. --mode verify: 실제 journal, final/progress, 비벨트 tensor와 기존 입력·보고서·소스를 보존했는지 확인한다.
8. report_unified_belt_normal.py: 결과·한계·현재 전달물SHA·재개 기준을 작성한다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_normal_roi.py
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_normal_learning.py
./.venv/Scripts/python.exe scripts/prepare_belt_normal_rois.py
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode freeze
# 소스와 protocol commit 후
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_normal.py --mode verify
./.venv/Scripts/python.exe scripts/report_unified_belt_normal.py
```

완료된 준비·protocol·평가·보고서는 배타적으로 생성한다. 같은 연구를 새 실험으로 다시 실행하거나 기존 결과를 덮어쓰지 않는다. 새 추가 자료나 손실 비교는 새 namespace와 사전조건으로 진행한다. 현 단계는 CAL3개와 촬영 구간 미검증으로 적용 금지된pilot이다. 전체 정상 사진 오탐은 여전히 미측정이다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_NORMAL_RESULTS_KO.md), [새 자료의 실제 출처](https://universe.roboflow.com/gofaones-workspace/belt-defects/dataset/2), [이전 모델 출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
