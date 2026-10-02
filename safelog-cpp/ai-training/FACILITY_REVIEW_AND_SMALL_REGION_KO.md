# 역할3: 오류 검수와 작은 손상 보강

## 프로젝트 범위와 자료 출처

서비스 목표는 **산업체 현장에서 사진을 찍고, AI의 손상 제안을 점검자가 확인해 점검 기록과 보고서를 만드는 것**이다. 이번 연구 모델은 그중 콘크리트 표면 사진의 균열·박락을 보강한다. 공장 벽·바닥·기둥에 나타날 수 있는 공통 손상 형태를 학습하기 위해 기존 교량·댐 사진을 보조 자료로 사용한다.

현재 시설 손상 학습 자료는 교량·댐 사진이 중심이다. 이 사진에서 공통 표면 손상을 학습해 산업체 현장 점검 보조에 연결하려는 단계다. 현재 결과는 해당 공개 자료에서 검증한 수치이며, **산업체 현장이나 공장 사진에서 같은 성능을 낸다는 점은 아직 검증하지 않았다.** 사진에 보이는 표면 손상의 제안과 시설의 안전 판정은 구분한다.

이번에는 원본 사진 수와 앱의7항목을 늘리지 않는다. 기존 격자 detail12,041행 중1,694행을 작은 손상 주변의 맥락 crop으로 교체하고, 대조군·보강군 한 쌍을 비교한다. 향후 새 자료를 확보한다면 산업체의 벽·바닥·기둥 등 사용 환경에 가까운 자료를 우선 검토한다. 이는 후속 방향이며, 이번 실험에서 산업체 자료를 새로 수집하거나 추가 학습한 것은 아니다.

## 이 PC에서 바로 볼 것

`ai-training/runs/facility-train-review-round9-final/index.html`을 Chrome/Edge에서 연다. 원본 TRAIN14,248행을 확인하고 소스별 균열·박락 오류와 정답 일치 비교를 포함한140장의 검수 표본을 만든 것이다. 원본 사진은 로컬 데이터 경로로 연결되므로 HTML만 따로 보내면 사진이 보이지 않는다.

왼쪽은 모델 입력 사진, 오른쪽은 출처의 주석이다. 오른쪽 표시가 모델이 찾은 위치는 아니다. 원래 정답·모델 확률·원본 태그를 함께 보고 `검수 제안`과 `근거 메모`에 의견을 기록한다. `검수 의견 JSON 저장`을 눌러 창을 닫기 전에 저장한다. JSON은 의견만 저장하며 원래 정답이나 학습에는 자동 반영하지 않는다.

FP=없는데 있다고 제안, FN=있는데 놓침, TP/TN=정답 일치. 모델 확률은 예측값이며 정답일 확률을 보장하는 신뢰도 수치가 아니다. 이 표본은 학습에 사용된 자료라 현장 정확도 측정용으로 사용하지 않는다. 원본 주석의 클래스 경계가 애매한 경우에는 의견으로 남기고 전문가 확인을 받는다.

## 팀원이 재현하려면

기존 자료 준비·auxiliary 학습이 완료된 환경에서 `safelog-cpp/ai-training` 폴더의 Python 가상환경을 사용한다. 큰 원본 데이터와 가중치는 Git에 포함되지 않는다. 이미 생성된 단계는 반복하지 않는다.

```powershell
./.venv/Scripts/python.exe scripts/score_facility_train.py --name facility-presence-target-auxiliary
./.venv/Scripts/python.exe scripts/build_facility_train_review.py --cache runs/facility-presence-target-auxiliary/TRAIN-REVIEW-PREDICTIONS.json --output runs/facility-train-review-team --per-error 12 --per-correct 3 --seed 49
```

처음 명령은 원본 TRAIN만 읽고, 두 번째 명령은 HTML과 검수 JSON을 만든다. 출력 파일이 있으면 보존을 위해 중단하므로 새 경로를 지정한다. HTML·주석·사진이 들어가는 `runs/`, `data/`는 로컬에 유지한다. CODEBRIM을 포함한 원본/가공 데이터의 이용 및 재배포 조건을 지킨다.

## 작은 손상 비교 실행

현재의 격자 확대 자료 중 일부를 작은 source polygon/mask 주변 맥락 crop으로 교체한다. 전체 사진·전체 행 수·TRAIN 부모를 유지한다. crop 정답은 내부 주석에서 다시 계산하고, 미확인 항목과 사진 전체의 보조 태그를 crop에 넘기지 않는다. 이미 준비된 manifest는 덮어쓰지 않는다.

원본14,248행과 전체26,289행,7종 출력 계약은 대조군·보강군에서 동일하다. 파생 crop1,694개를 새 현장 사진1,694장으로 설명하지 않는다. 이번 비교 결과만으로 공장 현장의 정확도를 발표하지 않는다.

```powershell
./.venv/Scripts/python.exe scripts/prepare_facility_small_regions.py
./.venv/Scripts/python.exe scripts/train_facility_spatial.py --name facility-presence-target-roi-control --seed 51 --epochs 6 --patience 6 --initial runs/facility-presence-target-auxiliary/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --backbone-lr .00004 --head-lr .00025
./.venv/Scripts/python.exe scripts/train_facility_spatial.py --name facility-presence-target-small-region --seed 51 --epochs 6 --patience 6 --initial runs/facility-presence-target-auxiliary/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --spatial-manifest data/facility-small-region-training/train.json --backbone-lr .00004 --head-lr .00025
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-roi-control --grids 1
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-small-region --grids 1
./.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-roi-control --aggregate-only
./.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-small-region --aggregate-only
./.venv/Scripts/python.exe scripts/report_facility_small_region.py
./.venv/Scripts/python.exe scripts/report_facility_target.py
./.venv/Scripts/python.exe scripts/plot_facility_target.py
```

대조군과 보강군은 한 번에 하나씩 GPU에서 실행한다. 동일6epoch 동안 각 후보의 가장 좋은 검증 epoch를 선택한다. 전체 사진(grid1) 평가는 미리 고정했다. 데이터 생성법을 바꾸면 클래스 비중과 pixel 가중치도 달라질 수 있어, 이를 순수 해상도 효과로 설명하지 않는다.

결과는 `reports/FACILITY_SMALL_REGION_RESULTS_KO.md`와 `reports/facility-small-region-comparison.json`에 기록한다. 최대 오류, 출처별 미탐·오탐, 작은 부위 미탐, 나머지 항목의 AP와 양성/음성 추출 비중을 함께 비교한다. 원래5% 기준과 보류 시험 실행 조건은 유지한다. 기본 앱 모델을 자동으로 교체하지 않는다.

## 역할4에게 전달할 내용

이번 변경은 AI 학습·검수 도구이므로 Qt 앱 출력 계약은 기존7종 photo presence를 유지한다. 역할4는 기존 연결로 사진→AI 제안→점검자 채택/수정/거절→보고서 흐름을 개발한다. 새 연구 모델을 앱 기본값으로 적용했다고 안내하지 않는다. 모델 채택은 전체7항목 및 기존 탐지 결과와의 결합을 검증한 후 결정한다.

앱 소개와 발표에서는 대상 서비스를 산업체 현장 점검 보조로 설명하고, AI 학습 자료의 출처와 산업체 현장 평가 여부를 함께 적는다. 교량·댐 자료의 검증 수치를 산업체 현장 성능으로 표시하지 않는다.
