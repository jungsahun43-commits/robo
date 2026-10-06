# 박락 음성 태그 표본 추출 대조 계획

이번 실험은 원래 박락 정답이 음성이고 Rockpocket·WConccor·Hollowareas·Cavity 중 하나 이상을 가진 DACL 전체 TRAIN 사진 666장의 학습 노출을 높인다. 관련 태그를 박락으로 바꾸지 않으며, 태그가 없는 사진을 정상 시설이라고 단정하지 않는다.

## 가설과 고정 조건

원래 모델의 전체 DACL TRAIN 점수 집계에서 이 집단의 박락 오탐은176/666이었다. 이는 학습 자료의 진단이며 현장 성능이나 오탐 원인을 증명한 수치가 아니다. 관련 표면 손상을 박락과 구분하는 데 해당 음성 사례의 추가 노출이 도움이 되는지 한 가지1.5배 조건으로 확인한다.

- 출발 모델: 기존 ROI 대조 모델0773. 두 모델 모두 같은 상태324개를 엄격하게 불러온다.
- 원래 640 입력,7종 사진·영역 출력, 원본19태그 보조 학습을 유지한다.
- 원래 TRAIN 전체14,248행과 파생 crop 포함26,289행을 사용한다. 앞선 원본 ROI 비교에서 생성한 PNG 자료로 바꾸지 않는다.
- 양군 각각6epoch, batch8, epoch당14,248draw, seed56, 원래 학습률·손실 가중치를 유지한다.
- 대조군 순서는 준비된 원래 Torch 다항 추출이다. 보강군은 별도 seed59의 최대 결합 방식으로 같은 DACL/full/균열·박락 결합 층 안에서만 대체한다.
- 출처·full/crop·원래 균열/박락 정답을 모든 추출 위치에서 보존한다. 원래 eligible 위치는 모두 남기고, 해당 층 밖과 박락 양성·crop 위치는 바꾸지 않는다.
- 준비된6epoch의 대상 노출은3,539→4,743회, 변경 위치는1,204곳이다. 조건부1.5배 가중치가 실제 전체 노출1.5배를 뜻하지는 않는다.
- 다른5종 사진 정답과 보조19태그의 실제 학습 노출은 달라질 수 있어 별도 AP 악화 검사를 수행한다.

## 실행 및 검증

먼저 새 코드의 테스트를 기록하고 고정 프로토콜을 선언한다. 원본 자료 SHA·크기·수정 시각을 보존한 뒤 양군에서 실제 CUDA1배치 업데이트와 CPU 재로딩을 확인한다. Git에 학습 전 소스를 저장하고 해당 Git 바이트와 프로토콜의 SHA가 일치하는 스냅샷을 만든 뒤6+6epoch를 순차 실행한다.

```powershell
.venv/Scripts/python.exe scripts/test_facility_subtype.py
.venv/Scripts/python.exe scripts/facility_subtype_study.py --declare
.venv/Scripts/python.exe scripts/preflight_facility_subtype.py
# 학습 전 Git 커밋 이후:
.venv/Scripts/python.exe scripts/verify_facility_subtype.py --snapshot-before-training
.venv/Scripts/python.exe scripts/run_facility_subtype_pair.py
.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-subtype-control --grids 1
.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-subtype-negative --grids 1
.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-subtype-control --aggregate-only
.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-subtype-negative --aggregate-only
.venv/Scripts/python.exe scripts/verify_facility_subtype.py --test-results runs/facility-subtype-test-results.json
.venv/Scripts/python.exe scripts/report_facility_subtype.py
.venv/Scripts/python.exe scripts/plot_facility_subtype.py
```

실제 추출 순서의 SHA, 대상 노출, 출처·full/crop·두 정답의 구성, attempted batch와 실제 optimizer 업데이트·AMP skip을 기록한다. 기존 frozen 소스·원본 정답·모델·앱 프로필을 덮어쓰지 않는다. 자료와 개별 사진의 순서·점수·검수 화면은 로컬에 보관한다.

## 결과 판단

동일한 DACL710/Dam424/CODEBRIM611의 grid1 검증에서 균열·박락 각각의 미탐률·오탐률을 비교한다. 원래 기준 모델과 새 대조군 모두에 대해 최대 비율0.5%p 이상 감소, 개별 비율 악화2%p 이내, 다른 알려진 항목 AP 악화0.02 이내, 작은 손상 FN 합계2건 이상 감소와 각 작은 손상 FNR 악화2%p 이내를 연구 후보 기준으로 사용한다.

별도로12개 비율이 모두 엄격하게5% 미만인지를 확인한다. 반복 사용한 공개 VAL은 독립 현장 성능이 아니며, 연구 후보 기준 통과가5% 목표 통과를 뜻하지 않는다. 이번 비교에서 TEST 추론과 앱 기본 모델의 자동 교체는 수행하지 않는다.
