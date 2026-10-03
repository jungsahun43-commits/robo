# 콘크리트 사진 보강 대조 실험 계획

작성일: 2026-10-03. 산업체 사진 점검이라는 제품 목표를 유지한다. 새 자료를 공장·독립 산업 현장 검증 자료라고 표시하지 않는다.

## 자료 취득과 사용 조건

산업유산 콘크리트의 [IHRCD-Det](https://github.com/eggnog1307/IHRCD-Det)은 배포 사이트 접근이 브라우저 보안 정책으로 차단되어 취득하지 못했다. 다른 접근 경로로 우회하지 않는다.

[PECCD V1](https://data.mendeley.com/datasets/w7549ryvx2/1)의 공식 RAR 3,767,446,444바이트는 공식 SHA256 `9acc37537c2a618c22bbda55d1353823d4e2dd82d98ae729d0c30315c72388ec`과 일치한다. 그러나 현재 읽을 수 있는 파일 및 [저자 초록](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5758816)에서 숫자 라벨의 클래스 대응을 확인하지 못했고 아카이브 목록 조회에도 읽을 수 없는 파일명 경고가 있었다. 설명에 나열된 클래스 순서를 숫자 ID라고 추측하지 않는다. 준비·학습 사진은 0장이다.

이번 비교는 [ConViD V4](https://data.mendeley.com/datasets/fx3rthfjhy/4)의 공식 `crack`·`Spalling` 폴더에서 사진을 제한적으로 취득한다. 출처는 인도 Pune의 야외 콘크리트를 스마트폰으로 촬영한 자료이며 공장 시설 종류나 원본 장면 ID는 확인되지 않았다. CC BY 4.0 출처를 유지한다.

- 검증 점수를 보기 전에 seed52로 폴더별 최대100장, 합계 최대200장을 선정한다. 파일별 공식 SHA256을 확인한다.
- 원본·기존 자료의 정확한 픽셀 및 해시 기반 유사 사진을 검사한다. 의심되는 사진은 제외한다. 이 검사가 다른 각도·잘린 장면의 독립성을 증명하지는 않는다.
- `crack`은 균열만1, `Spalling`은 박락만1이다. 각 사진의 나머지6항목은 미확인(-1)이다. 폴더에 없다는 이유로 음성(0)을 만들지 않는다.
- Honeycomb·VOID 폴더는 이번 실험에서 사용하지 않는다. 분류 폴더에서 픽셀 위치 정답을 만들지 않는다. 새 사진의7종 위치 정답과19종 보조 정답은 모두 미확인이다.
- 긴 변1280의 비율 보존 사진을 만든다. 원본 파일·정답·확인 기록은 Git에서 제외된 로컬 자료에 보존한다.
- 출처에 공식 train/val/test 및 부모 장면 구분이 없어 선별된 사진은 모두 TRAIN 보강만 한다. 새 출처 검증·시험 또는 공장 정확도 수치는 산출하지 않는다.

선별 후 실제 사진은192장(균열92/박락100)이다. 기존 자료와 dHash가 가까운3그룹의7장을 검수 대기로 제외하고, 새 자료 내 유사 그룹의 대표만 유지하며1장을 제외했다. 빈 자리를 다른 사진으로 채우지 않았다. 공식 설명은 동일 손상의 여러 각도 사진이 있음을 명시하므로192개의 독립 장면이라고 해석하지 않는다.

준비된 TRAIN 중8장을 AI가 육안으로 확인했다. 균열4장은 틈 형태가 보이며, 박락4장 중 일부는 어두운 거친 표면·페인트/미장층 벗겨짐과 구분이 애매하다. 원래 폴더 라벨은 전문가가 검증한 우리 앱의 엄밀한 콘크리트 박락 정답이라고 확정할 수 없다. 저자 라벨을 변경하거나 다른 항목으로 덮어쓰지 않고, 출처 정의의 차이를 비교 실험의 한계로 유지한다. 이 사진 확인은 성능 측정이 아니다.

최초 대조군은 source-picks·원본 사진·원본 분할·공개 감사 해시 검사 보강을 위해 첫epoch 완료 전에 중단했다.12800개 추출까지의 진행을 확인했지만 완료된epoch·체크포인트·평가는0이다. 로컬 `runs/facility-presence-target-building-control-preflight-aborted`에 중단 기록을 보존하고, 동일한 최종 검증 코드로 정식 한 쌍을 다시 시작한다. 부분 학습을 완료된epoch나 채택 모델로 세지 않는다.

## 같은 조건으로 비교

| 설정 | 대조군 | ConViD 보강군 |
|---|---|---|
| 실행 이름 |facility-presence-target-building-control |facility-presence-target-building-convid |
| 초기 체크포인트 |기존 roi-control best.pt |동일 |
| 초기 SHA256 |0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a |동일 |
| 기존 full/crop |14,248 /12,041 |동일 |
| 새 사진 |0 |검사 통과한 최대200장 |
| seed/학습 |52 /6epoch, patience6 |동일 |
| 입력/배치 |640 /8 |동일 |
| LR |backbone4e-5, head2.5e-4 |동일 |
| epoch당 추출 |14,248, 복원 추출 |동일 |
| 출처 추출 비중 |DACL .70 /Dam .10 /CODEBRIM .20 |기존 .63 /.09 /.18 +ConViD .10 |
| 기존 출처 내 full/crop·각 행 상대 비중 |기존 설정 |그대로 보존 |
| 보조 감독 |기존 DACL 원본19종·위치 지도 |동일, 새 사진은 미확인 |
| 평가 |원래 세 출처 전체 사진(grid1) |동일 |

각 epoch는 모든 고유 사진을 한 번씩 보는 의미가 아니다. 복원 추출 수를 동일하게 유지하며 보강군에서 기존 출처 추출의 총량은 90%로 줄어든다. 양성만 추가하므로 학습 분포·양성 클래스 가중치가 달라질 수 있고, 기존 항목의 오탐 증가를 함께 측정한다. 기존 출처 내 양성 가중치·위치·보조 감독 가중치는 보존한다.

## 판정과 보고

원래 검증 정답·분모·시험 분할을 바꾸지 않는다. 균열·박락 각각의 출처별 FN/(TP+FN) 및 FP/(FP+TN), 나머지5항목의 알려진 정답 AP, DACL의 작은 주석 면적별 미탐을 비교한다. 미확인은 음성 AP에 합치지 않는다.

6epoch와 전체 사진(grid1) 평가를 먼저 고정한다. 이번 결과를 보고 epoch나 확대 뷰를 늘리지 않는다. 동일 seed 한 쌍의 차이는 반복 실험이나 독립 현장 개선의 증거가 아니다.

각 출처·대상 항목의 미탐·오탐이 모두 엄격히5% 미만인 경우에만 기존 시험 사용 조건을 검토한다. 미달이면 보류 시험 및 조건부 추가1epoch를 실행하지 않는다. 새 자료를 넣었다는 이유로 앱 기본 모델을 바꾸지 않는다.

검수 화면의140개 TRAIN 중23장은 AI가 사진과 원본 주석을 확인했다. 전문가 검수나 정답 수정은 이루어지지 않았다. [AI 관찰 기록](FACILITY_TRAIN_VISUAL_REVIEW_KO.md)은 새 라벨의 근거로 쓰지 않는다.

## 실행 명령

`safelog-cpp/ai-training` 폴더에서 실행한다. 공식 V4 메타데이터를 받아 고정 선정·검증·준비한 결과의 해시를 확인한 뒤 아래 한 쌍만 실행한다. 기존 원본 자료가 있어야 중복 검사를 진행할 수 있다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_convid_supplement.py --fetch-metadata
.\.venv\Scripts\python.exe scripts/train_facility_spatial.py --name facility-presence-target-building-control --seed 52 --epochs 6 --patience 6 --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --draws-per-epoch 14248 --backbone-lr .00004 --head-lr .00025
.\.venv\Scripts\python.exe scripts/train_facility_spatial.py --name facility-presence-target-building-convid --seed 52 --epochs 6 --patience 6 --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --draws-per-epoch 14248 --backbone-lr .00004 --head-lr .00025 --photo-supplement data/convid-training/train.json
.\.venv\Scripts\python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-building-control --grids 1
.\.venv\Scripts\python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-building-convid --grids 1
.\.venv\Scripts\python.exe scripts/analyze_facility_target.py --name facility-presence-target-building-control --aggregate-only
.\.venv\Scripts\python.exe scripts/analyze_facility_target.py --name facility-presence-target-building-convid --aggregate-only
.\.venv\Scripts\python.exe scripts/report_facility_building_supplement.py
```

기존 실행을 덮어쓰지 않는다. `best.pt`가 이미 있으면 학습 도구는 중단한다. 위 명령은 초기 모델과 기존 원본 자료가 로컬에 있는 개발 환경용이며, Git 코드를 받는 것만으로 모델·원본 자료가 내려받아지지는 않는다.
