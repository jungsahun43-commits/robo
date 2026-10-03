# 균열·박락 구분 학습 계획

2026-10-04에 학습 전에 정한 한 쌍의 대조 실험이다. 이전 콘크리트 사진 보강은 최대 검증 미탐·오탐을 22.28%에서 낮추지 못했다. 보강군은 22.85%였으며 앱에 적용하지 않았다.

## 자료와 손상 기준 정리

기존 TRAIN 원본 사진 14,248장 중 균열과 박락이 둘 다 음성인 사진은 5,915장이다. 해당 두 항목의 출판자 정답이 음성이라는 의미이며, 다른 손상이 없거나 현장이 안전하다는 의미는 아니다. 사진 전체 정상 라벨을 새로 만들지 않는다.

원본 DACL의 19개 태그에는 도장·미장층 손상을 직접 구분하는 이름이 없다. ConViD Spalling 사진의 재료·손상 기준도 전문가가 확인하지 않았으므로 사진을 AI 판단으로 재분류하지 않는다. 이번 실험은 기존 출판자의 확인 가능한 TRAIN 정답으로 한정한다.

## 학습 변경 한 가지

같은 미니배치의 원본 전체 사진에서 같은 출처·같은 항목의 양성과 음성 출력 점수를 비교한다. 손상 양성 점수가 음성보다 높도록 `softplus(음성 logit - 양성 logit)` 손실을 추가한다. 이는 사진 두 그룹을 구분하는 순위 학습이며, 오차율·확률 보정·정밀 위치 정답을 대신하지 않는다.

- 출처가 다른 사진끼리는 비교하지 않는다.
- 해당 항목 정답이 미확인인 사진과 TRAIN crop은 추가 순위 손실에서 제외한다.
- 양성·음성 쌍이 없는 그룹은 추가 손실을 만들지 않는다.
- 기존 photo/pixel/auxiliary 손실, 클래스 가중치, sampler, full/crop 비중과 정답은 동일하다.
- 기존 7개 공개 출력, 19개 학습 보조 태그와 모델 구조를 유지한다.

## 고정 조건

| 조건 | 대조군 / 구분 학습군 |
|---|---|
| 초기 모델 | 기존 ROI 대조 후보, 최대 검증 오류 22.28% |
| seed | 53 / 53 |
| 학습 epoch | 6 / 6 |
| epoch당 복원 추출 | 14,248 / 14,248 |
| 크기·batch | 640·8 / 640·8 |
| backbone/head 학습률 | 0.00002 / 0.000125, 양쪽 동일 |
| 출처 비중 | DACL 70%, Dam 10%, CODEBRIM 20%, 양쪽 동일 |
| 순위 손실 가중치 | 0 / 0.25 |
| 추가 독립 사진 | 0 / 0 |

학습률은 지난 추가 학습보다 양쪽 모두 낮췄다. 과거 실험과 순수 한 요인 비교를 주장하지 않으며, 이번 두 실행 사이에서 추가 순위 손실을 비교한다. 고유 사진 수와 epoch당 추출 횟수는 다르며 모든 사진을 한 번씩 읽는다는 뜻은 아니다.

## 후보 선정 기준

세 출처의 균열·박락 각각 미탐·오탐을 모두 기록한다. 연구 후보로 인정하려면 기존 초기 모델과 이번 대조군 모두에 대해 최대 오류가 적어도 **0.5%p 감소**해야 한다. 어느 출처·항목의 미탐·오탐도 두 비교 대상보다 2%p 넘게 악화되면 탈락한다. 출판자가 정답을 제공한 나머지 5종의 AP도 항목별로 0.02 넘게 악화되면 탈락한다. AP는 오류율이 아니다.

모든 출처·두 손상의 미탐·오탐이 각각 5% 미만이어야 기존 엄격한 목표를 통과한다. 이 목표와 연구 후보 기준은 다르다. 목표에 미달하면 보류 시험을 실행해 반복 선택하지 않는다. 연구 후보 기준 통과만으로 앱 기본 모델을 교체하지 않는다.

오류율 옆에는 관측 개수와 Wilson 95% 구간을 함께 기록한다. 사진 간 상관과 반복된 검증·임계값 선택이 있으므로 독립 현장의 확정적 신뢰구간으로 해석하지 않는다. 공장 사진의 별도 현장 검증은 아직 없다.

이번 6epoch 대조 한 쌍을 완료한 뒤 결과를 정리한다. 결과에 따라 epoch를 무기한 늘리거나 새 recipe를 자동 재시도하지 않는다.

## 실행

AI training 폴더에서 두 명령을 순서대로 실행한다. 이미 완료된 run은 덮어쓰지 않는다.

```powershell
./.venv/Scripts/python.exe scripts/train_facility_spatial.py --name facility-presence-target-discrimination-control --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --seed 53 --epochs 6 --patience 6 --draws-per-epoch 14248 --backbone-lr .00002 --head-lr .000125 --target-ranking-weight 0 --study-protocol reports/facility-target-discrimination-protocol.json
./.venv/Scripts/python.exe scripts/train_facility_spatial.py --name facility-presence-target-discrimination-ranking --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --seed 53 --epochs 6 --patience 6 --draws-per-epoch 14248 --backbone-lr .00002 --head-lr .000125 --target-ranking-weight .25 --study-protocol reports/facility-target-discrimination-protocol.json
```

학습 전 고정 조건 원문은 [실험 protocol](facility-target-discrimination-protocol.json)에 있다. 실행 중 결과에 맞춰 이 조건 파일을 수정하지 않는다. 원본 사진·주석·모델과 개별 사례는 무시되는 로컬 자료로 유지하고, 코드·집계 결과만 Git에 기록한다.
