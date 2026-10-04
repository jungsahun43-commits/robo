# 작은 손상 특징을 읽는 모델 구조 비교

## 목적과 한 가지 변경

기존 MobileNetV3 LRASPP 기반 시설 모델에 stride4 특징을 읽는 잔차 경로 하나를 추가한다.
기존 연구 최대 검증 미탐·오탐 22.28%에서 개선되는지 실제 대조 학습으로 확인한다.
새 데이터나 정상·손상 정답을 추가하는 실험이 아니다. 성공이나 5% 달성을 약속하지 않는다.

원래 모델은 640 입력에서 stride8의 80×80 특징과 stride16의 40×40 특징을 사용한다.
새 경로는 같은 backbone의 block3, 24채널 160×160 특징에 1×1 투영, GroupNorm,
SiLU, depthwise 3×3, GroupNorm, SiLU, 7채널 투영을 적용한다. 이를 평균으로 80×80에
맞춰 기존 지도에 더하고 기존 top32·사진/지도 혼합·19종 보조 출력을 그대로 사용한다.
공개 출력은 7종의 사진 항목 존재 점수다. 160×160 정밀 위치 정답을 새로 만들지 않는다.

마지막 투영의 가중치·bias를 0으로 초기화한다. 원래 checkpoint의 모든 공통 가중치와
BN 상태를 동일하게 옮기므로 추가 학습 전 사진 점수·80×80 지도·19종 보조 출력은
대조군과 같아야 한다. 이는 초기화의 공정성 확인이며 정확도가 같다는 현장 검증은 아니다.

기본 구조는 [Torchvision 공식 LRASPP 설명](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html)과 연결된다.
실행에는 현재 설치된 Torch 2.12.0+cu126/Torchvision 0.27.0+cu126을 사용하며,
block3의 채널·해상도는 이 설치본에서 실제 확인한다. 외부 문서의 최신 버전으로 패키지를 변경하지 않는다.

## 학습 전에 고정한 비교

정확한 조건은 [고정 JSON](facility-detail-architecture-protocol.json)에 기록한다.

| 항목 | 대조군·보강군 공통 |
|---|---|
| 시작 가중치 | 기존 ROI 대조 모델, SHA `0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a` |
| 학습 | seed54, 각 8epoch, patience8 |
| 입력·예산 | 640, batch8, epoch당 복원 추출 14,248회 |
| 학습률 | backbone 0.00004 / 기존·신규 head 0.00025 |
| 손실 | 기존 사진·80×80 격자·19종 보조 손실, auxiliary0.5, ranking0 |
| 데이터 | 원본 TRAIN·파생 crop·원래 known/unknown 정답 그대로 |
| 출처 비중 | DACL/Dam/CODEBRIM 70/10/20% |
| 선택 | 같은 세 VAL의 균열·박락별 최대 미탐·오탐, 동률이면 합계, grid1 |

대조군 이름은 `facility-presence-target-detail-control`, 새 구조는
`facility-presence-target-detail-s4`다. 과거의 `facility-presence-target-detail` 증강 실험과 별개다.
추가 모듈의 난수 소비가 사진 증강을 바꾸지 않도록 sampler54 / TRAIN worker55 /
모델 구성 이후 전역56 / 각 검증 worker154~156의 전용 난수 상태를 사용한다.
같은 사진을 뽑았는지 epoch별 원본 행 인덱스 순서의 SHA와 출처·full/crop·정답 조합을 기록한다.

## 평가와 중단 기준

- 연구 후보는 초기 모델과 새 대조군 **둘 다**보다 최대 오류가 0.5%p 이상 낮아야 한다.
- 각 출처·항목의 미탐/오탐 악화는 2%p 이하, 정답이 있는 다른 5종 AP 하락은 0.02 이하로 제한한다.
- 작은 DACL 손상은 기존 `<1%` 폴리곤 구간의 균열93·박락105 항목별 양성 사례를 사용한다.
  두 항목 미탐 합계가 초기·대조 각각보다 2개 이상 줄고, 각 항목의 작은 손상 미탐률 악화는 2%p 이하이어야 한다.
  198은 고유 사진 수가 아니며 같은 사진이 두 항목에 포함될 수 있다.
- 엄격한 목표는 세 출처의 두 항목 미탐·오탐 **각각 모두 5% 미만**이다.
  한 seed의 연구 후보 기준 통과나 작은 손상 개선은 이 목표 통과와 다르다.
- 검증 사진별 오답을 보고 TRAIN 정답을 바꾸거나 표본을 선별하지 않는다. 목표 미달이면 test 추론을 하지 않는다.
- 이번 한 쌍을 완료하고 종료한다. 결과를 보고 epoch·손실·구조를 자동으로 다시 조정하지 않는다.
  앱 프로필은 연구 결과만으로 교체하지 않는다.

Wilson95 구간은 관측 비율의 기술 통계다. 연관 사진과 반복 VAL 선택 때문에
독립된 공장 현장의 오류 보장이나 유의성 검정으로 해석할 수 없다.
현재 교량·댐 콘크리트 자료가 일반 설비 전체, 법적 안전성, 도장·미장 구분을 검증하지 않는다.
정답 범위는 [라벨 범위](FACILITY_LABEL_SCOPE_KO.md)를 따른다.

## 실행

기존 데이터와 시작 모델이 준비된 `ai-training/`에서 실행한다.

```powershell
python scripts/preflight_facility_detail.py
python scripts/train_facility_detail.py --name facility-presence-target-detail-control --model-variant control --seed 54 --epochs 8 --patience 8 --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --draws-per-epoch 14248 --backbone-lr .00004 --head-lr .00025 --study-protocol reports/facility-detail-architecture-protocol.json
python scripts/train_facility_detail.py --name facility-presence-target-detail-s4 --model-variant detail --seed 54 --epochs 8 --patience 8 --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --draws-per-epoch 14248 --backbone-lr .00004 --head-lr .00025 --study-protocol reports/facility-detail-architecture-protocol.json
python scripts/evaluate_facility_target.py select --name facility-presence-target-detail-control --grids 1
python scripts/evaluate_facility_target.py select --name facility-presence-target-detail-s4 --grids 1
python scripts/analyze_facility_target.py --name facility-presence-target-detail-control --aggregate-only
python scripts/analyze_facility_target.py --name facility-presence-target-detail-s4 --aggregate-only
python scripts/verify_facility_detail.py
python scripts/report_facility_detail.py
python scripts/report_facility_target.py
python scripts/plot_facility_target.py
```

preflight의 disposable TRAIN backward/초기 출력/메모리 점검과 단위 테스트는 정확도 측정이 아니다.
완료 모델·원본 사진·개별 주석은 ignored `runs/`·`data/`에 보관하고 GitHub에는 코드·집계·조건만 올린다.
