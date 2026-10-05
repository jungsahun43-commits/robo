# 640·960 입력 해상도 대조 학습 계획

최근 TRAIN 사진 관찰에서 작은 균열·좁은 패치·표면 흔적의 구분을 다음 검토 후보로
남겼다. 이번에는 동일 ROI 초기 모델에서 입력 해상도 하나를 비교한다. 관찰 의견으로
원본 라벨을 고치거나 AI의 불확실 판단을 새 학습 정답으로 만들지 않는다.

## 바꾸는 계산

- 대조군은 기존 AuxiliaryClassifier, 입력 640, 사진 판단·픽셀 손실 격자 80×80이다.
- 보강군은 같은 가중치·파라미터를 사용하는 ResolutionClassifier, 입력 960이다.
  원시 120×120 logit을 **area 평균으로 80×80에 맞춘 뒤** 기존 top32·global·mix를
  계산한다. 픽셀 손실도 같은 80×80 logit과 기존 정답을 사용한다.
- 같은 32개 셀의 상대 면적을 유지한다. 960에서 원시 120×120의 top32를 고르는
  구조와 다르며, 확대 정답·새 고해상도 픽셀 주석을 만들지 않는다.
- 19종 보조 head와 공개 7종 출력은 유지한다. 새 파라미터는 0개다.
  640에서 두 구조의 초기 출력은 같아야 한다. 입력 960과 640의 초기 출력이 같다는
  뜻은 아니다. 고해상도 체크포인트는 전용 architecture로 저장·재로딩한다.

logit 면적 평균은 [PyTorch의 interpolate 공식 API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)의
`mode="area"`를 사용하며, 사진 변환·픽셀 정답 변환과 구분한다.
원본 자료의 좌표와 여러 손상 태그는 [DACL 공식 도구 설명](https://github.com/phiyodr/dacl10k-toolkit)의
주석 형식을 따른다. native polygon·19종 태그는 이전 학습에도 사용했다.

## 학습 전에 고정할 조건

| 항목 | 고정 조건 |
|---|---|
| 초기 모델 | 기존 ROI 대조군, SHA `0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a` |
| run 이름 | `facility-presence-target-resolution-control` / `facility-presence-target-resolution-highres` |
| 예산 | 각 6epoch, patience 6, 총 12epoch |
| 입력·batch | 대조 640 / 보강 960, 두 군 batch 8 |
| 표본 예산 | 두 군 매 epoch 복원 추출 14,248행 |
| 난수 | model·sampler 56, TRAIN worker 57, 모델 생성 이후 global 58 |
| VAL worker | DACL 156 / Dam 157 / CODEBRIM 158 |
| 학습률 | backbone 0.00004 / head 0.00025 |
| 손실 | 기존 사진·80×80 픽셀·19종 보조 손실; auxiliary 0.5, ranking 0 |
| 자료 | 기존 TRAIN full 14,248행 + 파생 crop 12,041행, 원본 태그·known/unknown 보존 |
| 출처 비중 | DACL / Dam / CODEBRIM 70 / 10 / 20% |
| 모델 선택 | 같은 세 VAL의 균열·박락 FNR/FPR minimax, 동률이면 비율 합계, 전체 사진 grid1 |

기존 processed 사진을 입력한다. DACL processed 최대 변은 1280이며 더 큰 native
원본의 확보 상태는 별도 [TRAIN 자료 감사](facility-resolution-source-audit.json)에 기록한다.
이 실험에서 원본 전체 사진을 새 입력으로 바꾸거나 새 현장 사진을 추가하지 않는다.
작은 native512·640 사진의 확대는 새로운 세부 정보를 만들지 않는다. 원시 map이
커져도 기존 80×80 주석을 확대하여 정밀 위치 정답이라고 주장하지 않는다.

같은 epoch·batch·표본 수는 같은 시간·메모리·모바일 추론 비용이 아니다. 학습 시간,
CUDA allocated peak, 실제 optimizer update와 AMP skip을 따로 기록한다.

## 비교와 종료 기준

초기 모델과 새 대조군 **각각**보다 최대 검증 미탐·오탐이 최소 0.5pp 낮고,
모든 target FNR/FPR 악화는 2pp 이하, 다른 알려진 항목 AP 악화는 0.02 이하여야
연구 후보의 공통 기준을 통과한다. 기존 작은 DACL 양성 198개 항목·사진 사례
(균열 93 / 박락 105)의 FN 합계가 두 비교 대상보다 각각 최소 2건 줄고, 작은
항목별 FNR 악화도 2pp 이하여야 한다. 198은 고유 사진 수가 아니다.

엄격한 목표는 균열·박락 × 세 출처 × FNR/FPR의 12개 비율이 **각각 모두 5% 미만**인
것이다. 연구 후보 선정과 앱 배포·현장 안전 보장은 별도다. 한 쌍을 끝낸 뒤 결과를
보고하며, 결과에 맞춰 해상도·epoch·pooling 격자를 다시 고르지 않는다.
미달이면 보류 TEST 추론을 하지 않고 앱 기본 `facility-validation-v2`를 유지한다.

한 seed, 반복 VAL 선택, 같은 원본의 crop 상관이 있다. Wilson95는 기술 통계이며
독립된 공장 현장의 오류 보장이나 모델 차이의 유의성 검정이 아니다. 사진 크기와
박락 정의 질문을 이번 학습으로 해결했다고 간주하지 않는다.

## 실행

기계 판독 조건과 코드 SHA는 [고정 JSON](facility-resolution-study-protocol.json)에 둔다.
GPU·저장·재로딩·라벨 보존 preflight를 통과하고 소스 commit을 고정한 뒤 순서대로 실행한다.

```powershell
./.venv/Scripts/python.exe scripts/preflight_facility_resolution.py
./.venv/Scripts/python.exe scripts/train_facility_resolution.py --name facility-presence-target-resolution-control --variant control --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --study-protocol reports/facility-resolution-study-protocol.json
./.venv/Scripts/python.exe scripts/train_facility_resolution.py --name facility-presence-target-resolution-highres --variant highres --initial runs/facility-presence-target-roi-control/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --study-protocol reports/facility-resolution-study-protocol.json
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-resolution-control --grids 1
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-resolution-highres --grids 1
./.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-resolution-control --aggregate-only
./.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-resolution-highres --aggregate-only
./.venv/Scripts/python.exe scripts/test_facility_resolution.py
./.venv/Scripts/python.exe scripts/verify_facility_resolution.py --test-results runs/facility-resolution-test-results.json
./.venv/Scripts/python.exe scripts/report_facility_resolution.py
./.venv/Scripts/python.exe scripts/plot_facility_resolution.py
./.venv/Scripts/python.exe scripts/report_facility_target.py
./.venv/Scripts/python.exe scripts/plot_facility_target.py
```

preflight는 TRAIN 8건의 제한된 기술 점검이며 새 학습 epoch·성능 측정에 더하지 않는다.
전체 6epoch의 표본 순서 동등성은 완료 history의 row index SHA로 별도 검증한다.
원본·파생 사진·개별 주석·경로·검수 메모·가중치는 로컬 ignored 폴더에 보관한다.

### 실행 중단 기록

최초 metadata 작성에서 문자열 경로를 `Path`로 감싸는 한 줄 오류를 수정했다.
그 시점의 완료 epoch·optimizer update는 0이었다. GPU preflight를 다시 수행하고
동일 학습 조건을 소스 commit `0097b4b851f69a5921ef9e9fe247a19c5eb1109b`에서 재고정했다.
그 뒤 실행 세션 중단으로 대조군의 완료 2epoch 기록이 남았다. optimizer 복원 정보가
없어 해당 기록을 보존하고, 같은 프로토콜·초기 모델에서 대조군 6epoch를 다시 시작했다.
이전 중단 결과는 최종 대조·보강 쌍에 포함하지 않으며 [별도 집계](facility-resolution-interruption.json)에 남긴다.
누적 완료 epoch에는 보존 2epoch를 별도로 더한다. 정량 기록이 없는 중단 당시 부분
epoch 작업량은 추정 합산하지 않는다. 성능 결과에 따라 epoch·해상도·정답을 바꾼 재시도는 아니다.

한 번의 백그라운드 순차 실행은 `scripts/run_facility_resolution_pair.py`를 사용한다.
이미 존재하는 run과 증거를 덮어쓰지 않으며, 한 단계가 실패하면 다음 단계로 넘어가지 않는다.
