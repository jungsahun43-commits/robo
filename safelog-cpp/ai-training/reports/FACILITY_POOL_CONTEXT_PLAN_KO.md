# 좁은 피크·넓은 증거 pooling 대조 계획

학습 결과를 보기 전에 한 가지 후보와 비교 조건을 고정한다. 이전 stride 4 구조가 최대 오류와 작은 손상 미탐을 개선하지 못했으므로, 이번에는 사진 단위 pooling만 바꾼다. 성능 향상은 아직 확인하지 않았다.

## 바꾸는 부분

기존 모델은 전체 사진 high feature 평균, spatial map의 상위 32개 셀 평균, 학습 mix를 이미 결합한다. 후보는 같은 map의 상위 256개 셀 평균과 상위 32개 평균의 차이에 클래스별 계수 7개를 학습한다.

```text
narrow = mean(top32(map))
broad  = mean(top256(map))
spatial = narrow + tanh(context_gate) * (broad - narrow)
photo = 기존 global/mix 결합
```

계수는 0으로 시작하여 초기 사진·map·19종 보조 출력이 기존 초기 모델과 정확히 같다. 새 상태 tensor 1개와 파라미터 7개만 추가한다. tanh 계수는 음수도 가능하므로 항상 넓은 영역을 선호하는 구조라고 설명하지 않는다. 상위 셀 통계는 공간 인접성이나 현장 의미를 보존하지 않는다.

Backbone·map·global head·mix·보조 head의 계산 구조는 유지한다. 두 군 모두 전체 모델을 fine-tune하므로 학습 후 map/보조 출력 값이 서로 같다는 뜻은 아니다. 이전 stride 4 분기와 순위 추가 손실은 함께 넣지 않는다.

## 고정 조건

| 항목 | 조건 |
|---|---|
| 초기 모델 | `facility-presence-target-roi-control`, SHA `0773b64f85bde27c256580be2fe36fabc0956c8b0bc3c087fc49716ba61d6c6a` |
| 대조/후보 | `facility-presence-target-context-control` / `facility-presence-target-context-pool` |
| 예산 | 각 6epoch, patience 6, 총 12epoch |
| 입력/배치/표본 | 640 / 8 / epoch당 복원 추출 14,248행 |
| seed | 모델 55, sampler 55, 학습 worker 56, 모델 생성 후 global 57 |
| VAL worker | DACL 155 / Dam 156 / CODEBRIM 157 |
| 학습률 | backbone 0.00004 / 모든 head·새 계수 0.00025 |
| 학습 정답 | 기존 원본 TRAIN·기존 파생 crop·DACL 원본 19종 보조 태그 |
| 손실 | 기존 사진/픽셀/보조 손실, 보조 가중치 0.5, 순위 손실 0 |
| 출처 비중 | DACL/Dam/CODEBRIM 70/10/20% |
| 모델·임계값 선택 | 같은 세 VAL에서 균열·박락 FNR/FPR minimax, 전체 사진 grid1 |

K=256·계수 범위·자료·정답·예산을 결과에 맞춰 재선택하지 않는다. 14,248행에는 저자 patch가 포함되며 복원 추출 횟수는 새 독립 사진 수가 아니다. 새 산업 현장 사진과 전문가 확정 라벨은 이번 비교에 추가하지 않는다.

## 채택·중단 기준

초기 모델과 대조군 **각각**보다 최대 검증 오류 최소 0.5pp 개선, 모든 target FNR/FPR 회귀 2pp 이하, 다른 알려진 항목 AP 회귀 0.02 이하를 요구한다. DACL 작은 손상 198개 사진·항목 사례의 미탐 합계도 각각보다 최소 2건 줄고 각 작은 항목 FNR 회귀는 2pp 이하여야 한다.

연구 후보 선정은 엄격한 5% 목표나 앱 배포 통과와 별도다. 균열·박락 × 세 출처 × FNR/FPR의 12개 비율이 모두 5% 미만이어야 기존 목표를 통과한다. 미달이면 시험 추론·반복 시험 선택·앱 프로필 교체를 진행하지 않는다. 사전에 정한 한 쌍을 마친 뒤 결과를 보고한다.

## 실행

```powershell
./.venv/Scripts/python.exe scripts/preflight_facility_context.py
./.venv/Scripts/python.exe scripts/train_facility_context.py --name facility-presence-target-context-control --model-variant control --seed 55 --epochs 6 --patience 6 --initial runs/facility-presence-target-roi-control/best.pt --draws-per-epoch 14248 --backbone-lr .00004 --head-lr .00025 --auxiliary-manifest data/facility-auxiliary-training/train.json --study-protocol reports/facility-pool-context-protocol.json
./.venv/Scripts/python.exe scripts/train_facility_context.py --name facility-presence-target-context-pool --model-variant context --seed 55 --epochs 6 --patience 6 --initial runs/facility-presence-target-roi-control/best.pt --draws-per-epoch 14248 --backbone-lr .00004 --head-lr .00025 --auxiliary-manifest data/facility-auxiliary-training/train.json --study-protocol reports/facility-pool-context-protocol.json
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-context-control --grids 1
./.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-context-control --aggregate-only
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-context-pool --grids 1
./.venv/Scripts/python.exe scripts/analyze_facility_target.py --name facility-presence-target-context-pool --aggregate-only
./.venv/Scripts/python.exe scripts/verify_facility_context.py
./.venv/Scripts/python.exe scripts/report_facility_context.py
```

완료된 run은 덮어쓰지 않는다. 기술 검증에는 실제 테스트 기록과 학습 전 Git 소스 고정 기록이 필요하다.

## 재현과 검수

같은 초기 상태·초기 출력·4 worker의 제한된 3batch 증강 재현·실제 6epoch 표본 순서를 비교한다. GPU AMP에서 새 계수의 gradient와 학습을 확인하고, 마지막에 저장 checkpoint를 CPU에서 엄격하게 로드한다. 이 점검은 정확도 측정과 별도다.

이전 stride 4 실험의 factory를 포함한 정확한 소스는 commit `78276772678802a9b9660ea46b195faba7ca79a0`에 보존했다. 이번 factory의 새 ARCH 등록을 이유로 과거 학습 코드나 source hash를 바꾸지 않는다. 과거 보고 재생성은 해당 고정 commit에서 진행한다.

기존 TRAIN 검수 묶음은 다른 초기 모델로 선택한 것이며 전문가의 원본 정답 수정은 0건이다. AI 오답은 검수 후보를 보여주는 근거이고 원저자 라벨을 자동 변경하지 않는다. 미확인 다른 항목을 정상·음성으로 바꾸지 않는다.

단일 seed·반복 VAL 선택·같은 사진에서 만든 patch 상관이 있다. Wilson95는 기술 통계이며 공장의 독립 현장 성능·모델 차이의 유의성·구조 안전을 보장하지 않는다. 공개 산출물에는 집계와 hash만 포함하며 사진·원주석·개별 점검 기록·개별 절대 경로는 포함하지 않는다.

[기계 판독용 고정 조건](facility-pool-context-protocol.json)
