# 정규화 통계 고정 후속 비교 계획

작성: 2026-10-07. 가중치4 결과를 완료한 뒤 다음 후보 하나를 학습 전에 고정한다.

## 앞선 결과와 이번 선택의 근거

보존 손실 가중치4의 최대 source-VAL 미탐·오탐은22.28%로 원본과 같았다. DACL 철근 노출 AP는0.6758로 가중치1의0.6715보다 조금 높아졌으나, 원본0.7010 대비 보존 허용 범위를 넘었다. CODEBRIM 백화 AP 하락은0.0367, 작은 손상 FN 합계는72건으로 기존 기준에 미달했다.

CPU에서 네 체크포인트의 상태만 비교했다. 모델 추론이나 VAL 재검증은 수행하지 않았다. BN은47개 층·12,328채널이고, 세 추가 학습 모델의 선택 epoch3에서 모든 BN counter가 초기보다5,343회 증가했다.

| 원본 대비 상태 벡터 상대 L2 변화 | 가중치0 | 가중치1 | 가중치4 |
|---|---:|---:|---:|
| BN 저장 평균 | 5.88% | 5.44% | 6.14% |
| BN 저장 분산 | 8.15% | 7.67% | 7.49% |
| BN 학습 계수 γ·β | 0.31% | 0.31% | 0.29% |
| head의 BN 외 파라미터 | 14.89% | 14.80% | 15.17% |

위 수치는 모델 상태의 변화이며 오류율이 아니다. head 변화도 있으므로 BN이 성능 하락의 원인이라고 단정하지 않는다. 측정된 정규화 정책 하나를 비교할 근거로 사용한다.

## 바꾸는 한 가지와 학습량

학생 모델의 BN만 매번 `model.train()` 직후 평가 모드로 설정한다. 학습 중 현재 배치의 평균·분산 대신 원본0773에 저장된 평균·분산을 사용한다. 저장 평균·분산·counter141개 텐서를 그대로 유지하면서 BN 계수 γ·β94개 텐서와 backbone·head 파라미터는 계속 학습한다. eps와 momentum은 바꾸지 않는다. **저장 버퍼뿐 아니라 학습 시 정규화 방식도 달라지는 조건**이다.

원본0773에서 새 학생을6epoch 학습한다. 완료된 가중치4·일반 BN 모델을 대조군으로 재사용하며 대조군 재학습은0epoch다. 입력640, T2·보존 손실 가중치4, 원래7종 사진/80×80 픽셀/19종 보조 정답, 원본 추출 배열·증강·배치8·seed56·학습률·스케줄러·교사 실행을 유지한다.

## 검사와 판정

사전 GPU 검사에서는8장짜리 TRAIN 배치1개로 업데이트를 수행한다. 실제 BN47개 층이 평가 모드로 실행됐는지,141개 버퍼가 원본 해시와 같은지,94개 계수의 경사가 존재하고 유한하며 일부가0이 아닌지 확인한다. 실제 학습에서는 매 epoch의 버퍼 불변·모드 적용·마지막 배치의 계수 경사를 기록한다. 초기 AMP 건너뜀은 실제 업데이트와 따로 센다.

원본과 재사용 대조군의 비교에는 기존 규칙을 적용한다. 성능 보존 기준은 알려진 다른5항목의 출처별 AP 하락이 원본 대비0.02 이하, DACL 철근 노출 AP가 이번 가중치4 대조군보다0.02 이상 회복, 균열·박락 최대 오류 악화가 원본 및 대조군보다2%p 이하이다. 기존 연구 기준·작은 손상 누락 기준·12개 누락/오탐 각각5% 미만 기준은 따로 보고한다. 결과를 본 뒤 기준을 완화하지 않는다.

이전 VAL 결과와 상태 감사를 보고 정한 후속 실험이다. 독립 공장 현장 성능이나 BN의 인과 효과를 입증하지 않는다. TEST 추론·원본 정답 수정·기본 앱 모델 교체는 수행하지 않는다.

## 재현 순서

원본 자료와 완료 가중치4 기록을 가진 `safelog-cpp/ai-training`에서 실행한다. 기존 산출물은 덮어쓰지 않는다.

```powershell
.venv/Scripts/python.exe scripts/test_facility_batchnorm.py
.venv/Scripts/python.exe scripts/facility_batchnorm_study.py --declare
.venv/Scripts/python.exe scripts/preflight_facility_batchnorm.py
# 고정 소스·프로토콜을 커밋한 뒤
.venv/Scripts/python.exe scripts/verify_facility_batchnorm.py --snapshot-before-training
.venv/Scripts/python.exe scripts/run_facility_batchnorm.py
```

마지막 실행은6epoch 학습 뒤 공개 VAL grid1 재추론, 작은 손상 집계, 완료 검증, 결과 보고서와 그림 생성을 차례로 수행한다.

GitHub의 그림 링크는 파일 이름의 대소문자를 구분한다. 최종 공개 그림은 아래처럼 입력·출력 이름을 명시해 생성했다. 기본 이름으로 생성한 첫 그림과 sidecar는 `runs/facility-batchnorm-original-plot-case`에 보존한 뒤 실행했다. 측정값과 고정 소스는 바꾸지 않았다.

```powershell
.venv/Scripts/python.exe scripts/plot_facility_batchnorm.py --input reports/facility-batchnorm-study-comparison.json --output reports/facility-batchnorm-study-comparison.png
```
