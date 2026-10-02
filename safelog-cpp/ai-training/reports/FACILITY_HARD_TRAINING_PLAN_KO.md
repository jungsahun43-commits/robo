# 어려운 학습 사례를 보강하는 추가 실험

## 학습 전 고정한 설정

- 실험: `facility-presence-target-hard`, seed 49, 최대18epoch, 개선 없는5epoch 뒤 종료.
- 초기 모델: 기존 세 자료 검증에서 가장 좋았던 `facility-presence-target-spatial/best.pt`.
- 자료: 기존 DACL TRAIN6,225 + Dam TRAIN1,585 + CODEBRIM TRAIN6,438, 해당 부모에서 만든 상세 조각12,041. S2DS 추가 효과와 분리해서 비교한다.
- 초기 모델의 **학습 사진** 확률을 먼저 계산한다. 균열·박락의 알려진 정답과 차이가 큰 사진의 추출 비중을 `1 + 2 * 최대차이²`로 조정한다. 증가량은 최대3배다.
- 미확인 정답(-1)은 비중 계산에서 제외한다. 정답을 자동으로 수정하지 않는다.
- 기존 양성 균형·상세 조각 비중 설정을 유지한 뒤 조정하고, 자료별 전체 추출 비중 DACL0.7/Dam0.1/CODEBRIM0.2로 다시 정규화한다.
- 위치 정답 학습·사진 판단 구조를 유지한다. 입력640, 배치8, backbone 학습률0.00004, head0.00025.
- TRAIN 전용 추출 계산의 원본 모델 SHA, 항목 순서·확률·비중 및 코드 SHA를 로컬 `TRAIN-MINING.json`과 학습 메타데이터에 보존한다. 원본 사진과 모델은 Git에 올리지 않는다.

## 평가와 종료

균열·박락 각각의 미탐률과 오탐률이 세 검증 자료에서 모두5% 미만이어야 통과한다. 검증 기준, 분리 및 항목을 이전 실험과 같게 유지한다. 검증 실패 후보의 시험 점수는 반복 선택에 쓰지 않는다.

한 실험의 epoch 증가는 성능 향상을 보장하지 않는다. 이 실험은 새로운 추출 방법의 비교이며5% 달성 약속이 아니다. 단순 학습 정답 오류도 반복해서 강조될 수 있어 비중을 제한했지만, 전문가 정답 검수가 필요할 수 있다.

메이커톤 자동 판단을 제한하고 불확실한 사진을 사람에게 돌리는 설계는 별도 평가가 필요하다. 이번 실험의 오류율 기준을 자동 판단한 사진만으로 바꾸지 않는다.

## 실행

```powershell
./.venv/Scripts/python.exe scripts/train_facility_spatial.py --name facility-presence-target-hard --seed 49 --epochs 18 --patience 5 --initial runs/facility-presence-target-spatial/best.pt --hard-mining-strength 2 --backbone-lr .00004 --head-lr .00025
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-hard
./.venv/Scripts/python.exe scripts/report_facility_target.py
./.venv/Scripts/python.exe scripts/plot_facility_target.py
```
