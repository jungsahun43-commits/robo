# 원본 세부 항목을 함께 구분하는 보조 학습

학습 시작 전에 다음 설정을 고정한다.

- 실험 `facility-presence-target-auxiliary`, seed50, 최대18epoch, 개선 없는5epoch 뒤 종료.
- 초기 가중치는 기존 최고 공간 모델 `facility-presence-target-spatial/best.pt`이다. 앞선 어려운 사진 반복 실험은 기존 최고를 넘지 못해 초기 가중치로 쓰지 않는다.
- 데이터·분리·기본 손실·입력640·배치8·추출량14,248/epoch는 공간 모델과 같다. 학습률 backbone0.00004/head0.00025, 자료 추출 비중0.7/0.1/0.2. 어려운 사진 추가 비중과 S2DS 추가는 끈다.
- 새 보조 선형 분류기는 같은 특징으로 DACL 원래19종의 사진 태그를 예측한다. 보조 손실 비중0.5, masked focal gamma1, 양성 가중치0.2~6이다. 태그는 상호 배타적인 단일 종류가 아닌 복수 태그다.
- 원래 TRAIN 전체 사진6,225개만 보조 정답을 받는다. 부분 조각과 다른 자료에는 이19항목의 정답을 미확인(-1)으로 두며, 부모 전체 사진의 태그를 부분 조각의 정답으로 복사하지 않는다.
- 기존 균열·박락 및 나머지5종의 정답은 수정하지 않는다. Rockpocket·Hollowareas 등 원본 표기를 다른 손상 종류로 자동 합치지 않는다.
- 공개 추론은 기존7개 항목만 반환한다. 보조 항목을 시설 진단으로 새로 발표하거나 API로 내보내지 않는다.
- 모델 선택은 동일한 세 검증 자료의 균열·박락별 미탐률·오탐률 최댓값이다. 모두5% 미만이어야 통과하고, 검증 실패 모델은 보류 시험으로 반복 선택하지 않는다.

이 방법은 검증 오류와 원본 태그의 동시 출현을 관찰해서 세운 가설이다. 동시 출현이 오류의 원인이라는 증거는 아니며, 성능 개선과5% 달성을 보장하지 않는다.

```powershell
./.venv/Scripts/python.exe scripts/prepare_facility_auxiliary.py
./.venv/Scripts/python.exe scripts/train_facility_spatial.py --name facility-presence-target-auxiliary --seed 50 --epochs 18 --patience 5 --initial runs/facility-presence-target-spatial/best.pt --auxiliary-manifest data/facility-auxiliary-training/train.json --auxiliary-weight .5 --backbone-lr .00004 --head-lr .00025
./.venv/Scripts/python.exe scripts/evaluate_facility_target.py select --name facility-presence-target-auxiliary
```
