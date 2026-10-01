# SafeLog 시설 AI — 3차 합성 공동 보강

## 적용 결과

3차 후보는 기본 모델로 채택하지 않았다. 기존 1차 facility-validation-v2를 유지한다.
판단: Per-label test acceptance failed

실험 프로필은 facility-inference-profile-round3-candidate.json이다. 기본 프로필과 분리해 시험했고 교체 판단 전에는 기본 서버에 적용하지 않았다.
## 사용한 데이터

[공식 synth-dacl 자료](https://doi.org/10.60776/9D6E4M)의 synthcavity 두 원본 ZIP을 발행처 MD5로 확인했다. 이용 조건은 CC BY-NC 4.0이다.
실제 학습 사진 6,225장 + 합성 학습 사진 5,000장. 실제 검증은 710장이다.
합성 공동 양성 5,000장, 음성 0장. 중복 픽셀 제외 0장, 실제 원본 분할과 일치 0장.
절차 생성 장면은 2,500개다. render/render_noise 변형은 같은 장면에 속하며 독립적인 촬영 현장이 아니다.
합성 자료는 별도의 실제 현장 사진이나 독립적인 평가 자료가 아니다. 실제 자료와의 중복 검사는 원본 픽셀 SHA256 기준이다.
render/render_noise 사진과 gt_cavity OR gt_render_cavities 마스크를 대응시켰다. 여섯 항목은 정답이 없는 -1로 보관하고 손실에서 제외했다.
물기 신규 현장 데이터는 이번 학습에 추가하지 않았다. 합성 공동 자료로 누수 인식 성능이 입증됐다고 해석하지 않는다.

## 학습

기존 1차 분류기로 초기화해 512px, 배치 12, seed 42, 최대 16 epoch에서 실제 8 epoch 수행했다.
합성 사진 손실 가중치는 0.2다. 실제 자료는 기존 양성 가중치 BCE, 합성 자료는 공동 항목의 BCE만 사용했다.
최고 실제 검증 사진 분류 macro AP는 71.1%다. 박스 mAP나 현장 안전 정확도가 아니다.
검증/시험 사진은 optimizer에 넣지 않았고 원본 라벨은 수정하지 않았다. 체크포인트 선택에는 실제 검증 macro AP를 사용했다.

## 검증 선택

교체 기준은 학습·시험 전에 FACILITY_ROUND3_PLAN_KO.md로 고정했다. 물기/공동만 후보로 검토했고 다른 다섯 항목은 유지했다.
오탐률이 증가하지 않고 미탐률이 최소 2%p 줄어든 기준만 자격을 부여했다. 동률에는 높은 기준을 선택했다.

| 항목 | 후보 자격 | 적용 기준 | 1차 미탐 | 선택 후 미탐 | 1차 오탐 | 선택 후 오탐 |
|---|---|---:|---:|---:|---:|---:|
| 젖은 표면 | 미통과·기존 유지 | 0.98 | 65.6% | 65.6% | 2.9% | 2.9% |
| 표면 공동·파임 | 통과 | 0.86 | 44.3% | 41.5% | 5.0% | 4.0% |

후보 자체의 진단값(기존 오탐률 이내에서 FNR+FPR이 가장 작은 기준):

- 젖은 표면: 기준 0.97, 검증 미탐 65.6%, 오탐 2.9%. 후보 자격과 실제 적용 여부는 위 표를 따른다.
- 표면 공동·파임: 기준 0.86, 검증 미탐 41.5%, 오탐 4.0%. 후보 자격과 실제 적용 여부는 위 표를 따른다.

## 고정 후보의 시험 비교

기존 시험 사진 975장을 재평가했다. 같은 사진에서 여러 항목을 평가하므로 집계 건수는 항목-사진 쌍이다.
미탐률=FN/(TP+FN), 오탐률=FP/(FP+TN). 항목 존재만 평가했고 박스 위치나 사진 전체 안전을 평가하지 않았다.

| 항목 | 1차 미탐 | 3차 미탐 | 1차 오탐 | 3차 오탐 | 회수/놓친 양성 | 제거/추가 오탐 |
|---|---:|---:|---:|---:|---:|---:|
| 표면 균열 | 45.1% | 45.1% | 4.5% | 4.5% | 0/0 | 0/0 |
| 콘크리트 박리 | 40.9% | 40.9% | 13.9% | 13.9% | 0/0 | 0/0 |
| 녹 얼룩 | 20.4% | 20.4% | 12.2% | 12.2% | 0/0 | 0/0 |
| 철근 노출 | 52.9% | 52.9% | 1.5% | 1.5% | 0/0 | 0/0 |
| 젖은 표면 | 69.4% | 69.4% | 4.1% | 4.1% | 0/0 | 0/0 |
| 백화 | 39.8% | 39.8% | 5.6% | 5.6% | 0/0 | 0/0 |
| 표면 공동·파임 | 45.5% | 43.7% | 7.1% | 7.4% | 8/5 | 8/11 |

합계: 미탐 732→729, 오탐 309→312.

## 한계와 다음 작업

- 합성 사진의 공동 크기·질감이 실제 현장과 다르다. 양성/음성 분포도 실제 촬영 분포를 대표하지 않는다.
- 같은 검증/시험을 반복 확인하고 시험을 교체 판단에 사용하므로 독립적인 성능 보증이 아니다. 새 현장/촬영 회차별 외부 평가가 필요하다.
- 검증이나 시험에서 오류가 늘어난 후보는 기본 모델로 채택하지 않는다. 시험 결과를 보고 임계값·항목을 다시 선택하지 않았다.
- 젖은 표면과 공동의 실제 정상/손상 사례를 함께 모으고 사람이 정답을 확인해야 다음 보강의 근거가 된다.
- 사진 분류의 box=null 의견은 위치를 확정하지 않는다. 최종 판단과 완료 확인은 점검자가 한다.

## 재현

```powershell
./.venv/Scripts/python.exe scripts/download_synthcavity.py
./.venv/Scripts/python.exe scripts/prepare_synthcavity.py
./.venv/Scripts/python.exe scripts/train_facility_presence.py --name facility-presence-synthetic --initial runs/facility-presence/best.pt --extra-manifest data/synthcavity-training/manifest.json --extra-weight 0.2 --imgsz 512 --batch 12 --epochs 16 --patience 5
./.venv/Scripts/python.exe scripts/refine_synthcavity.py select
./.venv/Scripts/python.exe scripts/refine_synthcavity.py test
./.venv/Scripts/python.exe scripts/refine_synthcavity.py export
./.venv/Scripts/python.exe scripts/refine_synthcavity.py release
./.venv/Scripts/python.exe scripts/build_synthcavity_report.py
```

채택하지 않은 이번 후보의 CPU ONNX 비교와 기본 설정 보존 확인은 `python scripts/verify_synthcavity.py`로 재현한다.
이 변환 비교는 모델 정확도나 Android 동작 검증이 아니다.

기존 완료 실험은 덮어쓰지 않는다. 원본 자료·학습 가중치·가상환경은 Git에서 제외한다.
