# SafeLog 시설 AI — 2차 피드백 보강

## 이번 출시 판단

기존 1차 모델을 기본 설정으로 유지했다. 2차 후보는 기본 모델로 채택하지 않았다.
항목-사진 쌍 합계에서 미탐은 732→731, 오탐은 309→317였다. 같은 사진에 여러 항목이 있을 수 있어 고유 사진 수가 아니다.
후보의 항목·기준은 검증으로 고정했다. 시험 오류를 보고 기준을 다시 조정하지 않고 전체 후보의 교체 여부를 판단했다.
시험 분할을 출시 판단에도 사용했으므로 독립적인 성능 보증 자료는 아니다. 실제 현장 성능 주장을 위해서는 새로운 외부 평가가 필요하다.

## 목적과 방법

1차에서 늘어난 오탐과 개선되지 않은 항목을 확인하고, 같은 학습 사진 6,225장으로 사진 분류 모델을 추가 학습했다.
1차 best.pt에서 시작해 입력을 384→512px로 키웠다. 양성 가중치 BCE 대신, 잘못 높은 점수를 주는 음성 항목의 BCE 손실을 최대 3배 강조했다.
검증/시험 사진은 optimizer에 입력하지 않았다. 원본 라벨도 바꾸지 않았다. 새 데이터가 추가된 실험은 아니다.
학습 10 epoch, 가장 좋은 검증 사진 분류 macro AP 71.8%. 박스 mAP·현장 안전 정확도가 아니다.
항목별 후보 임계값을 검증 사진 710장에서 탐색했다. 1차 대비 사진 미탐률·오탐률이 모두 증가하지 않는 후보 중 FNR+FPR이 최소인 모델을 선택했다.
동률이거나 나아진 후보가 없으면 기존 항목을 유지한다. 선택되지 않은 모델의 항목은 기준 1.00으로 비활성화한다.
선택된 모델과 항목별 기준을 고정한 뒤 기존 시험 사진 975장을 재평가했다. 시험 결과가 나빠진 항목도 공개하며 시험 결과로 설정을 다시 고르지 않았다.

## 검증에서의 선택

| 항목 | 선택 모델 | 기준 | 1차 검증 미탐 | 2차 검증 미탐 | 1차 검증 오탐 | 2차 검증 오탐 |
|---|---|---:|---:|---:|---:|---:|
| 표면 균열 | facility-presence-refined | 0.90 | 49.3% | 48.3% | 2.8% | 2.8% |
| 콘크리트 박리 | facility-presence-refined | 0.77 | 42.0% | 39.8% | 11.9% | 11.7% |
| 녹 얼룩 | facility-presence-refined | 0.68 | 23.2% | 23.0% | 12.2% | 11.9% |
| 철근 노출 | facility-presence | 1.00 | 58.8% | 58.8% | 0.3% | 0.3% |
| 젖은 표면 | facility-presence | 0.98 | 65.6% | 65.6% | 2.9% | 2.9% |
| 백화 | facility-presence-refined | 0.74 | 30.2% | 30.2% | 7.4% | 7.3% |
| 표면 공동·파임 | facility-presence | 0.86 | 44.3% | 44.3% | 5.0% | 5.0% |

## 같은 시험 사진에서의 비교

항목의 존재 여부만 평가한다. 박스 위치와 현장 전체 안전/불안전 정확도는 평가하지 않는다.
미탐률=FN/(TP+FN), 오탐률=FP/(FP+TN). 항목별 양성/음성 사진을 분모로 사용한다.

| 항목 | 1차 미탐 | 2차 미탐 | 1차 오탐 | 2차 오탐 | 추가 회수/놓친 양성 | 제거/추가 오탐 |
|---|---:|---:|---:|---:|---:|---:|
| 표면 균열 | 45.1% | 45.1% | 4.5% | 4.6% | 2/2 | 1/2 |
| 콘크리트 박리 | 40.9% | 39.8% | 13.9% | 15.3% | 7/2 | 3/10 |
| 녹 얼룩 | 20.4% | 21.3% | 12.2% | 12.4% | 2/6 | 5/6 |
| 철근 노출 | 52.9% | 52.9% | 1.5% | 1.5% | 0/0 | 0/0 |
| 젖은 표면 | 69.4% | 69.4% | 4.1% | 4.1% | 0/0 | 0/0 |
| 백화 | 39.8% | 39.8% | 5.6% | 5.5% | 2/2 | 7/6 |
| 표면 공동·파임 | 45.5% | 45.5% | 7.1% | 7.1% | 0/0 | 0/0 |

## 해석 및 다음 피드백

- 검증에서 지킨 오차 제한은 시험·실제 현장에서 보장되지 않는다. 검증의 작은 차이가 시험에서 반대로 나올 수 있다.
- 동일 검증·시험 분할을 반복 확인하고 있어 독립적인 외부 검증으로 해석할 수 없다. 장면 ID가 없어 사진 간 장면 독립성도 보장하지 못한다.
- 검증 오류 예시에서 물기·백화·녹 얼룩의 외관이 비슷해 보이는 사진과 작은 정답 영역이 보였다. 라벨에 없는 항목을 오탐으로 집계하지만 원본 주석의 완전성을 별도로 확인한 것은 아니다.
- 모델 출력으로 원본 라벨을 임의 수정하지 않았다. 현장 점검자가 판단한 정상/손상 사례를 확보해 다음 외부 평가 자료로 쓰는 것이 필요하다.
- 기존 시설 박스 모델·화재·PPE는 변경하지 않았다. 금속 부식은 정상 시험 사진이 없는 한계를 유지한다.
- `box=null`, `evidence_scope=photo_presence`인 의견은 위치를 찾은 결과가 아니다. 사람의 최종 확인이 필요하다.

## 팀원 실행

최신 feature/ai-engine 코드와 모델 ZIP을 함께 받은 뒤 기존처럼 실행한다.

```powershell
./start_ai_server.ps1 -FacilitiesOnly
```

`/health.photoClassifiers`에서 활성 모델·해상도·항목별 기준을 확인한다. 기본 설정은 facility-inference-profile.json, 1차 설정은 facility-inference-profile-round1.json, 실험 후보는 facility-inference-profile-round2-candidate.json이다.
실험 후보를 기본 설정으로 복사해 사용하지 않는다. 기본 서버는 출시 판단을 통과한 설정을 사용한다.

## 보강 재현

```powershell
./.venv/Scripts/python.exe scripts/train_facility_presence.py --name facility-presence-refined --initial runs/facility-presence/best.pt --hard-negatives --imgsz 512 --batch 12 --epochs 18 --patience 5
./.venv/Scripts/python.exe scripts/refine_facility_feedback.py select
./.venv/Scripts/python.exe scripts/refine_facility_feedback.py test
./.venv/Scripts/python.exe scripts/refine_facility_feedback.py export
./.venv/Scripts/python.exe scripts/refine_facility_feedback.py release
./.venv/Scripts/python.exe scripts/build_facility_feedback_report.py
./.venv/Scripts/python.exe scripts/package_models.py --require-facilities --require-optimization
```

이미 존재하는 학습 실험은 덮어쓰지 않는다. 순서대로 검증 선택 → 고정 설정의 시험 평가 → 모델 전달을 수행한다.
