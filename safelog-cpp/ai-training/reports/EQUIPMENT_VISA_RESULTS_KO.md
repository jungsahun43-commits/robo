# 산업 설비 외관 이상: 회로기판 첫 학습 결과

## 앱 범위와 데이터

일반 산업현장 점검의 우선 대상을 컨베이어·전기 설비로 전환했다. 이 결과는 **VisA PCB1~4의 회로기판 근접 사진**만을 대상으로 한다. 컨베이어 벨트·전체 기계 사진·내부 고장 진단의 성능으로 일반화하지 않는다. `pipe_fryum`은 식품이므로 배관으로 취급하지 않으며 식품·생활용품 자료를 이번 설비 학습에 포함하지 않았다.

공식 자료4,416장 중 정상3,250장을 학습하고, 정상364장을 학습과 분리해 경고 임계값을 정했다. 시험은 작성자의 공식 split에 있는 정상402장+결함400장=802장이다. 원본 RGB를 완전히 동일한 사진으로 비교했을 때 중복0건이다. 실제 기판 개체·공장·촬영 세션의 독립성은 확인되지 않았다.

데이터는 [Amazon 공식 VisA](https://github.com/amazon-science/spot-diff)의 **CC BY4.0**이다. 인용: Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer, *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation*, ECCV2022. [AWS 공식 데이터 안내](https://registry.opendata.aws/visa/)에도 같은 라이선스가 명시돼 있다. 제공 ZIP이 아닌 공식 TAR의 전체 SHA256은 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`이다.

## 실제 학습과 시험 분리

ResNet18 ImageNet 교사 모델은 동결하고, 새로운 학생 모델의 신경망 가중치를 정상 사진으로 **각 종류20epoch·총80개 종류별 모델 epoch**, optimizer **2,080회** 실제 업데이트했다. 전체 모델 학습·보정·CPU/GPU 재로드 시간은 **2.46분**이다. 사전 검증3회 업데이트는 학습 epoch에 포함하지 않는다.

정상 특징을 모사하는 학생·교사 특징 피라미드 방식이며, [STFPM 원 논문](https://arxiv.org/abs/2103.04257)을 참고한 별도 실험이다. optimizer·학습 횟수 등 원 논문 조건 전체를 재현한 것은 아니다. 단순 설정 변경이나 임계값 조절만 한 작업이 아니다.

사진은256×256 RGB·ImageNet 정규화한다. 모델은 최종20epoch로 고정했다. 정상 TRAIN의 별도91장씩으로 임계값을 계산하고, 네 모델과 임계값을 모두 기록한 다음 TEST를1회 추론했다. TEST를 학습·모델 선택·임계값 조절에 사용하지 않았다. 정상 보정 alpha0.05는 시험 오탐률5% 미만을 보장하지 않는다.

## 공식 TEST 사진 단위 결과

| 기판 종류 | 오탐 / 정상 | 오탐률 | 미탐 / 결함 | 미탐률 | 이미지 AUROC |
|---|---:|---:|---:|---:|---:|
| pcb1 | 7/100 | 7.00% | 22/100 | 22.00% | 0.9505 |
| pcb2 | 8/100 | 8.00% | 17/100 | 17.00% | 0.9584 |
| pcb3 | 4/101 | 3.96% | 35/100 | 35.00% | 0.9252 |
| pcb4 | 9/101 | 8.91% | 0/100 | 0.00% | 0.9848 |

합산 오탐 28/402 = **6.97%**, 합산 미탐 74/400 = **18.50%**. 합산 값으로 개별 종류의 취약점을 감추지 않는다. 네 종류의 오탐·미탐8개 비율이 모두5% 미만인 기준은 **미달**이다.

AUROC는 정상·결함 점수의 순위 분리 정도로, 0.95를 정확도95%나 오차5%로 해석하지 않는다. 픽셀 위치 정확도는 측정하지 않았다. 출력 heatmap은 상대적인 특징 불일치 시각화로 확정된 결함 위치 정답이 아니다.

## 앱 결과와 다음 단계

`equipment_category`를 사용자가 지정하고 해당 종류의 근접 사진을 넣는 별도 연구 CLI를 제공한다. 경고는 `appearance_anomaly_suspected`, 낮은 점수는 `no_visual_anomaly_detected`다. 둘 다 `mechanical_failure_state=not_assessed`, `human_confirmation_required=true`로 기록한다. 이상 점수는 확률이 아니며 일반 설비 종류는 거절한다.

앱은 설비 등록 → 종류별 촬영 → 외관 이상 후보 표시 → 점검자 확인 → 조치·재점검 → 점검 기록/보고서의 흐름으로 확장한다. 소음·진동·온도·전류·가동 상태는 추가 측정 또는 점검 항목으로 기록하며 사진 점수로 고장을 확정하지 않는다.

컨베이어는 **BeltCrack14ks**의 실제 벨트 균열 bbox 정답으로 별도의 모델을 준비한다. 벨트 사진을 건물의 콘크리트 균열로 재분류하지 않는다. 작성자 sequence 단위 split을 유지하고 인접 프레임을 임의로 섞지 않는다. 이 보고서의 PCB 결과는 벨트 모델 결과가 아니다.

## 검증·실행

임계값·누락 분모·동점 AUROC·교사 gradient 차단·지원하지 않는 설비 거절을 검증하는 단위 테스트8개를 통과했다. 기존 소스170개·파일79,412개와 새 입력9,237개의 SHA/크기/수정시각을 검증했다. 네 모델의 strict CPU/GPU 재로드 비교를 통과했다. 기존 앱 기본 모델을 변경하지 않았다.

```powershell
.venv/Scripts/python.exe scripts/infer_equipment_visa.py --category pcb1 --checkpoint runs/equipment-visa-stfpm-v1/pcb1/final.pt --image <pcb1-closeup.jpg>
```

[고정 프로토콜](equipment-visa-study-protocol.json), [시험 수치](equipment-visa-results.json), [보존·업데이트 검증](equipment-visa-verification.json), [학습 전 시험 미사용·임계값 고정 기록](equipment-visa-frozen-operating-points.json).
