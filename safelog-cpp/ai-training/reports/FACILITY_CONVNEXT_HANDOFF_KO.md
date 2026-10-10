# ConvNeXt 연구 모델 전달 안내

역할3이 준비한 시설 사진 분류 연구 후보다. 아래 지표는 같은 공개 VAL의 반복 탐색 결과이며 독립 산업현장 정확도가 아니다. AP는 양성 사진을 앞쪽에 배치하는 순위 지표이며 정확도 백분율과 다르다.

실제 학습6epoch, 선택6epoch. 최대 오탐·미탐률: 기존22.0207% → 신규22.2798%.
연구 후보 기준: 미달. 기존 항목 보존 기준: 통과. 각 비율5% 미만: 미달.

## 시설 항목별 AP

| 출처·항목 | 기존 대조군 | 신규 모델 | 차이 |
|---|---:|---:|---:|
| dacl · rust_stain | 0.8820 | 0.9194 | +0.0373 |
| dacl · exposed_rebar | 0.7129 | 0.7819 | +0.0690 |
| dacl · wet_surface | 0.5205 | 0.5579 | +0.0374 |
| dacl · efflorescence | 0.7601 | 0.8244 | +0.0644 |
| dacl · surface_cavity | 0.6030 | 0.6975 | +0.0944 |
| codebrim · rust_stain | 0.8940 | 0.8739 | -0.0201 |
| codebrim · exposed_rebar | 0.9691 | 0.9752 | +0.0061 |
| codebrim · efflorescence | 0.8486 | 0.8560 | +0.0074 |

## 로컬 GPU 추론 비용

같은 TRAIN 사진1장,640픽셀,FP32,batch1. 모델별10회 예열 후40회 측정했다. 파일 읽기·전송·앱 네트워크·초기 모델 로딩은 제외한다.

| 모델 | 파라미터 | 추론 중앙값 | p90 | checkpoint |
|---|---:|---:|---:|---:|
| facility-presence-target-head-lr-low | 3,244,151 | 4.23ms | 4.68ms | 12.55MiB |
| facility-presence-target-convnext-finetune | 28,259,976 | 15.61ms | 17.13ms | 107.85MiB |

## 파일 사용

1. GitHub의 feature/ai-engine 브랜치에서 코드를 받는다. 기존 AI Python 환경의 Torch/Torchvision을 사용한다.
2. 이 ZIP을 safelog-cpp/ai-training 폴더에 압축 해제한다. runs/facility-presence-target-convnext-finetune/best.pt가 있어야 한다. 가중치는 Git에 포함하지 않으며 별도 ZIP으로 전달한다.
3. 사진 경로를 지정해 아래 명령을 실행하면 검증된 가중치 해시를 확인하고7종 확률을 JSON으로 출력한다. GPU가 없으면 --device cpu를 사용한다.

```powershell
cd safelog-cpp/ai-training
.\.venv\Scripts\python.exe scripts/predict_facility_convnext.py --image "사진경로.jpg" --device cuda
```

199개 state tensor의 새 모델이며 이전324개 MobileNet 가중치와 구조가 다르다. 기존 PresenceClassifier에 새 가중치 파일을 넣는 방식으로 연결할 수 없다. ConvnextResearchPresence가 strict 재로딩과 기존640픽셀 전처리를 제공한다.
역할4의 앱 화면·공용 API 계약은 기존대로 사용한다. 본 ZIP과 명령은 별도 모델 검토용이며 앱 기본 모델이나 서버 설정을 교체하지 않는다. 앱 연결 시 역할3이 서버의 모델 로더와 프로필을 별도 구현·검증해야 한다.

사람 검토가 필요한 AI 제안이다. 사진 점수는 구조 안전성이나 산업현장 검증 정확도 자체를 의미하지 않는다. 원본 데이터·사람 검수 기록·sourceTEST는 ZIP에 포함하지 않는다.
