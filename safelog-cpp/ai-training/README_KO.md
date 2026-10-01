# SafeLog 산업안전 AI 학습

이 폴더는 역할 3의 데이터 다운로드, 검증, YOLO 파인튜닝, 평가, ONNX 변환과
C++ 앱용 로컬 API 서버를 재현하기 위한 코드다. 원본 데이터와 학습 결과는 Git에
올리지 않는다.

## 1. 설치

Windows에 Python 3.11 또는 3.12를 설치하고 PowerShell에서 실행한다.

```powershell
cd safelog-cpp/ai-training
./setup_windows.ps1
./.venv/Scripts/Activate.ps1
```

설치 스크립트는 NVIDIA GPU가 있으면 CUDA 12.6용 PyTorch를 설치하고, 없으면 CPU용을
설치한다. CPU용을 명시하려면 `./setup_windows.ps1 -CpuOnly`를 쓴다. 실제 검증된 패키지
버전은 `requirements-tested.txt`에 있으며 `python scripts/check_environment.py`로 CUDA
사용 여부를 확인한다. 앱은 C++/Qt, 모델 학습과 추론 서버는 Python으로 개발한다.

## 2. 공식 데이터 다운로드와 검사

```powershell
python scripts/download_dataset.py construction-ppe
python scripts/audit_yolo_dataset.py data/construction-ppe/data.yaml
python scripts/download_dataset.py chvg
python scripts/prepare_chvg.py
python scripts/audit_yolo_dataset.py data/chvg-yolo/data.yaml
python scripts/download_dataset.py indoor-fire-smoke
python scripts/prepare_fire_smoke.py
python scripts/audit_yolo_dataset.py data/indoor-fire-smoke/data.yaml
python scripts/download_dataset.py sh17
python scripts/prepare_sh17.py
python scripts/audit_yolo_dataset.py data/sh17/data.yaml
python scripts/prepare_sh17_training_copy.py
```

Construction-PPE 1,416장, CHVG 1,699장, 실내 화재·연기 5,000장과 SH17 8,099장을
각각 사용한다. 출처와 라이선스는 `datasets/sources.json`에 고정했다. 다운로드 결과는
`data/`에 저장되며 Git에서 제외된다. 대회 제출 또는 모델 공개 전에는 각 라이선스의
상업 이용 및 재배포 조건을 다시 확인한다.

## 3. 모델별 학습

```powershell
python scripts/train.py --data data/construction-ppe/data.yaml --epochs 100 --batch 16 --device 0 --name ppe-baseline
python scripts/train.py --data data/indoor-fire-smoke/data.yaml --epochs 100 --batch 16 --device 0 --name fire-smoke
python scripts/train.py --data data/chvg-yolo/data.yaml --epochs 100 --batch 16 --device 0 --name chvg-ppe
python scripts/train.py --data data/sh17-1280/data.yaml --model yolo11s.pt --epochs 100 --batch 16 --device 0 --name sh17-ppe --cache disk
python scripts/train.py --data data/construction-ppe/data.yaml --model runs/sh17-ppe/weights/best.pt --epochs 100 --batch 16 --device 0 --name ppe-sh17-transfer
```

라벨 뜻이 다른 데이터셋을 억지로 한 파일로 합치지 않는다. Construction-PPE는 미착용
판단, 화재·연기는 화재 위험 판단, CHVG와 SH17은 PPE 부품 탐지와 외부 검증에 쓴다.
각 학습 결과는 `runs/<이름>/weights/best.pt`에 저장된다. 조기 종료가 작동하므로 실제
학습 횟수는 100회보다 적을 수 있다.

추가 PPE 전이 실험은 SH17 학습 결과로 Construction-PPE를 다시 학습한다. 원본 PPE
모델과 전이 모델은 검증 분할의 네 미착용 클래스 평균 AP50으로 선택하며, 독립 test
분할은 모델 선택에 쓰지 않는다. `reports/selected-models.json`에 선택 근거를 저장하고
서버 실행 스크립트가 선택된 모델을 사용한다.

SH17은 8,099장의 고해상도 이미지와 75,994개 객체를 포함해 저장 공간과 시간이 많이
필요하다. 공식 분할은 train 6,479장과 val 1,620장이며 별도 test가 없다. 따라서 SH17
수치는 공식 val 기준이라고 명시한다. SH17은 더 큰 `yolo11s`를 사용하고 RTX 4070
SUPER 12GB에서 배치 16으로 학습한다. 메모리가 부족한 PC에서는 배치를 8로 낮춘다.

## 4. 평가와 ONNX 변환

```powershell
python scripts/evaluate.py runs/ppe-baseline/weights/best.pt data/construction-ppe/data.yaml
python scripts/export_onnx.py runs/ppe-baseline/weights/best.pt
```

평가 결과는 콘솔과 `runs/evaluation-<학습명>.json`에 저장된다. 모든 모델 평가 후
`python scripts/summarize_results.py`를 실행하면 비교표가 만들어진다. 발표에는
precision, recall, mAP50, mAP50-95와 테스트 이미지 수를 함께 기록한다.

## 5. C++ 앱 연결 서버

```powershell
$env:SAFELOG_MODEL_PATH="runs/ppe-baseline/weights/best.pt"
$env:SAFELOG_FIRE_MODEL_PATH="runs/fire-smoke/weights/best.pt"
# 선택: 추가 PPE 모델도 함께 추론할 때만 지정한다.
# $env:SAFELOG_AUX_MODEL_PATH="runs/chvg-ppe/weights/best.pt"
python -m uvicorn safelog_ai.server:app --host 0.0.0.0 --port 8080
```

앱 실행 전 서버 PC의 IP를 설정한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://127.0.0.1:8080"
```

서버는 C++ 앱이 요구하는 세 API를 제공한다.

- `POST /v1/analyze-hazard`
- `POST /v1/compare-action`
- `POST /v1/summarize`

서버는 지정된 모델을 차례로 실행하고 탐지 결과를 하나의 위험 판정으로 합친다. 운영
기본 조합은 미착용을 직접 학습한 Construction-PPE 모델과 화재·연기 모델이다. CHVG와
SH17은 클래스 의미가 다르므로 부품 탐지 근거를 추가하는 보조 모델로 선택한다.

한 번에 여섯 모델을 시작하려면 `./start_ai_server.ps1 -AllModels`를 실행한다. 기본
`./start_ai_server.ps1`은 PPE+화재 모델을 사용한다. 모델 ZIP을 `ai-training/`에 풀면
`models/`의 학습된 파일을 우선 사용한다. 원본 데이터셋 없이도 서버를 실행할 수 있다.

PPE 후보는 `ppe-baseline`과 SH17 가중치에서 전이 학습한 `ppe-sh17-transfer`다.
`reports/selected-models.json`이 있으면 서버 실행 스크립트가 선택된 후보를 사용한다.
선택 기준은 Construction-PPE val에서 미착용 네 클래스의 AP50 평균이며 test 지표는
선택에 사용하지 않는다. 두 후보와 모든 클래스의 평가 결과는
`reports/TRAINING_RESULTS_KO.md`에 기록된다. `no_boots`의 val 정답이 4개뿐이므로
선택 점수만으로 현장 성능을 확정하지 않는다.

Windows 앱을 같은 PC에서 실행할 때는 `SAFELOG_AI_BASE_URL=http://127.0.0.1:8080`을
설정한다. Android 휴대폰에서는 같은 Wi-Fi에 연결한 서버 PC의 실제 IP를 사용한다.
`127.0.0.1`은 휴대폰 자체를 가리킨다. 앱 설정 저장 또는 APK의 서버 주소 전달은
역할 4의 앱 시작 설정과 연결한다.

## 시설 위주의 추가 모델

시설 표면 손상과 금속 부식을 별도 데이터셋·모델로 학습한다.
출처와 라이선스는 `datasets/facility_sources.json`에 기록했다.
데이터 준비부터 학습·시험 평가·ONNX 변환까지 재현하려면 다음을 실행한다.

```powershell
./run_facility_training.ps1
```

이미 다운로드했다면 `-SkipDownload`, 자료 준비만 하려면 `-PrepareOnly`를 사용한다.
학습 중단 후 같은 명령을 다시 실행하면 last.pt에서 이어 학습한다.
GPU 학습은 순차 진행한다. 시설 학습은 입력 960px·YOLO11s·배치 8을 사용한다.
실제 학습 모델이 없는 초기 상태에서는 시설 서버를 실행할 수 없다.

완료된 모델 ZIP을 ai-training에 풀고 다음 명령으로 시설 모델 2개와 화재·연기를 시작한다.

```powershell
./start_ai_server.ps1 -FacilitiesOnly
```

`-Facilities`는 PPE+화재+시설 4개, `-AllModels`는 보조 PPE까지 6개를 실행한다.
`/health`에 모델 목록과 입력 크기가 표시된다. 시설 모델은 960px로 추론하며 기존 모델은 640px다.
시설 성능·고정 신뢰도 0.25에서의 미탐·오탐 비중·변환 규칙·자료 한계는
`reports/FACILITY_TRAINING_RESULTS_KO.md`에서 확인한다.

공개 교량 사진의 손상 폴리곤과 금속 부식 마스크를 영역 박스로 변환했다.
백화/젖은 표면을 배관 누수로, 철근 노출을 전선 노출로 해석하지 않는다.
위험 등급은 점검 우선순위 규칙이며 구조 진단 모델의 판단이 아니다.
모든 제안과 조치 후 상태는 점검자의 최종 확인을 거친다.

PPE/화재/시설 탐지는 실제 파인튜닝한 YOLO 모델이다. 위험도·조치 문장과 `/v1/summarize`는
현재 규칙과 템플릿으로 생성한다. 문장을 생성하는 별도의 언어 모델을 학습한 것은 아니다.
`detections`에는 탐지 모델, 클래스, 신뢰도와 좌표가 포함돼 앱의 원본 AI 분석 기록에
함께 저장된다. PPE 부품이 탐지되지 않았다는 사실만으로 미착용을 확정하지 않는다.

## 6. 다시 학습하거나 팀에 전달하기

데이터셋 준비 후 `./run_training_suite.ps1`은 학습·평가·ONNX 변환을 순서대로 실행한다.
기존 `last.pt`가 있으면 이어 학습하며 완료된 학습은 반복하지 않는다. 학습명을 새로
지정하면 별도의 실험을 시작한다. SH17 사본은 이미지의 최대 변을 1280으로 줄이고
정규화된 라벨은 유지한다. 이 과정에서 모든 이미지의 VOC 클래스와 YOLO 번호를 대조한다.
기존 PPE/화재 모델 입력 크기는 640이고 시설 모델은 960이다.

```powershell
python scripts/train.py --data data/sh17-1280/data.yaml --name sh17-ppe --resume --device 0
python scripts/evaluate.py runs/sh17-ppe/weights/best.pt data/sh17-1280/data.yaml --device 0 --split val
python scripts/summarize_results.py
python scripts/build_training_report.py
python scripts/package_models.py
```

시설 학습·시험·ONNX·시설 보고서까지 준비되면 ZIP에 시설 모델을 자동으로 포함한다.
시설 모델 포함을 반드시 검사하려면 `python scripts/package_models.py --require-facilities`를 사용한다.
기존 PPE/화재 학습만 완료한 상태에서는 기존 5개 모델로 ZIP을 만든다.

GitHub에는 코드·출처·결과 보고서가 올라가고, 데이터·가중치는 제외된다. 별도로 생성된
`artifacts/safelog-trained-models.zip`을 팀원에게 전달한다. 팀원은 동일 브랜치를 받고
설치 스크립트를 실행한 뒤 ZIP을 `ai-training/`에 풀어 서버를 시작한다. ZIP에는 PT와
ONNX, 모델별 SHA256과 평가 지표가 들어 있다.

검사는 `python -m unittest discover -s tests -v`로 실행한다. 직접 설치했다면 먼저
`pip install -r requirements-dev.txt`로 테스트 의존성을 설치한다.

## 재현성과 발표 시 기록할 내용

- 데이터셋 이름, 버전, 라이선스, 이미지 수와 분할 수
- 학습 모델(`yolo11n.pt` / SH17·전이·시설은 `yolo11s.pt`), 입력 크기(기존 640 / 시설 960), seed 42, 실제 epoch
- 학습에 쓰지 않은 test 분할의 precision, recall, mAP50, mAP50-95
- 잘 되는 클래스와 표본 부족으로 약한 클래스
- AI 결과는 제안이며 안전관리자가 최종 확인한다는 앱 흐름

## 클래스 주의사항

Construction-PPE에는 `no_helmet`, `no_gloves`, `no_boots`, `no_goggle`가 있지만
`no_vest`는 없다. 따라서 안전조끼 미착용 판단을 이 데이터만으로 학습했다고 발표하면
안 된다. `no_vest`는 별도 데이터 또는 팀 자체 라벨을 추가한 뒤 활성화한다.
