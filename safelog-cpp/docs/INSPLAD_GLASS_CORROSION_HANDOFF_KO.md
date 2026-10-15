# 역할4: 유리 애자 누락 + 연결고리 부식 연구 기능

기존 v16-r1 ZIP의 model.pt와 evaluation.json을 사용합니다. 이 ZIP에는 두 추가 분류부의 결합 차이 파일과 보존 검사용 부식 차이 파일이 있습니다. 한 공통 encoder를 읽는 한 모델입니다. 단독 앱이나 단독 모델 파일이 아닙니다.

기존 ZIP과 이 ZIP을 별도 폴더에 풀고 이 ZIP 폴더에서 기존 Python 환경으로 실행합니다. /BASE는 원래 v16-r1 폴더의 실제 경로로 바꿉니다.

```powershell
$overlay = Get-Content -Raw overlay.json | ConvertFrom-Json
python -m safelog_ai.insplad_glass_server --parent "/BASE/model.pt" --parent-sha $overlay.parent_file_sha256 --corrosion-delta corrosion-delta.pt --corrosion-delta-sha $overlay.preserved_corrosion_delta_sha256 --corrosion-calibration corrosion-calibration.json --delta insplad-glass-delta.pt --delta-sha $overlay.delta_sha256 --calibration calibration.json --evaluation "/BASE/evaluation.json" --port 8775
```

함께 든 safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp와 테스트를 기존 프로젝트에 반영하고 다시 빌드합니다. task 허용 목록에 insplad_glass_missing_cap을 추가했습니다. 기존 insplad_upper_shackle_rust와 15개 항목도 지원합니다. C++ SAFELOG_AI_BASE_URL은 http://127.0.0.1:8775입니다.

유리 애자는 inspectionTask=insplad_glass_missing_cap 또는 equipment_surface+같은 equipmentCategory로 요청합니다. 입력은 유리 애자 사슬을 자른 사진입니다. 종횡비를 보존한384입력과 실제 이미지 영역의 특징으로 분류합니다. 부식 기능의 입력은 고분자 애자 상부 금속 연결고리를 자른 사진이며 기존256입력/기준/가중치를 보존했습니다. 입력 자산의 자동 판별은 제공하지 않습니다.

API modelSha256은 overlay.json의 combined_tensor_state_sha256이며 부모 파일 SHA와 구분합니다. 새 물리적 결함 위치 지도나 전기/기계 건강 판단을 제공하지 않습니다. 후보·조치 여부는 점검자가 확인하며 보고서 요약은 기존 규칙 템플릿입니다.

원저자 유리 애자 평가 정상29/결함30장에서 오탐2/29=6.90%, 미탐6/30=20.00%였습니다. 정상CAL85 rank74의 고정 기준을 사용했습니다. 부식 기능의 기존 공개 평가19.35%/0%를 유지하는 가중치 보존과 연결 검사를 수행합니다. 새 현장 정확도나 0미탐 보장이 아닙니다. 물리 부품·현장 독립성과 날짜/배경 영향은 검증 전입니다.

InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1, 논문10.1080/01431161.2023.2283900, CC BY-NC3.0 비상업 학술 연구/원저자·출처·버전·변경 표시를 유지합니다. 기존 자료의 이용조건도 유지합니다. 새 pip 설치나 Android APK 빌드를 검증한 전달물은 아닙니다.
