# 역할4: 연결고리 부식 연구 추가 기능 v45

기존 v16-r1 ZIP의 model.pt와 evaluation.json이 필요합니다. 이 ZIP은 작은 추가 분류부이며 단독 앱/단독 모델이 아닙니다. 같은 공통 encoder를 사용합니다.

기존 ZIP과 이 ZIP을 각각 다른 폴더에 풉니다. 이 ZIP 폴더에서 기존 Python 환경을 사용해 실행합니다. 아래 /BASE는 실제 기존 ZIP 폴더로 바꿉니다. 설치 의존성은 requirements-api.txt이며 이번 검사는 새 pip 설치나 Android APK 빌드를 포함하지 않습니다.

```powershell
$overlay = Get-Content -Raw overlay.json | ConvertFrom-Json
python -m safelog_ai.insplad_server --parent /BASE/model.pt --parent-sha $overlay.parent_file_sha256 --delta insplad-delta.pt --delta-sha $overlay.delta_sha256 --calibration calibration.json --evaluation /BASE/evaluation.json --port 8775
```

C++ 앱에서는 이 ZIP의 safelog-cpp/apps/mobile-qt/adapters/ai/http_ai_safety_analyzer.cpp를 같은 프로젝트 경로에 반영하고 다시 빌드해야 합니다. 새 항목을 허용 목록에 추가한 변경이며 기존 15개 항목은 유지합니다. 테스트 파일도 함께 제공합니다.

C++의 SAFELOG_AI_BASE_URL은 http://127.0.0.1:8775, inspectionTask는 insplad_upper_shackle_rust입니다. equipment_surface + 같은 equipmentCategory도 가능합니다. modelSha256은 기존 파일 SHA가 아니라 overlay.json의 combined_tensor_state_sha256입니다. 원본 파일과 추가 파일 SHA는 별도 필드로 기록합니다. 기존 15개 alias도 같은 서버에서 유지합니다.

입력은 고분자 애자의 상부 금속 연결고리를 자른 사진이어야 합니다. 일반 컨베이어·다른 금속 부품·전력선 전체 사진의 부식 검출 모델로 쓰지 마세요. 입력 종류의 자동 판별은 구현하지 않았습니다. 결과는 사진상 부식 후보이며 점검자가 확인합니다. confirmed pixel 위치나 기계/전기 건강 판단은 제공하지 않습니다. 보고서 요약은 기존 규칙 템플릿입니다.

원저자 평가 정상31/부식33장에서 FP6/FN0, 즉 오탐19.35%/미탐0%였습니다. CAL 정상108장의 고정 rank93 기준을 사용합니다. 촬영 시기와 라벨의 상관, 다른 프레임의 같은 실제 부품 가능성이 남아 있습니다. 작은 공개 출처 평가 결과를 실제 현장 오차율이나 0미탐 보장으로 설명하지 않습니다. 기존 검정 브래킷/튜브 성능은 개선했다고 측정하지 않았습니다.

InsPLAD Mendeley V1 DOI10.17632/5n3fjgvfyz.1, 원저자 논문 DOI10.1080/01431161.2023.2283900, CC BY-NC3.0 비상업 학술 연구/인용/출처·버전·변경 표시를 유지합니다. 기존 모델의 다른 자료 이용조건도 유지합니다. 이 전달물은 추가 분류부와 명시적인 API 경로를 추가한 변경입니다.
