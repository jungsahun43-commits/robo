# 역할4: 현수 연결구 부식 + 유리 애자 캡 누락 + 상부 연결고리 부식

기존 v16-r1 ZIP의 model.pt/evaluation.json이 필요합니다. 새 ZIP에는 세 추가 분류부의 결합 delta와 이전 두 기능의 보존 검사용 delta·기준·코드가 들어 있습니다. 하나의 공통 encoder를 쓰는 한 모델이며 단독 앱이나 단독 모델 파일이 아닙니다. 기본 앱을 자동 교체하지 않습니다.

각 ZIP을 별도 폴더에 풀고 이 ZIP 폴더에서 기존 Python 환경으로 실행합니다. /BASE는 원래 v16-r1 폴더로 바꿉니다.

```powershell
$overlay = Get-Content -Raw overlay.json | ConvertFrom-Json
python -m safelog_ai.insplad_suspension_server --parent "/BASE/model.pt" --parent-sha $overlay.parent_file_sha256 --corrosion-delta corrosion-delta.pt --corrosion-delta-sha $overlay.preserved_corrosion_delta_sha256 --corrosion-calibration corrosion-calibration.json --glass-delta glass-delta.pt --glass-delta-sha $overlay.preserved_glass_delta_sha256 --glass-calibration glass-calibration.json --delta insplad-suspension-delta.pt --delta-sha $overlay.delta_sha256 --calibration calibration.json --evaluation "/BASE/evaluation.json" --port 8775
```

함께 든 C++ 어댑터와 테스트를 반영하고 재빌드합니다. 새 inspectionTask=insplad_lightning_suspension_rust이며 equipment_surface와 동일 equipmentCategory도 가능합니다. 기존 insplad_glass_missing_cap, insplad_upper_shackle_rust 및 15개 항목을 보존합니다. SAFELOG_AI_BASE_URL=http://127.0.0.1:8775입니다.

새 입력은 가공지선 현수 연결구를 자른 사진이며 256 RGB 입력으로 사진상 부식 후보를 제안합니다. 유리 애자는 384 종횡비 보존, 상부 연결고리 부식은 256 입력의 기존 가중치·기준을 보존합니다. 입력 자산의 종류는 사용자가 선택합니다. 자산 자동 판별·전체 설비 안전·전기/기계 작동·새 결함 위치 지도를 제공하지 않습니다. 점검자가 최종 확인합니다. 요약과 위험 등급은 기존 규칙 템플릿입니다.

API modelSha256은 overlay.json의 combined_tensor_state_sha256으로 부모 파일 SHA와 다릅니다. parentCheckpointSha256과 deltaSha256은 따로 제공합니다.

새 원저자 평가 정상231/부식20에서 오탐24/231=10.39%, 미탐0/20=0%였습니다. 정상CAL51 rank45의 고정 기준을 사용했습니다. 평가 날짜 중 정상/부식이 함께 있는 날짜는0개라 날짜·배경 혼동을 배제하지 못하며 물리 부품·촬영·현장 독립성도 검증 전입니다. 0미탐 보장이나 독립 현장 성능이 아닙니다. 기존 두 기능의 공개 평가 수치나 검사를 새 검사의 수량으로 합치지 않습니다.

InsPLAD MendeleyV1 DOI10.17632/5n3fjgvfyz.1, 논문10.1080/01431161.2023.2283900, CC BY-NC3.0 비상업 학술 연구와 원저자·출처·버전·변경 표시를 유지합니다. 기존 자료 이용조건도 유지합니다. 새 pip 설치나 Android APK 검증은 포함하지 않습니다.
