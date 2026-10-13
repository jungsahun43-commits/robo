# 사진별 손상 탐지 보호 비교

같은6개 양성 TRAIN 사진을 하나로 합산하던 보호 계산을 사진별6개 조건으로 분리한다. 매 조건은 해당 사진의 작성자 양성 pixel 수로 정규화한다. 실제 Adam 변화·float64 dual 좌표 계산·FP32 쓰기 후 조건 검사를 수행한다. 64cycle 안에 만족하지 못하면 정확히 이전 파라미터를 복원하고 raw optimizer moment는 유지한다. 이 조건은 TRAIN 일차 근사이며 검증 미탐을 보장하지 않는다.

## 완료된 실행 순서

1. test_belt_photo_guard.py: 작은/큰mask 정규화, 유효pixel, 다중조건·직교방향, 안전변화 보존, 반대조건 복원, Adam moment, 고차원FP32의7검사.
2. study_unified_belt_photo_guard.py --mode freeze: 기존control과같은draw·입력SHA·기존파일SHA/mtime·TRAIN정답크기 진단·사전8조건을 고정. 소스/protocol을commit한다.
3. --mode train:4epoch512optimizer업데이트.64단계마다 가중치·optimizer·scheduler·cursor·CPU/CUDA RNG저장. WindowsPID는 읽기전용process_exists사용. 완료final은 재학습하지 않는다.
4. --mode eval: knownVAL353와normalCAL3만 평가. 보관7·예약BeltCrack TEST 점수는 계산하지 않는다.
5. --mode verify:512journal의6개조건총3072개, final/progress, 비벨트tensor, 기존파일·소스 보존과 사전선택조건을 확인한다.
6. report_unified_belt_photo_guard.py: 전체/작은손상/촬영구간과 사진미탐ID변화를 보고한다. 선택된새후보가있으면 별도연결전달물검증을 진행한다. 앱기본모델을 자동교체하지 않는다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_photo_guard.py
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode freeze
# 소스/protocol commit 후
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_photo_guard.py --mode verify
./.venv/Scripts/python.exe scripts/report_unified_belt_photo_guard.py
```

완료된 freeze/eval/report를 다시 실행해 덮어쓰지 않는다. 새 학습은 새 run·사전 protocol·학습 전 commit을 사용한다. 사진 전체 정상 FPR은 미측정이다. 원자료를 Git/전달 ZIP에 추가하지 않는다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_PHOTO_GUARD_RESULTS_KO.md)
