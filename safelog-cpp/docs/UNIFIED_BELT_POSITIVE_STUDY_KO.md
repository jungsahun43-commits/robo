# 벨트 양성 구역·예측 보존 학습 파이프라인

기존 선택 가중치에서 벨트 출력만 실제6회 학습하고 현재 선택한 전체＋원본 구역 정책으로 비교한다. 새 데이터셋이나 새 정답은 추가하지 않는다.

1. `tests/test_belt_positive_retention.py`로 교사 gradient 차단·유효 좌표 제외·동일 점수·unknown 손실을 확인한다.
2. `study_unified_belt_positive.py --mode freeze`로 TRAIN draw, 양성 구역 선택, 고정6epoch, 기존 모델, 입력·소스SHA와 사전 선택 기준을 기록한다.
3. 소스·protocol을 Git에 먼저 commit한다. 아래 학습 명령은 현재HEAD와SHA가 같아야 실행된다.
4. `--mode train`: 실제768업데이트,64업데이트마다 모델·optimizer·scheduler·cursor·CPU/CUDA RNG를 보관한다. Windows PID 확인은 읽기 전용 process_exists를 사용한다. 완료된 final은 재학습하지 않는다.
5. `--mode eval`: 알려진 VAL353에 동일한0.5 기준·작성자640 mask를 적용한다. 예약 TEST는 사용하지 않는다.
6. `--mode select-verify`: 현재 선택 정책 대비8조건과 입력·기존 모델·소스 보존, 실제 journal, progress 일치, 비벨트 tensor를 확인한다.
7. `report_unified_belt_positive.py`: 실제 개선·악화·장수·학습 횟수와 선택 여부를 기록한다. 채택된 경우에만 역할4용 전달물을 새 버전으로 만들고 독립 ZIP의 CPU API·Qt/C++ 연결을 시험한다.

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p test_belt_positive_retention.py
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode freeze
# 소스와 protocol commit 후 실행
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode train
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode eval
./.venv/Scripts/python.exe scripts/study_unified_belt_positive.py --mode select-verify
./.venv/Scripts/python.exe scripts/report_unified_belt_positive.py
```

freeze·평가·보고서는 새 파일을 배타적으로 작성한다. 완료된 연구를 재실행해 기존 결과를 덮어쓰지 않는다. 동일 이름의 completed run을 새 실험으로 세지 않는다. 교사는 학습 제약용으로만 사용하며 추론은 모델 하나를 사용한다.

[실제 결과](../ai-training/reports/UNIFIED_BELT_POSITIVE_RESULTS_KO.md), [앞선 추론 비교](../ai-training/reports/UNIFIED_BELT_FINE_RESULTS_KO.md), [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
