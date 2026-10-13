# 원본 구역 학습·분석 대조

기준일: 2026-10-13. 출발 모델은 직전 native_mix 통합 모델이다. 같은 모델에 전체 사진과 원본 구역을 입력해 작은 손상 위치를 더 자세히 분석하는 연구다.

## 순서

1. 작성자 TRAIN 1,397장의 원본 JSON, 포함된 JPEG의 SHA·RGB, 기존 전체 사진·마스크·유효 영역을 비교한다. 대부분 4K이며 가로·세로 사진이 모두 있다.
2. 사진 크기로만 최대6개 정사각형 구역을 정한다. 긴 변3개·짧은 변2개, 한 변은 min(short, max(640, ceil(0.4long), ceil(0.6short)))이다. 균등 배치로 모든 픽셀을 덮으며 손상 도형·모델 점수로 위치를 고르지 않는다.
3. 학습 순서와 구역 선택을 정답·예측을 보기 전에 정한다. 배치8장 중4장 전체사진·4장 원본구역이다. 실제로 예약한 구역만 별도폴더에 만들고 작성자 도형을 같은 좌표로 옮긴다. 구역 mask가 비어 있어도 전체 사진 정상/기계 정상으로 저장하지 않는다.
4. 기존 모델의 벨트 출력만10epoch·1,280회 학습한다. BCE posweight20+Dice를 유지하고 배경 penalty·교사를 추가하지 않는다. 공통 분석부·시설·이물질과 모든 BN통계를 고정한다.64회마다 재개 상태를 저장한다.
5. 세 설정을 비교한다. accepted_tiles는 직전 가중치에 구역분석을 적용하고 신경망 추가학습0회다. trained_global과 trained_tiles는 이번에 한 번 학습한 같은 가중치를 사용한다. 세 설정을3개모델학습으로 합산하지 않는다.
6. 구역 결과는 원본좌표의 Hann 가중평균으로 합친 뒤 기존640유효영역으로 옮긴다. 전체 확률50%+구역확률50% 고정 평균, 기준0.5 유지다. 같은 모델을 여러 구역에 반복 사용하며 다른 모델을 묶지 않는다. maxedge640이하·3:1초과 비율은 전체 분석으로 fallback한다.
7. 같은 알려진VAL353장·4촬영구간에서 마지막10회차만 평가한다. 평가 mask는 기존640 raster를 그대로 쓴다. 구역 선택·확률 합치기에 정답을 넘기지 않는다. 예약TEST·이물질CAL/TEST는 사용하지 않는다.
8. IoU 최소0.5%p 개선 및 배경pixel오탐·사진미탐·pixel미탐·정답겹침실패가 모두 기준보다 나빠지지 않는 사전조건을 적용한다. 실패하면 직전 native_mix를 유지한다.
9. 다른 기능의 모든tensor·CPU예측,3기능CPU/GPU 및 원본 구역 CPU/GPU, 최종/재개 가중치·journal·입력/기존소스 보존을 확인한다.
10. 선택된 경우 모델1개와 분석정책을 새ZIP으로 전달하고 CPU API·실제Qt/C++9호출을 확인한다. 모델SHA와 정책SHA를 함께 기록한다. 기존 앱 기본 모델은 자동 교체하지 않는다.

## 명령

`safelog-cpp/ai-training`에서 기존 가상환경을 사용한다. prepare/freeze는 새실험을한번생성하며 완료파일을 덮어쓰지 않는다.

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_belt_tiles -v
./.venv/Scripts/python.exe scripts/study_unified_belt_tiles.py prepare
./.venv/Scripts/python.exe scripts/study_unified_belt_tiles.py freeze
# 소스·사전조건 커밋 뒤 실행
./.venv/Scripts/python.exe scripts/run_unified_belt_tiles.py train
./.venv/Scripts/python.exe scripts/study_unified_belt_tiles.py evaluate
./.venv/Scripts/python.exe scripts/study_unified_belt_tiles.py select
./.venv/Scripts/python.exe scripts/study_unified_belt_tiles.py verify
./.venv/Scripts/python.exe scripts/release_unified_belt_tiles.py report
./.venv/Scripts/python.exe scripts/release_unified_belt_tiles.py package
./.venv/Scripts/python.exe scripts/release_unified_belt_tiles.py smoke
```

train은 마지막progress에서 재개하고 완료final.pt가 있으면 학습을 반복하지 않는다. 같은 실험의 실행중lock이 있으면 중복실행하지 않는다.

Windows 재개 진입점은 `run_unified_belt_tiles.py`다. PID 존재 여부를 `OpenProcess`의 조회 권한과 `GetExitCodeProcess`로만 확인한다. Windows의 `os.kill(pid, 0)`은 종료 API를 호출할 수 있어서 조회 용도로 사용하지 않는다. [Python3.12 문서](https://docs.python.org/3.12/library/os.html#os.kill), [Microsoft 조회 API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess)에 맞춘 운영 보완이며 동결된 학습 소스·가중치·분석 정책은 바꾸지 않았다.

## 해석 범위

파생 구역은 새 독립 사진이 아니며 작성자 mask 변환은 새 사람/AI 라벨이 아니다. 정상벨트사진은0장이어서 정상사진오탐은 미측정이다. 알려진VAL을 여러 실험에서 재사용해 선택하므로 독립 산업현장5% 검증을 주장하지 않는다. 사진으로 내부 고장·가동 상태를 확정하지 않는다. 학습법 변화와 분석법 변화를 구분해서 기록한다.

[기준 모델 성능](../ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md), [출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
