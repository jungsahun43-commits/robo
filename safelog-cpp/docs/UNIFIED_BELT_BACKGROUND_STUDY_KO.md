# 벨트 배경 오탐 감소 · 고정 대조

기준일: 2026-10-13. 출발점은 원본 손상 부위 학습으로 선택한 native_mix 통합 모델 하나다. 직전 모델의 배경pixel 오탐0.90%, 손상pixel 미탐38.08%, 위치IoU40.46%, 사진미탐3/353(0.85%)를 기준으로 비교한다.

## 수행 내용

1. 기존 TRAIN1,397장과 원본crop1,397개를 SHA로 보존·재사용한다. 새 라벨·독립 사진은0개다.
2. 두 후보는 같은 모델에서 시작하고 같은10epoch·1,280회씩 추가 학습한다. 배치8장 중4원본crop·4전체사진, 순서·ID·반전·학습률이 같다. 모든 비벨트 가중치와BN통계는 고정한다.
3. 두 후보에 기존 BCE20+Dice와 손상 보존 항목을 사용한다. 작성자손상mask·valid와 동결기준점수0.5이상인 픽셀에서 기준점수보다 낮아지는 방향에 squared probability deficit를2비중으로 준다. 기준점수를 새 정답으로 만들지 않는다. 기준모델은 학습용이며 배포에는 넣지 않는다.
4. hard_background에만 추가0.05배경loss를 준다. 손상도형 주변3pixel을 제외하고valid인 작성자background 중 높은 negative BCE 상위1%를 매 학습배치에서 선택한다. anchor_control의 추가배경loss는0이다. 전체사진을 정상벨트 정답으로 만들지 않는다.
5. 패딩·도형·경계가 배경loss의 gradient에서 제외되는지, 기준모델은학습되지않는지, 실제두자료형식이유효한지, 선택조건이미탐악화를거절하는지 테스트5개를 먼저 확인한다.
6. 소스·조건을 학습전에Git에커밋한다.64회마다모델·옵티마이저·진행·난수상태를저장하고journal을남긴다.
7. 마지막10회차만 같은 알려진VAL353장·4촬영구간의 전체640사진에서기준0.5로평가한다.epoch별VAL튜닝·예약TEST사용·이물질CAL/TEST재보정은없다.
8. 배경pixelFPR이상대10%이상줄고IoU·사진FNR·pixelFNR·정답위치겹침실패가모두기준보다나빠지지않아야한다.통과후보는낮은FPR·높은IoU·control순으로선택하고실패하면기존native_mix를유지한다.
9. 최종모델/재개상태·기존입력/소스·비벨트전체tensor·3기능CPU/GPU및기존시설/이물질CPU예측을확인한다.
10. 선택한경우새단일모델ZIP과CPU API·실제Qt/C++호출을확인한다.기존앱기본모델은자동교체하지않는다.

## 명령

`safelog-cpp/ai-training`에서 기존 가상환경을 사용한다. freeze는 새실험을한번만생성하며 완료파일을덮어쓰지않는다.

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_belt_background -v
./.venv/Scripts/python.exe scripts/study_unified_belt_background.py freeze
# 소스·사전조건 커밋 뒤 실행
./.venv/Scripts/python.exe scripts/study_unified_belt_background.py train
./.venv/Scripts/python.exe scripts/study_unified_belt_background.py evaluate
./.venv/Scripts/python.exe scripts/study_unified_belt_background.py select
./.venv/Scripts/python.exe scripts/study_unified_belt_background.py verify
./.venv/Scripts/python.exe scripts/release_unified_belt_background.py report
./.venv/Scripts/python.exe scripts/release_unified_belt_background.py package
./.venv/Scripts/python.exe scripts/release_unified_belt_background.py smoke
```

train은 마지막progress에서재개하고final.pt가있으면재학습하지않는다. 실행중인같은실험lock을확인하면중복실행하지않는다.

## 해석 범위

배경pixel은작성자mask의모델좌표상비손상영역이며전체정상사진오탐분모가아니다. 정상벨트사진은0장이므로사진오탐은미측정이다. 알려진VAL을여러실험에서재사용해선택하므로독립산업현장5%검증을주장하지않는다. 내부고장·가동여부는별도측정/사람확인이필요하다. 출처사용조건을계승하고원본/파생사진·가중치는Git에넣지않는다.

[이전원본학습·다른기능수치](../ai-training/reports/UNIFIED_BELT_DETAIL_RESULTS_KO.md), [출처·사용조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md).
