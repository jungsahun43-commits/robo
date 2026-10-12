# 통합 메인 · 원본 벨트 손상 부위 학습

기준일: 2026-10-13. 기존 통합 v1의 공통 분석부와 시설·이물질 출력을 정확히 고정하고, 같은 모델 안의 벨트 출력 부분만 추가 학습하는 대조다.

## 실제 파이프라인

1. BeltCrack 작성자 TRAIN 1,397장의 원본 JSON, 포함된 원본 사진, 기존 640 사진·마스크·유효 영역을 SHA 및 픽셀로 대조한다.
2. 첫 작성자 손상 도형의 중심 주변을 원본 사진에서 정사각형으로 잘라 640으로 만든다. 최소 720 native pixel, 도형 폭의 1.5배, 사진 짧은 변 이내로 정한다. 사진 경계를 넘지 않게 옮기며 원래 모든 도형을 같은 좌표로 변환해 경계에서 자른다. 기존 640 사진을 다시 확대하는 처리와 다르다.
3. 작성자 사진마다 파생 crop 하나를 새 폴더에 저장한다. 새 독립 사진·새 정답은 0개다. VAL·예약 TEST로 crop을 만들지 않는다. 마스크·사진·유효 영역을 함께 변환한다.
4. 학습할 벨트 parameter만 옵티마이저에 넣고 공통 분석부, 시설 출력, 이물질 출력, 모든 BN 통계는 고정한다. 실제 원본 좌표·반전·다른 기능 보존·선택 조건 테스트를 먼저 통과시킨다.
5. 두 후보는 정확히 같은 v1에서 시작한다. full_control은 기존 전체 사진만, native_mix는 배치 8장 중 정확히 4장을 원본 crop으로 바꾼다. TRAIN ID·추출 횟수·반전·손실·학습률·seed는 같다.
6. 각각 10epoch, 1,280회 업데이트한다. epoch당 1,024회 추출하며 반복 추출을 새 사진으로 세지 않는다. 64회마다 모델·옵티마이저·진행 위치·난수 상태를 저장하고 실제 업데이트 journal을 남긴다.
7. 마지막 10회차만 같은 알려진 VAL 353장·4개 촬영 구간에서 평가한다. 입력은 원래 전체 640 사진이고 기준 0.5를 유지한다. epoch마다 VAL로 최적 회차를 찾지 않는다.
8. 사전 조건은 v1 대비 위치 IoU가 최소 0.5%p 높고, 사진 미탐·pixel 미탐·정답 위치와 전혀 겹치지 않는 사진 수가 나빠지지 않는 것이다. 통과 후보를 IoU, pixel 미탐, control 우선순으로 선택한다. 실패하면 v1을 유지한다.
9. 시설·이물질 가중치의 모든 tensor 바이트와 TRAIN 사진 CPU 예측의 정확한 일치를 확인한다. 기존 수치를 계승한다고 표시하며 새 시험을 했다고 소개하지 않는다. 이물질 CAL·TEST를 다시 최적화하지 않는다.
10. 연구 후보가 선택되면 모델 파일 하나를 별도 ZIP으로 전달하고 CPU API·실제 Qt/C++ 연결을 확인한다. 기존 앱 기본 모델은 자동 교체하지 않는다.

## 재현 명령

저장소의 `safelog-cpp/ai-training`에서 Python 가상환경을 사용한다. 준비·freeze는 새 실험을 한 번 생성하는 명령이다. 이미 완료된 실험의 입력과 결과를 덮어쓰지 않는다.

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_belt_detail -v
./.venv/Scripts/python.exe scripts/study_unified_belt_detail.py prepare
./.venv/Scripts/python.exe scripts/study_unified_belt_detail.py freeze
# 소스·사전 조건을 Git에 커밋한 뒤 실행
./.venv/Scripts/python.exe scripts/study_unified_belt_detail.py train
./.venv/Scripts/python.exe scripts/study_unified_belt_detail.py evaluate
./.venv/Scripts/python.exe scripts/study_unified_belt_detail.py select
./.venv/Scripts/python.exe scripts/study_unified_belt_detail.py verify
```

재개는 같은 train 명령을 사용한다. 완료 final.pt가 있으면 학습을 다시 하지 않는다. 실행 중인 같은 실험의 lock이 있으면 중복 실행을 거절한다.

## 해석 범위

- 정상 벨트 사진 정답은 0장이므로 정상 사진 오탐률은 미측정이다. background pixel 오탐률과 사진 오탐률을 혼용하지 않는다.
- IoU는 작성자 마스크와 모델 위치의 겹침 비율이며 사진 미탐과 별도 지표다. 약 38% IoU는 정밀 손상 위치 기능이 완성됐다는 뜻이 아니다.
- 학습 crop은 기존 사진의 다른 시야다. 새로운 현장·새 정상 사진·새 산업시설 종류가 추가됐다고 소개하지 않는다.
- 여러 실험에서 재사용한 VAL은 알려진 연구 비교다. 산업현장 일반화나 전체 오탐·미탐 5% 미만을 검증하지 않는다. 예약된 작성자 TEST는 유지한다.
- 내부 모터·롤러 고장, 정지·가동 상태를 사진으로 확정하지 않는다. 조치·위험 등급·요약은 기존 규칙 템플릿이다.
- 데이터는 BeltCrack CC BY-SA 4.0이며 원본 사진·라벨을 자르고 크기를 바꾼 파생 자료다. 원본과 파생 자료·가중치는 Git에 넣지 않는다. 다른 출처의 기존 사용 조건도 계승한다.

[작성자·출처·사용 조건](../ai-training/reports/UNIFIED_MAIN_ATTRIBUTION_KO.md)
