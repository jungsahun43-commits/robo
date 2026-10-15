# 역할4: 실제 마스크 학습을 반영한 v16 연구 엔진

기존15개 작업 ID와 C++ HTTP 규격을 유지한다. 변경 대상은 검정 브래킷 사진 후보 점수다. 미탐82.98%→63.83%, 오탐9.38%→12.50%의 부분 개선으로,20% 목표 통과 상태가 아니다. 전선 연결부·KSDD·볼스크루 등 기존 출력과 브래킷 기본 위치 지도를 보존한다.

## 실행

ZIP을 전체 압축 해제한다. 기존 API 환경에서 다음 명령을 사용한다. 정확한 SHA256은 동봉된 `model.json`의 `sha256` 값이다.

```powershell
python -m safelog_ai.real_mask_server --checkpoint model.pt --sha256 "model.json의 sha256 값" --evaluation evaluation.json --port 8775
```

환경이 없으면 기존 프로젝트의 Python 가상환경을 만들고 `python -m pip install -r requirements-api.txt`로 설치한다. 새 환경에서의 설치·Android APK 빌드는 이번 검사에 포함되지 않는다.

## 연결 항목

- `mpdd_bracket_black` 또는 `inspectionTask=equipment_surface`, `equipmentCategory=mpdd_bracket_black`을 사용한다.
- `/v1/analyze-hazard`, `/v1/compare-action`, `/v1/summarize`, `/health` 규격을 유지한다.
- `/v1/inference-policy`의 `realMaskTransfer.mpdd_bracket_black`에 새로운 기준과 실제 학습 이력이 있다.
- 사진 점수는 기존 점수와 새 분기의 결합이며 고장 확률이 아니다. `originalLocalizationChanged=false`, `transferIsValidatedPixelLocalization=false`를 유지한다.
- 사진 판정과 위치 지도를 구분하고 점검자가 최종 확인하도록 한다. 기존 위치 지도는 새로운 학습으로 정확해진 것으로 표시하지 않는다.

학습한 자기 타일 자료 자체는 TEST 오탐32.52%·미탐28.30%로 목표 미달이다. 지원 작업 목록에 새 유형으로 추가하지 않았다. 검정 브래킷 개선도 공개 TEST 반복 개발 결과이며 현장 성능은 미검증이다. 새 자료와 가중치의 용도는 학술 연구이며 원저자의 비상업 연구·논문 인용 조건을 따른다.
