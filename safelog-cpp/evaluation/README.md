# AI evaluation data

`cases.csv`에 팀이 직접 확인한 시연용 사진과 정답을 기록한다. 실제 사업장의 개인정보, 얼굴, 차량번호는 사용하지 않는다.

평가 절차:

1. 최소 20장의 가상 또는 공개 사용 가능한 안전 사진을 준비한다.
2. 팀원이 `expected_category`와 `expected_risk_level`을 합의한다.
3. 같은 프롬프트 버전으로 모든 사진을 분석한다.
4. 예측값과 분석 시간을 기록한다.
5. 결과를 합산하되 숫자를 임의로 만들지 않는다.

Qt 빌드에서 생성되는 평가 실행 파일을 사용한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://192.168.0.10:8080"
./build-qt/apps/mobile-qt/safelog_ai_eval evaluation/cases.csv evaluation/results.csv
```

입력 CSV 경로를 기준으로 `image_path`를 해석한다. 실제로 존재하는 이미지와 팀이 합의한
정답만 입력한다. 실행 결과에는 예측 분류, 위험 등급, 신뢰도, 지연시간과 오류가 기록된다.
콘솔에는 위험 종류 일치율, 위험 등급 평균 절대 오차, 평균 지연시간과 JSON 실패율이 출력된다.
