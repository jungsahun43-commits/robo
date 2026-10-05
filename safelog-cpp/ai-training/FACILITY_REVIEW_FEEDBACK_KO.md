# 균열·박락 검수 의견 사용 안내

## 박락 계측 감사 화면

이번에는 기존130장의 박락 오탐·미탐54장을 출처 주석·원래80격자 정답과 비교했다.
로컬 `runs/facility-spalling-train-audit/SPALLING-AUDIT.html`에서 출처·FP/FN 필터와
TP/TN 비교를 볼 수 있다. 계측은 판정이나 원인 확정이 아니며, 의견을 입력하려면
각 사례의 기존 검수 화면 링크를 사용한다. 기존 검수 의견을 새로 채우거나 원본 라벨에 반영하지 않는다.
CODEBRIM 양성의 빈 위치 마스크는 면적0이 아니라 위치 정답 미확인으로 표시한다.

[계측 결과·전체 TRAIN 태그 집계·다음 표본 추출 준비](reports/FACILITY_SPALLING_TRAIN_AUDIT_KO.md)

## 이번에 만든 기능

검수자별로 **균열과 박락의 판단·관찰 사유·메모·근거**를 기록하고 JSON으로 저장·불러온다. 집계기는 팀원 관찰과 분야 전문가라고 신고한 의견을 구분하고, 같은 항목의 있음/없음 의견이 충돌하면 별도 목록에 남긴다.

의견의 빈도는 모델 오류율이 아니다. 검수자 ID와 전문가 구분은 자기 신고이며 신원·자격 인증이 아니다. 원본 라벨을 바꿀 근거가 생기면 별도 승인·자료 버전 작업이 필요하다. 이 기능은 검수 의견을 모으는 단계다.

현재 PC에는 0773 모델로 고른 TRAIN **130건**의 새 화면이 준비돼 있다. 전문가 확인 정답·정답 변경·추가 학습은 이번 작업에서 **0건**이다. 기존 최선 연구 모델의 최대 검증 미탐·오탐률 **22.28%**, 5% 미만 목표 미달이라는 이전 측정 결과를 유지한다.

## 1. 화면 열기

이 문서와 명령의 기준 폴더는 `safelog-cpp/ai-training`이다.

현재 PC에서는 다음 로컬 HTML을 Chrome 또는 Edge에서 연다. 인터넷 서버나 GitHub 업로드는 필요 없다.

```text
runs/facility-train-review-roi-0773/review-workbench.html
```

새 TRAIN 검수 묶음에서 화면을 생성하려면:

```powershell
./.venv/Scripts/python.exe scripts/build_facility_review_workbench.py --package runs/facility-train-review-roi-0773/TRAIN-REVIEW.json
```

생성기는 사진·출판자 원본·주석의 실제 파일 SHA를 대조한다. HTML은 묶음 옆에 생성되므로 원본 상대 링크가 유지된다. 이미 생성된 화면이 있으면 다시 만들지 않고 해당 HTML을 연다. 검수 JSON·원본 묶음·기존 화면을 덮어쓰지 않는다.

사진·주석·모델·검수 묶음은 이용 조건에 따라 Git에서 제외돼 있다. 팀원이 코드를 clone한 것만으로 이 130건이 생기지는 않는다. 각자 이용 권한이 있는 원본 자료와 검수 묶음을 로컬에 준비해야 한다. 자료의 준비·샘플링은 기존 `build_facility_train_review.py`와 [검수 범위 문서](reports/FACILITY_CONTEXT_LABEL_REVIEW_KO.md)를 따른다.

## 2. 의견 입력

1. 같은 사람은 같은 **검수자 ID**를 쓰고 이름 또는 표시명을 적는다. ID는 대소문자·Unicode를 포함한 정확한 문자열로 구분한다.
2. `팀원 관찰` 또는 `관련 분야 전문가 의견`을 선택한다. 전문가 의견이면 전문 분야·관련 경험도 입력한다.
3. 처음에는 사진만 본다. 필요할 때 `원본 정답·주석 보기`, `모델 제안 보기`를 켠다. 모델 확률은 판단 근거나 정확도 보장이 아니다.
4. 균열과 박락을 각각 `손상이 보임 / 보이지 않음 / 사진으로 판단하기 어려움`으로 기록한다. 사유와 관찰 메모를 함께 적고, 전문가 의견에는 항목별 판단 근거도 적는다.
5. 확인하지 않은 항목은 `검수 전`으로 둔다. 잘못 입력한 항목은 `이 항목 의견 지우기`로 초기화한다.

팀원은 촬영 상태·눈에 보이는 특징·정의 질문을 먼저 기록할 수 있다. 사진만으로 판단하기 어렵거나 관련 전문 지식이 부족하면 불확실 의견으로 남긴다. 출판자 정답과 다른 의견은 곧바로 원본 라벨 오류가 되지 않는다.

화면의 원인별 숫자는 **작성 중인 항목 의견 수**다. 한 사진에 두 항목이 있으므로 사진 수와 다를 수 있으며, 유효한 검수 제출이나 전체 자료의 오류율로 해석하지 않는다.

## 3. 저장·불러오기

- `의견 JSON 저장`을 누른 뒤 실제 다운로드 파일을 확인한다.
- 다운로드를 지원하지 않는 브라우저에서는 `저장할 JSON 확인·복사`의 내용을 복사하여 메모장 등에서 **BOM 없는 UTF-8 `.json`** 파일로 저장한다.
- 다음 작업 때 `v2 의견 불러오기`에서 저장한 파일을 선택한다. 불러오기는 해당 검수자의 초안을 복원하고 현재 입력을 교체한다.
- 다른 묶음·모델의 의견, 중복 사례·항목, 잘못된 날짜·필드는 거부한다. 거부된 파일은 현재 입력을 변경하지 않는다.
- 입력은 창의 메모리에 있으므로 종료 전에 파일을 저장한다. 한 검수자의 수정본은 **가장 최근에 저장한 한 파일**만 집계에 넣는다.

현재 Codex 내장 브라우저 검사에서는 JSON 다운로드 요청의 디스크 저장을 확인하지 못했다. 복사 가능한 JSON 출력, 동일 내용의 로컬 파일 검증, 다시 불러오기 복원은 확인했다. 일반 Chrome/Edge의 실제 파일 다운로드는 이번 검증 범위에 포함하지 않았다.

## 4. 여러 사람 의견 집계

저장 파일은 로컬 `runs/review-opinions/` 같은 폴더에 보관한다. 아래 `team-a.json`, `expert-a.json`은 예시 이름이며 실제 저장 파일로 바꾼다. 매 집계에는 새 출력 폴더 이름을 사용한다.

```powershell
./.venv/Scripts/python.exe scripts/summarize_facility_review_feedback.py --package runs/facility-train-review-roi-0773/TRAIN-REVIEW.json --feedback runs/review-opinions/team-a.json runs/review-opinions/expert-a.json --output runs/feedback-round-01
```

생성 파일:

| 파일 | 내용 |
|---|---|
| `README_KO.md` | 역할·사유·자료·항목별 의견 수와 충돌·미해결 요약 |
| `feedback-summary.json` | 집계만 포함. 사례 ID·검수자 이름·메모·근거·사진 경로 제외 |
| `feedback-details.json` | 원문 의견, 충돌·미해결 사례 ID. 로컬 상세 검토용 |

같은 내용의 파일은 한 번만 집계한다. 같은 검수자 ID의 서로 다른 파일을 함께 넣으면 어느 수정본을 사용할지 알 수 없으므로 거부한다. 서로 다른 ID는 인증된 독립된 사람 수가 아니다. 의견 충돌을 다수결로 정답에 적용하지 않는다.

입력 파일 중 하나라도 잘못되면 새 출력 폴더를 만들기 전에 전체 집계를 거부한다. 기존 결과 폴더도 덮어쓰지 않는다.

이전 사유·메모만 있는 v1 의견을 별도 사진 단위 관찰로 집계하려면 `--allow-legacy`를 명시한다. v1에는 검수자·항목별 판정이 없어 새 v2 전문가 의견이나 균열·박락별 판단으로 변환하지 않는다.

## 5. 다음 학습으로 연결하는 기준

1. 원인 집계를 보고 어떤 사례를 추가로 확인할지 고른다.
2. 관련 전문가와 출판자 정의를 바탕으로 판단 근거와 이견을 검토한다.
3. 승인된 정답 변경·보강 자료만 원본과 분리한 새 TRAIN 버전으로 준비한다.
4. 평가 조건을 먼저 고정하고 대조 학습으로 개선 여부를 확인한다.

공장 시설 성능을 확인하려면 별도 현장 평가 자료가 필요하다. TRAIN 검수 사례나 의견에 맞춰 VAL/TEST 정답·어려운 사례를 바꾸어 5%를 맞추지 않는다.

## 6. AI 보조 관찰을 연결한 화면

최근 ROI 모델의 고정 TRAIN 묶음 130건 중 사진 23건을 AI가 확인하고, 항목 의견
31개를 별도 `facility_ai_train_observations_v1` 기록에 저장했다. DACL 8건에는 두
손상 항목을 기록하고, Dam 7건·CODEBRIM 8건에는 선택한 항목을 기록했다.
이전 AI 관찰 23건과 사진 SHA가 같은 사례는 6건이며, 나머지 17건은 그 기록에
포함되지 않았다. AI가 확인했다는 선언은 전문가 자격이나 정답 승인을 뜻하지 않는다.

이 PC에 생성한 새 화면은 다음 파일이다. 기존 화면과 원본 검수 묶음은 보존했다.

```text
runs/facility-roi-ai-observations-20261005/workbench/review-workbench.html
```

`AI 보조 관찰 보기`를 켜면 관찰한 사실·원인 후보·추가 확인 질문을 볼 수 있다.
기본은 꺼져 있고 AI 메모는 읽기 전용이다. 사람의 항목별 의견란에는 자동으로
입력되지 않으며, 사람 의견 JSON·검수 의견 집계에도 자동 포함되지 않는다.
먼저 사진을 판단한 뒤 참고 자료를 열고, 판단이 어려우면 불확실 의견을 남긴다.

각자의 로컬 묶음과 해당 묶음·모델·사진 SHA가 일치하는 AI 기록을 준비했다면
새 출력 폴더를 지정해 화면과 집계 보고서를 만든다. 아래 명령은 이 PC의 현재
자료 이름을 사용한 예시다. 원본 사진·주석·개별 AI 기록은 Git에 포함되지 않는다.

```powershell
./.venv/Scripts/python.exe scripts/build_facility_review_workbench.py --package runs/facility-train-review-roi-0773/TRAIN-REVIEW.json --ai-observations runs/facility-roi-ai-observations-20261005/AI-OBSERVATIONS.json --output runs/ai-review-round-02/review-workbench.html
./.venv/Scripts/python.exe scripts/report_facility_ai_observations.py --package runs/facility-train-review-roi-0773/TRAIN-REVIEW.json --observations runs/facility-roi-ai-observations-20261005/AI-OBSERVATIONS.json --prior-observations runs/facility-industrial-review/AI-OBSERVATIONS.json --output runs/ai-review-report-02
```

`--prior-observations`는 선택 옵션이다. 기존 기록과 사진 SHA의 겹침만 집계하며
이전 의견을 새 의견이나 전문가 판정으로 합치지 않는다. 출력의 `summary.json`과
`README_KO.md`는 집계용이며, `input-dimensions-details.json`은 사례별 로컬 기록이다.
다시 실행할 때는 기존 파일을 덮어쓰지 않도록 새로운 출력 위치를 사용한다.

실제 이번 관찰·크기 집계·한계는 [결과 보고서](reports/FACILITY_ROI_AI_REVIEW_KO.md),
코드·원본 보존 검증은 [기술 기록](reports/facility-roi-ai-review-verification.json)에 있다.
130건 중 107건은 크기와 SHA만 확인했다. 이번 작업은 새 학습·오류율 측정·라벨
수정 없이 검수 원인 후보를 준비한 단계이며, 5% 목표는 아직 미달이다.

## 역할4와 연결할 내용

역할3은 의견 형식 검증·오류 원인 집계·보강 자료 버전을 담당한다. 역할4는 앱에서 `AI 제안 → 사람 확인·수정 → 보고서 저장`을 연결하고, 저장할 때 점검 ID·사진 ID·판단 항목·작성자·시각·근거를 남기면 된다.

이번 v2 JSON은 **TRAIN 연구 검수용** 형식이다. 앱의 로그인 계정 인증이나 실제 점검 기록 API가 아니며, 실제 앱 의견을 TRAIN에 넣으려면 이용 동의·자료 분리·정답 승인 단계를 추가로 설계해야 한다.

## 개발 검증

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_facility_review_feedback tests.test_facility_review_workbench tests.test_facility_review_interop tests.test_facility_train_review -v
node --test tests/review_workbench.test.cjs
```

Python 26개와 Node 8개를 검증한다. 브라우저/Python 입력 계약 비교에는 잘못된 UTF-8·BOM·중복 JSON 키·모델 불일치·UTC 날짜·마이크로초 순서·검수자 ID·잘못된 추가 정답 필드 등 19개 합성 입력이 포함된다. Node가 없는 환경에서는 상호 검증 테스트가 skip되므로 실행 결과를 확인한다.
