# SafeLog 실제 AI Provider 연결

## 선택 방식

앱 시작 시 `SAFELOG_AI_BASE_URL`이 비어 있으면 기존 `MockAiSafetyAnalyzer`를 사용한다.
값이 있으면 `HttpAiSafetyAnalyzer`를 생성하여 역할 4의 `AppController`에 주입한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://192.168.0.10:8080"
$env:SAFELOG_AI_TIMEOUT_MS="25000"
```

API 키는 Android 앱에 넣지 않는다. 노트북의 프록시 서버가 클라우드 모델 키를 보관한다.

## 엔드포인트

### `POST /v1/analyze-hazard`

요청:

```json
{
  "image": "data:image/jpeg;base64,...",
  "userMemo": "통로에 자재가 적치되어 있음",
  "promptVersion": "safelog-hazard-v1"
}
```

응답:

```json
{
  "hazardCategory": "통로 적치물",
  "riskLevel": 4,
  "detectedHazards": ["걸림", "넘어짐"],
  "suggestedDescription": "통로에 자재가 적치되어 넘어질 위험이 있습니다.",
  "suggestedAction": "자재를 지정 구역으로 이동하세요.",
  "confidence": 0.87,
  "modelName": "사용한 모델명",
  "promptVersion": "safelog-hazard-v1"
}
```

### `POST /v1/compare-action`

요청 필드: `beforeImage`, `afterImage`, `actionNote`, `promptVersion`.

응답 필드: `likelyResolved`, `remainingRisks`, `assessment`, `confidence`,
`modelName`, `promptVersion`.

### `POST /v1/summarize`

요청 필드: `inspectionContext`, `promptVersion`.

응답 필드: `summary`, `keyRisks`, `modelName`, `promptVersion`.

## 검증과 실패 상태

- 사진은 기본 8MB 이하이며 이미지 형식이어야 한다.
- 위험 등급은 1~5 정수, 신뢰도는 0~1 범위다.
- 모델명, 프롬프트 버전과 모든 필수 필드를 검사한다.
- 서버 연결, 시간 초과, HTTP 4xx/5xx, 빈 응답, JSON 오류, 입력 오류를 구분한다.
- 연결·시간 초과·HTTP 5xx는 기본 한 번 재시도한다.
- UI 제한 시간 또는 로그아웃 시 실제 네트워크 요청도 취소한다.
- AI 결과는 제안이며 `Verified` 상태를 직접 만들지 않는다.

## 역할 4 연결

역할 4의 생성자 주입 구조는 유지된다. 별도 화면 변경 없이 앱 시작 환경변수만으로
Mock과 실제 Provider를 선택한다.

```text
AppController
  → AiController (QtConcurrent)
  → IAiSafetyAnalyzer
      ├─ MockAiSafetyAnalyzer
      └─ HttpAiSafetyAnalyzer
```
