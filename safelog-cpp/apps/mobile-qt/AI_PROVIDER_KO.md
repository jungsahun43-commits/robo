# SafeLog 실제 AI Provider 연결

## 선택 방식

앱 시작 시 `SAFELOG_AI_BASE_URL`이 비어 있으면 기존 `MockAiSafetyAnalyzer`를 사용한다.
값이 있으면 `HttpAiSafetyAnalyzer`를 생성하여 역할 4의 `AppController`에 주입한다.

```powershell
$env:SAFELOG_AI_BASE_URL="http://192.168.0.10:8080"
$env:SAFELOG_AI_TIMEOUT_MS="25000"
```

현재 학습 모델 서버는 API 키 없이 실행한다. `safelog-cpp/ai-training`에서 모델 ZIP을
풀고 `./start_ai_server.ps1 -AllModels`를 실행한다. 설치·학습·모델 전달 절차는
[AI 서버 안내](../../ai-training/README_KO.md)에 있다.

같은 PC의 Windows 앱은 `http://127.0.0.1:8080`, 휴대폰은 같은 Wi-Fi에 연결한
서버 PC의 실제 IP를 사용한다. Android에서 서버 주소를 전달하는 앱 설정은 역할 4와
연결해야 한다. 향후 클라우드 모델을 추가할 경우 비밀 키는 서버에만 보관한다.

## 엔드포인트

### `POST /v1/analyze-hazard`

요청:

```json
{
  "image": "data:image/jpeg;base64,...",
  "userMemo": "작업자의 안전모 착용 상태 점검",
  "promptVersion": "safelog-hazard-v1"
}
```

응답:

```json
{
  "hazardCategory": "안전모 미착용",
  "riskLevel": 4,
  "detectedHazards": ["안전모 미착용"],
  "suggestedDescription": "사진에서 안전모 미착용 위험이 탐지되었습니다.",
  "suggestedAction": "작업을 중지하고 규격에 맞는 안전모를 착용하세요.",
  "confidence": 0.87,
  "modelName": "사용한 모델명",
  "promptVersion": "safelog-hazard-v1"
}
```

실제 서버 응답에는 탐지 클래스·모델·좌표·신뢰도를 담은 `detections`와
`requiresHumanReview`도 포함된다. C++ 어댑터는 전체 JSON을 `rawJson`에 보관한다.
현재 학습 대상은 보호구·사람·신체 부품과 화재·연기다. 통로 장애물과 노출 전선은
전용 데이터로 추가 학습하기 전까지 탐지 대상으로 발표하지 않는다.

### `POST /v1/compare-action`

요청 필드: `beforeImage`, `afterImage`, `actionNote`, `promptVersion`.

응답 필드: `likelyResolved`, `remainingRisks`, `assessment`, `confidence`,
`modelName`, `promptVersion`.

### `POST /v1/summarize`

요청 필드: `inspectionContext`, `promptVersion`.

응답 필드: `summary`, `keyRisks`, `modelName`, `promptVersion`.

이 엔드포인트는 현재 규칙 템플릿이다. `modelName`은
`safelog-summary-template-v1`이며 별도로 학습한 언어 모델이 아니다.

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
