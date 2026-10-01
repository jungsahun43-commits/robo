# SafeLog AI 중심 설계

## 서비스 한 문장

현장 사진에서 AI가 위험요인과 개선안을 제안하고, 조치 전후 사진을 비교한 뒤, 사람이 최종 판단하고 전체 이력을 보고서로 남기는 안전점검 보조 앱이다.

## 핵심 원칙

- AI 결과는 제안이며 최종 판단은 점검자가 한다.
- AI 분석 원문, 모델명, 프롬프트 버전, 신뢰도, 사람의 채택·수정·거절을 함께 기록한다.
- AI 서버 장애 시 수동 입력 흐름은 계속 사용할 수 있다.
- 모바일 앱에는 클라우드 비밀 키를 넣지 않는다.

## AI 처리 단계

```text
조치 전 사진
→ 위험 종류·위험 등급·발견 문장·개선안 생성
→ 점검자의 채택/수정/거절
→ 담당자 조치
→ 조치 전후 사진 비교
→ 잔여 위험 제안
→ 점검자의 최종 확인
→ AI 보고서 요약
```

## C++ 인터페이스

`IAiSafetyAnalyzer`는 모델 공급자와 앱을 분리한다.

```cpp
analyzeHazard(imagePath, userMemo)
compareBeforeAfter(beforeImagePath, afterImagePath, actionNote)
summarize(inspectionContext)
```

현재 `MockAiSafetyAnalyzer`로 모델 서버 없이 전체 흐름을 개발할 수 있다.
실제 연결은 같은 인터페이스를 구현한 `HttpAiSafetyAnalyzer`를 사용하며,
앱 시작 시 `SAFELOG_AI_BASE_URL`이 설정되어 있으면 HTTP 어댑터를 주입한다.

학습 모델 서버는 `ai-training`에 있다. `start_ai_server.ps1 -FacilitiesOnly`는
표면 손상·금속 부식 모델과 화재·연기 모델을 실행한다. 시설 제안 8종은
표면 균열·박리·녹 얼룩·철근 노출·젖은 표면·백화·표면 공동·금속 부식이다.
위험 등급·개선 문구는 점검 우선순위 규칙이고, 보고서 요약은 현재 규칙 템플릿이다.
사진 기반 구조 안전 진단이나 별도로 학습한 보고서 LLM으로 발표하지 않는다.

모델별 실제 시험 성능과 고정 신뢰도 기준의 오탐·미탐은
`ai-training/reports/FACILITY_TRAINING_RESULTS_KO.md`와 기존 `TRAINING_RESULTS_KO.md`를 확인한다.

## 시연 구성

```text
Android Qt/C++ 앱
  → 같은 Wi-Fi의 노트북 AI 서버
  → 멀티모달 모델
  → 구조화된 JSON
  → Qt 앱에서 사람 검토
```

앱에는 `SAFELOG_AI_BASE_URL`만 설정한다. 클라우드 모델을 사용할 경우 비밀 키는 노트북의 프록시 서버에만 둔다.

## AI 응답 계약

위험 분석:

```json
{
  "hazardCategory": "통로 적치물",
  "riskLevel": 4,
  "detectedHazards": ["보행 통로 자재 적치", "넘어짐 가능성"],
  "suggestedDescription": "통로에 자재가 적치되어 넘어질 위험이 있습니다.",
  "suggestedAction": "자재를 지정 보관구역으로 이동하세요.",
  "confidence": 0.87
}
```

조치 비교:

```json
{
  "likelyResolved": true,
  "remainingRisks": ["통로 가장자리 추가 확인 필요"],
  "assessment": "주요 적치물이 제거된 것으로 보입니다.",
  "confidence": 0.81
}
```

모델 응답은 스키마 검증 후에만 저장한다. 위험 등급은 1~5, 신뢰도는 0~1 범위여야 한다.

## 발표용 AI 평가

직접 준비한 사진에 사람이 정답을 붙여 다음을 측정한다.

- 위험 종류 일치율
- 위험 등급 평균 오차
- AI 제안 채택·수정·거절 비율
- 전후 비교와 사람 판단의 일치율
- 평균 분석 시간
- JSON 파싱 실패율

실제 측정하지 않은 수치는 발표 자료에 넣지 않는다.
