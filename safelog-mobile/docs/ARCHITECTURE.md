# 모바일 아키텍처

`src/app`의 Expo Router 스택은 세션과 역할로 진입을 제어한다. `useFindings`에서 로그인 역할과 기록의 소유·배정을 기준으로 목록과 상세 조회를 제한한다. `WorkflowService`는 각 변경 시점에 역할·현재 상태·필수값을 다시 검증한다. Inspector는 자신이 등록한 기록, Manager는 현장 시연 기록, Assignee는 자신에게 배정된 기록을 본다.

```text
Expo Router → 세션 및 화면 → WorkflowService → FindingRepository → 메모리 구현
                  │                                  ↑
                  ├─ ImagePicker → 사진 URI          │
                  ├─ AiAnalyzer → Mock 제안 ──────────┘
                  └─ 보고서 데이터 → HTML → PDF/시스템 공유
```

`Open → InProgress → PendingReview → Verified`. 배정은 Open을 유지한다. 추가 조치 요청은 PendingReview에서 InProgress로 되돌리고, 이전 조치와 사진을 보관한다. 확인 작업에는 원 점검자, 전후 사진, 직접 확인 체크와 사람이 작성한 판단 내용이 필요하다. 비교 제안은 해당 조치 제출에 연결하므로 재조치 전의 제안은 최종 확인에 사용할 수 없다. AI 제안의 저장은 작업 상태를 바꾸지 않는다. 포트와 메모리 저장소 사이에서 기록·사진 목록·로그·AI 결정은 한 번의 스냅샷 교체로 적용한다.

## 모바일 화면 체계

`src/constants/safelog-theme.ts`와 `src/components/safelog-ui.tsx`에서 네이비, 흰색, 블루그레이의 공통 색상과 폼·버튼 스타일을 적용한다. 홈은 SafeLog 방패 로고, 현재 시연 현장, 로그인 역할의 안내와 조회 범위에 맞는 건수, 업무 바로가기를 표시한다. 작은 화면이나 확대 글자 설정에서는 카드가 한 열로 배치된다.

상태 이름과 색상은 `src/features/workflow/status-badge.tsx`를 공유한다. 미조치는 빨강, 조치 중은 파랑, 검토 대기는 주황, 확인 완료는 청록으로 표시한다. AI 분석 대기·채택 여부는 작업 상태와 분리한다. 시연 환경에는 기존 현장 한 곳이 등록되어 있으며 점검 위치는 등록 폼에서 입력한다.

## 영속화 전환

`feature/storage-ai`의 `database/schema.sql`은 sites, profiles, inspections, findings, photos, action_logs, ai_analyses를 정의한다. 사진과 로그가 finding 삭제를 연쇄 처리하고, AI subject는 분석 유형에 따라 대상을 해석한다. 버전 1 마이그레이션은 hazard_category와 risk_level을 추가하며, `feature/auth`는 자격 증명 저장을 확장한다. enum 문자열은 모바일의 PascalCase와 SQLite의 snake_case 사이에서 매핑한다.

오프라인이 필요하면 Expo SQLite 어댑터에서 foreign_keys를 활성화하고 user_version과 업그레이드 테스트를 운영한다. 구조화한 DB 행을 변경하는 작업은 한 트랜잭션으로 묶고 사진을 앱 관리 저장소에 복사한다. 파일과 DB의 커밋이 엇갈릴 때 보상 삭제와 고아 파일 정리를 설계한다. 협업이 필요하면 서버가 상태·역할·소유와 동시 갱신을 검사하고, 로컬 저장소는 캐시 또는 오프라인 큐가 된다. 그 단계에서 포트를 비동기화하고 임시 순번을 UUID로 바꾼다.

## 실제 AI 및 인증

AI 엔진의 Python 서버와 Qt HTTP 계약은 재사용 후보이다. 모바일 HTTP 구현은 `AiAnalyzer` 뒤에 두고, 기존 계약처럼 사진과 문맥을 요청에 포함한다. 응답의 필수값·등급·신뢰도·모델과 프롬프트 버전을 검증하고 원본 JSON과 검토 결정을 기록한다. 취소 신호, 타임아웃, 크기와 콘텐츠 형식 제한을 전송 어댑터에서 적용한다. 모바일 앱에는 비밀 키나 학습 가중치를 넣지 않는다. 실제 장치와 서버에서 입력 형식, 응답 및 지연을 확인한 뒤 적용한다.

현재 인증은 공개 시연 자격 증명과 메모리 세션이다. 운영 모드에서는 서버에서 자격 증명 및 권한을 검증하고, 세션 수명과 토큰 보관을 별도로 설계한다. 기존 Argon2id 포맷과 계정 관리 API는 서버 재사용 검토 대상이다.
