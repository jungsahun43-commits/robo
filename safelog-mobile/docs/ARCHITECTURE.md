# 모바일 아키텍처

`src/app`의 Expo Router 스택은 세션과 역할로 진입을 제어한다. `useFindings`에서 로그인 역할과 기록의 소유·배정을 기준으로 목록과 상세 조회를 제한한다. `WorkflowService`는 각 변경 시점에 역할·현재 상태·필수값을 다시 검증한다. Inspector는 자신이 등록한 기록, Manager는 현장 시연 기록, Assignee는 자신에게 배정된 기록을 본다.

```text
Expo Router → 세션 및 화면 → WorkflowService → FindingRepository → SQLite(네이티브) / 메모리(웹)
                  │                                  ↑
                  ├─ ImagePicker → 사진 URI          │
                  ├─ AiAnalyzer → 고정 CPU HTTP / 명시적 Mock ─┘
                  └─ 보고서 데이터 → HTML → PDF/시스템 공유

고정 CPU HTTP → loopback gateway 8780
                  ├─ /general → 기존 v16-r1 CPU 서버 8775
                  └─ /pcb     → 기존 PCB CPU 서버 8774
```

`Open → InProgress → PendingReview → Verified`. 배정은 Open을 유지한다. 추가 조치 요청은 PendingReview에서 InProgress로 되돌리고, 이전 조치와 사진을 보관한다. 확인 작업에는 원 점검자, 전후 사진, 직접 확인 체크와 사람이 작성한 판단 내용이 필요하다. 비교 제안은 제출 회차와 현재 조치 후 사진 URI에 연결하므로 재조치 전의 제안은 최종 확인에 사용할 수 없다. AI 제안의 저장은 작업 상태를 바꾸지 않는다. SQLite에서는 사진 복사 후 한 트랜잭션으로 저장하고 커밋 뒤에만 캐시와 구독자를 갱신한다. 실패하면 이번 저장이 만든 사진 사본만 보상 정리한다.

## 모바일 화면 체계

`src/constants/safelog-theme.ts`와 `src/components/safelog-ui.tsx`에서 네이비, 흰색, 블루그레이의 공통 색상과 폼·버튼 스타일을 적용한다. 홈은 SafeLog 방패 로고, 현재 시연 현장, 로그인 역할의 안내와 조회 범위에 맞는 건수, 업무 바로가기를 표시한다. 작은 화면이나 확대 글자 설정에서는 카드가 한 열로 배치된다.

상태 이름과 색상은 `src/features/workflow/status-badge.tsx`를 공유한다. 미조치는 빨강, 조치 중은 파랑, 검토 대기는 주황, 확인 완료는 청록으로 표시한다. AI 분석 대기·채택 여부는 작업 상태와 분리한다. 시연 환경에는 기존 현장 한 곳이 등록되어 있으며 점검 위치는 등록 폼에서 입력한다.

## 영속화 전환

`feature/storage-ai`의 `database/schema.sql`은 sites, profiles, inspections, findings, photos, action_logs, ai_analyses를 정의한다. 사진과 로그가 finding 삭제를 연쇄 처리하고, AI subject는 분석 유형에 따라 대상을 해석한다. 버전 1 마이그레이션은 hazard_category와 risk_level을 추가하며, `feature/auth`는 자격 증명 저장을 확장한다. enum 문자열은 모바일의 PascalCase와 SQLite의 snake_case 사이에서 매핑한다.

`repositories/sqlite/`는 Expo SQLite 어댑터에서 foreign_keys·WAL과 `user_version`을 사용한다. v1/v2 원형 마이그레이션은 유지하고 v3에 AI 분석 방식·분야·출처·후보·SHA·원응답 및 사진 메타데이터를 추가한다. confidence는 NULL을 허용해 PCB 점수나 전후 비교에 없는 확률을 만들지 않는다. 사진과 로그는 자연 키로 갱신해 기존 ID도 보존한다. [저장 계약과 마이그레이션](DB_AI_INTEGRATION.md)을 따른다. 여러 기기 협업에는 별도 서버가 필요하다.

## 실제 AI 연결

`src/features/ai/fixed-cpu.ts`가 기존 `AiAnalyzer` 포트를 구현한다. 사진 등록의 기본값은 `fixed-cpu`이며 통합 모델의 15항목과 PCB 5유형, 총 20항목을 연결한다. `src/contracts/inspection.ts`가 허용 ID와 유형 검사를 공유하고 `tasks.ts`가 분야·범위·연구 평가 집계를 제공한다. `researchTopTasks`는 양쪽 평가 사진이 각각 20장 이상인 항목의 최대 오탐·미탐률 순위다. `demoTasks`는 사용자 선정 발표 구성인 볼스크루·시설 균열·컨베이어 이물질·RobustAD PCB·케이블이며 숫자 순위와 구분한다. 기타 분야 `manual_other`는 `manual` 방식만 허용하고 AI를 호출하지 않는다. `Finding`에 분석 방식과 분야를 저장해 이후 화면에서도 유지한다. `mock`은 명시적으로 선택하며 실제 서버 오류의 fallback으로 사용하지 않는다.

`photo-data.ts`는 웹/네이티브 사진 URI에서 JPEG·PNG 바이트를 읽고 MIME·크기·이미지 입력 제한을 검사한다. 원본은 최대 8MiB, 통합 모델은 최대 12MP, PCB는 최대 40MP다. HTTP 분석기는 요청 전에 `/health`의 준비 상태·단일 모델 수·고정 모델 SHA를 확인하고, 분석 응답의 schema·필수값·숫자 범위·프롬프트 버전·모델 및 입력 식별값을 검사한다. 모델과 프롬프트, 원문 JSON, 후보 유무 및 사람의 검토 결정을 분석 기록에 보관한다.

일반 모델은 `/v1/analyze-hazard`와 `/v1/compare-action`, PCB는 `/v1/equipment/pcb/analyze`를 사용한다. PCB의 자동 전후 비교는 제공하지 않으며 직접 사진을 검토한다. 후보 미검출을 안전 보증으로 바꾸지 않고, 통신·입력·응답 오류를 정상으로 저장하지 않는다. AI 참고 점수는 보정된 확률이 아니며 PCB heatmap은 확정 결함 위치가 아니다. 실제 모델의 전후 비교 confidence 고정값을 성공 확률로 표시하지 않는다.

`use-analysis.ts`는 취소·시간 제한과 화면/점검 범위가 바뀐 뒤 도착하는 응답을 구분한다. 폐기된 요청의 응답이 다른 기록에 저장되지 않도록 저장 직전에 현재 요청인지 확인한다. 사람의 최종 확인 없이 AI가 Verified로 바꾸지 않는다.

`settings.ts`의 공개 환경변수는 `EXPO_PUBLIC_SAFELOG_AI_BASE_URL`과 `EXPO_PUBLIC_SAFELOG_PCB_BASE_URL`이며 기본값은 `http://127.0.0.1:8780/general`, `http://127.0.0.1:8780/pcb`다. `scripts/fixed-cpu-gateway.mjs`는 기존 8775/8774 서버를 변경하지 않고 허용된 경로만 전달한다. loopback 바인딩, 명시적 웹 origin, 요청 24MiB·응답 8MiB 제한, 60초 deadline과 클라이언트 취소 전달을 적용한다. [실행·연결 안내](FIXED_CPU_GATEWAY.md)를 따른다.

고정 모델 가중치는 기존 별도 Windows CPU 패키지에 있다. React Native 번들과 GitHub 소스에는 모델이나 비밀 키를 넣지 않는다. 이번 구현은 로컬 연결 코드이며 게시·병합은 별도로 확인한다. 2026-11-19 실제 HTTP 분석기·gateway·고정 CPU 서버를 연결해 시설·컨베이어·PCB 승인 사진 6장의 기대 판정과 컨베이어 메모리 워크플로를 확인했다. Expo Web export 생성도 통과했다. 이는 네트워크·계약·업무 상태 검사이며 브라우저 화면 조작·실제 현장 조치·독립 성능 평가가 아니다. Android 실기기·LAN·iOS 검증은 남아 있으며 현재 작업 환경의 Android SDK·실기기 미확보를 웹/모형 테스트 통과로 대체하지 않는다.

## 인증과 저장의 현재 범위

현재 로그인은 기존 scrypt 기반 기기 내 계정과 메모리 세션이다. 네이티브의 계정·점검은 SQLite, 사진은 앱 문서 폴더에 보관한다. 웹은 메모리 저장이다. `WorkflowService`에 계정 디렉터리를 주입해 현재 사용자 역할과 배정 대상의 역할을 확인한다. UI의 담당자 목록과 순수 `report-data.ts`의 이름도 같은 계정 목록을 사용한다. 운영 서버 인증·계정 동기화·다기기 공유는 남아 있다. 공개 시연 계정과 기존 인증 정책을 유지하며 고정 CPU 연결·기기 로컬 저장·운영 서버 인증을 구분한다.
