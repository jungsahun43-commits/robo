# 아이폰 Expo Go · 실제 AI HTTPS 시연

기존 고정 CPU 모델 2개와 SQLite·사람 검토 흐름을 그대로 사용한다. 아이폰에서 PC의 `127.0.0.1`에 직접 접속하는 대신, 토큰으로 보호한 gateway의 HTTPS 주소를 사용한다. 모델 재학습·판정 기준 변경은 없다.

## 구성

```text
iPhone Expo Go → Cloudflare HTTPS → 인증 gateway(PC loopback) → 기존 CPU 모델 8775 / 8774
```

통합 PC의 SSH 전달 포트 앞에 cloudflared를 두는 구성도 가능하다. AI 데스크탑에서 직접 터널을 열면 SSH 중간 단계는 필요 없다. 공개 연결에는 인증된 gateway만 사용하며 기존 로컬 전용 무인증 gateway 주소를 직접 공개하지 않는다.

## 아이폰에서 연결

1. 이 변경이 반영된 앱 번들을 Expo Go로 연다. `main`이나 이전 모바일 브랜치에 자동 반영된 것으로 가정하지 않는다.
2. 로그인 후 **내 계정 → 아이폰·외부 기기 AI 시연 연결**을 연다.
3. 전달받은 HTTPS 서버 주소(root 주소 하나)와 시연 연결 토큰을 입력하고 **시연 연결 적용**을 누른다. `/general`과 `/pcb`는 앱이 붙인다.
4. **모델 연결 확인**을 눌러 고정 모델 두 개의 준비 상태·SHA 확인 성공을 기다린다. 이 버튼은 사진을 보내지 않는다.
5. **안전점검 등록**에서 JPEG·PNG 사진을 고르면 앱이 점검 대상을 자동으로 판별하고, 해당 고정 모델의 손상 후보를 보여준다. 파일 크기는 8MiB 이하다.
6. AI 결과 검토·DB 저장·앱 종료 후 복원·PDF 공유를 실제 기기에서 확인한다. 연결·분석 실패는 오류이며 Mock으로 자동 전환하지 않는다.

토큰은 앱 메모리에만 보관한다. 앱 종료·완전 Reload 후 다시 입력하며, 같은 실행 중 역할 계정을 바꿔도 연결 설정은 유지된다. DB·보고서·로그·Git·`EXPO_PUBLIC_`에는 토큰을 넣지 않는다. 주소 변경·연결 지우기는 진행 중 AI 요청과 오래된 결과를 무효화한다.

## 사진 자동 판별 범위

시설 표면, 볼스크루 표면, 컨베이어 이물질 점검, RobustAD PCB 외관, 케이블 외관을 자동 판별 대상으로 사용한다. 짧은 변이 192픽셀 미만인 확대 크롭은 자동 판별을 보류한다. 모호한 사진은 재시도하거나 직접 검토할 수 있다. 기존 20개 항목은 **다른 검사 방법 → 점검 분야 직접 선택**에서 사용할 수 있다.

서버 담당자는 데스크탑의 루프백 Ollama에 `qwen3-vl:4b-instruct`가 준비된 상태에서 `SAFELOG_ENABLE_PHOTO_CLASSIFIER=1`을 설정하고 인증 게이트웨이를 실행한다. 분류기는 검사 대상을 선택하고 손상 결과는 기존 고정 모델이 생성한다. 분류기는 데스크탑 안에서 실행하므로 모바일은 기존 HTTPS 주소와 토큰을 사용하면 된다.

## 서버 담당자

정확한 토큰은 로컬 제한 파일에만 보관하고 최소 32byte 난수의 URL-safe 값(43자 이상)을 사용한다. CLI는 만료 시각을 현재부터 최대 24시간 이내로 제한한다.

```powershell
$env:SAFELOG_DEMO_TOKEN_FILE = 'C:\local-private\demo-token.txt'
$env:SAFELOG_DEMO_TOKEN_EXPIRES_AT = '실제 24시간 이내 UTC ISO 시각'
node scripts/fixed-cpu-gateway.mjs --port 8781 --require-demo-auth
```

환경 설정이 없거나 잘못되면 `--require-demo-auth` 실행은 중단된다. 토큰 누락·오류·만료는 health와 분석 모두 `401`이며 요청 본문 처리나 모델 호출 전에 거절한다. 토큰은 모델 서버에 전달하지 않는다. gateway는 `127.0.0.1`에만 바인딩한다.

시연용 Quick Tunnel의 origin은 인증 gateway의 root 주소로 지정한다. 주소는 실행마다 달라지므로 새 주소를 앱에 적용한다. 이메일 OTP 옵션 `--allowed-mail`은 브라우저용이며 현재 native API 호출에는 사용하지 않는다. Cloudflare 연결은 `info` 로그를 사용하고 인증 헤더를 기록할 수 있는 `debug`를 사용하지 않는다. 이 데스크탑의 테스트 실행은 2시간 뒤 터널을 자동 종료하고 인증도 만료한다. 재사용 시 이전 프로세스·PID·시작 시각을 확인하고 새로운 토큰·만료·주소로 준비한다.

Expo Go의 Metro/앱 번들 연결은 AI API 터널과 별개다. 이 프로젝트 SDK 58의 기본 Expo fetch를 사용한다. `EXPO_PUBLIC_USE_RN_FETCH=1` 또는 `true`로 구형 네트워크 구현을 강제하지 않는다.

## 확인된 범위

115개 자동 검사와 TypeScript·Expo lint가 통과했다. 실제 HTTPS에서 무토큰 `401`, 토큰 포함 health의 두 고정 모델 SHA 일치를 확인했다. 앱 분석기 자체의 HTTPS 연결 확인도 사진 없이 검증했다. Android·iOS·웹 번들 검사 결과와 원형 실패 기록은 [검증 보고서](TEST_REPORT.md)의 2026-11-24 절을 따른다.

**아이폰 실기기의 사진 분석·DB 재시작·공유 성공은 실제 리허설 후에만 완료로 표시한다.** 기존 PC 사진 9장 기록을 아이폰 검증으로 합산하지 않는다. Quick Tunnel은 개발·시험용이며 발표 전에 같은 아이폰·네트워크로 리허설하고 시연 중에만 실행한다.

공식 근거: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [Expo 환경변수와 Reload](https://docs.expo.dev/guides/environment-variables/).
