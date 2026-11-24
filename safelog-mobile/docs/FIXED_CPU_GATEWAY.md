# 기존 고정 CPU 모델과 모바일 앱의 연결

이 gateway는 기존 모델 서버를 같은 Windows PC의 Expo Web 또는 USB로 연결한 Android 개발 앱에 전달한다. 학습 가중치·모델 코드·사진을 GitHub나 모바일 번들에 넣지 않는다. 기존 서버의 고정 checkpoint와 판정 기준은 유지한다.

2026-11-24 아이폰 Expo Go 시연용으로 파일 기반 단기 토큰 인증과 HTTPS 연결 설정을 추가했다. 기본 로컬 전용 동작은 유지하며, 공개 연결은 `--require-demo-auth`로 시작한 인증 gateway만 사용한다. [아이폰 시연 절차](IPHONE_DEMO_HTTPS.md)에 서버 설정·앱 입력·만료·실기기 검증 범위를 구분했다.

## 같은 PC의 Expo Web

먼저 기존 전달본 안내에 따라 고정 CPU 서버를 실행한다.

- 통합 모델 `v16-r1`: `127.0.0.1:8775`
- PCB `pcb-multisource-v3`: `127.0.0.1:8774`

별도 터미널의 `safelog-mobile` 폴더에서 Node.js 24로 실행한다.

```powershell
node scripts/fixed-cpu-gateway.mjs
```

앱 분석기 주소는 다음과 같다.

| 용도 | 앱이 사용하는 base URL |
| --- | --- |
| 시설·컨베이어·전기 배선·금속 기계(볼스크루 포함) 15항목 | `http://127.0.0.1:8780/general` |
| PCB 5유형 | `http://127.0.0.1:8780/pcb` |

gateway는 `/general/health`, `/pcb/health`를 각 서버에 그대로 전달한다. 앱은 해당 응답의 준비 상태와 모델 SHA를 확인한다. gateway 시작 성공만으로 모델 준비나 사진 판정 성공을 뜻하지 않는다.

Expo Web의 `http://localhost:8081`, `http://127.0.0.1:8081`과 같은 두 호스트의 8082·19006 포트를 기본 허용한다. 다른 로컬 웹 포트를 사용하면 정확한 origin을 한 번 추가한다.

```powershell
node scripts/fixed-cpu-gateway.mjs --web-origin http://localhost:8083
```

기본 gateway 포트 8780을 바꿀 때는 `--port 8781`을 지정하고 앱 주소도 함께 바꾼다. 바인딩은 항상 `127.0.0.1`이며 LAN·외부 인터페이스로 열지 않는다. 방화벽 규칙이나 공개 터널을 만들지 않는다.

## Android USB 개발 연결

실제 Android 기기 또는 에뮬레이터, USB 디버깅, Android platform-tools가 준비된 개발 PC에서는 해당 기기에 다음 포트 연결을 설정할 수 있다.

```powershell
adb reverse tcp:8780 tcp:8780
```

이 구성을 확인한 앱에서 같은 `http://127.0.0.1:8780/general`·`/pcb` 주소를 사용한다. Expo 앱 실행 연결과 모델 API의 포트 연결은 별개다. Android의 HTTP 정책과 실제 기기 연결을 확인해야 하며, 이 문서는 실기기 실행 통과를 주장하지 않는다. USB 연결이 없거나 iOS 기기를 사용할 때는 별도 연결 구성이 필요하다.

## 실패와 보관

- 모델 준비 오류·503·422 등은 상태와 JSON을 그대로 전달한다. 오류를 정상 또는 후보 미검출로 변환하지 않는다.
- 브라우저 origin은 명시된 HTTP loopback 목록으로 제한하고 쿠키·인증 헤더를 모델 서버에 전달하지 않는다.
- 허용된 건강 확인·분석 경로만 전달하며 임의 URL이나 다른 서버 경로를 프록시하지 않는다.
- JSON 요청은 24MiB, 응답은 8MiB, 전체 요청 시간은 기본 60초로 제한한다. 앱과 모델 서버는 이와 별도로 사진 8MiB 제한과 입력·응답 검증을 적용한다.
- 클라이언트 취소/연결 종료 시 진행 중인 모델 요청도 중단한다. 사진 바이트·원문 요청·응답을 gateway 파일이나 로그에 기록하지 않는다.
- 실모델 연결 실패 시 앱에서 수동 검토를 선택한다. 사람의 최종 확인과 AI 제안은 분리한다.

격리된 loopback 모형 서버를 사용하는 gateway 테스트는 다음 명령이다. 실제 모델이나 평가 사진을 사용하지 않는다.

```powershell
node --test tests/fixed-cpu-gateway.test.mjs
```
