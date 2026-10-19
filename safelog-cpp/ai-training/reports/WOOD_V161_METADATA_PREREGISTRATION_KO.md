# v161 metadata 사전 등록 — 실행 전 검토본

## 정확한 범위

원배포 record의 라이선스·파일 선언을 확인하기 위해 `https://zenodo.org/api/records/4694695`에 **GET 최대1회**, JSON 본문 최대256KiB만 제안한다. 이 API endpoint는 record ID로 구성한 사전 고정 주소이며 아직 요청·응답 확인하지 않았다. 원배포 page의 실제 읽기와 새 API 요청은 구별한다.

스크립트 `scripts/inspect_wood_record_v161.py`, 정책 `reports/WOOD_V161_METADATA_PROTOCOL.json`, 이 문서의 **세 파일만** 사전 공개 전송 대상이다. 정확한 목적지는 **https://github.com/jungsahun43-commits/robo.git**, branch **feature/ai-engine**다. 기존 마지막 확인 remote `b7d0558c523f90d3dfc46c220535aed9243dd6c6`를 부모로 세 경로의 blob만 추가한 별도 publication commit을 준비한다. 승인 후에도 실제 remote ancestor/diff/동일 blob을 확인하며 비force 전송만 허용한다. remote가 바뀌었으면 임의 병합/force/다른 ref로 우회하지 않는다.

v157 이후의 로컬 검토 보고서·v160 보고서, 보관 검토2문서, source 원문·사진·GT·모델·weights·ZIP은 이 publication payload에 포함하지 않는다. 이전 exact6 보고서 승인은 완료/소진됐으며 이번 세 파일로 재사용하지 않는다.

## 요청·보존 guard

- 정확200/동일URL/application-json/identity, 단일 양수 Content-Length 필수·256KiB 상한·실제 길이 일치.
- redirect/재시도/alternate/HEAD/Range0. 오류·불일치 시1회 중단하고 원형 또는 명시한 상한의 header prefix+full-header SHA/부분 body/started/failure/stderr를 보존한다.
- strict UTF-8 JSON·중복 key/nonfinite 거부, recordID/DOI/title/creator token 및13파일 선언을 검증한다. 규격 파일의430bytes/선언MD5는 API 내부 선언 일관성 확인용이며 실제 그 파일을 받은 것이 아니다.
- License/version/service revision/파일 크기·MD5를 **선언으로만** 기록한다. 촬영일·실 file revision·실 archive CRC/GT·권리 전체 감사·학습 준비 통과로 승격하지 않는다.
- 저장/임시/run 합계4MiB+예비1GiB. 본문 최대256KiB와 보존 로그를 포함한 metadata 예산이며 full archive·pixel·학습 예산이 아니다.
- 요청 전 source family의 모든 가능 container/member/record/사본 UNASSIGNED 및 원TEST·publisherVAL 보호 영수증을 먼저 저장한다. 파일 링크는 추적하지 않는다.

## 실행 전 gate

사람의 exact 세 파일/목적지 전송 승인과 일반 사전 push·원격 검증이 필요하다. helper는 성공 publication 영수증·최대300초 내 원격 검증·exact 세 파일의 로컬 SHA/commit blob 일치를 검사한다. 새 성공 CIM preflight는 최대120초, 명령줄/시작시각·관련process0·AI root/runs와WORK outputs operation.lock0·PID23948부재·v85실제450/epoch30·실저장 예산을 확인해야 한다. helper는 현재 lock/여유를 재확인하고 자신의 PID-token operation.lock만 만들고 일치할 때만 해제한다. Windows os.kill(pid,0)·다른lock/deadtraining.lock23948/awakehelper/전원 변경0이다.

이 문서는 실제 요청 실행 기록이 아니다. 현재 새 API/규격TXT/사진/GT/archive 취득0이다. 기존 계속작업 승인은 유지하지만 사전 공개가 완료되기 전 `--run`을 실행하지 않는다. metadata 성공 뒤에도 source 권리/RGB/native GTunknown/배경·원역할/촬영group·전graph·고정 최소/full 예산·별도 pixel 정책/공통encoder 모델 조건·모든 평가/retention/runtime/삭제 승인 gate는 그대로다. **ready=false**다.

## 로컬 코드 확인

합성 metadata로 JSON/정체성/파일 선언·본문 header 상한·redirect 차단3개 fixture를1회 통과했다. 이어 redirect 중단 시 header 보존의 구체적 누락을 수정하고 해당 새 fixture1개만1회 통과했다. 앞3개를 재실행하지 않았으며 각 실행 당시 script SHA와 stdout/stderr는 별도 원형 영수증에 보존한다. 최종 코드는 앞3개에서 검사한 parser/header gate를 유지하고 새 redirect 관찰을 추가했다. 실제 HTTP·사진·GT·tensor·학습·모델 runtime 검사는 수행하지 않았다.

실제 helper/API가 중단·완료되면 재실행/상한완화/다른 endpoint/full 다운로드로 우회하지 않는다. 기존 모든 원형/부분실패·epoch/log/cache·GT/역할/graph·모델/앱과 A삭제완료/보관 조건은 유지한다. 이 사전 등록 전송은 추가 삭제나 앱 primary 모델 교체를 허가하지 않는다.
