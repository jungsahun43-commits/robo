# MDMCS v124 원 ZIP 중앙목록 조회 사전조건

Guo/Bao Mendeley V2 DOI10.17632/6x4dzzrs2h.2의 data CC BY4.0·출처·버전·변경·제삼자 권리 조건과 v123의 모든 보존조건을 유지한다. V1/V2가 같은 파일 id/이름/크기/SHA를 선언했다는 사실을 실제 원본 동일성 검사로 확대하지 않는다. v121 실패와 v122/v123 완료 metadata를 재요청·재실행하지 않는다.

## 사전 고정한 한 가지 조회

실행 전에 이 문서·JSON·코드를 일반 commit/push하고 HEAD와 원격 feature/ai-engine의 일치를 확인한다. 고정 v123 파일목록의 SHA와 원 URL을 재사용한다. 원 URL 하나에 HEAD 1회, 조건 통과 때만 마지막 22bytes와 중앙목록 Range 각 1회를 허용한다. redirect·retry·다른 endpoint·전체 ZIP 우회는 없다.

- HEAD: 응답 URL 동일, status200, Content-Length204919973, strong quoted ETag, Accept-Ranges bytes, Content-Encoding identity를 모두 요구한다. HEAD body는 읽지 않는다.
- GET: If-Range에 HEAD ETag, 정확206/Content-Range/Content-Length/동일 ETag/identity를 **body 읽기 전** 확인한다. 200 전체 응답·redirect·불일치는 body0으로 중단한다. HTTP 오류/redirect 헤더는 보존한다.
- EOCD: 정확 마지막22bytes만 허용한다. 단일 disk, ZIP64 아님, 주석0, 중앙목록 끝이 EOCD 직전이어야 한다. 더 큰 tail을 다시 받지 않는다.
- 중앙목록: 최대4MiB/20000멤버. 선언 개별 확장≤128MiB/합계≤4GiB. 지원 압축은 stored/deflate뿐이다. 안전한 상대경로·대소문자 무시 중복·암호화·symlink·ZIP64·대체 Unicode path·손상 extra/header를 검사한다. local header/멤버 body는 요청하지 않는다.
- 실제 Range body 합계≤4194326bytes. data/run 저장≤32MiB와 예비1GiB를 요구한다. 원본·부분·헤더·실패·로그는 보존한다. 완성 디렉터리와 선언 CRC/크기를 기록하되 archive SHA/실제 CRC 검증으로 부르지 않는다.

합성 metadata guard fixture3은 EOCD/목록 정상 구조, unsafe/encryption/duplicate 거절, 전체응답·ETag변경의 body 전 거절을 확인한다. source 사진·주석·synthetic GT/학습은 아니다. 기존 완료 fixture는 다시 돌리지 않는다.

## 보호와 한계

모든 멤버/가능한 사본을 UNASSIGNED 보호한다. 명시적인 소문자 경로 train/training/val/validation/cal/calibration/test/testing/not_used 토큰만 관찰값으로 저장하며 실제 사진역할로 확정하지 않는다. 명시 test/testing 멤버도 내용 전에 별도 보호한다. 숫자·무늬·파일명 유사성으로 원역할·물리그룹·사본관계를 추정하지 않는다. 원사진 TEST/최종분리·정상/배경·native GT/unknown·RGB·촬영그룹·전체 graph·최소는 미검증, readyfalse다.

source archive 전체·멤버·RGB/mask/JSON 내용·thumbnail·해독·시각QA·encoder/점수·gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0이다. source 픽셀 전 별도 고정 TEST/사본·역할·GT/사진/group/pixel 최소·좌표/해독·오류제외·예산 코드JSON 사전push, 자료통과 뒤 별도 단일모델 조건 사전push 원칙을 유지한다. old 지문/graph/잠긴 모든 역할·TEST/모델·실패·출력은 그대로 보존한다.

새 Python 전 fresh hostCIM 명령줄/시작시각·lockPID·v85 실제450/epoch30 장부를 확인한다. read-only process_exists만 사용하고 os.kill/dead training.lock23948 변경/awakehelper 재시작·연장·전원변경은 금지한다. GPU 작업0, 앱/alias/경고/primary/ZIP 자동교체0이다. 조건 미달은 metadata 접근 STOP이며 새 학습 실패가 아니다. 동일 실패 재시도나 전체다운로드로 해결하지 않는다. 계속승인과 자동후속 ACTIVE를 유지한다.
