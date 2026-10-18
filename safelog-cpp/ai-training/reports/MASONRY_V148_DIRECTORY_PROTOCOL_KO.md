# Masonry v148 — 고정 두 ZIP의 중앙목록만 읽는 사전정책

## 목적과 취득 범위

보존된 Zenodo record18458458 API의 정확한 두 파일 선언을 재사용한다. 원자료 DOI10.5281/zenodo.18458458, Storey, Katherine A / Snowball, Andrew L, dataCC BY4.0·출처/저자·라이선스·버전과 변경 표시·제삼자 권리를 유지한다. web의 v1 표시와 API metadata.version null/서비스 revision4는 별개다. 출판·갱신일을 촬영일로 쓰지 않는다.

보존 description은 100개 image-mask 쌍·수작업 균열255/배경0을 선언한다. 원역할·촬영 대응표는 이 보존 텍스트에서 확보하지 못했다. 두 container 중앙목록에서 원분리 이름/metadata 파일 이름이 실제 있는지 확인하고 모든 정확한 이름·가능한 사본을 보호하는 데만 사용한다. 이름/파일수/폴더를 검증 사진·원역할·촬영그룹으로 승격하지 않는다. 전체 저자 자료에 역할표가 없다는 증명도 아니다. EXIF 제거·pixelation을 촬영그룹 해소나 정상/unknown 처리 권위로 쓰지 않는다.

외부 요청 전 helper, JSON 정책, 이 문서를 일반 feature 브랜치에 사전 commit/push하고 원격 동일성을 확인한다. 정확한 URL 두 개는 JSON에 고정한다. 완료 API/페이지/기존 HEAD·CD·JSON·사진·mask/완료 helper·fixture를 재실행하지 않는다. 새 fixture는 이번 helper의 가짜200/잘못된206에서 body읽기0 및 ZIP64 marker/댓글 guard 3개뿐이다. 원저자 코드 실행은 없다.

## 새로운 HEAD/Range/ZIP 구조 검증

1. 파일 순서는 masksZIP(1052176bytes), imagesZIP(262056968bytes)다. 각각 HEAD 최대1회, 합2회. exact200/동일URL/identity·정확 선언 길이·명시 Accept-Ranges bytes/quoted strong ETag를 필수로 한다. HEAD body는 읽지 않는다. 오류·redirect·범위 미지원·ETag 부재면 즉시 종료하며 두 번째 파일도 자동 진행하지 않는다.
2. 모든 GET은 고정 Range와 If-Range ETag를 보낸다. body 전에 exact206·동일URL·정확 Content-Range·길이·HEAD와 같은 ETag/storage-version·identity를 확인한다. 200 full 응답을 다운로드하지 않는다. 원형 headers·byte 범위·partial·failure를 보존한다. ETag는 archiveSHA/MD5나 촬영일이 아니다. Last-Modified도 촬영일이 아니다.
3. 각 파일 마지막22bytes만 EOCD 후보로 읽는다. **최종22bytes가 실제 EOCD/comment0임을 검사**하며 미일치/댓글이면 종료한다. 가변 tail 탐색으로 member body를 같이 받는 우회는 하지 않는다. comment0/noZIP64를 기존 MDMCS 사실로 가정하지 않는다.
4. ZIP64 sentinel 또는 32bit CD 뒤 metadata gap을 관찰하면 별도로 locator20/정확 record-offset의 fixed56을 읽는다. singleDisk·count·record경계/크기<=4096·32/64 선언 일치를 검증하고 필요할 때만 선언 extension을 읽는다. ZIP64 지원은 이 새 코드의 제한된 구조 검사이며 서버 지원/새 source의 실제 구조 통과를 미리 주장하지 않는다. 다른 trailer/비정상 구조는 중단한다.
5. CD 정확 offset/길이만 읽는다. 각 CD<=1MiB·entries<=2000·정확 header/extra/name/count/경계·ZIP64 entry 크기/offset·singleDisk를 검사한다. 중복/위험경로·암호화·비정상 구조는 중단하며 rename/복구하지 않는다. JPEG/PNG member의 localheader/내용/IDAT/scan은 읽지 않는다. 목록 CRC/크기/압축방식은 선언이며 실제 memberCRC/전체 archive SHA/MD5 검증이 아니다.

GET 최대10회(두 파일이 모두 ZIP64 extension을 가질 경우), 총 body<=4MiB, data+run 저장<=16MiB, 예비1GiB다. 새 stage 전 저장/임시/run 상한과 reserve를 함께 확인하며 작은 목록 예산을 전체 ZIP/픽셀/학습 예산으로 쓰지 않는다. retries/redirect/alternate endpoint/fullZIP fallback/latest/parent/pagination은 0이다. 첫 실패로 전체 helper를 멈추고 같은 요청을 재실행하지 않는다.

## 역할과 안전 경계

새 요청 전에 기존 container/가능 record/사본 UNASSIGNED 보호를 기록한다. 목록 취득 후 모든 정확 member 이름과 가능한 사본을 UNASSIGNED 보호한다. 명시 train/test/val 같은 경로 토큰도 관찰일 뿐 원저자 원분리 검증/최종 FIT_VAL_CAL 또는 TEST 예약 완료를 의미하지 않는다. 사진 고유성·정답 권위/unknown·RGB/native좌표·물리그룹·source-oldheld 전체 graph·사진/group/validpixel 최소·ready는 계속 미검증이다. 새 pixel/split/model protocol, features/score/gradient/optimizer/CAL_TEST/CPU_GPU_CPP_ZIP_APK는 0이다.

사진/mask/annotation/fullarchive/PDF/원저자code/weights 취득·해독·시각QA/raster는 허가하지 않는다. ZIP metadata 대상 바이트의 실제 수신량은 기록하며 전체 source 수신0으로 설명하지 않는다. 실패한 구조 후보 바이트도 원형을 보존하고 유효한 metadata로 소급 통과시키지 않는다.

삭제는 사용자 승인 A의 6개 사본 12GiB만 이미 완료했다. B/C와 추가 삭제 허가는 없다. 최신 보호10과 기존 GT/역할/graph/잠긴TEST·승인 모델/앱·부모조건은 유지한다. 기존 source/모델 수용 기준을 완화하거나 source 검토를 앱 사진/현장독립/전체건강/0미탐 통과로 확대하지 않는다.

실행 전 fresh CIM 명령줄/시작시각·lock PID·실제 v85 장부450/epoch30·read-only process_exists·예비공간을 확인한다. Windows os.kill(pid,0), GPU 중복, deadtraining.lock23948 변경, awakehelper/전원 변경은 금지다. 전용 operation.lock은 자기 PID/token만 해제한다.

저장소 상태는 tracked clean과 **기존 로컬 검토문서2개만 untracked**인 상태를 구분해 기록한다. 두 문서의 경로/SHA는 정책에 고정하며 이 사전 commit에 넣지 않는다. 이 문서와 새 코드/JSON만 사전 일반 push한다. 원격 전송이 거절됐던 로컬 보관/삭제 문서를 대신 보내는 우회가 아니다. 원격 확인 전 source HTTP는 0이다.
