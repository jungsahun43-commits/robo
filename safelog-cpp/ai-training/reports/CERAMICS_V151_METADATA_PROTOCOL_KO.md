# v151 — 새 원배포 root-file 목록 한 번만, 사전 정책

원배포 DOI10.17632/2bkhytgwm8.1의 제목 V2와 publisher Version1을 구별한다. 원data CC BY-NC3.0/세 원저자·출처·버전·변경/제삼자 조건을 보존한다. 생산라인/녹색 paint-derived 약한 주석/회전·패치 설명은 author 선언이며 원본 GT·독립 그룹·고정 역할은 미검증이다.

정확 한 URL `https://data.mendeley.com/public-api/datasets/2bkhytgwm8/files?folder_id=root&version=1&%24start=0&%24limit=1000`만 GET한다. 이는 기존 완료 Mendeley API 경로 형식에서 새 ID/version을 고정한 것이며 현재 endpoint의 응답을 먼저 본 것은 아니다. 새 page/related record/API docs/folders/pagination/file/download/redirect/retry/fallback 요청은 허용하지 않는다. 파일 목록만으로 전체 source tree나 원사진별 TEST/VALHELD/최종 FIT_VAL_CAL·촬영 대응·GT·archive SHA/CRC를 통과했다고 주장하지 않는다.

source family와 가능한 related record/모든 container·record·사본은 본문 전에 UNASSIGNED 보호한다. exact200/동일URL/JSON MIME/identity 및 있으면 Content-Length 정수·512KiB 이하를 body 전에 요구한다. 실제 bounded 수신 길이와 header 길이를 대조하고 UTF8 strictJSON/중복 key/NaN을 거부한다. root-list는 1~999개, 각 file id/name 중복0/statusCOMPLETED/정수size≥0/publisher SHA25664hex가 있어야 완료로 기록한다. unknown schema/잘림가능1000/길이·SHA형식 불일치는 보존 중단하며 자동 보정하지 않는다. 선언 SHA는 실제 archive의 local SHA 검증이 아니다.

source metadata+임시/run 저장 예산4MiB 및 예비1GiB를 사전 고정한다. source binary/full-archive 저장 예산으로 쓰지 않는다. 새 fixture3개는 (1) redirect/wrong MIME/body cap header에서 body읽기0 (2) strictJSON의 중복key·NaN 거부 (3) truncation/중복ID/missing publisher SHA 거부를 검증한다. 기존 helper/fixture/HTTP를 재실행하지 않는다.

스크립트·JSON·이 문서·v150검토4파일의 commit 내용과 local SHA가 같고 일반 HEAD:feature/ai-engine push·원격 일치한 뒤만 helper1회 실행한다. 기존 로컬 보관 문서2개는 exact SHA로 확인해 미추적 유지한다. fresh CIM 명령줄/시작시각·현재 lock/PID·v85actual450_epoch30·예산을 확인한다. 새 helper는 Torch/NumPy/Pillow나 원저자 코드·의존성을 import/실행하지 않는다. OS read-only process 조회만 사용하며 os.kill0/deadtraining.lock23948/다른 lock 변경0이다.

헤더·rawJSON/부분·receipt·started/stdout/stderr·성공/실패를 보존하고 실패 재시도0, 자기 PID/token operation.lock만 해제한다. 사진/mask/annotation/archive/PDF/code/weights 취득·해독·시각QA·raster/features/우리score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, 새pixel/split/model protocol0/readyfalse다. paper/code/mirror를 data권리로 바꾸거나 약한 주석·patch수·정상 선언을 검증 자연 nativepixel/사진·배경/unknown으로 승격하지 않는다. A승인 소진/B_C미승인/추가삭제0·보호10/원형/GT역할graph·기존 모든 source/model gate를 유지한다.
