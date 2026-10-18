# v134 사전 고정 — Zenodo metadata와 작은 정적 텍스트만

대상은 원저자가 연결한 DOI10.5281/zenodo.17154852 V2 한 record다. 현재 code/JSON/이 문서를 일반 commit/push하고 remote feature와 HEAD 일치를 확인하기 전 HTTP 금지다. 고정 API `https://zenodo.org/api/records/17154852` 한 GET 뒤, 배포목록과 대조한 `LICENSE.txt` 및 `Indian Footpath Damage Segmentation Dataset.md` 두 개의 same-record 공식 self URL만 한 번씩 읽는다. 목록 네 파일 이름/관찰 MD5를 모두 맞추고 archive/가능한 포함record 전체 UNASSIGNED 보호를 텍스트 전에 저장한다. 이것은 사진별 TEST/최종역할 검증이 아니다.

API<=512KiB, 각 text<=16KiB/LICENSE931bytes exact, 합 network<=1MiB/data+run<=16MiB/실제 예비1GiB다. exact200·동일 URL·identity encoding·허용 MIME·선언길이(있으면 body 전)·실제길이·publisher MD5를 검사하고 local SHA256을 별도 기록한다. chunked면 cap 안에서 EOF까지 실제 길이를 확인한다. MD5를 SHA256으로 꾸미지 않는다. API identity/네 파일명/MD5·text size·same-record URL guard 불일치, redirect/non200/HTML·오류면 중단한다. 부분·원응답/헤더·실패/로그 보존, retry/alternate/fullZIP 우회0이다.

UTF8 strict 텍스트만 읽고 실행·렌더·링크 추적·사진/배열/IDAT/JPEG 해독0이다. CSV/ZIP/member/PDF/code/weights/이미지·GT는 금지다. 4.5GB archive를 전부 받거나 허가·내용을 예측하지 않는다. 논문 data조건과 실제 text/제삼자 권리를 구별해 결과를 검토하며 자동 dataset ready/최종역할/학습 승격0이다.

fresh host CIM 명령줄/시작시각·잠금/실제장부 확인 뒤 Python 하나만 실행한다. utility의 read-only OpenProcess만 재사용/no os.kill, 오래된 main/fixture 재실행0이다. dead training.lock23948 보존/awakehelper·전원변경0. 새 stage 경로가 이미 있으면 재실행/덮어쓰기 중단한다. 현재 PID/token의 임시 operation.lock만 종료시 해제하고 다른 lock은 건드리지 않는다. 모든 기존 원본/부분/실패·GT/잠긴 역할/graph·보호모델/앱 출력을 보존한다. features/score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이다.
