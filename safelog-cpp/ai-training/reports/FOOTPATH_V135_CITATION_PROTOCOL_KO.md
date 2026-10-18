# v135 사전 고정 — 인용 record와 버전 metadata 두 개만

기존 v134 API/text/보호/완료기록을 SHA로 재사용한다. 실제 LICENSE가 인용한 DOI15414238의 고정 record API `https://zenodo.org/api/records/15414238`와, 보존 current API가 제공한 `https://zenodo.org/api/records/17154852/versions` 두 GET만 새로 읽는다. old17154852 API/README/LICENSE·논문·latest·concept redirect 재조회0이다. 이 코드/JSON/문서를 일반push하고 HEAD/원격일치 확인 전 새 HTTP 금지다.

각<=512KiB/합<=1MiB/data+run<=16MiB/예비1GiB다. exact200·동일URL·identity·JSON MIME·헤더길이(있으면 body전)/실제길이·strictJSON을 확인한다. cited ID/DOI exact, 버전목록 hits<=25·명시 total/index 자료형을 검사한다. 상한/형식/redirect/HTTP오류는 중단하고 부분/원응답/헤더/실패/로그를 보존한다. retry/pagination/latest/alternate endpoint/HTML/PDF/코드·이미지·CSV/ZIP/member·어떤 nested link도 추적하지 않는다.

cited file 목록의 모든 container/가능한 record는 UNASSIGNED 보호한다. same concept ID/DOI를 실제 응답에서 대조하고 명시 version index만 기록한다. 날짜/순서/service revision을 dataset V1/V2로 추정하지 않는다. 파일명/선언bytes/checksum 비교는 publisher metadata 일치일 뿐 실제 archive/CRC/사진 동일성·독립성·새 source 수량 검증이 아니다. 일치하지 않아도 관계를 강제/수정하지 않고 보류를 유지한다. dataset ready·사진TEST/역할·GT/unknown·최소/graph·pixel/model protocol은 미확정/0이다.

fresh hostCIM 명령줄/시작시각·lockPID·실제장부 확인 뒤 Python 하나만 실행한다. utility의 read-only OpenProcess만 사용하고 old main/fixture/os.kill0, dead training.lock23948 보존·awakehelper/전원변경0이다. 새 stage 경로가 있으면 재실행/덮어쓰기 중단한다. 현재PID/token의 임시 operation.lock만 종료시 해제한다. 기존 모든 원형/부분/실패·GT/잠긴 역할/graph·모델/앱 출력은 보존한다. source픽셀/QA/features/우리모델score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이며 full4.5GB archive는 계속 예산 밖/취득0이다.
