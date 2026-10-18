# Turboshaft v142 — 고정 commit의 단일 정적 텍스트 사전조건

v141은 c6972e439d9036867d9be2d0460398f220658e37 사전 push 뒤 GET1/382bytes를 실제 받았으나 JSON의 내부 URL이 `git/refs/heads/main`이라 요청 URL `git/ref/heads/main`과 다르다는 고정 guard에서 실패했다. v141 정책/코드·원형·headers·failure·stderr487bytes를 변경하거나 소급 통과시키지 않는다. HTTP exact200/최종 요청URL동일/Content-Length382를 보존한 원문에서 ref `refs/heads/main`, object type `commit`, SHA `7a63ebc763b94b918b23ab79e3eb3abef4e1b01a`를 로컬 관찰했다. 이 단계는 그 정확 원문·헤더·실패 SHA를 별도 precondition으로 묶으며 이전 endpoint를 재요청하지 않는다.

원배포가 연결한 repo의 앞서 읽은 README에 명시된 경로 `segmentation/train_seg.py`를 위 commit에 고정한다. 정확 새 URL `https://raw.githubusercontent.com/matousc89/synthetic-corrosion-segmentation/7a63ebc763b94b918b23ab79e3eb3abef4e1b01a/segmentation/train_seg.py` 하나만 GET1로 요청한다. code/JSON/문서 3파일을 일반 commit/push·원격 feature 일치 확인 뒤 수행한다. commit은 repo identity이며 dataset revision·촬영일·저자 file checksum이 아니다. 사전 저자 file checksum은 제공받지 못했으므로 null로 남기고 수신 뒤 실제 길이/local SHA256을 별도로 기록한다. 파일이 없거나 조건이 다르면 실패 보존/재시도·대체 경로0이다.

exact200/동일URL/text_plain/identity와 Content-Length(decimal·상한 내, 있으면 실제길이 일치)를 확인한다. body64KiB 상한/cap+1 초과 즉시 중단/원형부분보존, UTF8·비어있지 않음·NUL없음만 검사한다. headers/receipt/started/stdout/stderr/failure를 보존한다. data+run2MiB/예비1GiB·fresh CIM 명령줄/시작시각·operation locks/실제v85장부·read-only process_exists를 확인한다. 자기 PID/token 임시lock만 해제하고 deadtraining.lock23948/다른lock/awakehelper/전원은 변경0이다.

원저자 파일은 Git 제외 data에 원문바이트 그대로 `.py.txt`로 보존하고 텍스트로만 읽는다. import/exec/eval/compile/원저자 함수·모델·의존성 실행/추가 import파일·links 취득0, 사진·mask decode/배열·히스토그램/시각QA/raster0이다. 포함 저작권/라이선스·인용 원형을 유지하되 code 권리를 data 권리로 대체하지 않는다. 자연발생 TRAIN/nativeGT·unknown/배경·원 exporter·분리/촬영 대응에 실제 명시된 범위만 관찰하며 모델 출력을 GT로 쓰지 않는다.

source DOI10.5281/zenodo.17834008/v0.1/Jiří, Pavlas 및 Cejnek, Matous/data CC BY4.0/출처·버전·변경·제삼자권리와 모든 기존 UNASSIGNED/container/copy 보호를 유지한다. 원TEST/VALHELD·최종역할/GT_unknown_RGBcoords/촬영그룹/전체source-oldheld graph/고정최소/full예산 미검증이다. source사진_mask_annotation_archive_PDF_weights0/실행·features_우리score_gradient_optimizer_CAL_TEST_delivery0/readyfalse. 새로운 pixel/모델 protocol은 이 문서가 아니다. 모든 이전 invariant·부모조건·닫힌source 제한을 유지한다.
