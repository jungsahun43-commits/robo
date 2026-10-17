# LIACi v94: 공식 관련 저장소의 정적 근거 조사

v90/v91 실패와 v92/v93 완료 수치를 재실행하지 않는다. ID107 COCO/BMP의 부식·marine_growth 존재 차이를 임의 수정하지 않으며 직접 정답 계약은 실패 상태다. 새로운 source RGB/mask 취득·pixel decode·시각 열람·특징·모델점수·gradient·optimizer·CAL_TEST는 모두0인 텍스트 조사만 수행한다.

공개 [SINTEF/LIACI_context](https://github.com/SINTEF/LIACI_context)는 원저자 LIACi 프로젝트의 관련 contextualization 코드다. 현재 배포 데이터 exporter라는 근거는 아직 없다. 저장소 설명상 inference와 inspection telemetry의 의미 연결이 주제이며, 이 설명을 bitmap/COCO 생성 이력으로 확대하지 않는다.

`inspect_liaci_public_code.py`와 `industrial-liaci-v94-public-code-policy.json`을 feature/ai-engine에 사전 일반 push/원격 확인한 뒤 실행한다. 정책 SHA256 `03e746c23a45d2ef9a4a506b2efd2d6e88d468f53c2c9de4544a16357bdd130d`이다.

익명 HTTPS GET로 공개 repo/default-branch commit/recursive tree를 받고 즉시 immutable commit/tree를 고정한다. 실제 목록의 root README/LICENSE와 path 키워드 coco20/dataset16/annotat12/label12/mask10/segment8 점수/파일명 순서 상위10개 py/md/rst/txt만 읽는다. 각 파일128KiB 이하, 응답2MiB/전체 네트워크4MiB/새 저장8MiB 이내와 예비1GiB를 유지한다. symlink/submodule/ipynb/이미지/모델/archive/전체clone은 취득하지 않는다. 미선택 관련 파일 목록을 남겨 조사 범위를 명시한다.

공식 API git blob의 base64는 소스 텍스트 전송 표현으로만 해독하고 Git SHA1/길이 및 로컬SHA256을 검사한다. 코드/import/eval/notebook/output 실행이나 모델 다운로드는 하지 않는다. 코드는 `.txt`로 저장한다. SHA1은 Git blob 동일성 검사이며 저자 제공 SHA256으로 바꾸지 않는다. 받은 응답·부분·실패·로그는 모두 보존한다. 인증/동의/접근우회/자동연락/PII제출은 하지 않는다.

자료의 원GT 권위·raster·overlap/unknown 의미는 정적 근거를 읽고 별도 판단한다. 코드의 MIT를 LIACi 데이터 CC BY-NC-SA4.0 또는 다른 부모 데이터 조건으로 바꾸지 않는다. 관련 프로젝트 코드라는 사실만으로 현재 데이터 생산 버전이나 물리 선박·촬영 대응을 증명하지 않는다. 전체 source 전이적 연결/oldheld 제외·고정 역할·최소 표본·저장량·최종 GT 계약이 미충족이므로 ready=false를 유지한다.

실행 전 CIM 명령줄/시작시각·lockPID/예산을 다시 확인한다. GPU 작업0, Windows os.kill(pid,0) 금지/read-only process_exists, 기존 training.lock23948과 모든 원본·실패·로그·모델/ZIP을 보존한다. awakehelper/전원계획 변경 및 기본앱/alias/경고/지도 교체는0이다.
