# v139 — Turboshaft Blade 공개 metadata 선별

새 공개 배포 metadata 탐색에서 발견한 [Turboshaft Blade Corrosion Image Collection](https://zenodo.org/records/17834008), DOI10.5281/zenodo.17834008/v0.1/2025-12-05를 선별했다. 원배포 creator 표기는 `Jiří, Pavlas`(Data collector), `Cejnek, Matous`(Project member), Czech Technical University in Prague다. 정확명칭·확인 저자/DOI/URL·연결 repo의 현재 repo 텍스트 검색은 정상 exit1 미일치였다. 모든 과거 commit/ignored 원형을 감사한 것은 아니다. 탐색에 노출된 기존 Ostrava는 다시 열지 않았고 다른 model/제한형 자료·PDF도 별도 단계/배포 근거로 채택하지 않았다.

원 record는 부식 양성/음성의 원본 stem 목록과 `dataset_patches.zip`262.1MB/배포자 MD5 `786d76733026b4c9732a7238bf811072`를 선언한다. stem 번호/접미사를 실제 부품/촬영 그룹으로 추정하지 않는다. 선언 patch 목록은 실제 확보·검증 사진/원역할·native GT/정상 wholephoto/독립 부품 수가 아니다. Rights/License 항목의 허가 값은 이번 web text에서 읽을 수 없었으며 허가가 어디에도 없다는 주장이 아니다.

원배포가 직접 연결한 [원저자 repo](https://github.com/matousc89/synthetic-corrosion-segmentation)의 공개 main README는 synthetic data를 이용하는 corrosion classification과 실행 경로를 설명한다. repo명/README 제목만으로 배포 사진 전부가 합성이거나 pixel mask가 없다고 단정하지 않는다. 반대로 segmentation 코드명·양성/음성 목록을 실제 자연 TRAIN pixel GT/명시 배경/unknown 계약으로 승격하지 않는다. branch 이름은 immutable revision 증명이 아니다. 실제 RGB·자연발생·원TRAIN/native GT/unknown·좌표/export·원분리/물리 그룹·원자료 권리/제삼자 권리·파일 bytes/checksum·선택 접근/임시량/예산은 미검증이다.

다음 v140은 이 새 원저자 record의 아직 미시도 공식 API 하나로 권리/version/정확 file 선언을 확인하는 별도 metadata 작업만 허용한다. 코드JSON·상한/실패 보존/예산을 사전 일반 push한 뒤만 수행한다. 새 binary/pixel/split/model protocol은 아니다. 원 Zenodo/README 반복·archive preview/download·코드 파일/실행·PDF/weights 취득0, 자동 연락·로그인/동의/PII·우회0이다. 웹 traffic bytes 미측정/전체 HTTP0 주장0이다. 관찰 메모는 publisher 고정 원문/자료 checksum과 구별한다. source pixel/QA/features/우리 score/gradient/optimizer/CAL_TEST/delivery0/readyfalse다.
