# v115–v116 Lusitano 공개 자료 계약 선별

2026-10-17 UTC. 실제 repository report/doc/source에서 Lusitano·정확 DOI/URL의 완료기록을 먼저 검색했다. 검색 오류 없이 미일치했고 v8–v114 완료단계는 다시 실행하지 않았다.

## v115 원판 Lusitano2024

[원논문](https://www.mdpi.com/2076-3417/14/12/5298), Carrilho/Hambarde/Proença, DOI10.3390/app14125298의 공개 본문은 실제 생산 원단의 자연 결함을 설명하지만 원래 TRAIN에는 정상만 넣고 결함은 TEST로 예약한다. 현재 자연 손상pixel TRAIN 확보 방향에서는 **취득 전 제외**한다. 정상 분포 학습을 다시 돌리거나 원TEST를 TRAIN으로 옮기지 않는다.

논문 내 TEST 수량 문구도 구간마다 달라 실제 수량으로 확정하지 않았다. dataset 고유 허가·실제 nativeGT/좌표·촬영그룹·file무결성은 미검사다. 발견한 원저자 project/download 링크를 열거나 원본을 받지 않았다. 논문의 모든 다른 활용이 불가능하다는 뜻은 아니다.

## v116 새 LUSITANOv2

[원논문 색인](https://www.mdpi.com/2079-9292/15/19/4403)의 Data Availability가 직접 연결하는 [원저자 Zenodo v1](https://zenodo.org/records/22150068), DOI10.5281/zenodo.22150068, 2026-09-23을 별도로 확인했다. 배포 설명은 생산 RGB 25,120개·YOLO 결함상자18,557개와 **pixel-level mask 미제공**을 명시한다. 이 수량은 배포자 선언이며 확보/검증 사진·독립 부품 수가 아니다.

따라서 **bbox-only 방향을 현재 native 손상pixel TRAIN 우선순위에서 취득 전 제외**한다. 상자를 양성mask로 채우거나 별도 검출과제를 자동 추가하지 않는다. 원판의 정상-only TRAIN을 새 버전에 그대로 가정해서 제외한 것이 아니다.

화면의 TRAIN/VAL/TEST·파일목록·37.6GB·MD5 표시는 실제 archive검증이 아니다. Rights/License 제목 아래 읽을 수 있는 허가값은 이번 text에서 확보하지 못했고 paper조건으로 대체하지 않았다. 직접 publisher HTML은429여서 재시도/다른PDF 접근을 하지 않았다. Zenodo archive/checksum/readme 링크도 열지 않았다.

## 실제 수행 범위

`data/industrial-source-v115-v116-metadata/primary-web-screens.json`은 웹 관찰메모이며 publisher 원문 고정파일/자료checksum이 아니다. source 사진·mask·archive·PDF/code/weight 취득, pixel/내용QA·encoder/점수·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 새 정식 취득·좌표·분리·모델protocol도 없다. 자료 선별 완료를 실제 학습 실패나 성능 개선으로 설명하지 않는다.

원모델·GT/TEST·역할·graph·모든 원본/실패/epoch/로그/cache/ZIP·앱 출력을 보존했다. 추적 완료 optimizer 부분합60430은 그대로이며 전체 역사 총량은 아니다. 다음은 `INDUSTRIAL_SOURCE_AFTER_V116_NEXT_KO.md`를 따른다.
