# v113–v114 산업 부식 자료 공개 근거 선별

2026-10-17 UTC. 실제 repository report/doc/source에서 정확 저자·제목·DOI를 검색한 뒤 새 후보를 순서대로 검토했다. v8–v112 완료단계는 재실행하지 않았다.

## v113 Fondevik 외 ICTAI2020

[원저자 소속 SINTEF 기록](https://www.sintef.no/en/publications/publication/0198cc7ab997-41a8cfe6-243c-4cee-9d8c-1a5122d2bde8/), DOI10.1109/ICTAI50040.2020.00125는608개의 부식 이미지와 instance별 binary segmentation mask를 설명한다. 이는 확보·검증 사진이나 독립 부품 수가 아니다. 연결된 NVA 기록은 이번 text 조회에서 JavaScript shell만 노출했다. JavaScript 실행·로그인·접근우회·author 연락은 하지 않았다.

실제 source 파일·data권리·checksum·원분리·정상/배경지원·native좌표·촬영/부품묶음은 미검증이므로 **취득 전 보류**한다. 논문 존재나 paper 접근 조건을 데이터 허가·readyTRAIN으로 대체하지 않으며 dataset이 어디에도 없다는 결론도 아니다.

## v114 CorrDetector

[원저자 arXiv 기록](https://arxiv.org/abs/2102.04686) 및 [저자 PDF의 색인텍스트](https://yongbinkang.github.io/files/research/CorrDetector_ESA_2021.pdf)는 drone 산업영상에서 격자별 부식 판정과 구조물/배경 분리를 다룬다. 관찰한 grid/object 라벨을 부식의 실제 pixel mask로 채우지 않는다. 현재 native손상pixel TRAIN 확보 방향에서는 **관찰한 계약을 취득 전 제외**한다. 다른 미검사 주석이 전혀 없다는 주장으로 확대하지 않는다.

직접 저자 PDF는404였고 재시도/다른PDF 취득은 하지 않았다. 실제 file/권리·원분리·nativeGT·물리그룹·무결성은 미검증이다. DOI10.1016/j.eswa.2021.116461는 arXiv가 연결한 논문 DOI이며 data checksum이 아니다.

## 실제 범위

`data/industrial-source-v113-v114-metadata/primary-web-screens.json`은 웹 관찰메모이며 publisher 원문 고정파일/자료 checksum이 아니다. source 사진/mask/archive/PDF/code/weight 취득·pixel/내용QA·encoder/점수·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 정식 취득·분리·학습 protocol도 없으며 실제학습 실패나 개선으로 설명하지 않는다.

원모델·GT/TEST·역할·graph·원본·실패·epoch·로그/cache/ZIP·앱 출력을 보존한다. 추적 완료 연구 optimizer 부분합60430은 그대로이고 전체 역사 총량은 아니다. 후속은 `INDUSTRIAL_SOURCE_AFTER_V114_NEXT_KO.md`를 따른다.
