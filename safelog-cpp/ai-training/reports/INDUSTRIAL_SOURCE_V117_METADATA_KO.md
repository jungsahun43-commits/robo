# v117 ZJU-Leaper 공개 근거 선별

2026-10-17 UTC. 실제 repository report/doc/source의 정확 명칭 검색에서 기존 ZJU-Leaper 완료기록은 없었다. 같은 검색에서 발견한 CableInspect와 magnetic-tile 완료기록은 재조사하지 않았다. v8–v116 완료단계를 다시 실행하지 않았다.

## 확인한 원저자 문서

[원저자 project](https://nico-zck.github.io/zju-leaper/)는 결함 mask를 쓰는 setting2/5를 명시한다. [API 문서](https://github.com/nico-zck/ZJU-Leaper-Dataset/blob/master/dataset_api/README.md)는 정상·결함 TRAIN과 TEST를 구별한다. 따라서 정상-only/bbox-only 자료로 오인해 제외하지 않는다. 원논문 DOI는10.1109/TAI.2021.3057027, 저자는 Chenkai Zhang/Shaozhe Feng/Xulongqi Wang/Yueming Wang이다.

project의 fabric group은 무늬 묶음이며 물리 원단 roll·촬영 session·독립 생산부품을 확인한 그룹이 아니다. API의 설정·seed·분리 함수 설명도 실제 파일 역할 예약이나 원주석 검사와 같지 않다. 문서의 명령이나 원저자 코드는 실행하지 않았다.

## 취득 전 보류

이번에 읽은 두 문서에서 data 고유 허가·동의 요구·자연발생 provenance·실제 RGB 의미·native mask/unknown/좌표·파일별 원역할·촬영묶음·목록/checksum·작은 익명 선택접근/저장량을 확정하지 못했다. **필수 자료 계약 확인 전 보류**한다. mask나 허가가 어디에도 없다는 결론은 아니다. paper/code/mirror 조건으로 데이터 허가를 대신하지 않는다.

루트 README 조회는 web tool의400/timeout으로 실패했고 같은 문서 재시도·raw/mirror 대체 접근은 하지 않았다. project가 연결한 단축 다운로드 링크·원본 archive·PDF·사진/mask·baseline 링크도 열지 않았다. 현재 project/master 웹 관찰이며 immutable revision/배포파일 감사를 완료한 것이 아니다. 검색에 나온 제삼자 논문의 수량·변환 분리나 mirror 라이선스는 판단 근거로 사용하지 않았다.

## 실제 범위

`data/industrial-source-v117-metadata/primary-web-screen.json`은 관찰메모이며 원배포자 고정 원문/자료 checksum이 아니다. source 사진·mask·archive·PDF/code/weight 취득, pixel/내용QA·encoder/score·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 새 정식 취득·좌표·분리·모델 protocol0이며 학습 실패나 성능 개선으로 설명하지 않는다.

원모델·GT/TEST·역할·graph·원본/실패/epoch/로그/cache/ZIP·앱 출력을 보존했다. 추적 완료 optimizer 부분합60430은 그대로이며 전체 역사 총량은 아니다. 다음은 `INDUSTRIAL_SOURCE_AFTER_V117_NEXT_KO.md`를 따른다.

후속 이름 S2DS도 기존기록 검색에서 이미 download/준비/감사/학습 기록이 확인돼 새 후보로 세지 않았다. 외부 페이지 재조회·자료 취득·감사/학습 재실행은0이며 별도 새source 단계도 부여하지 않았다.
