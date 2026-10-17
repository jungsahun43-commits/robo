# v107–v108 공개 원자료 선별

2026-10-17 UTC. 실제 repository의 report/doc/source 범위에서 정확 명칭과 제목을 검색해 기존 완료기록 미일치를 확인한 뒤 두 후보를 순서대로 검토했다. v8–v106 완료 단계는 재실행하지 않았다.

## v107 MIAD

[원저자 프로젝트](https://miad-2022.github.io/)는 자료를 3D graphics로 생성했다고 명시한다. 야외 산업 시나리오와 pixel GT가 있어도 자연 촬영 TRAIN 정답으로 바꾸지 않는다. 현재 자연발생 산업RGB 위치주석 우선순위에서 **취득 전 제외**했다. 원저자 data 조건은 CC BY-NC-SA4.0·비상업·인용이며 논문/코드 조건과 구분한다. 다운로드 링크를 열거나 실제 archive·주석·원분리·file/checksum을 검사하지 않았다.

## v108 AITEX/AFID

[원저자 기관 페이지](https://www.aitex.es/afid/)는 정상140/결함105의245이미지와 white-positive 결함 mask를 설명하며 연구 시 논문 인용을 요청한다. 이 수량은 파일·독립 부품 검증이 아니다. 확인한 text에서는 dataset 고유의 명시적 CC0 허락을 확보하지 못했다. 제삼자의 CC0/변환본247개 설명을 원저자의 권리나 실제 원본 수량으로 대체하지 않는다.

원저자 [논문 DOI](https://doi.org/10.2478/aut-2019-0035), archive/원사진·mask 링크를 열거나 받지 않았다. 자연발생 범위·실제 RGB·좌표·원분리·촬영묶음·파일 무결성도 미검사다. 명확한 원저자 조건과 실제 source 계약 확인 전 **취득 전 보류**한다. 권리나 유효 GT가 어디에도 없다는 결론은 아니다.

## 실제 범위

`data/industrial-source-v107-v108-metadata/primary-web-screens.json`은 웹 관찰메모이며 publisher 원문 고정파일/자료 checksum이 아니다. source 사진/mask/archive/PDF/code 취득·pixel/내용QA·encoder/점수·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 정식 취득·분리·학습 protocol도 등록하지 않았다. 추적 완료 연구 optimizer 부분합60430은 그대로이며 전체 역사 총량이 아니다.

기존 모델·원GT/TEST·역할·graph·원본·실패·epoch·로그/cache/ZIP·앱 출력을 보존했다. 새 학습 실패나 성능 개선으로 설명하지 않는다. 후속은 `INDUSTRIAL_SOURCE_AFTER_V108_NEXT_KO.md`를 따른다.
