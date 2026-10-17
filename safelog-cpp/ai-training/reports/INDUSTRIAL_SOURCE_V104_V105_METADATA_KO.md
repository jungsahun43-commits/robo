# v104–v105 원자료 공개 근거 선별

2026-10-17 UTC. 기존 reports/scripts/docs에서 정확 제목·URL·DOI를 검색한 뒤 두 후보를 순서대로 검토했다. 완료한 v8–v103 취득·감사·학습은 재실행하지 않았다.

## v104 Blade30

[원저자 저장소](https://github.com/cong-yang/Blade30)는 `mask/`를 블레이드 영역, JSON을 손상·오염 주석으로 구분한다. 물체 영역 mask를 손상 위치 GT로 바꾸지 않는다. 확인한 공개 저장소 화면에서 명시적 데이터 이용조건을 확보하지 못했고 실제 손상 좌표·음성지원·원분리·파일별 무결성도 미검사다. 다운로드 링크를 열거나 파일을 받지 않고 취득 전 보류했다. 모든 배포물에 조건이나 손상 mask가 없다는 결론은 아니다.

## v105 ISP-AD

[원논문](https://link.springer.com/article/10.1007/s10845-025-02778-z)은 실제 결함 TRAIN의 pixel GT를 제공하지 않는다고 명시한다. 합성 TRAIN의 pixel GT와 실제 TEST mask를 구분하며, 정상 라벨도 완전 무결함 보장은 아니다. 현재 자연발생 산업RGB TRAIN 위치주석 우선순위에서는 취득하지 않는다. TEST나 합성을 자연 TRAIN 정답으로 승격하지 않았다. 논문의 data 조건은 CC BY-NC-SA4.0이며 article CC BY4.0와 다르다. 연결된 [자료 DOI](https://doi.org/10.5281/zenodo.14911042)의 실제 archive·파일목록·checksum은 검사하지 않았다.

## 실제 작업 범위와 보존

웹 관찰메모는 `data/industrial-source-v104-v105-metadata/primary-web-screens.json`이다. 배포자 원문을 로컬 고정한 파일이나 배포자 checksum으로 설명하지 않는다. source 사진·mask·archive 취득, pixel 해독·내용 QA, encoder·모델점수·gradient·optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK는 모두 0이다. 정식 취득·좌표·분리·학습 프로토콜도 새로 등록하지 않았다. 추적 완료 연구 optimizer 부분합60430은 그대로이며 전체 역사 총량이 아니다.

기존 부모·승인 연구 overlay·실패 가중치·모든 원본/epoch/로그/cache/ZIP, TEST와 역할·원GT·출력을 보존했다. 새 학습 성능이나 앱 개선 완료를 주장하지 않는다. 후속은 `INDUSTRIAL_SOURCE_AFTER_V105_NEXT_KO.md`를 따른다.
