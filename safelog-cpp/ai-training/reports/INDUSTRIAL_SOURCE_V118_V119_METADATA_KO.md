# v118–v119 원자료 공개 근거 선별

2026-10-17 UTC. 실제 repo report/doc/source의 RSDDs/MVTec AD 2 검색은 오류 없이 미일치했다. DACL은 별도 이름검색에서 기존 취득·학습 이력이 확인돼 외부 재조사나 새source 단계로 세지 않았다. v8–v117 완료단계는 다시 실행하지 않았다.

## v118 RSDDs: 원출처 접근·계약 미확인

제삼자 catalog에서 원출처로 안내된 [기관 URL](http://icn.bjtu.edu.cn/Visint/resources/RSDDs.aspx)은 이번 web tool에서 접근되지 않았다. 반환된 https target도 같은 요청의 도구 결과이며 별도 재시도가 아니다. 원기관 본문·원저자·자료 계약을 확보하지 못했다. URL을 발견한 사실만으로 원저자 권위/현재 배포를 검증했다고 주장하지 않는다.

**원출처 접근·계약 미완료로 취득 전 보류**한다. RGB·nativeGT/unknown·정상/배경지원·원분리·촬영묶음·권리·file/checksum·작은 익명 선택접근/예산은 미검증이다. 제삼자 논문·mirror의 수량/확장사진/분리/조건을 원자료 근거로 대신하지 않았다. 재시도·alternate endpoint·mirror·다운로드·로그인·연락은 하지 않았다. 영구 접근불가나 mask/허가 부재를 증명한 것은 아니다.

## v119 MVTec AD 2: 정상-only TRAIN

[공식 배포사](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)의 공개 색인본문은 TRAIN/VAL이 결함 없는 사진이고, 결함과 해당 pixel 주석은 TEST 범위에 있음을 설명한다. 일부 TEST GT는 비공개다. 현재 자연 손상pixel TRAIN 확보 방향에서는 **취득 전 제외**한다. 공개 TEST mask를 TRAIN으로 옮기거나 기존 정상분포 실패를 다시 돌리지 않는다. 다른 목적의 benchmark 활용 가능성을 부정하는 결론은 아니다.

배포 data조건은 CC BY-NC-SA4.0이며 비상업·원저자 인용/출처·변경·동일조건을 보존하고 code/paper 조건과 구별한다. 공식 화면의 이름/email/form·비상업 확인란은 제출하지 않았다. 현재 배포사가 인용하는 논문 DOI10.1007/s11263-026-02743-0은 data파일 revision/checksum이 아니다. 원본·주석·PDF·code·평가서버를 열거나 내려받지 않았고 실제 좌표·그룹·file무결성도 미검사다.

## 실제 범위

`data/industrial-source-v118-v119-metadata/primary-web-screens.json`은 관찰메모이며 publisher 고정 원문/자료 checksum이 아니다. source 사진/mask/archive/PDF/code/weight 취득, pixel/내용QA·encoder/score·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 새 정식 자료·모델protocol0이며 접근 실패를 새학습 실패로 세지 않는다.

원모델·GT/TEST·역할·graph·모든 원본/실패/epoch/로그/cache/ZIP·앱 출력을 보존했다. 추적 완료 optimizer 부분합60430은 그대로이며 전체 역사 총량은 아니다. 후속은 `INDUSTRIAL_SOURCE_AFTER_V119_NEXT_KO.md`를 따른다.
