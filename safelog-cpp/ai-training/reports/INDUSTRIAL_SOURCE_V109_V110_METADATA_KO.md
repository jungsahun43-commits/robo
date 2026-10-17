# v109–v110 부식 자료 공개 근거 선별

2026-10-17 UTC. 실제 repository report/doc/source에서 정확 이름·저자·제목·URL의 기존 기록을 검색한 뒤 새 후보를 순서대로 검토했다. v8–v108 완료단계는 재실행하지 않았다.

## v109 RustSEG

[원저자 논문](https://arxiv.org/abs/2205.05426)은 분류 판단에 기여한 픽셀을 찾아 정제한 예측 mask를 만드는 방법을 설명한다. [프로젝트 설명](https://corrosionsegmenter.com/about)도 pixel TRAIN 정답 없이 학습하는 범위를 명시한다. 이 출력을 검증된 자연 segmentation GT로 승격하지 않는다. 현재 실제 TRAIN 위치주석 확보 우선순위에서 **취득 전 제외**했다. 실제 원파일·권리·원분리·촬영묶음·무결성은 미검사이며 demo 업로드/실행도 하지 않았다.

## v110 MaVeCoDD

[원저자 Mendeley V1](https://data.mendeley.com/datasets/ry392rp8cj/1), DOI10.17632/ry392rp8cj.1은 CC BY4.0이며 원저자/출처/버전/변경 표시를 유지한다. high/low 해상도 폴더가 서로 다른 현장의 각 한 선박에서 왔다고 설명하지만 실제 파일·물리그룹 감사는 아니다.

원저자 주석 설명은 실제 부식 **또는 가까운 미래에 녹이 생길 수 있는** 검사 ROI를 포함한다. 현재 부식만의 pixel 양성 정답으로 바꾸거나 일부를 추정 수정하지 않는다. 이를 확인한 [연결 논문](https://www.preprints.org/manuscript/202404.1320)과 원자료 페이지의 의미 범위에서, 실제 손상과 미래위험을 분리한 원저자 근거가 확보되기 전 **취득 전 보류**한다. 별도 검사ROI 학습 과제를 자동 추가하지 않았다. 자료 자체가 모든 과제에서 무용하다는 뜻은 아니다.

## 수행 범위

`data/industrial-source-v109-v110-metadata/primary-web-screens.json`은 웹 관찰메모이며 publisher 원문/자료 checksum이 아니다. source 사진·mask·archive·PDF/code 로컬취득, pixel/내용QA·encoder/점수·gradient/optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK 모두0이다. 실제 좌표·palette·원분리·file목록/checksum·oldheld 연결·최소수는 검사하지 않았고 정식 취득·분리·모델 protocol도 없다.

기존 모델·원GT·TEST·역할·graph·원본·실패·epoch·로그/cache/ZIP·앱 출력을 보존했다. 추적 완료 연구 optimizer 부분합60430은 그대로이고 전체 역사 총량이 아니다. 새 학습 실패나 성능 개선으로 설명하지 않는다. 후속은 `INDUSTRIAL_SOURCE_AFTER_V110_NEXT_KO.md`를 따른다.
