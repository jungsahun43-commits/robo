# v106 RealClutch 공개 근거 선별

2026-10-17 UTC. 기존 기록의 정확 이름/제목/저자 검색 후 새 후보 한 곳을 검토했다. v8–v105 완료 단계는 재실행하지 않았다.

[공식 CVF 논문 색인](https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Fulir_Synthetic_Data_for_Defect_Segmentation_on_Complex_Metal_Surfaces_CVPRW_2023_paper.html)과 PDF 색인텍스트의 3.1–3.2절은 RealClutch의 결함을 수작업으로 만들었고 회색조 카메라를 사용했다고 설명한다. 실제 촬영된 물체라는 사실을 자연발생 RGB 손상 정답으로 바꾸지 않는다. 짝을 이루는 SynthClutch도 자연사진이 아니다. 현재 자연 산업RGB TRAIN 위치주석 우선순위에서는 **취득 전 제외**했다.

[원저자 소속기관 목록](https://www.itwm.fraunhofer.de/en/departments/bv/staff/juraj-fulir.html)의 논문 DOI는10.1109/CVPRW59228.2023.00465다. 실제 archive의 data조건·주석·원분리·사진/물리그룹·파일목록/checksum은 검사하지 않았다. 해당 자료의 다른 활용 가능성까지 부정하거나 논문/코드 조건을 data허가로 대체하지 않는다.

직접 CVF HTML은403, PDF는timeout이었으며 재시도/대체접근은 하지 않았다. 근거는 공식 출처의 검색 색인텍스트 관찰이며 로컬 고정 publisher 원문이나 archive 검증이 아니다. 다운로드 링크/원자료/예시사진을 열지 않았다. 관찰메모는 `data/industrial-realclutch-v106-metadata/web-screen.json`이다.

새 사진·mask·archive/PDF·원문/코드 취득, pixel/내용QA·encoder·점수·gradient·optimizer·CAL/TEST·API/CPU/GPU/CPP/ZIP/APK는0이다. 정식 취득/분리/학습정책도 등록하지 않았다. 기존 부모·원GT·TEST·역할·graph·원본·실패·로그/cache/epoch/ZIP을 보존했고 완료 연구 optimizer 부분합60430은 그대로다. 전체 역사 총량이나 성능 개선으로 설명하지 않는다.

처음 두 history 검색은 존재하지 않는 docs 경로를 포함해 실패했으며0건 근거로 쓰지 않았다. 실제 repository report/doc/source 범위 검색을 다시 수행해 오류 없이 미일치를 확인했다. source해독/학습 오류는 아니다. 다음은 `INDUSTRIAL_SOURCE_AFTER_V106_NEXT_KO.md`를 따른다.
