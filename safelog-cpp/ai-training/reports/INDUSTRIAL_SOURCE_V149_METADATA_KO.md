# v149 — UTE 풍력 블레이드 논문 공개 text 검토, 자료 취득 전 보류

정확 제목·확인 저자·DOI·URL을 기존 repo report/script/config/dataset/source와 outputs의 실제 텍스트에서 검색해 정상 exit1 미일치를 확인한 뒤 새 후보 한 개를 검토했다. 전체 과거 commit·source 바이너리 감사나 자료 부재 증명은 아니다. 단순 탐색 토큰 BladeSeg/WTB는 확인된 datasetbrand나 별도 source 단계로 세지 않았다. 기존 Blade30·무관 검색결과·다른 논문은 재조사하지 않았다.

## 원저자 공개 근거와 판단

[A Hierarchical Selection of Instance Segmentation Models for Wind Turbine Blade Defect Detection](https://www.researchgate.net/publication/404197757_A_Hierarchical_Selection_of_Instance_Segmentation_Models_for_Wind_Turbine_Blade_Defect_Detection), DOI `10.1109/ICIT64854.2026.11491076`, ICIT2026. 원문 저자는 Juan Pablo Carballal / Juan Pedro Maestrone / Mauricio Olivera / Agustín López / Juan Pirotto / Guillermo Carbajal / Facundo Benavides / Rafael Canetti다. RG 요약 이름 Juan Maestrone와 원문 Juan Pedro 표기를 구별했다. Facundo Benavides의 author-upload HTML만 읽었고 April2026/업로드2026-04-25를 촬영일·data revision으로 쓰지 않았다.

원문은 UTE 점검의 **proprietary 자료**라고 명시한다. 5,049사진·TRAIN3,829/VAL813/TEST407·수작업 polygon/Roboflow·12주석종류는 저자 선언이며 실제 파일/사진수·독립부품·역할 검증이 아니다. 일부 주석은 손상 여부와 무관한 부품이다. 모든 objectmask를 손상으로 바꾸지 않는다. 논문의 동일 결함 재촬영 없음 설명도 실 촬영그룹/전graph 감사가 아니다. 맑은 하늘 장면은 정상 GT 계약이 아니다.

혼동행렬의 **NULL**은 누락·허위 예측을 표현하는 평가 범주다. 원 native mask의 수치 배경/unknown·미주석 권위로 채택하지 않았다. mask부재/bbox-only로도 오인하지 않는다. 실제 data 고유허가·제삼자 권리·공개 release/revision·RGB/nativeGT/unknown·정상지원/coords/export·원분리/물리촬영묶음·file/bytes/checksum·작은 익명 접근·보존/임시/run 예산은 미검증이다. **취득 전 보류/readyfalse**이며 모든 공개 접근·유효GT가 영구 부재하다는 결론은 아니다. paper 권리를 data 허가로 대체하지 않았다.

## 실행 범위

정확 DOI의 IEEE-domain 검색 미일치는 논문/DOI 부재 증명이 아니다. 직접 publisher body/DOI redirect·중복 RG 페이지·PDF/그림·프로필·연락처·reference/related·자료/code/model 링크는 열지 않았다. 새 직접 HTTP 오류0·재시도/대체endpoint/mirror0·자동연락/login/동의/PII제출/우회0이다. web traffic bytes는 미측정이며 전체 HTTP0이라고 하지 않는다.

웹 관찰 메모는 `data/industrial-source-v149-metadata/primary-web-observations.json`에 저장한다. 이는 publisher 고정 원문이나 자료 checksum이 아니다. rawHTML/API/PDF/code/weights·원사진/mask/annotation/archive 로컬 취득0, sourcepixel/시각QA/raster/features/우리score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이다. dataset fetch/decode/training helper0·새 unit0·새 정식 취득/pixel/split/model protocol0이다. 기록·host/Git PowerShell은 datasetjob이 아니다. 논문 모델·증강/loss/회차·평가 기준/점수를 SafeLog protocol·성능/전체건강으로 채택하지 않았다.

v8~v148 완료·실패중단 helper/fixture/HTTP 재실행0, v127 JPG1+mask1/v126 JSON1/v141 ref/v142v143 text2 등 기존 실제 취득과 모든 원형·부분·실패·epoch/log/cache·GT/역할/graph·모델/앱을 보존했다. A6파일12GiB 삭제는 이전 승인 완료이며 이번 추가삭제0/B_C미승인이다. 기존 로컬 보관 검토2문서는 untracked로 유지하고 이 보고서와 다음 문서만 일반 `HEAD:feature/ai-engine` push한다. 최종 host/보호10/Git/재개 영수증은 outputs에 별도로 남긴다.
