# MDMCS v124 — 원 URL HEAD302에서 정책 중단

코드·정책·프로토콜을 `a31437bf7869491abead4e02afe8e4bcda40e2df`로 사전 일반push/원격 확인한 뒤 한 번 실행했다. 정책 SHA256은 `e7903e961e136edf7f48d8722356111d01d648520bfcfc62ce4373fe9f76922e`다.

2026-10-17T22:21:18Z 원배포 URL의 HEAD는 **302**와 공개 S3 Location을 반환했다. status200/같은 URL 조건을 통과하지 못해 body를 읽기 전에 중단했다. HEAD 요청1/Range 요청0/실제 body0bytes다. 중앙목록·EOCD·멤버/전체ZIP은 취득하지 않았다. HTTP302 자체는 배포자 연결 응답이며 영구접근 불가나 ZIP 손상을 증명하지 않는다.

헤더 `data/industrial-mdmcs-v124-directory/publisher-HEAD-headers.json`, 원형 코드/정책, guard fixture3, `runs/industrial-mdmcs-v124-directory/failure.json`, stdout/stderr를 보존한다. fixture3은 통과했다. helper exit1은 사전 중단 규칙에 따른 metadata 접근 STOP이며 학습 runtime 실패가 아니다. operation.lock은 자신의 PID를 확인한 뒤 정리했고 dead training.lock23948은 건드리지 않았다. 같은 실행·원 URL HEAD·완료 v121–v123 HTTP를 다시 돌리지 않는다.

source 사진/mask/archive/PDF/code/weight 취득·pixel/시각QA·features/score·gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0이다. 기존 v85 장부450/epoch30과 추적완료 일부60430은 그대로이며 후자는 전체 역사 총량이 아니다. 원 TEST/사본·모든 역할·GT/graph·모델·실패·출력은 보존했다. readyfalse다.

원배포자가 이 HEAD의 Location으로 새 공개 저장소 URL을 직접 지정한 근거가 생겼다. 다음 문서의 별도 v125는 이 **새로 관찰한 정확 URL**의 조건을 사전 코드JSON으로 고정한 뒤 검토하는 제안이다. v124를 소급 통과하거나 자동 redirect/다른 경로 추측/전체 ZIP 다운로드를 허용하는 변경이 아니다. data CC BY4.0·Guo/Bao·DOI10.17632/6x4dzzrs2h.2·출처/버전/변경/제삼자 권리 조건과 모든 이전 보존 제약을 유지한다.
