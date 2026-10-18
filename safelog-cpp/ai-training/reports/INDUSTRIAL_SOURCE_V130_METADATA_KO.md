# v130 — CRACK500 원저자 공개 metadata 검토

최신 v128–v129 재개/Git/보호/host 기록과 다음 문서를 읽고 실제 저장소 상태를 확인했다. v8–v129 완료 취득·감사·학습·평가와 fixture는 재실행하지 않았다. 이번은 새 후보 한 개의 웹 텍스트 선별이며 성능 개선이나 새 학습 실패가 아니다.

## 기존 이력과 원출처

실제 존재하는 `ai-training/reports`, `ai-training/scripts`, `safelog-cpp/docs`에서 CRACK500 정확명칭과 확인된 원저자 저장소·논문 제목/arXiv 식별자를 검색했다. 정상 exit1(미일치)이었으며 경로/권한 실패를 미일치로 세지 않았다. KSDD2와 CSB는 기존 사용/검토 기록이 확인돼 새 source로 세거나 외부 재조사하지 않았다. SewerSeg 이름도 초기 기존기록 검색에 포함됐지만 외부 조사·별도 새 source 단계는 수행하지 않았다.

[Fan Yang의 원저자 저장소](https://github.com/fyangneil/pavement-crack-detection)는 [Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection](https://arxiv.org/abs/1901.06340)을 직접 연결하고 Fan Yang, Lei Zhang, Sijia Yu, Danil Prokhorov, Xue Mei, Haibin Ling의 논문 인용을 요청한다. CRACK500에는 Lei Zhang, Fan Yang, Yimin Daniel Zhang, Ying Julie Zhu의 ICIP2016 논문 인용도 안내한다. 논문 식별자는 dataset revision/checksum이 아니다.

현재 master README의 공개 웹 텍스트는 CRACK500의 `train.txt`와 `test.txt`를 구별하고 데이터·결과·모델·주석도구를 여러 저장소 링크로 안내한다. TRAIN 손상 주석이 없거나 bbox-only라고 판정하지 않았다. 실제 배포 파일별 역할·원사진/주석과 픽셀 손상 의미·RGB/자연발생 범위는 이번에 검증하지 않았다. 제삼자의 수량·재분리·변환/합성 noisy mask·license는 원배포 근거로 채택하지 않았다.

## 취득 전 보류

검토한 README 텍스트에서는 dataset 고유의 명시적 이용 허가와 native GT 값·unknown/미주석·정상/배경 지원 계약을 확보하지 못했다. README의 밝은 crack/검은 background 안내는 **평가 도구에 넣는 예측지도**에 대한 설명이다. 이를 원GT palette·배경 또는 wholephoto 정상 정의로 옮기지 않는다. 일반 HED/Caffe 명령도 원저자의 실제 annotation 생성 revision이 아니다. 이 제한 검토는 모든 원저자 자료 어디에도 허가나 유효 GT가 없다는 증명이 아니다.

원분리의 실제 목록, 촬영 session/물리 도로 구간, source/oldheld 전이 graph, file bytes/checksum·작은 익명 선택접근/임시량·실제 예산은 미검증이다. 도로 표면 과제를 일반 산업설비·전기/기계 전체 건강으로 확대하거나 새 도로 과제를 자동 추가하지 않는다. 필수 권리·GT·역할·접근/예산 gate 전이므로 archive/member/픽셀·학습은 보류한다.

원저자 repository README 웹 관찰 한 번 외에 연결 Drive/OneDrive/Baidu·archive·모델/주석도구·source code/notebook·이미지 링크를 열거나 실행하지 않았다. 연결 arXiv abstract 직접 조회 한 번은 도구의 `UnexpectedStatusCode`로 실패했다. 재시도·다른 endpoint/PDF/mirror·로그인/우회·author 연락/동의·PII 제출0이다. 실패를 논문/데이터의 영구 접근 불가로 설명하지 않는다. immutable commit/blob SHA·원문 바이트/길이 고정·전체 저장소 감사를 수행한 것은 아니다.

## 저장과 실제 수행량

`data/industrial-source-v130-metadata/primary-web-observations.json`은 URL·관찰 범위·검색/조회 실패·보류 근거를 남긴 로컬 관찰기록이다. publisher 고정 원문 SHA나 dataset checksum이 아니다. 웹 도구 traffic bytes는 미측정이다. 새 raw HTML/API/PDF/code/weights·source 원사진/mask/annotation/archive 로컬 취득0, source 픽셀해독/시각QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0이다. 새 dataset fetch/decode/training helper·unit fixture0, 새 정식 취득/pixel/split/model protocol0, readyfalse다. 관찰기록 작성과 host/Git 조회 PowerShell은 dataset job 수에 포함하지 않는다.

새 작업 전 성공한 host CIM 명령줄/시작시각 조회와 실제 v85 장부·AI root/runs 및 WORK outputs 잠금 범위를 확인했다. 관련 Python/C++·operation.lock0/PID23948 없음, v85 actual450/epoch30 그대로다. dead training.lock23948·awakehelper/전원설정을 바꾸지 않았다. 완료추적 일부 optimizer60430은 실패포함 부분합이며 전체 역사 총량이 아니다. 보호10파일·모든 GT/잠긴 역할/graph·모델/앱 출력과 원본·부분·실패·epoch/log/cache/ZIP을 보존했다.

새 명확한 원저자 data 권리·실제 GT/배경·접근 근거 없이 같은 README/arXiv 실패·다운로드/mirror를 반복하지 않는다. 다음 후보는 여전히 별도 새 공개 원저자 근거가 있을 때만 검토한다.
