# v137 — RustNet 원논문 자료 취득 전 보류

RustNet은 v136까지 이름 이력 검색만 있었고 외부 source 검토가 없던 항목이다. 이번 정확제목/확인 저자·DOI/URL의 실제 repo 텍스트 검색은 정상 exit1 미일치였다. 이름의 기존 메모를 완료된 자료 연구로 세거나, 모든 과거 commit/ignored raw source를 감사했다고 주장하지 않는다.

원저자 Touqeer Abbas, Muhammad Mubeen, Abdul Razzaq, Umair Javed, Muneeb Raza, Mohammad Tabish, Khalid M. Alotaibi, Ghulam Yasin, Hui Li의 [RustNet: an automated approach for rust detection and quantification in steel structures using feature encoding and multilayer attention networks](https://link.springer.com/article/10.1007/s43452-025-01312-5), Archives of Civil and Mechanical Engineering25/article264, DOI10.1007/s43452-025-01312-5/2025-09-02의 공식 공개 preview text를 읽었다. 논문/방법명 확인이며 RustNet이라는 별도 공개 dataset brand/release를 확인한 것은 아니다.

공식 abstract는 서로 다른 조명의 이미지로 녹 검출/정량화를 검증했다고 설명한다. 이는 자연발생 산업 RGB 원TRAIN/native 손상 GT·명시 배경/unknown·사진 수/파일별 역할·촬영 물리그룹의 실제 검증이 아니다. 경계/연결영역 분석 출력이나 encoder/decoder/attention 설계·모델 점수를 실제 원GT/새 SafeLog protocol로 가져오지 않는다. mask 부재/bbox-only라고 단정하지 않는다.

공식 Data availability는 자료를 교신저자에게 합리적으로 요청해 제공받도록 안내한다. **취득 전 보류**한다. 공개 dataset 고유 허가/제삼자 권리·원출처/revision·자연발생·RGB/nativeGT/unknown·정상지원/coords/export·원분리/촬영그룹·file/bytes/checksum·작은 익명 접근/임시량/보존 예산은 미검증이다. 논문 rights/permissions를 data 허가로 대체하지 않는다. 요청형 안내는 모든 유효 GT/권리/공개 접근의 영구 부재 증명이 아니다. 출판일을 촬영일·dataset revision으로 쓰지 않는다.

검색 색인의 ResearchGate 관련논문/인용 섹션에 노출된 별도 semi-supervised/현미경 patch/mean-teacher 설명을 RustNet 자체의 data 계약으로 연결하지 않았다. 동명의 네트워크 모니터와 RustNet-3D 밀 녹병 LiDAR도 이 자료가 아니다. 해당 문서/그림/논문/원자료는 열지 않았고 별도 source 단계로 세지 않는다. 공식 페이지의 그림/PDF·구독/구매/기관로그인·연락처/related/reference 링크는 추적하지 않았다. 직접 web 도구 오류0, 재시도/대체 PDF/mirror/우회0, 자동 author 연락·로그인/동의·PII 제출0이다. 새 명확한 원저자 public data release 근거 없이 같은 RustNet 논문/페이지/검색·저장소를 반복하지 않는다.

관찰기록 `data/industrial-source-v137-metadata/primary-web-observations.json`은 웹 관찰 메모이며 publisher 고정 원문/자료 checksum이 아니다. 웹 traffic bytes는 미측정이며 전체 HTTP0이라고 설명하지 않는다. 사진/mask/annotation/archive/datasetPDF/rawHTML/API/code/weights 로컬 취득0, source pixel/시각QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, dataset fetch/decode/training helper0/새 unit0/새 정식 취득·pixel·split·model protocol0/readyfalse다. 기록 작성·host/Git PowerShell은 dataset job 수에 넣지 않는다. 새 학습 실패/성능 개선이 아니며 모든 기존 원형/부분/실패/epoch/log/cache/보고서/ZIP·GT/역할/graph/모델/앱·v127 JPG1+mask1/v126 JSON1 취득을 보존한다.
