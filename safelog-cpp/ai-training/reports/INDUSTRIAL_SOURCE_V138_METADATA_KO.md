# v138 — CorroNet 신호 연구를 현재 RGB pixel TRAIN 우선순위에서 제외

CorroNet은 이전에 이름만 검색했고 source 검토는 없던 항목이다. 이번 실제 repo의 정확제목/확인 저자·DOI/URL 검색은 정상 exit1 미일치였다. 모든 과거 commit/ignored 원문을 감사한 것은 아니다. SteelSeg 이름 검색은 관련 원저자 자료를 식별하지 못했으며 검색 결과 부재를 dataset 부재 증명이나 새 source 단계로 세지 않았다. 무관한 회사/개인 정보를 이력이나 자료로 저장·사용하지 않았다.

원저자 Rui Wang, Guangbin Shan, Feng Qiu, Linqi Zhu, Kang Wang, Xianglong Meng, Ruiqin Li, Kai Song, Xu Chen의 [Corrosion State Monitoring Based on Multi-Granularity Synergistic Learning of Acoustic Emission and Electrochemical Noise Signals](https://www.mdpi.com/2227-9717/12/12/2935), Processes2024/12(12)/2935, DOI10.3390/pr12122935의 공식 publisher 검색 색인 text를 확인했다. CorroNet은 모델/연구명이며 별도 공개 RGB dataset brand/release를 확인한 것이 아니다.

원저자 관찰 계약은 **음향방출(AE)·전기화학 잡음(EN) 시계열로 부식 단계를 분류**하는 것이다. 학습에는 paired AE-EN과 추가 EN을 쓰고 평가에는 EN을 쓴다. 실험은 304 stainless steel/NaCl 환경의 실험실 부식을 사용한다. 이를 자연발생 산업 RGB 원TRAIN의 native 손상 pixel/배경 GT로 전환하지 않고 **현재 우선순위에서 취득 전 제외**했다. C1/C2/C3 단계 label·신호 window/파생 segment를 사진/독립 부품·native pixel GT/wholephoto 정상으로 바꾸거나 별도 신호 과제를 자동 추가하지 않는다. 논문의 모델 정확도·encoder/분기/loss/회차를 SafeLog 성능이나 새 protocol로 채택하지 않는다.

Data Availability Statement는 **코드를 요청해 제공**받는 안내다. 이를 dataset 공개 허가나 실제 source 파일 확보로 확대하지 않는다. 실제 data 고유 권리/제삼자 권리·배포 revision/목록/checksum·원분리/물리 실험그룹·작은 익명 접근/임시량/예산은 미검증이다. 원문에 RGB 사진/유효 GT가 어디에도 없다는 주장도 하지 않는다. 출판/HTML update/논문 revision을 촬영일·dataset revision으로 쓰지 않는다.

추가 직접 publisher body 요청/HTTP실패·PDF/그림/부록·reprint·원신호/사진/GT/archive/API/rawHTML/code/weights 로컬 취득·실행0, 자동 연락·로그인/동의/PII·우회0이다. 검색에 노출된 제삼자 catalog/관련 연구를 원자료 배포로 연결하지 않았다. 새 명확한 원저자 자연 RGB native TRAIN release 근거 없이 같은 CorroNet 논문/publisher/검색/PDF/code/저장소를 반복하지 않는다.

`data/industrial-source-v138-metadata/primary-web-observations.json`은 웹 관찰 메모이며 publisher 고정 원문/자료 checksum이 아니다. 웹 traffic bytes 미측정/전체HTTP0 주장0이다. source pixels/시각QA/raster/features/우리 모델 score/gradient/optimizer/CAL_TEST/API_CPU_GPU_CPP_ZIP_APK0, dataset fetch/decode/training helper0/새 unit0/새 정식 취득·pixel·split·model protocol0/readyfalse다. 기록·host/Git PowerShell은 dataset job 수에 넣지 않는다. 새 학습 실패/성능 개선이 아니며 모든 원형/부분/실패/epoch/log/cache/보고서/ZIP·GT/역할/graph/모델/앱·기존 v127 JPG1+mask1/v126 JSON1 취득을 보존한다.
