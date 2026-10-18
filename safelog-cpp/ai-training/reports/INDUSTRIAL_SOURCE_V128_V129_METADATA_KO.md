# v128–v129 공개 텍스트 검토 — MDMCS 원GT 보류 유지 / CarDD 취득 전 보류

최신 v127 완료 기록·보호10파일·Git/host 상태와 다음 문서를 읽었다. v8–v127의 완료 학습·평가·취득·metadata/fixture를 재실행하지 않았다. 이번은 웹 텍스트 관찰이며 새 source 취득·픽셀·학습 단계가 아니다.

## v128 — 새 MDMCS 결과를 원배포 mask 계약으로 채택하지 않음

기존 report/script/doc의 MDMCS/HRCDS·저자/DOI 이력을 먼저 검색했다. 아직 확인하지 않은 저자 exporter/codebook을 찾는 제한 검색에서 새 문서 하나를 확인했다. [Kalyan Khatry의 저자 업로드 preprint](https://www.researchgate.net/publication/403423550_Comparative_Evaluation_of_Generative_Augmentation_Techniquesfor_Imbalanced_Concrete_Damage_Segmentation_ClassicalMethods_GANs_and_Diffusion_Models)는 별도 연구에서 JSON을 학습용 mask로 변환한 방법을 설명한다. 이 문서의 저자는 원배포 Guo/Bao와 다르며, 서술한 변환을 실제 배포 PNG의 exporter/revision·원GT class/배경 정의로 연결할 근거는 확인되지 않았다.

따라서 그 연구의 class 부여·background·raster/overlap 정책이나 학습조건을 가져오지 않았다. 공개 HTML 본문의 저자/관련 방법 범위만 읽었으며 PDF·이미지·코드/weights/download link는 열거나 로컬에 받지 않았다. 이 연구의 자체 방법에 대한 primary 설명을 원데이터 권위로 바꾸지 않는다. 검색에서 원하는 근거를 못 찾았다는 사실은 원저자 계약이 어디에도 없다는 증명이 아니다.

원 MDMCS page/V1/V2metadata/ASCE403·HEAD/CD/JSON/JPG/mask의 직접 재조회0이다. 보존된 v126/v127 구조 관찰과 모든 원형·실패를 유지했다. 원GT/unknown·미주석·정상지원·물리그룹/전graph·최소는 계속 미검증이다. mask0/검은색을 무부식/wholephoto정상으로 선언하거나 IDAT/픽셀을 먼저 해독해 계약을 추정하지 않는다. MDMCS pixel/학습은 보류 상태다.

## v129 — CarDD 원저자 페이지의 라이선스 신청 조건

CarDD 정확명칭/제목·확인된 원저자/DOI/URL을 실제 존재하는 report/script/doc에서 검색해 완료 이력 미일치를 확인했다. 최초 단순 문자열 검색의 ReinAD `cardData` key 일치는 다른 자료였으며 CarDD 완료 근거로 쓰지 않았다. 단어경계·정확 URL 및 저자/DOI 검색은 정상 exit1(미일치), 경로 오류/접근 실패가 아니었다.

[원저자 프로젝트](https://cardd-ustc.github.io/)와 [원저자 논문 abstract의 arXiv v2](https://arxiv.org/abs/2211.00945)는 Xinkuang Wang/Wenjing Li/Zhongcheng Wu, DOI10.1109/TITS.2023.3258480의 CarDD를 연결한다. 원문은 고해상도 차량손상 이미지4000·9000이상 instance·6종 손상과 detection/segmentation을 선언한다. 이는 확보·검증한 고유사진/물리부품 수량이나 native TRAIN GT·정상지원 계약의 통과가 아니다. bbox-only/mask부재로 오인해 제외하지 않는다.

프로젝트는 다운로드 전에 라이선스 신청서를 작성해 저자에게 보내 링크를 받으라고 명시한다. 이번 검토는 이 **원저자 접근 조건**에서 취득 전 보류했다. dataset link·form/PDF·로그인·신청/동의·이메일/PII 제출·접근우회0이다. 실제 data 계약/제삼자 권리·RGB/native GT/unknown·원역할·촬영/물리그룹·원파일 목록/bytes/checksum·작은 익명 접근·저장예산은 미검증이다. paper/code/mirror 조건을 data 허가로 대신하지 않았다.

CarDD는 차량 표면손상 자료의 metadata 검토이며 새 차량 과제를 자동 추가하거나 일반 산업설비·전기/기계 전체 건강으로 확대하지 않는다. 원저자의 새 명확한 공개 data release/허가 근거 없이 동일 page/form/download/mirror를 반복하지 않는다.

## 저장·보존 범위

`data/industrial-source-v128-v129-metadata/primary-web-observations.json`에 정확 URL·관찰·검색 한계·미수행 범위를 저장했다. 이것은 웹 관찰기록이지 publisher 고정 원문 SHA/자료 checksum이 아니다. 실제 웹 도구의 traffic bytes를 측정하지 않았으므로 전체 HTTP bytes0이라고 주장하지 않는다. dataset raw HTML/API/PDF/code/weights·원사진/mask/annotation/archive의 새 로컬 취득은0이다.

새 source 픽셀해독·시각QA/특징·우리 모델 점수/gradient/optimizer·CAL_TEST·API_CPU_GPU_CPP_ZIP_APK0, 새 dataset fetch/decode/training helper·unit fixture0이다. 관찰기록 작성과 host 조회용 PowerShell은 이 dataset job 수에 포함하지 않는다. 새 정식 취득/좌표/지문/분리/모델 protocol도 미정/readyfalse다. v85 실제450/epoch30과 실패포함 완료추적 일부60430은 불변이며 후자는 전체 역사 총량이 아니다. 원본·부분·실패·epoch/log/cache/ZIP·GT/역할/graph·모델/앱 출력은 모두 보존했다.

다음은 아직 미검토인 원저자의 자연발생 산업RGB TRAIN native손상 위치+명시배경 지원을 가진 새 공개 근거 한 후보의 metadata다. 기존 완료·보류·권리 미해결 source/검색/실패·공간 통지를 반복하지 않는다. 계속승인과 자동후속 ACTIVE를 유지한다.
