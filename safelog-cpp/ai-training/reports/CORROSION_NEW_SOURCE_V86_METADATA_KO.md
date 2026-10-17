# v86: 추가 실제 위치 주석 자료의 공개 메타데이터

v85는 실제450 optimizer 후 epoch4/60update를 선택했다. 제공 ROI의 pooled FP11.2679%/FN16.9879%였으나 최악 사진 FP56.675%가 고정50% 한도를 넘어서 전체 제외했다. 새 Ostrava retention/장치/API/C++/ZIP/APK 검사는0이다. 같은 TEST의 cut·stride·계수 탐색은 종료한다.

## 우선 확인할 LIACi

[SINTEF 프로젝트](https://www.sintef.no/en/projects/2021/liaci/)와 [원저자 논문](https://doi.org/10.1109/JOE.2022.3219129)은 실제 수중 선박 검사 영상에서 뽑은1,893장과 부식 등을 포함한10종류의 pixel 주석을 설명한다. 이는 설명상의 수량이다. 현재 실제 ZIP 내부 사진·주석·분리·정상 지원·선박별 묶음은 미검사다. 수중 도장·해양 생물·부식의 중첩과 영상 프레임 상관이 있어 일반 금속이나 독립 부품으로 확대하지 않는다.

[공식 공개 앱](https://liaci.sintef.cloud/)의 정적 HTML과 실제 Dash renderer가 지정한 공개 layout JSON만 읽었다. JavaScript는 실행하지 않았고 layout의 image data URI는0개다. layout이 명시한 **데이터 조건은 CC BY-NC-SA4.0**이다. 논문이나 제삼자 코드의 라이선스로 바꾸지 않는다. 학술 비상업 연구·저자/출처/배포 상태/변경 표시·동일조건 및 제삼자 권리를 유지한다.

layout의 실제 링크 `/download_data/data.zip`에 HEAD만 요청했다. 2026-10-17 UTC 응답은200,1,044,750,953bytes, ETag `"1726140227.0-1044750953-416943642"`, Last-Modified2024-09-12이다. 배포자 SHA/MD5는 이 응답에서 제공되지 않았다. ETag를 내용 SHA로 부르지 않는다. 현재 여유 약2.14GB에서 전체 ZIP+작업공간을 저장하면1GiB reserve가 빠듯하므로 전체 다운로드하지 않는다.

다음 v87은 **공개206 Range와 ZIP 중앙 목록만** 검증하는 별도 코드/JSON을 먼저 push한다. 16MiB 총수신 상한,8MiB 중앙 목록 상한, 정확 Content-Range/ETag/바이트수, 안전경로/중복명/암호화·다중볼륨·지원외 형식 중단을 고정한다. 200 전체 응답을 받으면 본문을 읽지 않고 중단한다. Range/선택 접근이 가능하다고 아직 주장하지 않는다. 실제 원TEST/VAL 유무와 파일 목록을 확인한 뒤 별도 취득·TEST 예약·그룹·분리·최소 조건을 정한다. 현 단계는 모델 프로토콜이 아니다.

## 다른 확인 후보

- [CSB 공식 V3](https://doi.org/10.4121/6162a9b6-2a20-4600-8207-e9dcd53a264a.v3): 실제 철교 균열과 비균열 사진, 원Train/Test 및 좌표 JSON을 설명한다. 공식 DataCite/배포 페이지는 **CC BY-NC-ND4.0**이며 제삼자 요약의 CC BY로 대체하지 않는다. 실제 주석은 반자동 도구에 기초하며 비균열이 무부식/구조 전체 정상을 뜻하지 않는다. 주자료 RAR10,218,410,544bytes라 전체 취득하지 않았다. README3,106bytes와 정적 loader825bytes만 배포자 MD5를 검증해 읽었고 실행하지 않았다. 작은 선택 접근/배포 조건이 미해결이다.
- [CSM 공식 Zenodo](https://zenodo.org/records/15627907): 공식 API는 CC BY4.0이나 SEM 영상 연구이고 파일1,260,097,515bytes다. 현장 RGB/충분한 정상·원TEST 지원과 맞는지 확인되지 않아 다운로드하지 않았다.
- [논문이 연결한 배수관 저장소](https://github.com/Planet82/sewer-datasets): 공개 저장소 metadata의 license는 null이다. 논문 공개 조건을 데이터에 대신 붙이지 않고 원자료는 받지 않았다.

원메타데이터/HTTP 영수증은 `data/industrial-new-surface-v86-metadata`에 보존했다. 새 사진·주석 파일 취득/픽셀 해독/시각 열람/분리/encoder 특징/모델 점수/gradient/optimizer0이다. v85 완료 재개/10개 보호 파일 SHA 검증을 재사용하며 기존 모델·출력·위치·실패를 보존한다. 새 자료의 공식 조건·파일 무결성·실제 주석 계약과 모든 현재 held 연결 제외가 통과하기 전에는 학습하지 않는다.
