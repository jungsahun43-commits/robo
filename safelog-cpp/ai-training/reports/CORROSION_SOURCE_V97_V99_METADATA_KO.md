# v97–v99 새 후보 공개 메타데이터 선별

기존 reports/scripts/safelog-cpp docs의 정확 제목·DOI·URL 검색에서 완료 기록을 찾지 못한 후보 세 개를 공식 웹 텍스트로만 선별했다. 새 자료 취득·주석 검증·분리·학습 프로토콜은 아니다. 원문 응답을 로컬에 고정하지 않았으므로 관찰 메모의 SHA를 배포자 원문이나 자료 checksum으로 설명하지 않는다.

## v97 강관 RGB 부식 자료: 요청형 배포

[원저자 논문](https://www.mdpi.com/2624-5558/7/1/21/xml), DOI **10.3390/cmd7010021**, *Detection of Uniform Corrosion in Steel Pipes Using a Mobile Artificial Vision System*의 공식 검색 색인 텍스트는 640×640 부식 사진600장과 비부식 사진600장, 수작업 위치 주석을 설명한다. 통제된 시험 환경이며, 증강1195개를 새 자연사진으로 세지 않는다. 파일별 분리·실제 좌표·물리 강관/촬영 묶음은 미검사다.

같은 공식 텍스트의 자료 제공 방식은 **교신저자에게 요청**이다. 직접 XML 조회는 HTTP429로 제한되어 재시도하지 않았고, 공개 archive·파일목록·data 이용조건·무결성은 검증하지 못했다. 논문의 CC BY를 데이터 이용허가로 바꾸지 않는다. 현재 취득 전 보류이며 저자 연락·로그인·개인정보 제출은 하지 않았다.

## v98 CorroSeg: 초음파 두께 지도

[SLB의 공식 Challenge Data 페이지](https://challengedata.ens.fr/challenges/144)는20개 유정의 초음파 관 두께 지도와 대응 이진 주석,36×36 NPY patch/CSV 형식을 설명한다. 깊이/방위각 측정 지도이며 일반 RGB 사진이 아니다. 일부 no-groove 자료도 장비 전체 정상이나 RGB 무부식 정답으로 바꾸지 않는다. 파일은 로그인과 대회 등록이 필요하다. 현재 RGB 자료 취득 우선순위에서 제외했고, 등록·다운로드·license 동의는 하지 않았다.

## v99 IVD-SEG: 표준화 모음의 원출처 미확정

[Zenodo v1](https://zenodo.org/records/10547589), DOI **10.5281/zenodo.10547589**, 및 [공식 연결 저장소](https://github.com/KLIVIS/IVD-SEG)는43제품/5686이미지/12하위자료를256×256으로 표준화했고 그중 두 자료를 저자팀이 새로 만들었다고 설명한다. 기존 Kolektor/MVTec/DAGM 사본을 새 독립자료로 세거나 기존 TEST·이용조건을 피하는 경로로 쓰지 않는다.

v1의 최신판 링크는 HTTP429였고 재시도하지 않았다. 공식 검색 색인에서 보인 record12704122의 Good.zip 약1.5GB도 취득하지 않았다. 저장소 Apache2.0은 compiled data 전체 이용조건 검증이 아니다. 실제 두 원자료의 출처·권리·native 좌표·명시적 정상지원·원분리/촬영그룹은 미확인이라 준비 통과가 아니다. 전체 모음 취득은 하지 않는다.

## 이번 실제 작업 범위

관찰 메모는 `data/industrial-source-v97-v99-metadata/new-primary-screens.json`에 저장했다. 새 source 사진/주석 취득·픽셀/시각QA·encoder/점수·gradient/optimizer·CAL/TEST·API/장치/C++/ZIP/APK는 모두0이다. 추가 source 연구가 있었다는 사실을 실제 학습으로 세지 않는다. 추적 완료 일부 optimizer60430은 그대로이며 전체 역사 총량이 아니다.

LIACi의 주석 권위/전역 연결/저장량 조건은 계속 미충족이고, 기존 모델·GT·분리·경고·위치는 보존한다. 같은 완료 후보/실패/공간 알림을 반복하지 않는다. 다음은 `INDUSTRIAL_SOURCE_AFTER_V99_NEXT_KO.md`이며 자동후속은 ACTIVE를 유지한다.
