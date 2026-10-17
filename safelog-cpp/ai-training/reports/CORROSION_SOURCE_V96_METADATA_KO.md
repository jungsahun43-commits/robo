# v96 새 후보 1개의 공개 메타데이터 사전 선별

기존 reports/scripts/docs에서 WPI/ARL 원URL·BMVC21 저장소·정확제목의 완료 연구를 찾지 못한 뒤, **한 후보**를 원저자 근거로만 검토했다. 이는 source 취득/학습 프로토콜이 아니라 공개 웹 텍스트 선별이다. 파일목록·원분리·주석좌표·실제pixel·정상지원·무결성은 아직 검사하지 않았다.

[BMVC 공식 논문 페이지](https://bmva-archive.org.uk/bmvc/2021/conference/papers/paper_1544.html), DOI **10.5244/C.35.417**은 Biao Yin 외 저자의 자료를600사진/10년간 실험실 부식시험/전문가 부식등급을 사용하는 분류 과제로 설명한다. 현재 우선인 실제TRAIN 손상위치·명시적 배경지원의 근거는 이 metadata에서 확인하지 못했다. 분류등급을 pixelmask로 바꾸거나 실험실 데이터를 독립 현장사진으로 세지 않는다.

[원저자 저장소](https://github.com/Boaz-Yin/BMVC21--Corrosion-Image-Data-Set-for-Automating-Scientific-Assessment-of-MaterialsCorrosion-Image)는 [WPI 원자료 페이지](https://arl.wpi.edu/corrosion_dataset/)를 직접 연결하고 data/citation 구역에 MIT/WPI2021을 명시한다. 실제 자료 archive의LICENSE/제삼자 권리/배포자checksum은 미검사이므로 데이터 이용조건 검증 통과로 부르지 않는다. WPI 주소의 이번 직접 web open은 JavaScript landing shell이었으며 script를 실행하지 않았다. 이름이 같은 공개 텍스트의 주장과 실제 native 주석파일 확인은 구분한다.

이 후보는 **현재 위치정답 우선순위에서는 source 취득 전에 보류**했다. pixelGT가 세계 어디에도 없다는 증명이 아니라 확인한 원저자 metadata만으로 위치학습 계약을 정할 수 없다는 뜻이다. 좋은 분류성능이 있었다고 별도모델을 자동학습하거나 mask를 생성하지 않았다. rawHTML/PDF/원사진/주석 archive를 로컬에 받지 않았고, web으로 확인한 사실·URL만 `data/industrial-corrosion-source-v96-metadata/wpi-bmvc21-web-screen.json`에 남겼다. 로컬 publisher원문SHA 또는 전체archive검증으로 꾸미지 않는다.

새 source사진/mask/픽셀·시각QA·특징·점수·gradient·optimizer·CAL_TEST·API/장치/C++/ZIP은0, ready=false다. 자동연락/로그인/PII/접근우회/외부코드실행0이며 기존 모든 GT·분리·원본·실패·모델과 이용조건을 보존한다. 같은등급자료를 위치정답으로 다시 조사하거나 기존완료source를 되풀이하지 않는다. 다음은 `INDUSTRIAL_SOURCE_AFTER_V96_NEXT_KO.md`이다.
