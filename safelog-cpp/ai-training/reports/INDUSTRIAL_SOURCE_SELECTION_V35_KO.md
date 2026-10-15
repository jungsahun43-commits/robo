# v35 원저자 자료 선정 — 실제 연결 부품 다양성

## 우선 후보: G-Links 원본 한 종류

[원저자 Zenodo 배포본](https://zenodo.org/records/15693611), DOI10.5281/zenodo.15693611, version1.0의 metadata를 원문으로 보존했다. API의 license id는 `cc-by-nc-sa-4.0`, access는 open이다. 비상업 연구·저자/출처/버전/변경 표시·동일조건을 유지한다. 연결된 소프트웨어 저장소의 GPL은 별개의 코드 조건이며 코드나 환경을 실행하지 않는다.

실제 단조 강철 연결 부품을 로봇으로 이동·회전 촬영한 자료여서 기어와 다른 부품의 표면 상태 연구에 관련성이 있다. 전체 하중 안전성·내부 고장·파손 시점의 정답으로 확대하지 않는다. 원본 `g-link_anomaly.zip` 306,973,826bytes와 배포 MD5 `7faf736f2da8e3fb98b12f73bbf35889`를 고정했다. 자동 배경제거/중앙배치 변형을 추가 사진으로 더하거나 함께 다운로드하지 않는다.

현재 디스크 약3.9GB에서 원본307MB만 우선 검토한다. ZIP은 MD5/전체 CRC/안전 경로를 확인하고 실제 멤버 수·주석 대응·부품 ID·촬영 묶음을 감사한다. 공식 페이지의 표제 수와 본문 개수는 합계가 일치하지 않아 그 숫자를 학습 사진 수로 그대로 쓰지 않는다.

**원저자 TEST는 그대로 TEST로 예약**하고 실제 TEST mask나 그 중복을 학습 정답으로 가져오지 않는다. 정상 publisher TRAIN을 FIT/VAL/CAL에 쓸 수 있는지는 부품 그룹과 정확RGB/회전·반전 근접 감사를 마친 뒤 결정한다. 현재 보존 TEST와의 중복도 확인한다. 원래 TEST가 같은 부품의 다른 각도일 가능성을 사진 독립성으로 설명하지 않는다. 새 pixel 해독·최종 분리·모델 구조·cache·optimizer는 다운로드 및 파일명 감사와 별도 단계다.

## 다른 후보의 현재 상태

- [MPDD2 원저자 저장소](https://github.com/stepanje/MPDD2): README와 CC BY-NC-SA4.0 LICENSE를 원문 보존했다. 원저자 링크의 공개 Drive 폴더만 확인했으며 사진 파일은 받지 않았다. 기존 MPDD와 겹치지 않는다고 아직 주장하지 않는다. DOI10.1109/ICUMT57764.2022.9943437.
- [GC10-DET 원저자 저장소](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets): 직접 이용 조건을 확정하지 못해 다운로드/학습에 사용하지 않았다. 제3자 미러의 CC 표기를 원저자 조건으로 바꾸지 않는다.
- [Bearing defect dataset 배포자 페이지](https://figshare.com/articles/dataset/Bearing_defect_dataset/25335076): 페이지의4.69GB는 현재 여유보다 커 다운로드하지 않았다. 픽셀·분리·표본 수·주석은 미검사다. 진동/음향 자료를 사진 학습 자료로 바꾸어 세지 않는다.

후보 확인·파일 다운로드·라벨 검사·실제 학습·앱 반영은 각각 구분해 기록한다. 현재 채택 모델과 역할4 ZIP은 v16-r1 그대로이며 이 자료 선정만으로 오탐/미탐 개선을 주장하지 않는다.
