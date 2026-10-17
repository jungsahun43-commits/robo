# v102 단일 TRAIN HDF5 구조 확인: 예비 최소 미달

사전등록 **a012ea09962d51c0a06c29fcccbc66c4a99f9dee**/정책 SHA **cef542be13de5e5cbdc8cccad7d7e8432aa69e4ca0f79f6ff388da4aa2adbe3e**를 원격에서 확인한 뒤 한 번 실행했다. 고정 원revision **cfdff64a23f293bf0d32fc53ac1898c470da5bfb**의 `train/cable_2.h5` 하나만 취득했다. 이 선정은 v101의 선언 바이트수 최소/경로 순서이며 사진·점수·TEST 결과와 무관하다.

원파일 **8,343,076bytes**의 로컬 SHA256은 서버 LFS **dbfde7dbcf3f01e9af85a43e6057eaf4f6b468ba007ec5ba6cf5ec4c3fabbb9b**와 일치했다. h5py3.16.0 wheel3,182,868bytes도 공식 PyPI SHA **96b422019a1c8975c2d5dadcf61d4ba6f01c31f92bbde6e4649607885fe502d6**와 일치했다. 기존 venv/numpy를 변경하지 않고 별도 reader-runtime에 의존성 없이 설치했다. reader의 HDF5는2.0.0이다. 원저자 코드/loader/notebook/model은 실행하지 않았다.

## 확인한 구조

| 원 HDF5 dataset | shape | dtype |
|---|---|---|
| `/Images/Anomaly_0` | 39×400×304×3 | uint8 |
| `/Images/Normal_0` | 100×400×304×3 | uint8 |
| `/Images/Normal_1` | 22×400×304×3 | uint8 |
| `/Masks/Anomaly_0` | 39×400×304 | uint8 |

gzip/chunk metadata만 읽었다. root split/category는 variable-length object 속성이므로 고정 길이 제한에 따라 **값을 읽지 않고** shape/dtype만 기록했다. shape가 같다는 사실은 mask 값0/1·실제 native 위치 정합·RGB의 색 의미·정상 라벨·물리 독립성을 검증한 것이 아니다.

정상122/이상39는 **헤더의 선언행**이다. 고유 자연사진161장 또는 독립 부품161개를 확인한 수가 아니다. 이상39가 사전 예비 최소50보다 작으므로 이 작은 파일 방향은 **픽셀 검사/분리/학습 전에 중단**한다. 최소를 낮추거나 더 큰 파일을 골라 보충하지 않는다. 전체 ReinAD가 쓸 수 없다는 결론은 아니며, 미확인123.67GB 전체 취득으로 진행하지 않는다.

## 실제 수행량과 보존

v102 다운로드 **11,525,944bytes**는 source8,343,076 + reader wheel3,182,868이다. v101 metadata46,788bytes와 구별하며 세션 전체 웹 트래픽 총량으로 부르지 않는다. dataset 값 읽기 API를 차단했고 실제 시도0이다. RGB/mask 배열값 해독·시각QA·새 encoder/점수·gradient/optimizer·CAL/TEST·API/장치/C++/ZIP/APK는0이다. 합성unit0, 실행 오류0이다. source 파일을 받았으므로 새source취득0이라고 쓰지 않는다.

원TEST22 HDF5/모든 포함 record는 v101 byte식별 예약을 유지했고 취득/내용 열람0이다. oldheld 연결·전체source전이 graph·부품/촬영그룹·GT좌표 검사는 아직 하지 않았다. ready=false, 앱/alias/경고/위치 교체0, 추적 완료 일부 optimizer60430은 그대로다.

원본·wheel·isolated reader·헤더/영수증·stdout/stderr를 보존했다. v102 data19,776,597bytes와 run1,219bytes로64MiB 예산 이내이며 마무리 시 실제 예비1GiB를 확인한다. 기존 source/실패/epoch/캐시/ZIP은 삭제하지 않는다. data CC BY-NC-SA4.0과 h5py dependency 조건을 구별하고 원저자·출처·버전·변경을 유지한다. 결과 JSON은 `data/industrial-reinad-v102-structure/structure-complete.json`, SHA **4b901eed47720489cc0e61b6497bf77fdfb6dc726da5a06c1e502f864eed9dde**다.

다음은 `INDUSTRIAL_SOURCE_AFTER_V102_NEXT_KO.md`다. 완료 v100–v102의 검색·metadata 취득·구조 run을 재실행하지 않는다.
