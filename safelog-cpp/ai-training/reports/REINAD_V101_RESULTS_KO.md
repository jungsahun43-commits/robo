# v101 공개 metadata 고정과 원TEST 파일 예약

사전등록 **c024de91b1f74a8e63a94740ec5c4f6e570b72ef**를 원격 feature/ai-engine에서 확인한 뒤 한 번 실행했다. 정책 SHA는 **62698c91bd54d6ab7f10890d2d87a1404a9ac29c8ea8f811708c085a451b2f16**이다. 원저자가 직접 연결한 `Tocmac/ReinAD`의 고정 revision **cfdff64a23f293bf0d32fc53ac1898c470da5bfb**에서 info/tree의 경로·크기·gitblob/LFS SHA와 README 바이트를 교차 확인했다.

- 실제 HTTP 수신 **46,788bytes**, source HDF5 취득0, 오류0이다.
- TRAIN34/TEST22 HDF5, 선언 합계 **123,668,352,496bytes**다. 실제 사진 수를 새로 센 것이 아니다.
- 모든 TEST 파일의 경로/길이/서버 LFS SHA/원revision을 먼저 예약했다. 예약 JSON SHA는 **040d237de87f5fb1b1c61c3dc4bd77b81103afe0b9e9b89e8ad4a8816fdb8cc3**이다. 파일에 들어 있는 모든 원TEST를 묶지만 사진별 ID는 아직 모른다.
- data README1,380bytes는 gitblob SHA1/길이와 일치했고 로컬 SHA256은 **076bfc2ce75e1a2a4d803f355df103a512fcde1708e8a74332097396da054a63**다. CC BY-NC-SA4.0을 유지한다.

메타데이터가 제시한 가장 작은 TRAIN은 `train/cable_2.h5`, **8,343,076bytes**, 서버 LFS SHA256 **dbfde7dbcf3f01e9af85a43e6057eaf4f6b468ba007ec5ba6cf5ec4c3fabbb9b**다. 이 checksum은 아직 실제 source 바이트와 비교하지 않았다. 크기/경로만의 선정이며 사진/TEST성능으로 선택하지 않았다. 파일 하나로 원source의 모든 전이 연결·물리부품 독립성을 검증했다고 주장하지 않는다.

## v102의 별도 구조 확인

새 코드/정책을 먼저 push한 뒤 이 TRAIN 파일 하나의 바이트 SHA와 HDF5 **구조만** 확인한다. 모든 TEST 파일은 받지 않는다. dataset 값 읽기 API를 막고 shape/dtype/chunk와 제한된 속성만 읽는다. RGB/mask 해독·표시·라벨값/좌표 검증·분리·특징·학습·점수0이며 ready=false다. 최소70 normal/50 anomaly 선언행과 정합한 NHWCuint8/NHWuint8 구조는 후속 픽셀정책 검토의 예비조건일 뿐, 실제 label/group/validpixel 조건의 통과가 아니다. 미달하면 더 큰 파일로 보충하지 않는다.

기존 환경에 HDF5 reader가 없어 공식 PyPI의 h5py3.16.0 WindowsCPython3.12 wheel3,182,868bytes와 SHA를 확인했다. 별도 reader-runtime에만 의존성 없이 설치하여 기존 venv/numpy를 바꾸지 않는다. 이 표준 reader와 원저자 데이터 코드는 구분한다. 구조 작업 전체 저장64MiB·예비1GiB를 고정하며 원본·실패·wheel·로그를 보존한다. 정식 source 픽셀/학습 프로토콜은 아직 없다.
