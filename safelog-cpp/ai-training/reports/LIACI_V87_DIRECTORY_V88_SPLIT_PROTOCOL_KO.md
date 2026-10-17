# LIACi v87 공개 목록 결과와 v88 원분리 메타데이터 조건

`cb7f49d`에 v87 코드/JSON을 사전 일반 push한 뒤 원저자 공개 ZIP의206 Range만 읽었다. 총수신2,647,370bytes는 꼬리65,557와 중앙 목록2,581,813의 합이며 일부 겹침이 있다. 전체1,044,750,953byte ZIP이나1,893원본 JPEG는 받지 않았다. 정확 Content-Range/ETag/길이/Last-Modified·안전경로·중복명·단일볼륨/지원 압축 조건을 통과했다.

중앙 목록26,522멤버, 선언 확장4,490,928,334bytes다. `images`에는1,893JPEG/948,381,354원본bytes가 있다. thumbnail1,893과 각 분류 mask/segmentation/saliency 사본은 새 자연사진으로 더하지 않는다. 전체 원본·주석 개별 CRC나 전체 ZIP SHA를 검증했다고 주장하지 않는다. 중앙 목록에 있는 CRC는 아직 선언값이다.

원저자 목록에 `train_test_split.csv`50,824bytes/압축9,955/CRC8de40ea5가 있다. `coco-labels.json`15,612,447bytes와 원저자 모델의 유사도·embedding 파일도 있지만 내용은 읽지 않았다. embedding/유사도/모델 출력은 새 GT로 사용하지 않는다.

## v88: CSV 하나만 취득

`scripts/fetch_liaci_split_metadata.py`와 `industrial-liaci-v88-split-policy.json`을 별도 사전 push한 뒤 실제 목록이 지정한 CSV 하나만 읽는다. 로컬 ZIP 헤더30bytes→실제 파일명/extra→그 멤버 압축구간만 exact206으로 받아 원래 파일명·flags·압축·size·CRC/UTF8을 확인한다. 총 Range1MiB/텍스트128KiB 이하,1GiB reserve,200전체응답/정체성변경/알수없는형식/불일치에서 중단한다. 파일 경로를 쉘이나 임의 접근에 사용하지 않고 공개 링크를 추측하지 않는다.

CSV를 확보한 뒤 실제 스키마/원TEST·VAL ID를 명시 예약한다. 그 예약과 원저자 GT/촬영 묶음 의미를 확인하는 다음 코드/JSON 전에 COCO 주석 내용·사진·mask를 읽지 않는다. v88은 모델·픽셀·최종 자료 분리 프로토콜이 아니다. 표본과 독립성, CC BY-NC-SA4.0의 데이터 조건을 유지하며 기존 모든 held 지문/graph/역할을 재사용한다.

v87 새 합성unit3개 통과. 최초2/3 통과 실패는 Windows ZipInfo가 fixture의 backslash를 쓰기 전에 정상화한 사례였다. 원형/실패 기록을 outputs에 보존하고 raw중앙목록 이름 fixture로 수정했으며 실제 source Range 전이었다. v88은 별도3개 합성unit으로 stored/deflate text, CRC·이름 불일치, 한도 초과 시 요청0을 확인했다. 과거 모델/장치 검사를 새 검사 수로 더하지 않는다.

v86/v87의 새 source 사진·주석 본문·RGB 해독·시각 열람·encoder 특징·모델 점수·gradient·optimizer0이다. 기존 앱/승인 overlay/v62/실패 모델과 로그를 보존한다. 공개 자료의 작은 취득 가능성만 확인했으며 학습 성능이나 현장 독립성 증거는 아니다.
