# v156 — FabriD 원배포 공개 text 검토, 주석·수량 선언 미해소로 취득 전 보류

2026-10-19 한국시간. 제목/DOI/원 record ID·확인 contributor·파일명 토큰을 실제 report/script/config/dataset/output 텍스트에서 오류 없이 검색해 정상 exit1을 확인했다. 일반 후보 탐색에서 다른 결과와 기존 보류 source가 노출됐지만 추가 source 단계나 재조회로 이어가지 않았다.

최초 검토한 primary page는 [FabriD:Fabric Defect Detection Dataset](https://data.mendeley.com/datasets/vncdwps24r/1), DOI `10.17632/vncdwps24r.1`, publisher Version1/2026-10-14다. contributor Tahmina Islam / Sabbir Ahmed / Ismam Ahmed / Nishat Mahzabin 및 data CC BY4.0 선언을 보존한다. 출처·저자·version·license·변경·제삼자 권리 조건을 유지하며 웹 사이트 footer나 다른 자료의 paper/code 조건으로 dataset 허가를 바꾸지 않는다. 공개일을 촬영일·실파일 revision/checksum으로 쓰지 않는다.

## 서로 다른 선언을 구별

원배포는 실제 fabric 촬영, 결함4종과 defect-free, Roboflow 수작업 주석을 설명하고 boxes/polygons/segmentation masks도 언급한다. 같은 설명에 각 이미지를 dominant fabric type의 단일 class label로 지정했다는 문장도 있다. 어느 선언이 실제 배포의 TRAIN native 손상 mask와 연결되는지, exporter·GT값/unknown·미주석·명시 배경/좌표 규약은 미확인이다. segmentation GT 부재·bbox-only·classification-only로 단정하지 않고, mask 선언만으로 pixel TRAIN을 통과시키지도 않는다.

표시된 분류별 수량 Needle Mark467 / Hole459 / Stains407 / Missing Yarn454 / Defect-free624의 정수합은 **2411**이며, 같은 page의 전체 수량 **2407**과 4 차이가 난다. 두 원선언을 그대로 보존한다. 합계를 실제 사진 수·중복 수·독립 물리부품으로 해석하거나 네 장을 임의 제외·수정하지 않는다. defect-free라는 이름도 실제 원사진·정상 wholephoto 또는 background/unknown 검증은 아니다.

검색 색인의 `Febric Defect Detection and Segm.yolov.zip`/187MB는 배포자 표시이며 원 spelling을 보존한다. 웹 page의 Files 부분만으로 실제 목록을 검증하지 않았다. 확장자·이름으로 YOLO detection/segmentation 종류·실 archive 구조/멤버·CRC/checksum을 추정하지 않는다. API·파일 링크·archive·Roboflow workspace를 열지 않았고 현재 사용자의 다른 Roboflow 탭과도 연결하지 않았다.

**취득 전 보류/readyfalse**다. 실제 자연산업 RGB 원촬영·원출처/revision·data 제삼자 권리·native GTunknown/정상 배경/coords·원TRAIN/TEST/VAL 역할·사진/촬영 물리그룹·source-oldheld 전체 graph·파일별 bytes/checksum·작은 익명 선택 접근·고정 최소/full 저장임시run 예산은 미통과다. data CC BY4.0 공개 선언 자체를 무시하거나 전 자료 권리 부재를 주장하지 않는다. source 전체 GT 부재·영구 접근 불가·조작을 증명한 검토도 아니다.

## 범위와 다음 조건

공개 publisher page를 web으로 한 번 읽었으며 local 고정 rawHTML/API/PDF/code/weights·사진/mask/annotation/archive 취득0이다. 전체 web traffic bytes는 미측정/전체HTTP0 주장은 하지 않는다. 새 직접 web 오류·재시도·alternate/mirror0, author code 실행·연락/login/동의/PII/우회0이다. sourcepixel/시각QA/raster/features/우리score/gradient/optimizer/CAL_TEST/모델API_CPU_GPU_CPP_ZIP_APK0, 새 dataset helper/unit/fixture/정식 취득/pixel/split/model protocol0이다. web 관찰 메모는 publisher 원문이나 dataset checksum이 아니다.

새 명확한 원저자 주석·수량/원역할·GTunknown 배경·촬영대응 release 없이 같은 page/search/API/파일/archive/Roboflow/변환/mirror/code/model을 반복하지 않는다. 아직 안 읽은 파일을 이용해 미해소 계약을 우회하지 않는다. fabric metadata를 일반 설비 전체건강·새 섬유 분류/판매 과제로 확대하지 않는다. 모든 이전 원형·부분·실패·epoch/log/cache·GT/역할/graph·모델/앱을 보존하고 A12GiB 이전 완료 외 추가 삭제0이다. 보고서2개는 로컬 보관이며 소진된 v152/v153 두 파일 전송 승인을 재사용하지 않는다. 새 학습 실패·성능 개선은 아니다.
