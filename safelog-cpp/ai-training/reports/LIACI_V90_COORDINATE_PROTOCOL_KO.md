# LIACi v90: 고정 TRAIN 6장의 수치 좌표 계약 조사

## 원논문과 정답 범위

원자료는 SINTEF LIACi이며 데이터 조건은 **CC BY-NC-SA 4.0**이다. 저자·출처·버전·변경, 비상업 및 동일조건과 제삼자 권리를 유지한다. 논문의 CC BY 4.0을 데이터 조건으로 바꾸지 않는다.

- 공식 프로젝트: https://www.sintef.no/en/projects/2021/liaci/
- 공식 데이터 앱과 실제 다운로드 링크: https://liaci.sintef.cloud/ 및 https://liaci.sintef.cloud/download_data/data.zip
- 원논문: https://doi.org/10.1109/JOE.2022.3219129
- 공개 원논문 보존본: https://nva.sikt.no/registration/0198cc6adfe2-9b2a60a8-4a59-4f57-b098-67c36b2e957c

원논문 텍스트는 상업 선박검사 16회와 연구선 영상, 두 주석자의 겹치는 polygon, 전체 주석 재검토와 일부 100장 전문가 검사를 설명한다. 모든 사진의 전문가 검수나 공개 선박별 ID는 확인되지 않았다. 선체·부품과 corrosion/paint_peel/marine_growth/defect는 겹칠 수 있다. 물 배경 등 미주석 영역이 있으므로 corrosion 주석이 없다는 사실을 사진 전체 무부식·장비 정상 정답으로 바꾸지 않는다. 도장 박리·해양 생물·부식을 자동 병합하지 않는다.

저자는 ResNet101 유사도와 라벨로 332장을 걸러 원 TRAIN 1370/TEST 191을 만들었다고 설명한다. not_used 332장은 학습에 보충하지 않는다. 임베딩/숫자 ID/해상도는 선박·촬영 독립성 증명이 아니다. 이 단계는 논문 텍스트와 클래스 정의 표 한 쪽만 읽었으며 원자료 예시를 시각 열람하지 않았다.

## 픽셀 전에 고정하는 범위

원 CSV/COCO/중앙목록과 v89 TEST 191장 및 각 14사본 예약 SHA를 재사용한다. 전체 1893행의 역할은 원 TRAIN에서 corrosion 주석 존재/미존재 층마다 SHA256(`liaci90:`+파일명) 순서로 첫 ceil15% VAL, 다음 ceil15% CAL, 나머지 FIT로 사전 고정한다. 현재 배정은 FIT956/VAL207/CAL207/원TEST191/not_used332이다. 이는 최종 근접 제외 전 역할 잠금이며 readyFIT 목록이 아니다.

조사는 presetFIT 중 선체 주석이 있는 각 층의 파일명 순서 첫 3장, **COCO ID 52/107/178 및 1/3/8**로 고정한다. 제외가 생겨도 다른 사진으로 보충하거나 역할을 바꾸지 않는다. 각 사진에서 원 JPEG, corrosion/ship_hull/paint_peel/marine_growth/defect BMP, segmentation PNG만 받는다. 총 42멤버이며 원본·받은 압축 범위·헤더를 모두 보존하는 저장 상한은 **16,084,493 bytes**이다. 네트워크 상한 32MiB, 디스크 예비량 1GiB와 추가 16MiB 여유를 확인한다. 전체 ZIP과 전체 확장은 금지한다.

정확한 공개 206/Content-Range/길이/ETag/identity와 ZIP 로컬 파일명·방식·크기·CRC가 맞아야 한다. 2개 HTTP 작업자만 사용하며 불일치·중단·부분 파일은 보존하고 전체 다운로드로 대체하지 않는다.

## 수치 검사와 중단 조건

JPEG는 RGB/단일 프레임/EXIF 없음 또는 orientation1, 최대 40MP/32MiB이다. BMP는 원본 1비트와 0/255, PNG는 제공된 모드·팔레트를 그대로 기록한다. 모두 COCO의 원본 크기와 같아야 하며 resize/flip/좌표 배율 추정은 하지 않는다. COCO는 iscrowd0, 유한한 원래 float polygon 좌표가 캔버스 경계 안에 있어야 한다. 알려지지 않은 형식·좌표·범위·category 존재 불일치는 중단한다.

제공 BMP와 독립적인 Pillow polygon raster의 차이를 **진단값**으로 기록한다. Pillow가 원저자 exporter라는 근거는 없으며 이 raster로 원본 mask를 교체하거나 불일치를 자동 수정하지 않는다. layer 중첩과 선체의 미검사영역 수치는 녹 부재/건강 정답이 아니다. PNG의 색과 class 대응도 임의로 추정하지 않는다.

새 원본 RGB 6장에만 RGB SHA와 8방향 pHash를 저장한다. 구현상 JPEG decode 호출은 layer별 5회로 총 30회, BMP30/PNG6회이며 새 자연사진은 6장이다. 기존 전역 RGB/거리 감사, VAL/CAL/TEST 픽셀·시각 QA, encoder/모델점수/gradient/optimizer는 모두 0이다. 이 단계는 항상 `ready_for_training=false`이다.

## 이후 준비 단계의 고정 조건

전체 1893 원본 지문과 모든 사본의 전이적 연결, 기존 모든 held 지문/graph/역할(Ostrava15/VT44/Rios7 TEST 포함)을 재사용한 제외가 필요하다. 미선택 원본을 통한 연결은 아직 미검사다. TEST/not_used 연결 TRAIN, 개발 역할 충돌은 제외하고 하나의 기존 역할은 보존한다. 기존 전체 해독·거리 감사를 다시 돌리지 않는다. 사진/그룹 수나 유효 픽셀이 부족하면 역할 이동·재배정·최소 완화 없이 학습 전에 중단한다.

사전 최소: FIT 각 주석 조건 30사진, VAL/CAL 각각 corrosion10/그 외20사진, 원TEST 각 조건20사진, 각 분리·조건 연결그룹3개 및 각 유효 native class4096픽셀. 주석 조건은 전체사진 정상/결함 정답이 아니다. 실제 음성/unknown 계약과 전체 지문·좌표·클래스 의미가 확인되기 전 최종 manifest나 모델 프로토콜을 만들지 않는다.

합성 unit 3개를 통과했다. 최초 JPEG `n_frames` 속성 오류는 source 읽기 전 발생했고 원형을 outputs에 보존한 뒤 단일 JPEG/BMP 인터페이스를 `getattr(...,1)`로 수정했다. 데이터 원본·기존 모델·출력에 대한 변경은 없다. 별도의 학습 후보는 아직 미확정이며 코드/JSON 사전 push 없이는 특징·학습을 시작하지 않는다.
