# LIACi 원래 분리와 주석 메타데이터: v88–v89

## 원래 평가 예약

`5dfbb0d` 사전 push 후 CSV 하나를 exact206으로 읽었다. 원래50,824bytes/압축9,955bytes와 CRC8de40ea5를 검증했다. 로컬 CSV SHA256은 `a53e08d4a666f0b0a7b1e4c38d258b82d0511f2fe7c900e789ef17ce72477ffb`이다. 요청한 실제 Range 합10,026bytes에는 로컬헤더/파일명도 포함된다.

실제 스키마는 빈 pandas index, `file_name`, `split`이다. 원래 TRAIN1,370/TEST191/`not_used`332를 보존했다. `not_used`를 TRAIN이나 정상으로 바꾸지 않았다. v89에서 모든1,893ID와 ID당14개 원본/thumbnail/mask 사본을 연결하고 **TEST191×14사본**을 JSON 내용과 픽셀 전에 예약했다. 예약SHA는 `2b7bab9a88db75e1a465eac0f82ebbbf40d92813c73510e17488f191976e5706`이다. TRAIN1,370은 미배정 후보이며 readyFIT가 아니다.

## 원저자 JSON만 읽은 결과

`5979b21`에 v89 코드·정확 정책을 사전 push한 뒤 `coco-labels.json` 한 멤버만 취득했다. 원래15,612,447bytes/압축1,791,261bytes/CRC d9eba479/UTF8·중복 key/비유한 상수 검사를 통과했다. 로컬 JSON SHA256 `4acd1fd3d0498c091803d4d5f2617bffcef5427cab0bd4d209c32d72e917de94`는 배포자 제공 SHA가 아니다. 전체 ZIP SHA/전체 멤버 CRC를 검증한 것은 아니다.

사진 metadata1,893개/주석12,727개/원래10category다. 실제 `corrosion`은 category5이며 이름을 바꾸지 않았다. 부식 polygon이 있는 사진의 metadata 수량은 TRAIN155/TEST23/not_used33이다. **나머지 사진을 전체 무부식 정상으로 확정하지 않는다.** 겹친 marine_growth/paint_peel, 가림과 경계·주석 완전성을 확인해야 한다. 물리·전기·기계·선박 전체 건강 정답이 아니다.

원래 image metadata 크기는640×480284장/1280×720837장/1920×1080772장이다. 모든 `date_captured`가0이며 `coco_url`/`flickr_url`은 빈 문자열이다. 선박·비행·촬영 UUID/세션을 입증하는 metadata는 확인하지 못했다. 숫자 ID/같은 해상도를 같은 물리부품·촬영 날짜로 바꾸지 않고 영상 프레임 상관과 물리 독립성 미검증을 유지한다. COCO의 licenses 배열도 빈 이름/URL이며 별도의 공식 앱이 명시한 CC BY-NC-SA4.0를 보존한다.

v89 초기 freeze는 파일명 정규식에서 BMP를 빠뜨려14사본 assert에서 중단됐다. 실제 목록에10분류+saliency BMP와segmentation PNG가 있음을 확인하고 BMP 매핑만 고쳤다. 원형/실패를 outputs에 남겼고 예약/정식 정책/JSON 내용/픽셀 실행 전이었다. v88 freeze의 cwd 경로 오류도 Python 시작 전에 보존·복구했으며 학습/평가 재실행은 없다.

새 source JPEG·BMP·PNG 취득/픽셀 해독·mask raster·시각 열람·새 encoder 특징·모델 점수·gradient·optimizer는 모두0이다. source COCO 주석 텍스트 한 파일을 읽은 것과 pixel 정답 준비·학습을 구분한다. 원TEST 수량이나 표시는 사람/선박 독립성, 최소조건 통과, 새 앱 유형 채택을 뜻하지 않는다.

원문·CRC/Range영수증·예약·실패는 `data/industrial-new-surface-v86-metadata`, `industrial-liaci-v87-source`, `industrial-liaci-v88-split`, `industrial-liaci-v89-metadata`에 보존했다. 기존 앱·승인 세분류·원v62·실패v81/v83/v85의 모델·출력·위치와 모든 이용조건은 보존한다.

출처: [SINTEF 원저자 프로젝트](https://www.sintef.no/en/projects/2021/liaci/), [공개 배포 앱](https://liaci.sintef.cloud/), [원논문 DOI10.1109/JOE.2022.3219129](https://doi.org/10.1109/JOE.2022.3219129). 데이터는 비상업 학술연구/저자·출처·배포 시점·변경/동일조건과 제삼자 권리를 유지한다.
