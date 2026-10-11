# 새 컨베이어 이물질 원본 확보 · 2026-10-12

[작성자 공개 Mendeley V1](https://data.mendeley.com/datasets/s25x2bnshz/1), DOI10.17632/s25x2bnshz.1, 페이지 라이선스 CC BY4.0. [작성자·라이선스 기록](CONVEYOR_DATA_ATTRIBUTION_KO.md)을 함께 유지한다.

## 실제 확보

공개 원본 ZIP **6,228,464,374바이트(6.23GB)**를 내려받고 작성자가 공개한 전체 SHA256 **f4ced924f274bcd729fd77ab0aca8fa72a48b5419077e2862b87ab175fd12480**과 일치함을 확인했다. 부분 전송으로 받은 바이트를 보존하고 남은 구간을 최대4개씩 받았다. 압축 전체69,636개 파일은 여러 복사본·동영상·사진·보조 파일을 포함하므로 독립 사진69,636장이라고 세지 않는다.

이번 점검은 작성자가 선별한 `split-ds-normal-filtered`만 대상으로 했다. 정상 TRAIN **12,951장**, 정상 publisherTEST **2,991장**, 이물질 이상 **2,991장**, 총 **18,933장**의 사진 파일을 확인했다. 이 자료의 정상은 **광석만 흐르는 장면**이며 벨트 균열 없음·모터/롤러 정상·안전 가동의 정답은 아니다. 기존 벨트 균열 모델의 정상 사진이나 오탐 분모로 사용하지 않았다.

## 학습 전에 해결할 항목

1. 이름에서 추정한 촬영명19개가 작성자 정상 TRAIN과 TEST에서 **모두 겹친다**. 같은 촬영 구간의 이웃 프레임이 섞인 채 성능을 높게 측정하지 않도록, 촬영명/원본 동영상의 관계를 확인하고 구간 단위로 분리해야 한다. 실제 물리적 촬영·기계의 독립성까지 확인된 것은 아니다.
2. `yolo-dataset-center/data.yaml`의 유일한 클래스는 `conveyor`다. 이 주석을 이물질·벨트 손상의 bbox 정답으로 사용하지 않는다. YAML의 `license: Private` 표시도 기록하고 해당 YOLO 하위자료는 사용하지 않았다. 공개 페이지의 전체 CC BY4.0과 하위 export 표기가 동일하다고 가정하지 않는다.
3. 영상 이름에서 추정한 그룹, 프레임 번호, 정상/이상 의미를 고정한 뒤 decoded RGB·근접 프레임 중복과 흐림·조명을 조사한다. 전체 원본 동영상과 처리 사진의 완전 동등성은 검증하지 않았다.
4. 새 클래스는 `conveyor_foreign_object`로 분리한다. sourceTEST를 학습·보정에 넣지 않는다. 이번에 본 샘플은 작성자 TRAIN 한 장이며 모델 점수는 계산하지 않았다.

**이 새 데이터의 모델 학습은 아직0epoch다.** 현재 실제 완료된 학습은 별도 [벨트 원본 확대40epoch](BELTCRACK_EXPANSION_RESULTS_KO.md)다. 새 정상/이물질 사진을 확보했다고 기존 벨트 균열 오탐률이 검증된 것은 아니다.

[체크섬·파일 수·촬영명 분리 점검](conveyor-foreign-acquisition.json). 원본은 `data/equipment-conveyor-foreign/image-data.zip`, 사진별 작업용 목록은 같은 폴더의 `curated-label-inventory.json`에 있다.
