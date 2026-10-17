# LIACi v94–v95 원저자 관련 코드·문서 조사 완료

## 고정 출처와 실행량

v94 조건은 `7c26eb9e6ae29fc1c6fd325def34a1e8eff23b5b`, v95 조건은 `391d80701b97d3f07dfe8dfb9b7f8ed5e88c4aab`에 각각 일반 push/원격SHA 확인 후 실행했다. 정책 SHA256은 각각 `03e746c23a45d2ef9a4a506b2efd2d6e88d468f53c2c9de4544a16357bdd130d`와 `c2f67a4ffd75c31ca197e71655215b68e9c448a0d288ebe1a6ed7b0d4f2bb97c`이다.

SINTEF/LIACI_context의 공개 commit **7bd7e3ec82b3a094ce67b6141dabbbf20d099f24**, tree113항목을 고정했다. v94는 LICENSE/루트README/LIACi_segmenter.py 3파일과 repo/commit/tree/blob응답66,845bytes, v95는 이미 확인된5개 README의 blob응답36,236bytes를 읽었다. 합계 **103,081bytes**이며 각 blob의 Git SHA1/크기·로컬SHA256을 검사했다. SHA1은 Git객체 동일성이지 저자제공 SHA256이 아니다. 원본 응답·정적 텍스트·로그·정책을 모두 보존했다. 네트워크/코드 실행 오류0이다. v94에서 받은 repo/tree/세 파일은 v95에서 다시 취득하지 않았다.

## 정적 검토 결론

[루트README](https://github.com/SINTEF/LIACI_context/blob/7bd7e3ec82b3a094ce67b6141dabbbf20d099f24/README.md)는 원래 검사 영상에 inference·telemetry·지식그래프를 결합하는 프로젝트라고 설명한다. [분할 코드](https://github.com/SINTEF/LIACI_context/blob/7bd7e3ec82b3a094ce67b6141dabbbf20d099f24/pipeline/computer_vision/LIACi_segmenter.py)는 외부ONNX와labels.txt를 읽고 예측 threshold/resize/색상화를 수행한다. 이를 데이터 생산용 polygon raster나 저자 수정 GT로 사용하지 않았다.

[computer_vision README](https://github.com/SINTEF/LIACI_context/blob/7bd7e3ec82b3a094ce67b6141dabbbf20d099f24/pipeline/computer_vision/README.md)는 pretrained 모델을 별도로 제공해야 한다고 설명한다. `get_coco`는 예측된 mosaic mask를 저장하기 위한 경로다. [pipeline README](https://github.com/SINTEF/LIACI_context/blob/7bd7e3ec82b3a094ce67b6141dabbbf20d099f24/pipeline/README.md)의 pycocotools2.0.4도 Neo4j에 저장하는 분할 표현을 위한 의존성이다. 이 버전을 현재 배포되는 COCO/BMP의 실제 주석 exporter/생산 버전으로 주장할 근거는 없다.

[video_input README](https://github.com/SINTEF/LIACI_context/blob/7bd7e3ec82b3a094ce67b6141dabbbf20d099f24/pipeline/video_input/README.md)는 사용자 입력 inspection metadata와 telemetry 처리를 설명한다. 문서의 예시 ID·이름·날짜는 현재1893개 데이터셋 원사진의 대응표가 아니다. 숫자ID/해상도/embedding이나 예시를 이용해 선박·촬영그룹을 추정하지 않았다. data/README와 demonstrator/README도 읽었지만 GT 생성·revision 대응은 확인되지 않았다.

**확인한 8개 정적 파일에서 ID107 COCO/BMP의 주석 존재 차이를 설명하거나 원GT 권위를 결정할 근거는 확보하지 못했다.** 전체 저장소의 모든 코드·과거 커밋·비공개 생산 과정을 조사했다는 뜻은 아니다. 관련 코드를 더 많이 읽으면 해결된다는 가정으로 동일 감사를 반복하지 않는다. 문서의 예측 mask/COCO 경로를 새 자연GT로 승격하지 않는다.

## 그대로 유지하는 경계

새 source RGB/mask 취득·pixel decode·시각QA·encoder 특징·모델점수·gradient·optimizer·CAL_TEST·API/장치/C++/ZIP/APK는 전부0이다. 원저자 코드/ONNX/CLI/문서 명령은 실행하지 않았고 notebook·저장된 모델outputs·문서 내 예시 이미지/동영상은 열지 않았다. 외부 연락/로그인/PII제출0이다. 코드 MIT와 LIACi 데이터 CC BY-NC-SA4.0 및 논문 CC BY4.0를 구별한다.

v90 직접BMP↔COCO 계약 실패와 v91 JPEG한정 실패는 그대로 보존한다. v92/v93 숫자·헤더 완료 결과를 다시 계산하지 않았다. 기존6source 원본/42파일/주석/원TEST191·각14사본/not_used332/고정개발역할/모든oldheld 제외를 유지한다. 제공 BMP 수정·색반전·좌표 배율·일부좋은사진 채택·COCO를 근거 없이 우선하는 조치는0이다. 전체1893source 전이적 연결·최소표본·GT 권위는 미확정이며 **ready_for_training=false**다.

전체 자료준비/후속run 예산도 여전히 부족하다. 실행 중 여유는 약2.04GB이며 v93 고정 계획은 준비2,135,089,959bytes/256MiB run 포함2,403,525,415bytes가 필요하다. mask 추가 비용 전이므로 최소 약0.4GB/실제1GB 이상의 추가 여유 조건은 유지한다. 모든 원본/실패/epoch/로그/cache/ZIP을 삭제하거나 reserve1GiB를 침범하지 않는다.

기본v16·승인3head·원v62·실패모델/ZIP은 변경하지 않았다. 이번 완료 일부연구 optimizer는 **60,430 그대로**, 전체역사총량이 아니다. 다음은 `LIACI_AFTER_V95_NEXT_KO.md`에 따른 변경 근거/공간 확인 또는 다른 원저자 자료의 작은 메타데이터 검토다. 완료 단계만을 이유로 자동후속을 일시중지하지 않는다.
