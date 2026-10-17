# v92 헤더 결과 및 v93 남은 로컬 수치 조사

v92는 `cf2c2c099207a7dafd7efba99c2ca91f54ad4bd2`에 사전 등록 후 남은21파일의 헤더만 읽었다. source pixel decode/새 네트워크/시각 열람/모델 작업은0이다. ID1과3의 `.jpg` 파일은 실제 PNG/RGB/1920×1080/단일 프레임/orientation없음이며 ID8은 JPEG/RGB/640×480이다. 모두 COCO 크기와 같고 다른18 mask헤더도 native크기와 같다. v91의 JPEG 한정 조건 실패 원인만 확인했으며 이를 GT 불일치가 해결됐다는 뜻으로 사용하지 않는다.

v93은 `complete_liaci_local_probe.py`/`industrial-liaci-v93-local-policy.json`을 먼저 일반 push한 뒤, 같은 ID1/3/8만 실제 헤더 형식 그대로 픽셀을 읽는다. 정책 SHA256 `6df04e2a36142c2b2f92424ee1b8464cdda6a9767af1264c1b48d876c0e66c2e`. PNG1/3과JPEG8을 변환 파일로 다시 저장하지 않고 원본 RGB값을 읽는다. native크기/단일프레임/EXIFNone1/40MP/32MiB 제한을 유지한다. BMP1비트의 흰색 영역과 독립 Pillow COCO raster를 진단하되 정답의 polarity/클래스·좌표를 추정 변경하지 않는다. segmentationPNG는 색별 수량만 기록하고 classmap을 만들지 않는다.

새 해독은 RGB3/BMP15/PNG3회이고 v91의 이미 완료된 세 수치 결과를 SHA로 재사용한다. 기존 전체 지문/거리감사, 다운로드, VAL/CAL/TEST 내용·모델점수·gradient·optimizer·시각 QA는0이다. 저장상한8MiB/예비1GiB를 유지한다. v90 및 v91의 실패/완료 원형은 모두 보존하며 다시 실행하지 않는다. 준비 통과/최종 GT/학습 프로토콜은 아직 없다.

원 중앙목록과 현재 여유만으로 향후 전체 지문 연결의 저장량을 계산한다. 아직 받지 않은 원본1887개의 압축 바이트+각128byte헤더 여유, graph16MiB/임시32MiB/활동64MiB/예비1GiB와 별도256MiB 학습run 예산을 표시한다. 추가 mask나 새 GT 계약 비용은 미포함임을 명시한다. 이 계산으로 전체다운로드/새source/학습을 시작하지 않는다. 실제 예산/정답 계약/전이적 held 제외가 충족되고 별도 코드JSON을 사전 push해야 한다.

원TEST191/14사본, not_used332 제외, v90 고정 개발 역할과 모든 이전 GT/모델/실패를 유지한다. LIACi 데이터 CC BY-NC-SA4.0 및 논문 별도조건/부모 조건을 보존한다.
