# v103 VAD 원저자 metadata: 취득 전 보류

v102 최소 미달 뒤 별도 후보로 [CVPR2024 원논문](https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html) *Supervised Anomaly Detection for Complex Industrial Images*와 직접 연결된 [저자 저장소](https://github.com/abc-125/segad)를 확인했다. 기존 정확제목/저자/VAD 검색에서는 이 새 후보의 완료 조사를 찾지 못했다. ReinAD 인용표를 VAD 권리·GT 근거로 대신 쓰지 않았다.

README는 실제 생산라인 자료와 TRAIN정상2000/이상1000, TEST정상1000/이상1000(미출현결함165 포함)을 설명한다. common ground_truth 폴더 설명만으로 실제 TRAIN사진의 native 손상mask 대응/라벨값·좌표/그룹을 확인한 것은 아니다. 논문 방법이 사진 단위 정답을 쓴다는 사실도 위치GT가 어디에도 없다는 증명이 아니다.

**조건 불일치**가 있다. README는 CC BY-NC-SA4.0이라고 하지만 [LICENSE-DATASET](https://github.com/abc-125/segad/blob/main/LICENSE-DATASET)의 제목·본문은 Attribution-NonCommercial4.0/CC BY-NC4.0이다. 둘을 임의로 대체하거나 데이터조건 검증 통과로 선언하지 않는다.

원저자의 [VAD.zip 배포 링크](https://drive.google.com/file/d/1sWNZCUnb2u4hcr6ak7gjpwsm6jtx6IF1/view?usp=sharing)는 이번 web 조회에 Sign in 링크만 노출했다. 익명 접근이 항상 불가능하다는 결론은 아니지만 로그인·다운로드URL 우회·다른 미러 취득은 하지 않았다. 직접 CVF PDF 조회403도 재시도하지 않았다. 파일 크기·checksum·원멤버/주석·분리·물리촬영묶음은 미확인이다.

현재 위치정답 우선순위에서 **취득 전 보류**, source사진/주석/ZIP0·픽셀/시각QA/특징/gradient/optimizer/새CAL_TEST0이다. 원저자 코드·노트북·저장모델 출력을 실행/열람하거나 자동연락/PII를 제출하지 않았다. 웹 관찰 사실·URL만 `data/industrial-vad-v103-metadata/web-screen.json`에 저장했으며 원배포 원문SHA·전체자료검증으로 설명하지 않는다. 다음 `INDUSTRIAL_SOURCE_AFTER_V103_NEXT_KO.md`의 보존조건을 유지한다.
