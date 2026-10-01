# 현장 사진 피드백 수집 — 역할 3·4 연결

## 먼저 모을 사례

물기가 있는 표면, 마른 얼룩·그림자·백화, 작은 공동·파임, 정상 표면을 함께 모은다.
AI가 잘 맞힌 사진과 틀린 사진을 모두 보관한다. 오류 사진만 모은 자료의 비율을 실제 현장 오차율로 발표하지 않는다.

역할 4는 사진 등록·AI 제안·사람의 검토 이력을 연결하고, 역할 3은 원본 사진과 사람이 확인한 항목별 정답을 받아 검사한다.
기존 앱의 채택/수정/거절 이력은 정답 확인의 참고 자료다. 문장을 채택했다고 사진의 모든 항목이 정답인 것은 아니다.

## 사진마다 필요한 항목

| 필드 | 내용 |
|---|---|
| photo_id | 원본 사진의 고유 ID |
| photo_path | 데이터 폴더에 보관한 원본 사진 경로 |
| site_id | 시설/현장 식별자 |
| capture_session_id | 같은 날·같은 구역에서 촬영한 묶음 |
| before_after_pair_id | 조치 전후 사진이면 같은 묶음 ID |
| labels | 사람이 확인한 7개 항목의 존재 여부 |
| reviewed_by | 확인한 사람 |
| reviewed_at | 확인한 시각 |
| model_name, raw_json | 당시 AI 모델과 원문 결과 |
| review_note | AI 오류와 사람이 확인한 근거 |

정답 값은 true=보임, false=확인했으나 안 보임, null=확인하지 못함이다.
null은 정상이라는 뜻이 아니며 학습 손실과 항목별 평가에서 제외한다.
안전/불안전, 붕괴 위험, 배관 누수 확정, 손상 깊이를 사진 존재 라벨로 대신하지 않는다.

```json
{
  "photo_id": "photo-0001",
  "photo_path": "data/field-feedback/photos/photo-0001.jpg",
  "site_id": "site-a",
  "capture_session_id": "session-a-01",
  "before_after_pair_id": null,
  "labels": {
    "concrete_crack": null,
    "concrete_spalling": null,
    "rust_stain": null,
    "exposed_rebar": null,
    "wet_surface": null,
    "efflorescence": null,
    "surface_cavity": null
  },
  "reviewed_by": null,
  "reviewed_at": null,
  "review_note": "형식 예시이며 학습에 사용한 실제 사진이 아님"
}
```

내부 학습 키 concrete_crack은 앱에서 surface_crack으로 반환된다.
물기가 보인다는 라벨은 누수 원인을 확정한 라벨이 아니다.

## 역할 3이 확인할 순서

1. 원본 사진과 사람의 확인 기록을 대응시킨다. 모델 출력으로 정답을 자동 생성하지 않는다.
2. 같은 사진의 파일/픽셀 중복과 연속 촬영·전후 사진의 묶음을 확인한다.
3. 현장 단위로 학습/검증/외부 평가 묶음을 고정한다. 같은 현장·같은 전후 묶음을 분할 사이에 섞지 않는다.
4. 새 외부 평가 묶음은 학습과 임계값 탐색에 쓰지 않는다. 항목별 양성/음성 분모를 함께 보고한다.
5. 학습 묶음의 확인된 정답만 보강에 쓰고 후보마다 오탐·미탐 비교와 적용 여부를 기록한다.

원본 사진·내보낸 DB·개발용 가상환경은 Git에 올리지 않는다.
이 문서는 수집 형식과 연결 절차이며, 앱의 DB/API 계약을 변경한 구현은 아니다.
