# 역할4 · 두 데이터셋으로 학습한 단일 PCB 모델

## 정확한 모델 구성

이번 `pcb-multisource-v3`는 **VisA와 RobustAD 두 출처의 정상 회로기판 사진을 하나의 학생 신경망에 함께 학습한 모델**이다. 고정된 ImageNet 교사와 학습한 학생을 하나의 검출 모델·체크포인트로 전달한다. PCB1~4와 `robustad_pcb`가 같은 학생 가중치를 사용한다. 유형별로 정상 사진을 이용한 경고 기준만 다르다.

벨트 균열·컨베이어 이물질·건물 균열은 이 모델에 넣지 않았다. 이들은 대상·정답·출력 기능이 달라 기존 별도 모델을 사용한다. 하나의 앱 AI 엔진이 설비 종류와 점검 항목에 맞게 모델을 선택한다.

자료의 양과 실제 성능은 [통합 모델 보고서](../ai-training/reports/PCB_MULTISOURCE_RESULTS_KO.md)에 기록한다. `pcb-multisource-role4-handoff-2026-10-12-v1.zip`은 기존 전달 파일과 별도로 보관한다. 기존 모델을 자동 교체한 것은 아니다.

## 실행

새 ZIP을 압축 해제하고 해당 폴더에서 의존성을 설치한다. Python3.12, CPU 실행 기준이다. 완전히 새 환경의 pip 설치 시험과 현재 환경에서의 실제 CPU 실행 시험은 구분한다. ZIP의 `requirements-host-tested.txt`는 실제 제작 환경 기록이며 GPU 버전도 포함한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
$modelRecord = Get-Content -Raw model.json | ConvertFrom-Json
.\.venv\Scripts\python.exe -m safelog_ai.pcb_multisource_server --checkpoint $modelRecord.path --sha256 $modelRecord.sha256 --port 8774
```

모델 파일은 `model.pt` 하나다. `GET http://127.0.0.1:8774/health`의 `ready=true`, `neuralModelCount=1`과 각 유형의 동일 `modelSha256`을 확인한다. 별도 기판별 모델5개를 로드하지 않는다.

## 앱 요청

`POST http://127.0.0.1:8774/v1/equipment/pcb/analyze`

```json
{
  "image": "JPEG 또는 PNG 사진의 base64",
  "equipmentCategory": "robustad_pcb",
  "sceneType": "registered_pcb_closeup",
  "promptVersion": "pcb-multisource-app-v1"
}
```

- `equipmentCategory`: `pcb1`, `pcb2`, `pcb3`, `pcb4`, `robustad_pcb`. 공개 자료에 있는5개 기판 유형의 촬영 조건을 선택한다. 자동으로 임의 기판 유형을 식별하거나 새 모델을 생성하는 기능은 아니다.
- 원본8MB·40MP 이내, JPEG/PNG만 받는다. EXIF 방향을 적용해256×256 RGB·ImageNet 정규화한다.
- 422: 유형·촬영 조건·사진 오류, 413: 파일 크기 초과, 503: 고정 모델 없음. 분석 실패를 정상 결과로 저장하지 않는다.
- 기본 서버는 해당 PC의127.0.0.1에서 실행한다. 스마트폰 연결에는 별도 서버 주소·인증·HTTPS 배포 구성이 필요하다. 현재 ZIP은 스마트폰 내부 C++/ONNX 모델 실행 버전이 아니다.

## 응답

| 필드 | 의미와 앱 처리 |
|---|---|
| `schema` | `safelog.equipment.pcb-visual.v3` |
| `trainedSourceDatasets` | `["VisA", "RobustAD"]` |
| `neuralModelCount` / `categorySpecificNeuralWeights` | `1` / `false` |
| `categorySpecificNormalCalibration` | true. 하나의 신경망에 유형별 정상 CAL 경고 기준 |
| `candidateDetected` | 서버의 고정 판정. 클라이언트에서 기준을 임의로 바꾸지 않음 |
| `anomalyScore` / `threshold` / `pixelThreshold` | 외관 특징 차이 점수·사진 기준·위치 평가 기준 |
| `scoreIsProbability` | false. 95% 확률처럼 표시하지 않음 |
| `modelName` / `modelSha256` | `pcb-multisource-v3` / 하나의 모델 SHA |
| `inputSha256` / `inputWidth` / `inputHeight` | 업로드 원본 SHA·EXIF 적용 후 크기 |
| `heatmapPngBase64` | 원본 크기로 확대한 회색 PNG. 특징 차이 확인용 |
| `heatmapIsConfirmedDefectLocalization` | false. 확정 결함 위치가 아님 |
| `localizationReadyForApp` / `heatmapUse` | false / `feature_difference_review_only`. 정밀 결함 위치 기능으로 사용하지 않음 |
| `mechanicalFailureState` | `not_assessed` |
| `humanConfirmationRequired` / `fieldPerformanceValidated` | true / false |

낮은 점수는 “AI 외관 이상 후보 없음”으로 표시한다. 실제 작동·절연·전류·내부 고장 상태는 사람이 별도 점검한다. heatmap은256 모델 좌표의 특징 차이이며 원본 크기로 확대해도 세부 정보가 늘어나지 않는다. 실제 위치 정확도는 결과 보고서에 별도 기록한다.

이번 RobustAD TEST0는 사진 오탐3.33%·미탐1.33%였지만 위치 IoU는0.25%에 불과하다. VisA도 IoU8.79~16.31%다. 앱에서 결함 위치를 정확히 집어주는 기능으로 소개하지 않는다. 현재는 외관 이상 후보 유무와 사람이 검토할 특징 차이 확인용으로만 제공한다.

## Qt/C++ 연결

기존 Qt Network 요청 구조를 그대로 사용하되 주소8774와 schema v3를 선택한다. `photoBytes`는 실제 JPEG/PNG 바이트, `selectedCategory`는 위5종 중 사용자가 선택한 값이다. manager/context 수명은 응답까지 유지한다.

```cpp
QJsonObject body{
    {"image", QString::fromLatin1(photoBytes.toBase64())},
    {"equipmentCategory", selectedCategory},
    {"sceneType", "registered_pcb_closeup"},
    {"promptVersion", "pcb-multisource-app-v1"}
};
QNetworkRequest request(QUrl("http://127.0.0.1:8774/v1/equipment/pcb/analyze"));
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
request.setTransferTimeout(30000);
auto *reply = manager.post(request, QJsonDocument(body).toJson(QJsonDocument::Compact));
QObject::connect(reply, &QNetworkReply::finished, context, [reply]() {
    const auto raw = reply->readAll();
    if (reply->error() != QNetworkReply::NoError ||
        reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() != 200) {
        reply->deleteLater();
        return; // 분석 실패 상태 표시, 원본 보존
    }
    QJsonParseError error;
    const auto doc = QJsonDocument::fromJson(raw, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject() ||
        doc.object().value("schema").toString() != "safelog.equipment.pcb-visual.v3") {
        reply->deleteLater();
        return;
    }
    const auto result = doc.object();
    QPixmap heatmap;
    heatmap.loadFromData(QByteArray::fromBase64(
        result.value("heatmapPngBase64").toString().toLatin1()), "PNG");
    // candidateDetected·점수·기준·SHA·heatmap을 검토 UI와 DB에 전달
    // 점검자의 확인·수정·조치 이력은 별도 입력으로 저장
    reply->deleteLater();
});
```

## 역할별 작업

- **역할1:** 기판 유형과 촬영 조건 선택, 후보·특징 차이 표시, 채택·수정·거절·판단 불가 화면.
- **역할2:** 설비 ID/기판 유형/입력 SHA, 모델 SHA/schema, 두 학습 출처, 점수/기준/후보, 점검자 판단/조치 저장. 실제 설비 작동 상태는 별도 필드.
- **역할3:** 두 출처 원본·정답·중복 확인, 실제 단일 모델 학습, 출처별 오탐·미탐·위치 평가, CPU 서버 전달.
- **역할4:** API 선택·실패 처리, 앱 흐름·보고서·재점검 연결, 선택한 모델 버전 관리.

ZIP에 원본 데이터셋은 넣지 않는다. 학습 사진5장으로 별도 압축 해제 폴더에서 실제 CPU/HTTP 호출과 하나의 모델 SHA를 확인한다. 시험 사진은 연결 시험에서 다시 추론하지 않는다.
