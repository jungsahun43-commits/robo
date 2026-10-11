# 역할3 → 역할4: 컨베이어 외관 분석 API 연결

## 담당 범위

역할3은 실제 모델·추론 서버·결과 형식·성능 기록을 전달한다. 역할4는 Qt/C++ 앱의 설비 선택·사진 전송·위치 표시·점검자 확인·조치·보고서를 연결한다. 역할2는 설비 ID·사진·모델 SHA·AI 결과·사람 확인·조치 이력을 DB에 저장한다.

이 문서는 **연구 후보를 명시적으로 실행하는 로컬 API**의 연결 안내다. 기존 앱 서버나 기본 모델을 자동 교체하지 않는다. 컨베이어 표면의 균열·찢어짐 후보만 분석하며, 사진으로 모터·롤러의 내부 고장·정상 가동을 확정하지 않는다.

## 1. 역할3 PC에서 서버 실행

`safelog-cpp/ai-training` 폴더에서 실행한다. 데이터와 모델은 Git에 없으므로 같은 PC를 사용하거나 `final.pt`를 별도로 전달받아야 한다. SHA는 전달하는 모델의 검증된 결과 파일과 비교한다.

```powershell
$beltResult = Get-Content reports/beltcrack-expansion-training.json -Raw -Encoding utf8 | ConvertFrom-Json
.venv/Scripts/python.exe -m safelog_ai.equipment_server --checkpoint runs/beltcrack-cached-expansion-v2/final.pt --sha256 $beltResult.weights_sha256 --port 8771
```

- 주소: `http://127.0.0.1:8771`
- 상태: `GET /health`
- API 설명: `http://127.0.0.1:8771/docs`
- 사진 분석: `POST /v1/equipment/belt/analyze`
- CPU 추론. 모델 해시·구조·단일 `belt_crack`·640입력·0.5임계값과 strict state 재로드를 확인한다.

같은 PC의 Qt 앱에서 먼저 연결한다. Android에서 같은 개발 PC 서버를 사용하려면 USB 디버깅 연결 후 `adb reverse tcp:8771 tcp:8771`을 설정하고 앱에서 `127.0.0.1:8771`을 사용한다. 스마트폰 자체에서 Python 모델이 실행되는 구성은 아니다. 별도 원격 서버 배포는 후속 통합 작업이다.

## 2. 요청

```json
{
  "equipmentCategory": "conveyor_belt",
  "image": "<JPEG/PNG 파일 바이트의 base64 또는 data:image/jpeg;base64,...>",
  "promptVersion": "belt-appearance-v1"
}
```

원본 파일 바이트 기준 최대8MB, 최대12MP다. 원본 EXIF 방향을 반영한다. 다른 설비 종류나 읽을 수 없는 사진은422, 이미지 크기 초과는413, 모델 미설정은503이다. `promptVersion`은 호출/화면 흐름을 추적하는 클라이언트 버전 표식이며 신경망 학습용 텍스트 프롬프트가 아니다.

## 3. 응답과 UI 처리

| 필드 | 역할4의 처리 |
|---|---|
| `schema` | `safelog.equipment.belt-visual.v1`인지 확인 |
| `equipmentCategory`, `className` | `conveyor_belt`/`belt_crack` 전용 결과로 표시. 콘크리트 균열과 섞지 않음 |
| `candidateDetected` | 손상 후보 있음/이번 사진에서 후보 미검출. 미검출을 정상 가동 확정으로 표시하지 않음 |
| `maximumPixelScore`, `pixelThreshold` | 640 모델 좌표의 패딩 제외 최대 점수와 고정0.5 기준. 점수를 고장 확률·안전 점수로 사용하지 않음 |
| `heatmapPngBase64` | PNG로 decode해 원래 사진에 투명 색상 지도나 위치 표시로 겹침 |
| `inputWidth`, `inputHeight` | EXIF 방향 적용 후 크기. 같은 종횡비와 `AspectFit` 영역으로 사진/heatmap을 정렬 |
| `heatmapScale` | 원래 sigmoid 점수0..255. 사진마다 대비를 늘린 지도가 아님. 확대 지도의 원본 픽셀 정확도가 검증됐다고 해석하지 않음 |
| `normalPhotoFprValidated` | 현재 `false`. 정상 사진 오탐률 미검증을 모델 성능 정보에 기록 |
| `mechanicalFailureState` | 항상 `not_assessed`. 실제 고장 확인은 점검자가 별도 입력 |
| `humanConfirmationRequired` | 항상 `true`. 채택·수정·거절과 최종 점검을 기록 |
| `modelName`, `modelSha256`, `inputSha256`, `promptVersion` | DB·보고서에 사진/모델/처리 버전으로 함께 저장 |

사진 전체에서 한 픽셀이라도0.5를 넘으면 `candidateDetected=true`다. 사진 양성 여부와 손상 위치의 정확도는 별도다. 현재 정상 사진 분모0이며5% 미만의 전체 오류가 검증된 모델은 아니다. 구체적인 사진·픽셀 성능은 [확대 학습 결과](../ai-training/reports/BELTCRACK_EXPANSION_RESULTS_KO.md)를 사용한다.

## 4. Qt/C++에서 요청하는 최소 형태

사진을 선택한 뒤 실행하는 예시다. 네트워크 오류와 JSON 형식 오류를 UI에 반환하고, 성공 시 위 필드를 화면 상태와 DB 모델에 연결한다.

```cpp
QFile photo(photoPath);
if (!photo.open(QIODevice::ReadOnly)) {
    // 사진 읽기 오류를 화면에 반환
    return;
}
const QByteArray bytes = photo.readAll();
QJsonObject body{
    {"equipmentCategory", "conveyor_belt"},
    {"image", QString::fromLatin1(bytes.toBase64())},
    {"promptVersion", "belt-appearance-v1"}
};
QNetworkRequest request(QUrl("http://127.0.0.1:8771/v1/equipment/belt/analyze"));
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
request.setTransferTimeout(60000);
QNetworkReply* reply = networkManager.post(request, QJsonDocument(body).toJson());
connect(reply, &QNetworkReply::finished, this, [this, reply]() {
    if (reply->error() != QNetworkReply::NoError) {
        // 오류 상태를 보여주고 기존 점검 입력 흐름 유지
        reply->deleteLater();
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(reply->readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        reply->deleteLater();
        return;
    }
    const QJsonObject result = document.object();
    const QByteArray heatmapPng = QByteArray::fromBase64(
        result.value("heatmapPngBase64").toString().toLatin1());
    // result를 검토 화면/DB에 넘기고, heatmapPng를 원본 사진과 정렬해 표시
    reply->deleteLater();
});
```

필요한 Qt 모듈은 Core·Network다. 실제 앱의 클래스/사진 경로/상태 저장 함수는 역할4의 틀에 맞춰 넣는다. 이 예시는 앱 전체 빌드나 스마트폰 설치 검증을 대신하지 않는다.

## 5. 정상 사진 데이터가 준비되면

[정상 데이터 조사와 다음 검증 기준](../ai-training/reports/BELT_NORMAL_DATA_AUDIT_KO.md)을 따른다. 공식 COCO ZIP을 다운로드한 뒤 다음 명령으로 주석 혼합·증강 부모·완전 동일 RGB의 분리 누락을 조사한다.

```powershell
.venv/Scripts/python.exe scripts/audit_belt_normal_coco.py --archive <belt-defects-coco.zip> --output data/belt-normal-audit-v1/audit.json
```

이 도구는 원본 ZIP을 읽기만 하고 모델 학습이나 정상 정답 생성을 수행하지 않는다. 작성자가 표시한 정상 사각형은 후보 정상 ROI로 기록하고 사진 전체의 정상 정답은 `unknown`으로 유지한다.
