# 역할4 · 등록 PCB 외관 분석 API 연결

## 전달 내용

`pcb-ai-role4-handoff-2026-10-12-v1.zip`에 실제 학습한 PCB1~4 모델4개, CPU 서버, 모델별 SHA 등록표, 성능 보고서를 제공한다. 기존 컨베이어 전달 파일과 함께 사용한다. 컨베이어 균열8771, 이물질8772, PCB8773으로 구분한다.

앱 기본 모델에 자동 적용하지 않은 연구 후보다. 종류별 오탐·미탐과 위치 성능은 [실제 결과](../ai-training/reports/PCB_REVERSE_RESULTS_KO.md)를 확인한다. 모델4개 각각40epoch를 실제 학습했으며 정상 CAL로 기준을 먼저 고정했다.

## 앱에서 연결할 흐름

1. 설비 등록에 `equipmentType=registered_pcb`, `pcbCategory=pcb1|pcb2|pcb3|pcb4`를 둔다. 현재 공개 자료의 기판 유형과 촬영 조건을 맞춘 시연용 분류이며 새 현장 기판의 등록만으로 새 모델이 학습되는 기능은 아니다.
2. 종류에 맞춘 근접 촬영 안내를 제공한다. 전체 배전반 사진이나 종류가 다른 기판은 이 API 대상으로 선택하지 않는다.
3. 사진을 보내고 `candidateDetected`와 고정 기준, 위치 표시를 받는다.
4. `candidateDetected=true`면 외관 이상 후보 검토를 요청한다. false면 “AI 외관 이상 후보 없음”으로 표시한다. 실제 작동·내부 고장 상태는 별도 점검 항목으로 남긴다.
5. 점검자가 채택·수정·거절·판단 불가를 선택한다. 담당자 조치·재점검·보고서 연결은 기존 앱 흐름을 따른다.

현재 구현은 Python CPU 서버와 Qt/C++ HTTP 클라이언트 연결이다. 스마트폰 안에 직접 실행되는 C++/ONNX 모델로 변환한 버전은 아니다. 데스크톱 시연에서는 localhost를 쓴다. 스마트폰의127.0.0.1은 스마트폰 자신이므로 향후 서버 배포 주소·인증·HTTPS 연결을 별도로 구성해야 한다. 서버 실행 기본값은 해당 PC의127.0.0.1이다.

## 서버 실행

ZIP을 새 폴더에 압축 해제하고 해당 위치에서 실행한다. 의존성 설치는 ZIP의 README_KO.md를 따른다.

```powershell
.\.venv\Scripts\python.exe -m safelog_ai.pcb_reverse_server --registry models.json --port 8773
```

`GET http://127.0.0.1:8773/health`의 `ready=true`와4개 `models`를 확인한다. 모델 경로는 models.json 폴더 안에 있어야 하며 SHA가 맞지 않으면 서버 시작이 실패한다. 서버 실행 중 다른 가중치로 몰래 바뀌지 않도록 종류별 SHA를 기록한다.

## 요청

`POST http://127.0.0.1:8773/v1/equipment/pcb/analyze`

```json
{
  "image": "JPEG 또는 PNG 사진의 base64 문자열",
  "equipmentCategory": "pcb3",
  "sceneType": "registered_pcb_closeup",
  "promptVersion": "pcb-app-v1"
}
```

사진은8MB·12MP 이하다. `data:image/jpeg;base64,...`와 PNG data URL도 받는다. 서버가 EXIF 방향을 적용하고256×256 RGB로 변환해 학습 때와 같은 ImageNet 정규화를 사용한다. UI에는 원본 크기로 복원한 회색 PNG를 반환한다. 422는 종류·촬영 조건·사진 입력 오류, 413은 크기 초과, 503은 등록 모델 없음이다. 실패 응답을 정상 점검 결과로 저장하지 않는다.

## 응답과 표시

| 필드 | 앱 처리 |
|---|---|
| `schema` | `safelog.equipment.pcb-visual.v2` 확인 |
| `equipmentCategory` / `sceneType` | 요청한 종류·근접 촬영 조건 확인 |
| `className` / `analysisType` | `pcb_appearance_anomaly` / `appearance_anomaly` |
| `candidateDetected` | 서버의 고정 기준 판정을 그대로 표시 |
| `anomalyScore` / `threshold` | 특징 차이 점수와 정상 CAL 기준 저장 |
| `pixelThreshold` | 위치 평가에 사용한 별도 기준 저장 |
| `scoreIsProbability` | false. 점수에100을 곱해 “확률”로 표시하지 않음 |
| `heatmapPngBase64` | 원본 크기 회색 PNG. 사용자 화면에서 별도 색상·투명도로 표시 가능 |
| `heatmapScale` | 특징 차이/종류별 사진 기준,0~255 clipping. 사진마다 최대값 재정규화하지 않음 |
| `heatmapIsConfirmedDefectLocalization` | false. 사람이 확정한 결함 위치로 저장하지 않음 |
| `modelName` / `modelSha256` / `promptVersion` | 모델·클라이언트 버전 추적 |
| `inputSha256` / `inputWidth` / `inputHeight` | 업로드 원본의 SHA 및 EXIF 적용 후 크기 |
| `mechanicalFailureState` | `not_assessed`. 실제 작동·고장 판단과 분리 |
| `humanConfirmationRequired` / `fieldPerformanceValidated` | true / false |

heatmap은 신경망 특징 차이를 보여준다. 원본 크기로 확대해도256 모델의 정보량이 늘어나지 않으며 작은 결함이 사라질 수 있다. `pixelThreshold`와 사진 최대값 판정 기준은 서로 다르다. 역할1은 이를 확인용 시각화로 사용한다.

## Qt/C++ 최소 호출

Qt Network 모듈을 링크한다. manager와 호출 객체의 수명은 응답까지 유지한다. 아래는 앱에 붙일 요청 예시이며 기존 앱에 자동 삽입된 코드는 아니다.

```cpp
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPixmap>

QJsonObject payload{
    {"image", QString::fromLatin1(photoBytes.toBase64())},
    {"equipmentCategory", "pcb3"},
    {"sceneType", "registered_pcb_closeup"},
    {"promptVersion", "pcb-app-v1"}
};
QNetworkRequest request(QUrl("http://127.0.0.1:8773/v1/equipment/pcb/analyze"));
request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
request.setTransferTimeout(30000);
auto *reply = manager.post(request, QJsonDocument(payload).toJson(QJsonDocument::Compact));
QObject::connect(reply, &QNetworkReply::finished, context, [reply]() {
    const auto raw = reply->readAll();
    const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if (reply->error() != QNetworkReply::NoError || status != 200) {
        // UI에 분석 실패를 표시하고 사진을 보존한다. 정상 결과로 저장하지 않는다.
        reply->deleteLater();
        return;
    }
    QJsonParseError parseError;
    const auto document = QJsonDocument::fromJson(raw, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        reply->deleteLater();
        return;
    }
    const auto result = document.object();
    if (result.value("schema").toString() != "safelog.equipment.pcb-visual.v2") {
        reply->deleteLater();
        return;
    }
    const bool candidate = result.value("candidateDetected").toBool();
    QPixmap heatmap;
    heatmap.loadFromData(QByteArray::fromBase64(
        result.value("heatmapPngBase64").toString().toLatin1()), "PNG");
    // candidate·heatmap·점수·모델 SHA를 검토 UI/DB에 전달한다.
    // 사람 확인과 실제 설비 작동 상태는 별도 입력으로 받는다.
    reply->deleteLater();
});
```

## 역할별 후속 작업

- **역할1:** 기판 종류·근접 촬영 안내, 후보 유무·heatmap 표시, 점검자 검토 화면.
- **역할2:** 설비 ID/유형/기판 종류, 원본 사진 SHA, AI schema/모델 SHA/점수/기준, 후보 여부, 사람 확인 상태·수정 내용·시각 저장. 사진 정상 판정과 실제 가동 상태를 별도 필드로 저장.
- **역할3:** 같은 시험802장 비교 보고서·실제 모델4개·서버 제공. 이후 새 기판·새 현장의 독립 자료로 검증.
- **역할4:** 앱 요청·응답 연결, 재시도/오류 표시, 담당자 조치·재점검·보고서 연결, 대상 선택과 시연 패키징.

ZIP 압축을 푼 별도 폴더에서 학습 사진4장으로 실제 CPU HTTP 호출을 검증한다. 공식 TEST를 이 연결 시험에 다시 추론하지 않는다. 재현 결과는 저장소 `ai-training/reports/pcb-reverse-standalone-v1.json`에 기록한다. ZIP 안에 원본 데이터셋을 넣지 않는다.
