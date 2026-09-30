#include "adapters/ai/http_ai_safety_analyzer.hpp"

#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QtTest>

#include <deque>
#include <variant>

using namespace safelog::qtapp;

namespace {
class FakeHttpClient final : public IJsonHttpClient {
public:
  using Result = std::variant<JsonHttpResponse, AiProviderError>;
  std::deque<Result> results;
  int calls{0};
  std::string lastUrl;
  std::string lastBody;

  JsonHttpResponse post(const std::string& url, const std::string& body, int,
                        const std::atomic_bool&) override {
    ++calls;
    lastUrl = url;
    lastBody = body;
    if (results.empty()) return {};
    Result result = std::move(results.front());
    results.pop_front();
    if (std::holds_alternative<AiProviderError>(result))
      throw std::get<AiProviderError>(result);
    return std::get<JsonHttpResponse>(std::move(result));
  }
};

std::string json(std::initializer_list<std::pair<QString, QJsonValue>> values) {
  QJsonObject object;
  for (const auto& [key, value] : values) object.insert(key, value);
  return QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
}
}

class HttpAiSafetyAnalyzerTests final : public QObject {
  Q_OBJECT
private slots:
  void initTestCase() {
    QVERIFY(directory_.isValid());
    imagePath_ = directory_.filePath("site.png");
    QImage image(2, 2, QImage::Format_RGB32);
    image.fill(Qt::red);
    QVERIFY(image.save(imagePath_));
  }

  void parsesHazardAndBuildsImageRequest() {
    auto client = std::make_shared<FakeHttpClient>();
    client->results.push_back(JsonHttpResponse{200, json({
      {"hazardCategory", "통로 적치물"}, {"riskLevel", 4},
      {"detectedHazards", QJsonArray{"걸림", "넘어짐"}},
      {"suggestedDescription", "통로에 자재가 있습니다."},
      {"suggestedAction", "자재를 이동하세요."}, {"confidence", 0.87},
      {"modelName", "test-vlm"}, {"promptVersion", "safelog-hazard-v1"}})});
    HttpAiSafetyAnalyzer analyzer({"http://127.0.0.1:8080"}, client);

    const auto result = analyzer.analyzeHazard(imagePath_.toStdString(), "현장 메모");
    QCOMPARE(result.riskLevel, 4);
    QCOMPARE(result.modelName, std::string("test-vlm"));
    QCOMPARE(result.promptVersion, std::string("safelog-hazard-v1"));
    QCOMPARE(result.detectedHazards.size(), std::size_t(2));
    QCOMPARE(client->lastUrl, std::string("http://127.0.0.1:8080/v1/analyze-hazard"));
    const QJsonObject request = QJsonDocument::fromJson(QByteArray::fromStdString(client->lastBody)).object();
    QVERIFY(request.value("image").toString().startsWith("data:image/png;base64,"));
    QCOMPARE(request.value("promptVersion").toString(), QString("safelog-hazard-v1"));
  }

  void parsesComparisonAndSummary() {
    auto client = std::make_shared<FakeHttpClient>();
    client->results.push_back(JsonHttpResponse{200, json({
      {"likelyResolved", true}, {"remainingRisks", QJsonArray{"가장자리 확인"}},
      {"assessment", "주요 적치물이 제거되었습니다."}, {"confidence", 0.8},
      {"modelName", "test-vlm"}, {"promptVersion", "safelog-comparison-v1"}})});
    client->results.push_back(JsonHttpResponse{200, json({
      {"summary", "통로 위험을 개선했습니다."}, {"keyRisks", QJsonArray{"넘어짐"}},
      {"modelName", "test-vlm"}, {"promptVersion", "safelog-summary-v1"}})});
    HttpAiSafetyAnalyzer analyzer({"http://127.0.0.1:8080"}, client);

    const auto comparison = analyzer.compareBeforeAfter(imagePath_.toStdString(),
      imagePath_.toStdString(), "자재 이동");
    QVERIFY(comparison.likelyResolved);
    QCOMPARE(comparison.promptVersion, std::string("safelog-comparison-v1"));
    QCOMPARE(comparison.remainingRisks.front(), std::string("가장자리 확인"));
    const auto summary = analyzer.summarize("점검 문맥");
    QCOMPARE(summary.keyRisks.front(), std::string("넘어짐"));
    QCOMPARE(summary.promptVersion, std::string("safelog-summary-v1"));
    QCOMPARE(client->calls, 2);
  }

  void rejectsInvalidSchema() {
    auto client = std::make_shared<FakeHttpClient>();
    client->results.push_back(JsonHttpResponse{200, R"({"riskLevel":9})"});
    HttpAiSafetyAnalyzer analyzer({"http://127.0.0.1:8080"}, client);
    try {
      analyzer.analyzeHazard(imagePath_.toStdString(), "메모");
      QFAIL("invalid response was accepted");
    } catch (const AiProviderError& error) {
      QCOMPARE(error.code, AiProviderErrorCode::InvalidJson);
    }
  }

  void retriesTransientFailure() {
    auto client = std::make_shared<FakeHttpClient>();
    client->results.push_back(AiProviderError{AiProviderErrorCode::ConnectionError, "offline"});
    client->results.push_back(JsonHttpResponse{200, json({
      {"summary", "요약"}, {"keyRisks", QJsonArray{"위험"}}, {"modelName", "test-vlm"},
      {"promptVersion", "safelog-summary-v1"}})});
    HttpAiSafetyAnalyzer analyzer({"http://127.0.0.1:8080"}, client);
    QCOMPARE(analyzer.summarize("문맥").summary, std::string("요약"));
    QCOMPARE(client->calls, 2);
  }

  void distinguishesHttpAndInputErrors() {
    auto client = std::make_shared<FakeHttpClient>();
    client->results.push_back(JsonHttpResponse{401, "unauthorized"});
    HttpAiSafetyAnalyzer analyzer({"http://127.0.0.1:8080"}, client);
    try {
      analyzer.summarize("문맥");
      QFAIL("HTTP 401 was accepted");
    } catch (const AiProviderError& error) {
      QCOMPARE(error.code, AiProviderErrorCode::HttpClientError);
      QCOMPARE(error.httpStatus, 401);
    }
    try {
      analyzer.analyzeHazard("missing.png", "메모");
      QFAIL("missing image was accepted");
    } catch (const AiProviderError& error) {
      QCOMPARE(error.code, AiProviderErrorCode::InputError);
    }
  }

private:
  QTemporaryDir directory_;
  QString imagePath_;
};

QTEST_MAIN(HttpAiSafetyAnalyzerTests)
#include "http_ai_safety_analyzer_tests.moc"
