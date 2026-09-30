#include "http_ai_safety_analyzer.hpp"

#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QMimeDatabase>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <QUrl>

#include <algorithm>
#include <cmath>
#include <utility>

namespace safelog::qtapp {
namespace {

class QtJsonHttpClient final : public IJsonHttpClient {
public:
  JsonHttpResponse post(const std::string& url, const std::string& jsonBody,
                        int timeoutMs, const std::atomic_bool& cancelled) override {
    const QUrl endpoint(QString::fromStdString(url));
    if (!endpoint.isValid() || endpoint.scheme().isEmpty() || endpoint.host().isEmpty())
      throw AiProviderError(AiProviderErrorCode::InputError, "AI 서버 주소가 올바르지 않습니다.");

    QNetworkAccessManager manager;
    QNetworkRequest request(endpoint);
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json; charset=utf-8");
    request.setRawHeader("Accept", "application/json");
    request.setTransferTimeout(timeoutMs);

    QNetworkReply* reply = manager.post(request, QByteArray::fromStdString(jsonBody));
    QEventLoop loop;
    QTimer deadline;
    QTimer cancellationPoll;
    bool deadlineTriggered = false;
    deadline.setSingleShot(true);
    deadline.setInterval(std::max(1, timeoutMs));
    cancellationPoll.setInterval(50);
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    QObject::connect(&deadline, &QTimer::timeout, &loop, [&] {
      deadlineTriggered = true;
      reply->abort();
      loop.quit();
    });
    QObject::connect(&cancellationPoll, &QTimer::timeout, &loop, [&] {
      if (cancelled.load()) {
        reply->abort();
        loop.quit();
      }
    });
    deadline.start();
    cancellationPoll.start();
    loop.exec();
    deadline.stop();
    cancellationPoll.stop();

    const auto networkError = reply->error();
    const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    const QByteArray body = reply->readAll();
    reply->deleteLater();

    if (cancelled.load())
      throw AiProviderError(AiProviderErrorCode::Cancelled, "AI 분석 요청이 취소되었습니다.");
    if (deadlineTriggered || networkError == QNetworkReply::TimeoutError)
      throw AiProviderError(AiProviderErrorCode::Timeout, "AI 서버 응답 시간이 초과되었습니다.");
    if (networkError != QNetworkReply::NoError && status == 0)
      throw AiProviderError(AiProviderErrorCode::ConnectionError, "AI 서버에 연결할 수 없습니다.");
    return {status, body.toStdString()};
  }
};

QJsonObject parseObject(const std::string& json) {
  if (json.empty())
    throw AiProviderError(AiProviderErrorCode::EmptyResponse, "AI 서버가 빈 응답을 반환했습니다.");
  QJsonParseError error;
  const QJsonDocument document = QJsonDocument::fromJson(QByteArray::fromStdString(json), &error);
  if (error.error != QJsonParseError::NoError || !document.isObject())
    throw AiProviderError(AiProviderErrorCode::InvalidJson, "AI 응답이 올바른 JSON 객체가 아닙니다.");
  return document.object();
}

std::string requiredString(const QJsonObject& object, const char* key) {
  const QJsonValue value = object.value(QString::fromUtf8(key));
  if (!value.isString() || value.toString().trimmed().isEmpty())
    throw AiProviderError(AiProviderErrorCode::InvalidJson,
                          std::string("AI 응답 필드가 없거나 비어 있습니다: ") + key);
  return value.toString().trimmed().toStdString();
}

double requiredNumber(const QJsonObject& object, const char* key) {
  const QJsonValue value = object.value(QString::fromUtf8(key));
  if (!value.isDouble() || !std::isfinite(value.toDouble()))
    throw AiProviderError(AiProviderErrorCode::InvalidJson,
                          std::string("AI 응답 숫자 필드가 올바르지 않습니다: ") + key);
  return value.toDouble();
}

std::vector<std::string> requiredStrings(const QJsonObject& object, const char* key) {
  const QJsonValue value = object.value(QString::fromUtf8(key));
  if (!value.isArray())
    throw AiProviderError(AiProviderErrorCode::InvalidJson,
                          std::string("AI 응답 배열 필드가 없습니다: ") + key);
  std::vector<std::string> result;
  for (const QJsonValue& entry : value.toArray()) {
    if (!entry.isString() || entry.toString().trimmed().isEmpty())
      throw AiProviderError(AiProviderErrorCode::InvalidJson,
                            std::string("AI 응답 배열 값이 올바르지 않습니다: ") + key);
    result.push_back(entry.toString().trimmed().toStdString());
  }
  return result;
}

void requirePromptVersion(const QJsonObject& object, const std::string& expected) {
  if (requiredString(object, "promptVersion") != expected)
    throw AiProviderError(AiProviderErrorCode::InvalidJson, "AI 프롬프트 버전이 요청과 다릅니다.");
}

void validateConfidence(double confidence) {
  if (confidence < 0.0 || confidence > 1.0)
    throw AiProviderError(AiProviderErrorCode::InvalidJson, "AI 신뢰도는 0에서 1 사이여야 합니다.");
}

std::string compactJson(const QJsonObject& object) {
  return QJsonDocument(object).toJson(QJsonDocument::Compact).toStdString();
}

bool retryable(AiProviderErrorCode code) {
  return code == AiProviderErrorCode::ConnectionError || code == AiProviderErrorCode::Timeout ||
         code == AiProviderErrorCode::HttpServerError;
}

} // namespace

std::string aiProviderState(AiProviderErrorCode code) {
  switch (code) {
  case AiProviderErrorCode::Cancelled: return "Cancelled";
  case AiProviderErrorCode::ConnectionError: return "ConnectionError";
  case AiProviderErrorCode::Timeout: return "Timeout";
  case AiProviderErrorCode::HttpClientError: return "HttpClientError";
  case AiProviderErrorCode::HttpServerError: return "HttpServerError";
  case AiProviderErrorCode::EmptyResponse: return "EmptyResponse";
  case AiProviderErrorCode::InvalidJson: return "InvalidJson";
  case AiProviderErrorCode::InputError: return "InputError";
  }
  return "Failure";
}

std::shared_ptr<IJsonHttpClient> makeQtJsonHttpClient() {
  return std::make_shared<QtJsonHttpClient>();
}

HttpAiSafetyAnalyzer::HttpAiSafetyAnalyzer(HttpAiSafetyAnalyzerConfig config,
                                           std::shared_ptr<IJsonHttpClient> client)
  : config_(std::move(config)), client_(std::move(client)) {
  while (!config_.baseUrl.empty() && config_.baseUrl.back() == '/') config_.baseUrl.pop_back();
  if (config_.baseUrl.empty() || !client_)
    throw AiProviderError(AiProviderErrorCode::InputError, "AI 서버 주소와 HTTP 클라이언트가 필요합니다.");
  config_.timeoutMs = std::max(1, config_.timeoutMs);
  config_.maxRetries = std::max(0, config_.maxRetries);
}

void HttpAiSafetyAnalyzer::cancel() { cancelled_.store(true); }

std::string HttpAiSafetyAnalyzer::imageDataUrl(const std::string& path) const {
  QFile file(QString::fromStdString(path));
  if (!file.exists() || !file.open(QIODevice::ReadOnly))
    throw AiProviderError(AiProviderErrorCode::InputError, "AI 입력 사진을 읽을 수 없습니다.");
  if (file.size() <= 0 || static_cast<std::size_t>(file.size()) > config_.maxImageBytes)
    throw AiProviderError(AiProviderErrorCode::InputError, "AI 입력 사진 크기가 허용 범위를 벗어났습니다.");
  const QByteArray bytes = file.readAll();
  const QString mime = QMimeDatabase().mimeTypeForFile(QFileInfo(file), QMimeDatabase::MatchContent).name();
  if (!mime.startsWith("image/"))
    throw AiProviderError(AiProviderErrorCode::InputError, "AI 입력 파일이 지원되는 이미지가 아닙니다.");
  return ("data:" + mime + ";base64," + bytes.toBase64()).toStdString();
}

JsonHttpResponse HttpAiSafetyAnalyzer::request(const std::string& endpoint, const std::string& body) {
  cancelled_.store(false);
  for (int attempt = 0;; ++attempt) {
    try {
      JsonHttpResponse response = client_->post(config_.baseUrl + endpoint, body,
                                                 config_.timeoutMs, cancelled_);
      if (response.statusCode >= 400 && response.statusCode < 500)
        throw AiProviderError(AiProviderErrorCode::HttpClientError,
                              "AI 서버가 요청을 거부했습니다.", response.statusCode);
      if (response.statusCode >= 500)
        throw AiProviderError(AiProviderErrorCode::HttpServerError,
                              "AI 서버 내부 오류가 발생했습니다.", response.statusCode);
      if (response.statusCode < 200 || response.statusCode >= 300)
        throw AiProviderError(AiProviderErrorCode::ConnectionError,
                              "AI 서버의 HTTP 상태가 올바르지 않습니다.", response.statusCode);
      return response;
    } catch (const AiProviderError& error) {
      if (cancelled_.load() || attempt >= config_.maxRetries || !retryable(error.code)) throw;
    }
  }
}

ai::HazardSuggestion HttpAiSafetyAnalyzer::analyzeHazard(const std::string& imagePath,
                                                         const std::string& userMemo) {
  const std::lock_guard operationLock(operationMutex_);
  QJsonObject payload{{"image", QString::fromStdString(imageDataUrl(imagePath))},
                      {"userMemo", QString::fromStdString(userMemo)},
                      {"promptVersion", QString::fromStdString(config_.hazardPromptVersion)}};
  const JsonHttpResponse response = request("/v1/analyze-hazard", compactJson(payload));
  const QJsonObject object = parseObject(response.body);
  requirePromptVersion(object, config_.hazardPromptVersion);
  const double riskNumber = requiredNumber(object, "riskLevel");
  const double confidence = requiredNumber(object, "confidence");
  if (std::floor(riskNumber) != riskNumber || riskNumber < 1 || riskNumber > 5)
    throw AiProviderError(AiProviderErrorCode::InvalidJson, "AI 위험 등급은 1에서 5 사이의 정수여야 합니다.");
  validateConfidence(confidence);
  return {requiredString(object, "hazardCategory"), static_cast<int>(riskNumber),
          requiredStrings(object, "detectedHazards"), requiredString(object, "suggestedDescription"),
          requiredString(object, "suggestedAction"), confidence, response.body,
          requiredString(object, "modelName"), requiredString(object, "promptVersion")};
}

ai::ActionAssessment HttpAiSafetyAnalyzer::compareBeforeAfter(
    const std::string& beforeImagePath, const std::string& afterImagePath,
    const std::string& actionNote) {
  const std::lock_guard operationLock(operationMutex_);
  QJsonObject payload{{"beforeImage", QString::fromStdString(imageDataUrl(beforeImagePath))},
                      {"afterImage", QString::fromStdString(imageDataUrl(afterImagePath))},
                      {"actionNote", QString::fromStdString(actionNote)},
                      {"promptVersion", QString::fromStdString(config_.comparisonPromptVersion)}};
  const JsonHttpResponse response = request("/v1/compare-action", compactJson(payload));
  const QJsonObject object = parseObject(response.body);
  requirePromptVersion(object, config_.comparisonPromptVersion);
  const QJsonValue resolved = object.value("likelyResolved");
  if (!resolved.isBool())
    throw AiProviderError(AiProviderErrorCode::InvalidJson, "AI 개선 여부 필드가 올바르지 않습니다.");
  const double confidence = requiredNumber(object, "confidence");
  validateConfidence(confidence);
  return {resolved.toBool(), requiredStrings(object, "remainingRisks"),
          requiredString(object, "assessment"), confidence, response.body,
          requiredString(object, "modelName"), requiredString(object, "promptVersion")};
}

ai::ReportSummary HttpAiSafetyAnalyzer::summarize(const std::string& inspectionContext) {
  const std::lock_guard operationLock(operationMutex_);
  if (inspectionContext.empty())
    throw AiProviderError(AiProviderErrorCode::InputError, "AI 보고서 요약 문맥이 비어 있습니다.");
  QJsonObject payload{{"inspectionContext", QString::fromStdString(inspectionContext)},
                      {"promptVersion", QString::fromStdString(config_.summaryPromptVersion)}};
  const JsonHttpResponse response = request("/v1/summarize", compactJson(payload));
  const QJsonObject object = parseObject(response.body);
  requirePromptVersion(object, config_.summaryPromptVersion);
  return {requiredString(object, "summary"), requiredStrings(object, "keyRisks"),
          response.body, requiredString(object, "modelName"), requiredString(object, "promptVersion")};
}

} // namespace safelog::qtapp
