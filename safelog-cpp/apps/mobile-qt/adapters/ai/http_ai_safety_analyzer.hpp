#pragma once

#include "safelog/ai/ai_analyzer.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace safelog::qtapp {

enum class AiProviderErrorCode {
  Cancelled,
  ConnectionError,
  Timeout,
  HttpClientError,
  HttpServerError,
  EmptyResponse,
  InvalidJson,
  InputError
};

class AiProviderError final : public std::runtime_error {
public:
  AiProviderError(AiProviderErrorCode code, std::string message, int httpStatus = 0)
    : std::runtime_error(std::move(message)), code(code), httpStatus(httpStatus) {}

  AiProviderErrorCode code;
  int httpStatus;
};

std::string aiProviderState(AiProviderErrorCode code);

struct JsonHttpResponse {
  int statusCode{0};
  std::string body;
};

class IJsonHttpClient {
public:
  virtual ~IJsonHttpClient() = default;
  virtual JsonHttpResponse post(const std::string& url, const std::string& jsonBody,
                                int timeoutMs, const std::atomic_bool& cancelled) = 0;
};

std::shared_ptr<IJsonHttpClient> makeQtJsonHttpClient();

struct HttpAiSafetyAnalyzerConfig {
  std::string baseUrl;
  int timeoutMs{25000};
  int maxRetries{1};
  std::size_t maxImageBytes{8U * 1024U * 1024U};
  std::string hazardPromptVersion{"safelog-hazard-v1"};
  std::string comparisonPromptVersion{"safelog-comparison-v1"};
  std::string summaryPromptVersion{"safelog-summary-v1"};
};

class HttpAiSafetyAnalyzer final : public ai::IAiSafetyAnalyzer {
public:
  HttpAiSafetyAnalyzer(HttpAiSafetyAnalyzerConfig config,
                       std::shared_ptr<IJsonHttpClient> client);

  ai::HazardSuggestion analyzeHazard(const std::string& imagePath,
                                     const std::string& userMemo) override;
  ai::ActionAssessment compareBeforeAfter(const std::string& beforeImagePath,
                                          const std::string& afterImagePath,
                                          const std::string& actionNote) override;
  ai::ReportSummary summarize(const std::string& inspectionContext) override;
  void cancel() override;

private:
  JsonHttpResponse request(const std::string& endpoint, const std::string& body);
  std::string imageDataUrl(const std::string& path) const;

  HttpAiSafetyAnalyzerConfig config_;
  std::shared_ptr<IJsonHttpClient> client_;
  std::atomic_bool cancelled_{false};
  std::mutex operationMutex_;
};

} // namespace safelog::qtapp
