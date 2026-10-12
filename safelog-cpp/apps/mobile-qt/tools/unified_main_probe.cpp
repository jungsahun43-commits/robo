#include "adapters/ai/http_ai_safety_analyzer.hpp"
#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QTextStream>

int main(int argc, char** argv) {
  QCoreApplication app(argc, argv);
  const auto args = app.arguments();
  if (args.size() != 4 && args.size() != 5) return 2;
  safelog::qtapp::HttpAiSafetyAnalyzerConfig config;
  config.baseUrl = "http://127.0.0.1:8775";
  config.inspectionTask = args[1].toStdString();
  config.timeoutMs = 30000;
  config.maxRetries = 0;
  safelog::qtapp::HttpAiSafetyAnalyzer analyzer(config, safelog::qtapp::makeQtJsonHttpClient());
  try {
    const auto hazard = analyzer.analyzeHazard(args[2].toStdString(), "통합 메인 실제 C++ 연결 시험");
    const auto comparison = analyzer.compareBeforeAfter(args[2].toStdString(), args[2].toStdString(), "동일 TRAIN 사진 연결 시험");
    const auto summary = analyzer.summarize("점검자 확인·조치·재점검 기록");
    const auto raw = QJsonDocument::fromJson(QByteArray::fromStdString(hazard.rawJson)).object();
    if (raw.value("modelSha256").toString() != args[3] || raw.value("neuralModelCount").toInt() != 1 ||
        raw.value("sharedBackboneCount").toInt() != 1 || comparison.likelyResolved ||
        summary.summary.empty()) return 3;
    QJsonObject result{{"status", "passed"}, {"inspectionTask", args[1]},
                      {"modelSha256", raw.value("modelSha256")}, {"operations", 3},
                      {"samePhotoDoesNotResolve", !comparison.likelyResolved}};
    QTextStream(stdout) << QJsonDocument(result).toJson(QJsonDocument::Compact) << '\n';
    if (args.size() == 5) {
      QFile output(args[4]);
      if (!output.open(QIODevice::WriteOnly)) return 5;
      const auto bytes = QJsonDocument(result).toJson(QJsonDocument::Indented);
      if (output.write(bytes) != bytes.size()) return 6;
    }
  } catch (const std::exception& error) {
    QTextStream(stderr) << error.what() << '\n';
    return 4;
  }
  return 0;
}
