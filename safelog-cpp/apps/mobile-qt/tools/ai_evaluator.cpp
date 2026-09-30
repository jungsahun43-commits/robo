#include "adapters/ai/http_ai_safety_analyzer.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QTextStream>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace safelog::qtapp;

namespace {
QStringList parseCsvLine(const QString& line) {
  QStringList values;
  QString value;
  bool quoted = false;
  for (int i = 0; i < line.size(); ++i) {
    const QChar ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') { value += '"'; ++i; }
      else quoted = !quoted;
    } else if (ch == ',' && !quoted) { values << value; value.clear(); }
    else value += ch;
  }
  values << value;
  return values;
}

QString csv(QString value) {
  if (value.contains('"')) value.replace("\"", "\"\"");
  if (value.contains(',') || value.contains('"') || value.contains('\n')) return '"' + value + '"';
  return value;
}
}

int main(int argc, char* argv[]) {
  QCoreApplication app(argc, argv);
  const QStringList args = app.arguments();
  if (args.size() != 3) {
    QTextStream(stderr) << "사용법: safelog_ai_eval <cases.csv> <results.csv>\n";
    return 2;
  }
  const QString baseUrl = QProcessEnvironment::systemEnvironment().value("SAFELOG_AI_BASE_URL").trimmed();
  if (baseUrl.isEmpty()) {
    QTextStream(stderr) << "SAFELOG_AI_BASE_URL을 설정하세요.\n";
    return 2;
  }
  QFile input(args[1]);
  QFile output(args[2]);
  if (!input.open(QIODevice::ReadOnly | QIODevice::Text) ||
      !output.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
    QTextStream(stderr) << "평가 CSV를 열 수 없습니다.\n";
    return 2;
  }

  HttpAiSafetyAnalyzer analyzer({baseUrl.toStdString()}, makeQtJsonHttpClient());
  QTextStream in(&input), out(&output);
  const QString header = in.readLine();
  const QStringList columns = parseCsvLine(header);
  const int caseIndex = columns.indexOf("case_id");
  const int imageIndex = columns.indexOf("image_path");
  const int categoryIndex = columns.indexOf("expected_category");
  const int levelIndex = columns.indexOf("expected_risk_level");
  if (caseIndex < 0 || imageIndex < 0 || categoryIndex < 0 || levelIndex < 0) {
    QTextStream(stderr) << "필수 CSV 열이 없습니다.\n";
    return 2;
  }
  out << "case_id,image_path,expected_category,expected_risk_level,predicted_category,"
         "predicted_risk_level,confidence,latency_ms,error\n";

  int attempted = 0, succeeded = 0, categoryMatches = 0, jsonFailures = 0;
  double levelError = 0.0, latencyTotal = 0.0;
  const QDir inputDir = QFileInfo(input).absoluteDir();
  while (!in.atEnd()) {
    const QString line = in.readLine();
    if (line.trimmed().isEmpty()) continue;
    const QStringList values = parseCsvLine(line);
    if (values.size() <= std::max({caseIndex, imageIndex, categoryIndex, levelIndex})) continue;
    bool levelOk = false;
    const int expectedLevel = values[levelIndex].toInt(&levelOk);
    const QString path = inputDir.absoluteFilePath(values[imageIndex]);
    ++attempted;
    QString predictedCategory, error;
    int predictedLevel = 0;
    double confidence = 0.0;
    QElapsedTimer timer;
    timer.start();
    try {
      const auto result = analyzer.analyzeHazard(path.toStdString(), "평가 사례 " + values[caseIndex].toStdString());
      predictedCategory = QString::fromStdString(result.category);
      predictedLevel = result.riskLevel;
      confidence = result.confidence;
      ++succeeded;
      if (predictedCategory == values[categoryIndex]) ++categoryMatches;
      if (levelOk) levelError += std::abs(predictedLevel - expectedLevel);
    } catch (const AiProviderError& e) {
      error = QString::fromStdString(aiProviderState(e.code)) + ": " + QString::fromUtf8(e.what());
      if (e.code == AiProviderErrorCode::InvalidJson || e.code == AiProviderErrorCode::EmptyResponse)
        ++jsonFailures;
    } catch (const std::exception& e) { error = QString::fromUtf8(e.what()); }
    const qint64 latency = timer.elapsed();
    latencyTotal += latency;
    out << csv(values[caseIndex]) << ',' << csv(values[imageIndex]) << ','
        << csv(values[categoryIndex]) << ',' << values[levelIndex] << ','
        << csv(predictedCategory) << ',' << (predictedLevel ? QString::number(predictedLevel) : QString()) << ','
        << (confidence ? QString::number(confidence, 'f', 4) : QString()) << ',' << latency << ','
        << csv(error) << '\n';
  }

  QTextStream(stdout)
    << "attempted=" << attempted << " succeeded=" << succeeded
    << " category_accuracy=" << (succeeded ? double(categoryMatches) / succeeded : 0.0)
    << " risk_level_mae=" << (succeeded ? levelError / succeeded : 0.0)
    << " average_latency_ms=" << (attempted ? latencyTotal / attempted : 0.0)
    << " json_failure_rate=" << (attempted ? double(jsonFailures) / attempted : 0.0) << '\n';
  return attempted > 0 ? 0 : 1;
}
