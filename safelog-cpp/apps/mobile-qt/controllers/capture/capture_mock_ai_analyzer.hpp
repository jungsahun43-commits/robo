#pragma once

#include "safelog/ai/mock_ai_analyzer.hpp"
#include "safelog/contracts/errors.hpp"

#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QThread>

#include <cmath>
#include <stdexcept>

class CaptureMockAiAnalyzer final
    : public safelog::ai::IAiSafetyAnalyzer
{
public:
    safelog::ai::HazardSuggestion analyzeHazard(
        const std::string& imagePath,
        const std::string& userMemo) override
    {
        // 실행 옵션이 있을 때만 실패를 재현한다.
        if (testArguments_.contains("--capture-ai-fail")) {
            throw std::runtime_error(
                QStringLiteral(
                    "모델 서버에 연결할 수 없습니다. (Mock 실패 테스트)"
                    ).toStdString());
        }

        // 시간 초과 또는 취소 동작을 확인하기 위한 지연이다.
        if (testArguments_.contains("--capture-ai-timeout"))
            QThread::msleep(32000);
        else if (testArguments_.contains("--capture-ai-slow"))
            QThread::msleep(10000);

        auto result = mock_.analyzeHazard(imagePath, userMemo);

        const QString category =
            QString::fromStdString(result.category);

        const QString description =
            QString::fromStdString(result.suggestedDescription);

        const QString action =
            QString::fromStdString(result.suggestedAction);

        if (category.trimmed().isEmpty()
            || description.trimmed().isEmpty()
            || action.trimmed().isEmpty()
            || result.riskLevel < 1
            || result.riskLevel > 5
            || !std::isfinite(result.confidence)
            || result.confidence < 0.0
            || result.confidence > 1.0) {
            throw safelog::ValidationError(
                "Mock AI returned an invalid hazard suggestion");
        }

        QJsonArray hazards;

        for (const auto& hazard : result.detectedHazards)
            hazards.append(QString::fromStdString(hazard));

        QJsonObject json;
        json["hazardCategory"] = category;
        json["riskLevel"] = result.riskLevel;
        json["detectedHazards"] = hazards;
        json["suggestedDescription"] = description;
        json["suggestedAction"] = action;
        json["confidence"] = result.confidence;

        result.rawJson =
            QJsonDocument(json)
                .toJson(QJsonDocument::Compact)
                .toStdString();

        // JSON 오류 처리 화면을 확인하기 위한 응답이다.
        if (testArguments_.contains("--capture-ai-bad-json"))
            result.rawJson = "{ invalid JSON";

        return result;
    }

    safelog::ai::ActionAssessment compareBeforeAfter(
        const std::string& beforeImagePath,
        const std::string& afterImagePath,
        const std::string& actionNote) override
    {
        return mock_.compareBeforeAfter(
            beforeImagePath,
            afterImagePath,
            actionNote);
    }

    safelog::ai::ReportSummary summarize(
        const std::string& inspectionContext) override
    {
        return mock_.summarize(inspectionContext);
    }

private:
    safelog::ai::MockAiSafetyAnalyzer mock_;

    const QStringList testArguments_{
        QCoreApplication::arguments()
    };
};