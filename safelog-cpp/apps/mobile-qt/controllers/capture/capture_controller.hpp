#pragma once

#include "safelog/capture/capture_service.hpp"
#include "safelog/ai/ai_safety_service.hpp"

#include <QObject>
#include <QString>
#include <QUrl>
#include <QThread>
#include <QTimer>
#include <QVariantMap>

class CaptureController final : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QString errorMessage READ errorMessage NOTIFY stateChanged)
    Q_PROPERTY(QString findingId READ findingId NOTIFY stateChanged)
    Q_PROPERTY(QString originalMemo READ originalMemo NOTIFY stateChanged)
    Q_PROPERTY(QString analysisState READ analysisState NOTIFY stateChanged)
    Q_PROPERTY(QVariantMap suggestion READ suggestion NOTIFY stateChanged)
    Q_PROPERTY(bool working READ working NOTIFY stateChanged)

    Q_PROPERTY(bool reviewSaving READ reviewSaving NOTIFY stateChanged)
    Q_PROPERTY(QVariantMap reviewResult READ reviewResult NOTIFY stateChanged)

public:
    explicit CaptureController(
        safelog::capture::CaptureService& service,
        safelog::ai::AiSafetyService& aiService,
        QObject* parent = nullptr);

    ~CaptureController() override;

    QString errorMessage() const;
    QString findingId() const;

    QString originalMemo() const { return originalMemo_; }
    QString analysisState() const { return analysisState_; }
    QVariantMap suggestion() const { return suggestion_; }
    bool working() const { return analysisThread_ != nullptr; }

    bool reviewSaving() const { return reviewSaving_; }
    QVariantMap reviewResult() const { return reviewResult_; }

    Q_INVOKABLE void saveFinding(
        const QString& location,
        const QString& memo,
        const QUrl& photoUrl);

    Q_INVOKABLE void resetDraft();
    Q_INVOKABLE void startAnalysis();
    Q_INVOKABLE void cancelAnalysis();

    Q_INVOKABLE void saveReview(
        const QString& decision,
        const QString& description,
        const QString& actionOpinion);

    Q_INVOKABLE void saveManual(
        const QString& description,
        const QString& actionOpinion);

signals:
    void stateChanged();
    void findingSaved(const QString& findingId);
    void analysisReady();
    void reviewSaved();
    void manualSaved();

private:
    safelog::capture::CaptureService& service_;
    safelog::ai::AiSafetyService& aiService_;

    safelog::Id inspectionId_;

    QString errorMessage_;
    QString findingId_;
    QString originalMemo_;
    QString analysisId_;
    QString analysisState_{QStringLiteral("idle")};

    QVariantMap suggestion_;
    QVariantMap reviewResult_;

    bool saving_{false};
    bool ignoreAnalysis_{false};
    bool reviewSaving_{false};

    QThread* analysisThread_{nullptr};
    QTimer analysisTimer_;
};