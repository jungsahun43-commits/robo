#include "capture_controller.hpp"

#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <cmath>
#include <exception>
#include <memory>

namespace {

struct AnalysisOutcome {
    QString id;
    QVariantMap suggestion;
    QString error;
    QString state;
};

bool validHazardJson(const QJsonObject& object)
{
    for (const auto* key : {
                             "hazardCategory",
                             "suggestedDescription",
                             "suggestedAction"}) {
        if (!object.value(key).isString()
            || object.value(key).toString().trimmed().isEmpty()) {
            return false;
        }
    }

    const auto risk = object.value("riskLevel");
    const auto confidence = object.value("confidence");

    if (!risk.isDouble() || !confidence.isDouble())
        return false;

    const double level = risk.toDouble();
    const double score = confidence.toDouble();

    if (!std::isfinite(level)
        || level < 1
        || level > 5
        || std::floor(level) != level
        || !std::isfinite(score)
        || score < 0
        || score > 1) {
        return false;
    }

    if (!object.value("detectedHazards").isArray())
        return false;

    for (const auto& item : object.value("detectedHazards").toArray()) {
        if (!item.isString())
            return false;
    }

    return true;
}

} // namespace

CaptureController::CaptureController(
    safelog::capture::CaptureService& service,
    safelog::ai::AiSafetyService& aiService,
    QObject* parent)
    : QObject(parent),
    service_(service),
    aiService_(aiService)
{
    analysisTimer_.setSingleShot(true);
    analysisTimer_.setInterval(30000);

    connect(&analysisTimer_, &QTimer::timeout, this, [this] {
        if (analysisState_ != "analyzing")
            return;

        ignoreAnalysis_ = true;
        analysisState_ = "timeout";
        errorMessage_ = QStringLiteral(
            "AI 분석 시간이 초과됐습니다. 사진과 원래 메모는 저장돼 있습니다.");

        emit stateChanged();
    });
}

CaptureController::~CaptureController()
{
    analysisTimer_.stop();

    // 서비스가 사라지기 전에 진행 중인 작업의 종료를 기다린다.
    if (analysisThread_)
        analysisThread_->wait();
}

QString CaptureController::errorMessage() const
{
    return errorMessage_;
}

QString CaptureController::findingId() const
{
    return findingId_;
}

void CaptureController::resetDraft()
{
    if (saving_ || working() || reviewSaving_)
        return;

    errorMessage_.clear();
    findingId_.clear();
    originalMemo_.clear();
    analysisId_.clear();
    suggestion_.clear();
    reviewResult_.clear();
    analysisState_ = "idle";

    emit stateChanged();
}

void CaptureController::saveFinding(
    const QString& location,
    const QString& memo,
    const QUrl& photoUrl)
{
    if (saving_ || working() || !findingId_.isEmpty())
        return;

    saving_ = true;
    errorMessage_.clear();
    emit stateChanged();

    const QString cleanLocation = location.trimmed();
    const QString cleanMemo = memo.trimmed();

    const auto fail = [this](const QString& message) {
        errorMessage_ = message;
        saving_ = false;
        emit stateChanged();
    };

    if (cleanLocation.isEmpty()) {
        fail(QStringLiteral("장소를 입력해 주세요."));
        return;
    }

    if (cleanMemo.isEmpty()) {
        fail(QStringLiteral("현장 메모를 입력해 주세요."));
        return;
    }

    if (!photoUrl.isLocalFile()) {
        fail(QStringLiteral("PC에 저장된 사진을 선택해 주세요."));
        return;
    }

    const QString photoPath = photoUrl.toLocalFile();
    const QFileInfo photoInfo(photoPath);

    if (!photoInfo.isFile() || !photoInfo.isReadable()) {
        fail(QStringLiteral("사진 파일을 읽을 수 없습니다."));
        return;
    }

    QImageReader reader(photoPath);
    reader.setAutoTransform(true);

    if (reader.read().isNull()) {
        fail(QStringLiteral(
            "정상적인 이미지 파일이 아닙니다. 다른 사진을 선택해 주세요."));
        return;
    }

    try {
        if (inspectionId_.empty()) {
            const auto inspection =
                service_.startInspection("site-1", "inspector-1");

            inspectionId_ = inspection.id;
        }

        safelog::capture::NewFindingInput input;
        input.inspectionId = inspectionId_;
        input.actorId = "inspector-1";
        input.location = cleanLocation.toStdString();
        input.description = cleanMemo.toStdString();
        input.actionOpinion = "";
        input.beforePhotoLocalPath = photoPath.toStdString();

        const auto bundle = service_.addFinding(input);

        findingId_ = QString::fromStdString(bundle.finding.id);
        originalMemo_ = cleanMemo;
    }
    catch (const std::exception& error) {
        fail(QStringLiteral("기록 저장에 실패했습니다: ")
             + QString::fromUtf8(error.what()));
        return;
    }
    catch (...) {
        fail(QStringLiteral("기록 저장 중 알 수 없는 오류가 발생했습니다."));
        return;
    }

    saving_ = false;
    emit stateChanged();
    emit findingSaved(findingId_);

    // 기록이 저장된 후 AI 분석을 시작한다.
    startAnalysis();
}

void CaptureController::startAnalysis()
{
    if (saving_ || working() || reviewSaving_
        || !reviewResult_.isEmpty()
        || findingId_.isEmpty()
        || !analysisId_.isEmpty()) {
        return;
    }

    errorMessage_.clear();
    suggestion_.clear();
    ignoreAnalysis_ = false;
    analysisState_ = "analyzing";

    auto outcome = std::make_shared<AnalysisOutcome>();
    auto* service = &aiService_;
    const auto id = findingId_.toStdString();

    auto* thread = QThread::create([service, id, outcome] {
        try {
            const auto analysis = service->analyzeFinding(id);

            const auto document = QJsonDocument::fromJson(
                QByteArray::fromStdString(analysis.resultJson));

            const auto object = document.object();

            if (!document.isObject()
                || !validHazardJson(object)
                || !analysis.riskLevel
                || *analysis.riskLevel != object.value("riskLevel").toInt()
                || QString::fromStdString(analysis.category)
                       != object.value("hazardCategory").toString()
                || !std::isfinite(analysis.confidence)
                || std::abs(
                       analysis.confidence
                       - object.value("confidence").toDouble()) > 1e-9) {
                outcome->state = "jsonError";
                outcome->error = QStringLiteral(
                    "AI 응답의 JSON 형식 또는 필수 항목이 올바르지 않습니다.");
                return;
            }

            outcome->id = QString::fromStdString(analysis.id);
            outcome->suggestion = object.toVariantMap();
            outcome->state = "success";
        }
        catch (const std::exception& error) {
            outcome->state = "error";
            outcome->error = QStringLiteral("AI 분석 실패: ")
                             + QString::fromUtf8(error.what());
        }
        catch (...) {
            outcome->state = "error";
            outcome->error = QStringLiteral(
                "AI 분석 중 알 수 없는 오류가 발생했습니다.");
        }
    });

    analysisThread_ = thread;
    thread->setParent(this);

    connect(thread, &QThread::finished, this, [this, thread, outcome] {
        thread->wait();
        analysisTimer_.stop();
        analysisThread_ = nullptr;

        // 취소하거나 시간 초과된 작업은 화면에 반영하지 않는다.
        if (!ignoreAnalysis_) {
            analysisState_ = outcome->state;
            errorMessage_ = outcome->error;

            if (outcome->state == "success") {
                analysisId_ = outcome->id;
                suggestion_ = outcome->suggestion;
            }
        }

        thread->deleteLater();
        emit stateChanged();

        if (analysisState_ == "success")
            emit analysisReady();
    });

    emit stateChanged();
    analysisTimer_.start();
    thread->start();
}

void CaptureController::cancelAnalysis()
{
    if (analysisState_ != "analyzing")
        return;

    ignoreAnalysis_ = true;
    analysisTimer_.stop();
    analysisState_ = "cancelled";
    errorMessage_ = QStringLiteral(
        "AI 결과 수신을 취소했습니다. 사진과 원래 메모는 저장돼 있습니다.");

    emit stateChanged();
}

void CaptureController::saveReview(
    const QString& decision,
    const QString& description,
    const QString& actionOpinion)
{
    // 분석 중이거나 이미 검토를 저장했다면 중복 실행하지 않는다.
    if (saving_ || working() || reviewSaving_
        || !reviewResult_.isEmpty()) {
        return;
    }

    const auto fail = [this](const QString& message) {
        errorMessage_ = message;
        reviewSaving_ = false;
        emit stateChanged();
    };

    if (analysisState_ != "success"
        || analysisId_.isEmpty()
        || findingId_.isEmpty()) {
        fail(QStringLiteral("저장할 AI 분석 결과가 없습니다."));
        return;
    }

    safelog::ai::HazardReview review;

    if (decision == "Rejected") {
        // 거절 시 서비스가 기존 발견 문장과 개선안을 유지한다.
        review.decision = safelog::AiReviewDecision::Rejected;
        review.description = "";
        review.actionOpinion = "";
    }
    else if (decision == "Accepted" || decision == "Edited") {
        const QString cleanDescription = description.trimmed();
        const QString cleanAction = actionOpinion.trimmed();

        if (cleanDescription.isEmpty() || cleanAction.isEmpty()) {
            fail(QStringLiteral(
                "최종 발견 문장과 개선안을 모두 입력해 주세요."));
            return;
        }

        const QString suggestedDescription =
            suggestion_.value("suggestedDescription")
                .toString().trimmed();

        const QString suggestedAction =
            suggestion_.value("suggestedAction")
                .toString().trimmed();

        const bool changed =
            cleanDescription != suggestedDescription
            || cleanAction != suggestedAction;

        if (decision == "Accepted" && changed) {
            fail(QStringLiteral(
                "문장을 변경했다면 수정한 내용으로 저장해 주세요."));
            return;
        }

        if (decision == "Edited" && !changed) {
            fail(QStringLiteral(
                "변경된 내용이 없습니다. AI 제안 그대로 채택을 선택해 주세요."));
            return;
        }

        review.decision =
            decision == "Accepted"
                ? safelog::AiReviewDecision::Accepted
                : safelog::AiReviewDecision::Edited;

        review.description = cleanDescription.toStdString();
        review.actionOpinion = cleanAction.toStdString();
    }
    else {
        fail(QStringLiteral("올바른 검토 결정을 선택해 주세요."));
        return;
    }

    errorMessage_.clear();
    reviewSaving_ = true;
    emit stateChanged();

    try {
        // reviewHazard에는 findingId가 아닌 AI 분석 ID를 전달한다.
        const auto finding = aiService_.reviewHazard(
            analysisId_.toStdString(),
            "inspector-1",
            review);

        // 서비스가 반환한 실제 저장 결과를 완료 화면에 전달한다.
        reviewResult_ = QVariantMap{
            {QStringLiteral("findingId"),
             QString::fromStdString(finding.id)},

            {QStringLiteral("description"),
             QString::fromStdString(finding.description)},

            {QStringLiteral("actionOpinion"),
             QString::fromStdString(finding.actionOpinion)},

            {QStringLiteral("decision"), decision}
        };
    }
    catch (const std::exception& error) {
        fail(QStringLiteral("검토 결과 저장에 실패했습니다: ")
             + QString::fromUtf8(error.what()));
        return;
    }
    catch (...) {
        fail(QStringLiteral(
            "검토 결과 저장 중 알 수 없는 오류가 발생했습니다."));
        return;
    }

    reviewSaving_ = false;
    emit stateChanged();
    emit reviewSaved();
}

void CaptureController::saveManual(
    const QString& description,
    const QString& actionOpinion)
{
    if (saving_ || reviewSaving_ || !reviewResult_.isEmpty())
        return;

    const auto fail = [this](const QString& message) {
        errorMessage_ = message;
        reviewSaving_ = false;
        emit stateChanged();
    };

    if (working()) {
        fail(QStringLiteral("이전 분석 작업이 끝난 후 저장해 주세요."));
        return;
    }

    const bool canUseManual =
        analysisState_ == "error"
        || analysisState_ == "jsonError"
        || analysisState_ == "connectionError"
        || analysisState_ == "timeout"
        || analysisState_ == "cancelled";

    if (findingId_.isEmpty() || !canUseManual) {
        fail(QStringLiteral(
            "AI 분석 실패 또는 취소 후 수동 입력을 사용할 수 있습니다."));
        return;
    }

    const QString cleanDescription = description.trimmed();
    const QString cleanAction = actionOpinion.trimmed();

    if (cleanDescription.isEmpty() || cleanAction.isEmpty()) {
        fail(QStringLiteral("발견 문장과 개선안을 모두 입력해 주세요."));
        return;
    }

    errorMessage_.clear();
    reviewSaving_ = true;
    emit stateChanged();

    try {
        const auto finding = service_.saveManualFinding(
            findingId_.toStdString(),
            "inspector-1",
            cleanDescription.toStdString(),
            cleanAction.toStdString());

        reviewResult_ = QVariantMap{
            {"findingId", QString::fromStdString(finding.id)},
            {"description", QString::fromStdString(finding.description)},
            {"actionOpinion", QString::fromStdString(finding.actionOpinion)},
            {"decision", QStringLiteral("Manual")}
        };
    }
    catch (const std::exception& error) {
        fail(QStringLiteral("수동 입력 저장에 실패했습니다: ")
             + QString::fromUtf8(error.what()));
        return;
    }
    catch (...) {
        fail(QStringLiteral(
            "수동 입력 저장 중 알 수 없는 오류가 발생했습니다."));
        return;
    }

    analysisState_ = "manualSaved";
    reviewSaving_ = false;
    emit stateChanged();
    emit manualSaved();
}