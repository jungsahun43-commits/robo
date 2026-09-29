#include "app_controller.hpp"
#include "safelog/ai/mock_ai_analyzer.hpp"
#include <QtTest>
#include <QSignalSpy>
#include <QImage>
#include <QTemporaryDir>
#include <QStandardPaths>
#include <QFileInfo>
#include <thread>

using namespace safelog;
using namespace safelog::qtapp;
namespace {
class SlowAnalyzer final : public ai::IAiSafetyAnalyzer {
public:
  ai::HazardSuggestion analyzeHazard(const std::string& p, const std::string& m) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return mock.analyzeHazard(p, m);
  }
  ai::ActionAssessment compareBeforeAfter(const std::string& a, const std::string& b, const std::string& m) override {
    return mock.compareBeforeAfter(a, b, m);
  }
  ai::ReportSummary summarize(const std::string& c) override { return mock.summarize(c); }
  ai::MockAiSafetyAnalyzer mock;
};
struct Fixture {
  QTemporaryDir dir;
  storage::InMemoryRepository repo;
  storage::SystemClock clock;
  storage::SequentialIdGenerator ids;
  storage::LocalPhotoStore photos{dir.path().toStdString() + "/photos"};
  std::shared_ptr<ai::IAiSafetyAnalyzer> provider{std::make_shared<ai::MockAiSafetyAnalyzer>()};
  capture::CaptureService service{repo, clock, ids, photos};
  ai::AiSafetyService reviews{repo, clock, ids, *provider};
  AuthController auth{repo};
  AiController ai{provider};
  CaptureController capture{repo, clock, ids, service, reviews, auth, ai};
  Fixture() {
    repo.saveSite({"site-1", "세이프 금속 가공공장", ""});
    repo.saveProfile({"inspector-1", "김안전", UserRole::Inspector});
    repo.saveProfile({"manager-1", "박관리", UserRole::Manager});
    repo.saveProfile({"assignee-1", "이조치", UserRole::Assignee});
    QImage image(20, 20, QImage::Format_RGB32); image.fill(Qt::green); image.save(path());
  }
  QString path() const { return dir.path() + "/before.png"; }
};
}
class ControllerTests final : public QObject {
  Q_OBJECT
private slots:
  void initTestCase() { QStandardPaths::setTestModeEnabled(true); }
  void loginAndLogout() {
    Fixture f;
    f.auth.login("inspector-1", "wrong");
    QVERIFY(f.auth.loading());
    QTRY_VERIFY(!f.auth.loading());
    QVERIFY(f.auth.session().isEmpty()); QVERIFY(!f.auth.error().isEmpty());
    for (const auto& user : {"inspector-1", "manager-1", "assignee-1"}) {
      f.auth.login(user, "demo1234"); QTRY_VERIFY(!f.auth.loading());
      QCOMPARE(f.auth.session().value("userId").toString(), QString(user));
      QCOMPARE(f.auth.session().size(), 5);
      f.auth.logout(); QVERIFY(f.auth.session().isEmpty());
    }
    f.auth.login("inspector-1", "demo1234"); f.auth.logout();
    QTest::qWait(200); QVERIFY(f.auth.session().isEmpty());
  }
  void capturePermissionAndInvalidPhoto() {
    Fixture f;
    QVERIFY(!f.capture.createFinding(f.path(), "통로", "메모", "이동"));
    f.auth.login("manager-1", "demo1234"); QTRY_VERIFY(!f.auth.loading());
    QVERIFY(!f.capture.createFinding(f.path(), "통로", "메모", "이동"));
    f.auth.logout(); f.auth.login("inspector-1", "demo1234"); QTRY_VERIFY(!f.auth.loading());
    QVERIFY(!f.capture.createFinding("/missing.jpg", "통로", "메모", "이동"));
    QVERIFY(!f.capture.error().isEmpty());
  }
  void humanReview_data() {
    QTest::addColumn<QString>("decision"); QTest::addColumn<QString>("description"); QTest::addColumn<int>("expected");
    QTest::newRow("accepted") << "Accepted" << "" << int(AiReviewDecision::Accepted);
    QTest::newRow("edited") << "Accepted" << "사람이 수정한 설명" << int(AiReviewDecision::Edited);
    QTest::newRow("rejected") << "Rejected" << "최초 메모" << int(AiReviewDecision::Rejected);
  }
  void humanReview() {
    QFETCH(QString, decision); QFETCH(QString, description); QFETCH(int, expected);
    Fixture f; f.auth.login("inspector-1", "demo1234"); QTRY_VERIFY(!f.auth.loading());
    QVERIFY(f.capture.createFinding(f.path(), "통로", "최초 메모", "이동"));
    const auto id = f.capture.draft().value("findingId").toString().toStdString();
    QTRY_COMPARE(f.ai.state(), QString("Success"));
    QCOMPARE(f.repo.analysesForSubject(id).size(), size_t(1));
    QCOMPARE(f.repo.analysesForSubject(id).front().decision, AiReviewDecision::Pending);
    if (description.isEmpty()) description = f.ai.result().value("description").toString();
    const auto action = f.ai.result().value("action").toString();
    QVERIFY(f.capture.saveReview(decision, description, action));
    QCOMPARE(int(f.repo.analysesForSubject(id).front().decision), expected);
    QCOMPARE(f.repo.findFinding(id)->status, FindingStatus::Open);
    QCOMPARE(f.repo.findFinding(id)->description, description.toStdString());
    QVERIFY(!f.capture.saveReview(decision, description, action));
  }
  void failuresAndManual_data() {
    QTest::addColumn<QString>("failure");
    for (const char* value : {"Failure", "Timeout", "InvalidJson", "ConnectionError"}) QTest::newRow(value) << QString(value);
  }
  void failuresAndManual() {
    QFETCH(QString, failure);
    Fixture f; f.auth.login("inspector-1", "demo1234"); QTRY_VERIFY(!f.auth.loading());
    f.ai.setDemoFailure(failure);
    QSignalSpy failed(&f.ai, &AiController::analysisFailed);
    QVERIFY(f.capture.createFinding(f.path(), "통로", "메모", "이동"));
    QTRY_COMPARE(failed.size(), 1); QCOMPARE(f.ai.state(), failure);
    const auto id = f.capture.draft().value("findingId").toString().toStdString();
    QVERIFY(f.repo.analysesForSubject(id).empty());
    f.ai.manualFallback();
    QVERIFY(!f.capture.saveReview("Manual", "", ""));
    QVERIFY(f.capture.saveReview("Manual", "수동 설명", "수동 조치"));
    QCOMPARE(f.repo.findFinding(id)->description, std::string("수동 설명"));
    QCOMPARE(f.repo.findFinding(id)->status, FindingStatus::Open);
  }
  void retryKeepsFinding() {
    Fixture f; f.auth.login("inspector-1", "demo1234"); QTRY_VERIFY(!f.auth.loading());
    f.ai.setDemoFailure("ConnectionError");
    QVERIFY(f.capture.createFinding(f.path(), "통로", "메모", "이동"));
    QTRY_COMPARE(f.ai.state(), QString("ConnectionError"));
    const auto draft = f.capture.draft(); f.ai.setDemoFailure(""); f.capture.retry();
    QTRY_COMPARE(f.ai.state(), QString("Success")); QCOMPARE(f.capture.draft(), draft);
  }
  void canceledResultIsIgnored() {
    Fixture f;
    AiController controller(std::make_shared<SlowAnalyzer>());
    QSignalSpy success(&controller, &AiController::analysisSucceeded);
    controller.analyzeHazard(f.path(), "memo"); QVERIFY(controller.loading());
    controller.manualFallback(); QTest::qWait(350);
    QCOMPARE(controller.state(), QString("ManualFallback")); QCOMPARE(success.size(), 0);
    controller.analyzeHazard(f.path(), "memo"); controller.reset(); QTest::qWait(350);
    QCOMPARE(controller.state(), QString("Idle")); QCOMPARE(success.size(), 0);
  }
  void deadlineIgnoresLateReply() {
    Fixture f;
    AiController controller(std::make_shared<SlowAnalyzer>(), nullptr, 30);
    QSignalSpy failed(&controller, &AiController::analysisFailed);
    QSignalSpy success(&controller, &AiController::analysisSucceeded);
    controller.analyzeHazard(f.path(), "memo");
    QTRY_COMPARE(failed.size(), 1);
    QCOMPARE(controller.state(), QString("Timeout"));
    QTest::qWait(300);
    QCOMPARE(success.size(), 0);
    QCOMPARE(controller.state(), QString("Timeout"));
  }
  void dashboardTracksSession() {
    AppController app;
    app.auth()->login("inspector-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    const auto photo = app.capture()->demoPhoto();
    QVERIFY(app.capture()->createFinding(photo, "통로", "메모", "이동"));
    QTRY_COMPARE(app.ai()->state(), QString("Success"));
    QCOMPARE(app.dashboard().front().toInt(), 1);
    app.auth()->logout(); QCOMPARE(app.dashboard().front().toInt(), 0);
    QVERIFY(app.capture()->draft().isEmpty()); QCOMPARE(app.ai()->state(), QString("Idle"));
    app.auth()->login("manager-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    QCOMPARE(app.dashboard().front().toInt(), 1);
    app.auth()->logout();
    app.auth()->login("assignee-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    QCOMPARE(app.dashboard().front().toInt(), 0);
  }
  void workflowAndReportIntegration() {
    AppController app;
    app.auth()->login("inspector-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    QVERIFY(app.capture()->createFinding(app.capture()->demoPhoto(), "통로", "적치물", "이동 필요"));
    QTRY_COMPARE(app.ai()->state(), QString("Success"));
    const auto findingId = app.capture()->draft().value("findingId").toString();
    const auto inspectionId = app.capture()->draft().value("inspectionId").toString();
    QVERIFY(app.capture()->saveReview("Accepted", app.ai()->result().value("description").toString(), app.ai()->result().value("action").toString()));
    QVERIFY(app.workflow()->selectFinding(findingId));
    QVERIFY(!app.workflow()->assign("assignee-1"));
    app.auth()->logout(); app.auth()->login("manager-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    QVERIFY(app.workflow()->selectFinding(findingId)); QVERIFY(app.workflow()->assign("assignee-1"));
    QCOMPARE(app.workflow()->selected().value("status").toString(), QString("open"));
    app.auth()->logout(); app.auth()->login("assignee-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    QVERIFY(app.workflow()->selectFinding(findingId)); QVERIFY(app.workflow()->beginWork());
    QVERIFY(!app.workflow()->submitAction("", "", ""));
    QVERIFY(app.workflow()->submitAction("통로 확보", app.capture()->demoPhoto(true), ""));
    QTRY_COMPARE(app.ai()->state(), QString("Success"));
    QVERIFY(!app.workflow()->verify("Accepted", "확인", true));
    app.auth()->logout(); app.auth()->login("inspector-1", "demo1234"); QTRY_VERIFY(!app.auth()->loading());
    QVERIFY(app.workflow()->selectFinding(findingId)); app.workflow()->compare();
    QTRY_COMPARE(app.ai()->state(), QString("Success"));
    QVERIFY(!app.workflow()->verify("Accepted", "확인", false));
    QVERIFY(app.workflow()->verify("Accepted", "현장 최종 확인", true));
    QCOMPARE(app.workflow()->selected().value("status").toString(), QString("verified"));
    app.reporting()->generate(inspectionId); QTRY_COMPARE(app.reporting()->state(), QString("Success"));
    QVERIFY(QFileInfo::exists(app.reporting()->outputPath()));
    app.reporting()->share("ShareCancelled"); QCOMPARE(app.reporting()->shareState(), QString("ShareCancelled"));
    QVERIFY(app.reporting()->error().isEmpty());
    app.reporting()->share("ShareFailed"); QVERIFY(!app.reporting()->error().isEmpty());
    app.auth()->logout(); QVERIFY(app.reporting()->outputPath().isEmpty());
  }
  void comparisonInterface() {
    Fixture f; QSignalSpy success(&f.ai, &AiController::comparisonSucceeded);
    f.ai.compareBeforeAfter(f.path(), f.path(), "조치"); QTRY_COMPARE(success.size(), 1);
    QVERIFY(f.ai.result().value("likelyResolved").toBool());
  }
};
QTEST_MAIN(ControllerTests)
#include "controller_tests.moc"
