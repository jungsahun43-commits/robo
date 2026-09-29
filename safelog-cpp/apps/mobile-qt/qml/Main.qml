import QtQuick
import QtQuick.Controls
import "auth"
import "home"
import "capture"
import "common"
import "workflow"
import "reporting"
ApplicationWindow {
    id: window
    width: 390; height: 760; visible: true
    title: "SafeLog AI"; color: "#f4f7f6"
    property string navigationError: ""
    function goHome() {
        captureController.reset(); workflowController.reset(); reportingController.reset()
        navigationError = ""; stack.clear(); stack.push(authController.session.userId ? homePage : loginPage)
    }
    function goBack() {
        if (stack.depth <= 2) { goHome(); return }
        aiController.reset(); stack.pop()
    }
    function showReport(inspectionId) {
        stack.push(reportPage, { inspectionId: inspectionId })
        reportingController.generate(inspectionId)
    }
    function navigate(menu) {
        const role = authController.session.role
        const allowed = role === "Inspector" ? ["점검 등록", "AI 검토", "최종 확인", "완료 보고서"] :
                        role === "Manager" ? ["담당자 지정", "전체 진행 상황", "검토 상태", "보고서"] :
                        role === "Assignee" ? ["배정된 작업", "조치 제출", "진행 중 작업"] : []
        if (allowed.indexOf(menu) < 0) { navigationError = "권한 없는 메뉴 접근입니다."; return }
        if (menu === "점검 등록") { captureController.reset(); stack.push(capturePage) }
        else if (menu === "AI 검토" && captureController.draft.findingId) stack.push(reviewPage)
        else stack.push(tasksPage, { menuTitle: menu })
    }
    header: ToolBar {
        Row {
            spacing: 12
            ToolButton { text: "‹ 뒤로"; visible: stack.depth > 1; onClicked: window.goBack() }
            ToolButton { text: "홈"; visible: !!authController.session.userId; onClicked: window.goHome() }
            Label { text: "SafeLog AI"; padding: 14; font.bold: true }
            ToolButton { text: "로그아웃"; visible: !!authController.session.userId; onClicked: authController.logout() }
        }
    }
    footer: ErrorPanel { text: window.navigationError }
    StackView { id: stack; anchors.fill: parent; initialItem: loginPage }
    Connections {
        target: authController
        function onSessionChanged() { stack.clear(); stack.push(authController.session.userId ? homePage : loginPage) }
    }
    Connections { target: captureController; function onSaved() { stack.replace(savedPage) } }
    Connections { target: workflowController; function onActionSubmitted() { stack.replace(comparisonPage) } }
    Component { id: loginPage; LoginPage {} }
    Component { id: homePage; HomePage { onNavigate: menu => window.navigate(menu) } }
    Component { id: capturePage; FindingCapturePage { onCaptured: stack.replace(reviewPage) } }
    Component { id: reviewPage; AiHazardReviewPage {} }
    Component { id: savedPage; FindingSavedPage { onHomeRequested: window.goHome() } }
    Component {
        id: tasksPage
        AssignedTasksPage {
            onActionRequested: stack.push(actionPage)
            onReviewRequested: stack.push(comparisonPage)
            onReportRequested: inspectionId => window.showReport(inspectionId)
            onHazardRequested: findingId => { if (captureController.resumeFinding(findingId)) stack.push(reviewPage) }
        }
    }
    Component { id: actionPage; ActionSubmitPage {} }
    Component { id: comparisonPage; AiComparisonReviewPage { onReportRequested: inspectionId => window.showReport(inspectionId) } }
    Component { id: reportPage; ReportPreviewPage {} }
    Component { id: placeholderPage; PlaceholderPage {} }
}
