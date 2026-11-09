import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window

    property bool captureOpen: false
    property bool reviewOpen: false
    property bool savedOpen: false

    readonly property bool busy:
        captureController.working || captureController.reviewSaving

    width: 390
    height: 760
    visible: true
    title: "SafeLog"
    color: "#f4f7f6"

    function returnHome() {
        if (window.busy)
            return

        window.captureOpen = false
        window.reviewOpen = false
        window.savedOpen = false
    }

    Connections {
        target: captureController

        function onAnalysisReady() {
            if (window.captureOpen)
                window.reviewOpen = true
        }

        function onReviewSaved() {
            Qt.callLater(function() {
                if (window.captureOpen)
                    window.savedOpen = true
            })
        }

        function onManualSaved() {
            Qt.callLater(function() {
                if (window.captureOpen)
                    window.savedOpen = true
            })
        }
    }

    header: ToolBar {
        contentHeight: 62

        background: Rectangle {
            color: "#126b58"
        }

        Label {
            anchors.centerIn: parent
            text: "세이프로그"
            color: "white"
            font.pixelSize: 21
            font.bold: true
        }
    }

    ColumnLayout {
        visible: !window.captureOpen
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16

        Label {
            text: "현장 안전점검"
            font.pixelSize: 26
            font.bold: true
            color: "#17212b"
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            text: "사진 등록부터 조치 확인, 보고서 생성까지 한 흐름으로 관리합니다."
            color: "#526069"
        }

        Frame {
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent

                Label {
                    text: "세이프 금속 가공공장"
                    font.bold: true
                    font.pixelSize: 18
                }

                Label {
                    text: appController.statusMessage
                    color: "#126b58"
                }
            }
        }

        Button {
            Layout.fillWidth: true
            text: "통합 시나리오 실행"
            enabled: !window.busy
            onClicked: appController.runDemoScenario()
        }

        Label {
            text: "개발 메뉴"
            font.bold: true
            font.pixelSize: 18
        }

        Repeater {
            model: [
                "A · 새 점검/사진 등록",
                "B · 로컬 저장소 상태",
                "C · 담당 업무/조치 확인",
                "D · 보고서 미리보기"
            ]

            delegate: Button {
                required property string modelData
                required property int index

                Layout.fillWidth: true
                text: modelData
                enabled: !window.busy

                onClicked: {
                    if (index === 0) {
                        window.savedOpen = false
                        window.reviewOpen = false
                        window.captureOpen = true
                    }
                }
            }
        }

        Item {
            Layout.fillHeight: true
        }

        Label {
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            text: "학술제 프로토타입 · 법적 적합성은 별도 검토 필요"
            color: "#7b878d"
            font.pixelSize: 11
            wrapMode: Text.Wrap
        }
    }

    Page {
        anchors.fill: parent
        visible: window.captureOpen

        header: ToolBar {
            RowLayout {
                anchors.fill: parent

                ToolButton {
                    text: "← 돌아가기"
                    enabled: !window.busy
                    onClicked: window.returnHome()
                }

                Item {
                    Layout.fillWidth: true
                }
            }
        }

        Loader {
            id: captureLoader

            anchors.fill: parent
            active: window.captureOpen

            source: window.savedOpen
                    ? "capture/FindingSavedPage.qml"
                    : window.reviewOpen
                      ? "capture/AiHazardReviewPage.qml"
                      : "capture/FindingCapturePage.qml"

            onLoaded: {
                if (!item)
                    return

                if (window.savedOpen) {
                    const saved = captureController.reviewResult

                    item.showSaved(
                        saved.findingId,
                        saved.description,
                        saved.actionOpinion,
                        saved.decision
                    )
                } else if (window.reviewOpen) {
                    // 저장 상태와 오류 메시지를 계속 화면에 반영한다.
                    item.saving = Qt.binding(function() {
                        return captureController.reviewSaving
                    })

                    item.errorMessage = Qt.binding(function() {
                        return captureController.errorMessage
                    })

                    const result = captureController.suggestion

                    item.showSuggestion(
                        result.hazardCategory,
                        result.riskLevel,
                        result.suggestedDescription,
                        result.suggestedAction,
                        captureController.originalMemo
                    )
                }
            }
        }

        Connections {
            target: captureLoader.item

            // 각 화면이 제공하는 신호만 연결한다.
            ignoreUnknownSignals: true

            function onReviewRequested(decision, description, actionOpinion) {
                captureController.saveReview(
                    decision,
                    description,
                    actionOpinion
                )
            }

            function onHomeRequested() {
                Qt.callLater(function() {
                    window.returnHome()
                })
            }
        }
    }
}