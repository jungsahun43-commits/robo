import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Page {
    id: page

    title: "현장 사진 등록"

    property url photoUrl: ""
    property bool manualMode: false
    property bool manualInitialized: false

    // 다음 단계에서 C++ Controller에 연결할 요청
    signal analysisRequested(string location, string memo, url photoUrl)
    readonly property bool recordSaved:
        captureController.findingId.length > 0

    Component.onCompleted: captureController.resetDraft()

    onAnalysisRequested: function(location, memo, photoUrl) {
        captureController.saveFinding(location, memo, photoUrl)
    }

    background: Rectangle {
        color: "#f4f7f6"
    }

    FileDialog {
        id: photoDialog
        title: "현장 사진 한 장 선택"
        fileMode: FileDialog.OpenFile
        nameFilters: ["사진 파일 (*.jpg *.jpeg *.png)"]

        onAccepted: page.photoUrl = selectedFile
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 20
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: parent.width
            spacing: 16

            Label {
                text: "현장 사진 등록"
                font.pixelSize: 26
                font.bold: true
                color: "#17212b"
            }

            Label {
                text: "장소와 짧은 메모를 입력하고 사진을 선택하세요."
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                color: "#526069"
            }

            Label {
                text: "장소"
                font.bold: true
            }

            TextField {
                id: locationInput
                readOnly: page.recordSaved

                Layout.fillWidth: true
                placeholderText: "예: 2층 가공라인 통로"
            }

            Label {
                text: "현장 메모"
                font.bold: true
            }

            TextArea {
                id: memoInput
                readOnly: page.recordSaved

                Layout.fillWidth: true
                Layout.preferredHeight: 110
                placeholderText: "예: 통로에 자재가 쌓여 있어 넘어질 위험이 있음"
                wrapMode: TextEdit.Wrap
            }

            Button {
                Layout.fillWidth: true
                enabled: !page.recordSaved

                text: page.photoUrl.toString().length > 0
                      ? "다른 사진 선택"
                      : "사진 선택"

                onClicked: photoDialog.open()
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 230
                color: "#e8eeeb"
                radius: 8
                border.color: "#c6d2cc"

                Image {
                    id: photoPreview
                    anchors.fill: parent
                    anchors.margins: 8
                    source: page.photoUrl
                    fillMode: Image.PreserveAspectFit
                    asynchronous: true
                    autoTransform: true
                }

                Label {
                    anchors.centerIn: parent
                    visible: photoPreview.status !== Image.Ready

                    text: {
                        if (photoPreview.status === Image.Loading)
                            return "사진을 불러오는 중입니다."
                        if (photoPreview.status === Image.Error)
                            return "사진을 읽을 수 없습니다.\n다른 사진을 선택해 주세요."
                        return "선택한 사진이 여기에 표시됩니다."
                    }

                    horizontalAlignment: Text.AlignHCenter
                    color: "#526069"
                }
            }

            Button {
                Layout.fillWidth: true

                text: page.recordSaved
                      ? "현장 기록 저장됨"
                      : "저장 후 AI 분석"

                enabled: !page.recordSaved
                         && locationInput.text.trim().length > 0
                         && memoInput.text.trim().length > 0
                         && photoPreview.status === Image.Ready

                onClicked: page.analysisRequested(
                    locationInput.text.trim(),
                    memoInput.text.trim(),
                    page.photoUrl
                )
            }

            Label {
                Layout.fillWidth: true
                visible: captureController.errorMessage.length > 0
                text: captureController.errorMessage
                color: "#b42318"
                wrapMode: Text.Wrap
            }

            Label {
                Layout.fillWidth: true
                visible: page.recordSaved
                text: "현장 기록 저장 완료\n기록 ID: "
                      + captureController.findingId
                color: "#126b58"
                wrapMode: Text.Wrap
            }

            BusyIndicator {
                Layout.alignment: Qt.AlignHCenter
                visible: captureController.working
                running: visible
            }

            Label {
                Layout.fillWidth: true
                visible: captureController.working

                text: captureController.analysisState === "analyzing"
                      ? "AI가 사진을 분석하고 있습니다."
                      : "이전 분석 작업이 끝나기를 기다리고 있습니다."

                wrapMode: Text.Wrap
            }

            Button {
                Layout.fillWidth: true
                visible: captureController.analysisState === "analyzing"
                text: "AI 결과 수신 취소"
                onClicked: captureController.cancelAnalysis()
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 12

                visible: page.recordSaved
                         && ["error", "jsonError", "connectionError",
                             "timeout", "cancelled"]
                            .indexOf(captureController.analysisState) >= 0

                Button {
                    Layout.fillWidth: true
                    visible: !page.manualMode
                    enabled: !captureController.working
                             && !captureController.reviewSaving
                    text: "AI 분석 다시 시도"
                    onClicked: captureController.startAnalysis()
                }

                Button {
                    Layout.fillWidth: true
                    visible: !page.manualMode
                    text: "수동 입력으로 전환"

                    onClicked: {
                        if (!page.manualInitialized) {
                            manualDescriptionInput.text =
                                captureController.originalMemo

                            page.manualInitialized = true
                        }

                        page.manualMode = true
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 12
                    visible: page.manualMode
                    enabled: !captureController.reviewSaving

                    Label {
                        Layout.fillWidth: true
                        text: "AI 제안 없이 발견 문장과 개선안을 직접 작성해 주세요."
                        wrapMode: Text.Wrap
                    }

                    Label {
                        text: "최종 발견 문장"
                        font.bold: true
                    }

                    TextArea {
                        id: manualDescriptionInput
                        Layout.fillWidth: true
                        Layout.preferredHeight: 110
                        wrapMode: TextEdit.Wrap
                    }

                    Label {
                        text: "최종 개선안"
                        font.bold: true
                    }

                    TextArea {
                        id: manualActionInput
                        Layout.fillWidth: true
                        Layout.preferredHeight: 110
                        placeholderText: "현장에서 필요한 개선안을 입력하세요."
                        wrapMode: TextEdit.Wrap
                    }

                    Button {
                        Layout.fillWidth: true
                        text: "수동 입력 저장"

                        enabled: !captureController.working
                                 && !captureController.reviewSaving
                                 && manualDescriptionInput.text.trim().length > 0
                                 && manualActionInput.text.trim().length > 0

                        onClicked: captureController.saveManual(
                            manualDescriptionInput.text,
                            manualActionInput.text
                        )
                    }

                    Button {
                        Layout.fillWidth: true
                        text: "수동 입력 접기"
                        onClicked: page.manualMode = false
                    }
                }
            }
        }
    }
}