import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Page {
    id: page

    title: "AI 위험 제안 검토"

    property string originalMemo: ""
    property string hazardCategory: ""
    property int riskLevel: 0
    property string suggestedDescription: ""
    property string suggestedAction: ""

    property bool editing: false
    property bool saving: false
    property string errorMessage: ""

    readonly property bool suggestionReady:
        hazardCategory.trim().length > 0
        && riskLevel >= 1
        && riskLevel <= 5
        && suggestedDescription.trim().length > 0
        && suggestedAction.trim().length > 0

    readonly property bool editedTextValid:
        descriptionInput.text.trim().length > 0
        && actionInput.text.trim().length > 0

    readonly property bool textChanged:
        descriptionInput.text.trim() !== suggestedDescription.trim()
        || actionInput.text.trim() !== suggestedAction.trim()

    // 다음 단계에서 C++의 검토 저장 함수와 연결한다.
    signal reviewRequested(
        string decision,
        string description,
        string actionOpinion
    )

    // C++에서 검증한 AI 결과를 이 화면에 전달할 때 사용한다.
    function showSuggestion(category, risk, description, action, memo) {
        hazardCategory = category
        riskLevel = risk
        suggestedDescription = description
        suggestedAction = action
        originalMemo = memo

        descriptionInput.text = description
        actionInput.text = action
        editing = false
    }

    function startEditing() {
        descriptionInput.text = suggestedDescription
        actionInput.text = suggestedAction
        editing = true
    }

    background: Rectangle {
        color: "#f4f7f6"
    }

    ScrollView {
        id: reviewScroll

        anchors.fill: parent
        anchors.margins: 20
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: reviewScroll.availableWidth
            spacing: 16

            Label {
                text: "AI 위험 제안 검토"
                font.pixelSize: 25
                font.bold: true
                color: "#17212b"
            }

            Label {
                Layout.fillWidth: true
                text: "AI 제안을 확인하고 채택, 수정 또는 거절해 주세요."
                wrapMode: Text.Wrap
                color: "#526069"
            }

            Label {
                Layout.fillWidth: true
                visible: !page.suggestionReady
                text: "아직 검토할 AI 제안이 없습니다."
                wrapMode: Text.Wrap
                color: "#526069"
            }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 14
                visible: page.suggestionReady
                enabled: !page.saving

                Label {
                    text: "처음 입력한 메모"
                    font.bold: true
                }

                Label {
                    Layout.fillWidth: true
                    text: page.originalMemo
                    wrapMode: Text.Wrap
                    color: "#526069"
                }

                Label {
                    Layout.fillWidth: true
                    text: "위험 종류: " + page.hazardCategory
                    font.bold: true
                    wrapMode: Text.Wrap
                }

                Label {
                    text: "위험 등급: " + page.riskLevel + " / 5"
                    font.bold: true
                }

                Label {
                    text: "AI가 제안한 발견 문장"
                    font.bold: true
                }

                Label {
                    Layout.fillWidth: true
                    text: page.suggestedDescription
                    wrapMode: Text.Wrap
                }

                Label {
                    text: "AI가 제안한 개선안"
                    font.bold: true
                }

                Label {
                    Layout.fillWidth: true
                    text: page.suggestedAction
                    wrapMode: Text.Wrap
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    visible: page.editing
                    spacing: 10

                    Label {
                        text: "최종 발견 문장 수정"
                        font.bold: true
                    }

                    TextArea {
                        id: descriptionInput
                        Layout.fillWidth: true
                        Layout.preferredHeight: 110
                        wrapMode: TextEdit.Wrap
                        placeholderText: "최종 발견 문장을 입력하세요."
                    }

                    Label {
                        text: "최종 개선안 수정"
                        font.bold: true
                    }

                    TextArea {
                        id: actionInput
                        Layout.fillWidth: true
                        Layout.preferredHeight: 110
                        wrapMode: TextEdit.Wrap
                        placeholderText: "최종 개선안을 입력하세요."
                    }

                    Button {
                        Layout.fillWidth: true
                        text: "수정한 내용으로 저장"
                        enabled: page.editedTextValid && page.textChanged

                        onClicked: page.reviewRequested(
                            "Edited",
                            descriptionInput.text.trim(),
                            actionInput.text.trim()
                        )
                    }

                    Button {
                        Layout.fillWidth: true
                        text: "수정 취소"
                        onClicked: page.editing = false
                    }
                }

                Button {
                    Layout.fillWidth: true
                    visible: !page.editing
                    text: "AI 제안 그대로 채택"

                    onClicked: page.reviewRequested(
                        "Accepted",
                        page.suggestedDescription,
                        page.suggestedAction
                    )
                }

                Button {
                    Layout.fillWidth: true
                    visible: !page.editing
                    text: "문장 수정하기"
                    onClicked: page.startEditing()
                }

                Button {
                    Layout.fillWidth: true
                    text: "AI 제안 거절 · 원문 유지"

                    onClicked: page.reviewRequested(
                        "Rejected",
                        "",
                        ""
                    )
                }
            }

            BusyIndicator {
                Layout.alignment: Qt.AlignHCenter
                visible: page.saving
                running: page.saving
            }

            Label {
                Layout.fillWidth: true
                visible: page.saving
                text: "검토 결과를 저장하고 있습니다."
                wrapMode: Text.Wrap
            }

            Label {
                Layout.fillWidth: true
                visible: page.errorMessage.length > 0
                text: page.errorMessage
                color: "#b42318"
                wrapMode: Text.Wrap
            }

            Label {
                Layout.fillWidth: true
                text: "AI 결과는 참고 제안이며, 최종 판단은 점검자가 합니다."
                wrapMode: Text.Wrap
                font.pixelSize: 12
                color: "#68766f"
            }
        }
    }
}