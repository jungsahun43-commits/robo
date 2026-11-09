import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Page {
    id: page

    title: "검토 결과 저장"

    property string findingId: ""
    property string finalDescription: ""
    property string finalActionOpinion: ""
    property string reviewDecision: ""

    readonly property bool knownDecision:
        ["Accepted", "Edited", "Rejected", "Manual"]
            .indexOf(reviewDecision) >= 0

    readonly property bool hasSavedResult:
        findingId.length > 0
        && finalDescription.trim().length > 0
        && knownDecision
        && (reviewDecision === "Rejected"
            || finalActionOpinion.trim().length > 0)

    readonly property string decisionLabel: {
        if (reviewDecision === "Accepted")
            return "AI 제안 채택"

        if (reviewDecision === "Edited")
            return "사용자 수정 후 저장"

        if (reviewDecision === "Rejected")
            return "AI 제안 거절 · 원문 유지"

        if (reviewDecision === "Manual")
            return "수동 입력으로 저장"

        return "저장 결과 확인 중"
    }

    signal homeRequested()

    // 다음 단계에서 저장에 성공한 결과만 전달한다.
    function showSaved(id, description, actionOpinion, decision) {
        page.findingId = id
        page.finalDescription = description
        page.finalActionOpinion = actionOpinion
        page.reviewDecision = decision
    }

    background: Rectangle {
        color: "#f4f7f6"
    }

    ScrollView {
        id: savedScroll

        anchors.fill: parent
        anchors.margins: 20
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: savedScroll.availableWidth
            spacing: 18

            Label {
                Layout.fillWidth: true

                text: page.hasSavedResult
                      ? "기록 저장 완료"
                      : "저장 결과를 기다리고 있습니다."

                font.pixelSize: 25
                font.bold: true
                color: "#126b58"
                wrapMode: Text.Wrap
            }

            ColumnLayout {
                Layout.fillWidth: true
                visible: page.hasSavedResult
                spacing: 16

                Label {
                    text: "	저장 방식"
                    font.bold: true
                    color: "#17212b"
                }

                Label {
                    Layout.fillWidth: true
                    text: page.decisionLabel
                    wrapMode: Text.Wrap
                    color: "#126b58"
                }

                Label {
                    text: "기록 ID"
                    font.bold: true
                    color: "#17212b"
                }

                Label {
                    Layout.fillWidth: true
                    text: page.findingId
                    wrapMode: Text.Wrap
                }

                Label {
                    text: "최종 발견 문장"
                    font.bold: true
                    color: "#17212b"
                }

                Label {
                    Layout.fillWidth: true
                    text: page.finalDescription
                    wrapMode: Text.Wrap
                }

                Label {
                    text: "최종 개선안"
                    font.bold: true
                    color: "#17212b"
                }

                Label {
                    Layout.fillWidth: true

                    text: page.finalActionOpinion.trim().length > 0
                          ? page.finalActionOpinion
                          : "작성된 개선안이 없습니다."

                    wrapMode: Text.Wrap
                }

                Label {
                    Layout.fillWidth: true
                    visible: page.reviewDecision === "Rejected"

                    text: "AI 제안을 거절하여 처음 입력한 원문을 유지했습니다."

                    wrapMode: Text.Wrap
                    color: "#526069"
                }

                Label {
                    Layout.fillWidth: true

                    text: "현장 기록이 저장되었습니다. "
                          + "담당자의 조치와 점검자의 최종 확인은 이후에 진행합니다."

                    wrapMode: Text.Wrap
                    color: "#526069"
                }
            }

            Button {
                Layout.fillWidth: true
                text: "처음 화면으로"
                enabled: page.hasSavedResult

                onClicked: page.homeRequested()
            }
        }
    }
}