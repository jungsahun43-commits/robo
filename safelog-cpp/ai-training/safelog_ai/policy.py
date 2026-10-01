from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    model: str = ""
    box: tuple[float, float, float, float] | None = None


RISK = {
    "no_helmet": (4, "안전모 미착용", "작업을 중지하고 규격에 맞는 안전모를 착용하세요."),
    "no_gloves": (3, "안전장갑 미착용", "작업에 맞는 안전장갑을 착용하세요."),
    "no_boots": (4, "안전화 미착용", "작업 전 안전화를 착용하세요."),
    "no_goggle": (3, "보안경 미착용", "비산물 위험에 맞는 보안경을 착용하세요."),
    "fire": (5, "화재", "즉시 작업을 중지하고 비상 절차에 따라 대피·신고하세요."),
    "smoke": (5, "연기", "발생원을 확인하지 말고 우선 대피한 뒤 관리자에게 신고하세요."),
    "blocked_aisle": (4, "통로 장애물", "통로의 자재와 장애물을 지정 보관구역으로 이동하세요."),
    "exposed_cable": (4, "노출 전선", "전원을 차단하고 자격을 갖춘 담당자가 절연·정리하도록 하세요."),
}


def analyze_detections(detections: list[Detection]) -> dict[str, object]:
    hazards = [detection for detection in detections if detection.label in RISK]
    if not hazards:
        confidence = max((d.confidence for d in detections), default=0.0)
        return {
            "category": "명확한 위험 미탐지",
            "risk_level": 1,
            "hazards": ["학습된 위험이 명확히 탐지되지 않음"],
            "description": "현재 모델이 명확한 안전 위반을 찾지 못했습니다. 점검자가 사진 전체를 확인해야 합니다.",
            "action": "AI 결과와 관계없이 현장 점검표에 따라 직접 확인하세요.",
            "confidence": confidence,
        }

    hazards.sort(key=lambda item: (RISK[item.label][0], item.confidence), reverse=True)
    primary = hazards[0]
    risk_level, category, _ = RISK[primary.label]
    unique_categories = list(dict.fromkeys(RISK[item.label][1] for item in hazards))
    return {
        "category": category,
        "risk_level": risk_level,
        "hazards": unique_categories,
        "description": "사진에서 " + ", ".join(unique_categories) + " 위험이 탐지되었습니다.",
        "action": " ".join(dict.fromkeys(RISK[item.label][2] for item in hazards)),
        "confidence": primary.confidence,
    }


def likely_resolved(before: list[Detection], after: list[Detection]) -> tuple[bool, list[str], float]:
    before_hazards = {item.label: item.confidence for item in before if item.label in RISK}
    after_hazards = {item.label: item.confidence for item in after if item.label in RISK}
    remaining = [RISK[label][1] for label in after_hazards]
    resolved = bool(before_hazards) and not after_hazards
    confidence = max(before_hazards.values(), default=0.5) if resolved else max(after_hazards.values(), default=0.5)
    return resolved, remaining, confidence
