from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    model: str = ""
    box: tuple[float, float, float, float] | None = None
    evidence_scope: str = "region_box"


RISK = {
    "no_helmet": (4, "안전모 미착용", "작업을 중지하고 규격에 맞는 안전모를 착용하세요."),
    "no_gloves": (3, "안전장갑 미착용", "작업에 맞는 안전장갑을 착용하세요."),
    "no_boots": (4, "안전화 미착용", "작업 전 안전화를 착용하세요."),
    "no_goggle": (3, "보안경 미착용", "비산물 위험에 맞는 보안경을 착용하세요."),
    "fire": (5, "화재", "즉시 작업을 중지하고 비상 절차에 따라 대피·신고하세요."),
    "smoke": (5, "연기", "발생원을 확인하지 말고 우선 대피한 뒤 관리자에게 신고하세요."),
    "blocked_aisle": (4, "통로 장애물", "통로의 자재와 장애물을 지정 보관구역으로 이동하세요."),
    "exposed_cable": (4, "노출 전선", "전원을 차단하고 자격을 갖춘 담당자가 절연·정리하도록 하세요."),
    "surface_crack": (3, "표면 균열 의심", "표면의 균열 위치와 폭·변화를 현장에서 확인하고 시설 담당자에게 점검을 요청하세요."),
    "concrete_spalling": (4, "콘크리트 박리 의심", "박리 부위 아래의 접근을 제한하고 탈락 가능성을 시설 담당자가 확인하도록 하세요."),
    "rust_stain": (3, "녹물·녹 얼룩 의심", "녹 얼룩의 발생원과 주변 철근 노출 여부를 시설 담당자가 확인하도록 하세요."),
    "exposed_rebar": (4, "철근 노출 의심", "노출 부위 접근을 제한하고 철근 상태와 보수 필요성을 시설 담당자가 확인하도록 하세요."),
    "wet_surface": (2, "젖은 표면 의심", "표면의 물기와 미끄럼 위험을 확인하고 유입 원인을 점검하세요. 사진만으로 배관 누수를 확정하지 마세요."),
    "efflorescence": (2, "백화 의심", "표면 백화와 수분 유입 경로를 현장에서 확인하고 경과를 기록하세요."),
    "surface_cavity": (3, "표면 공동·파임 의심", "표면 결손의 크기와 주변 손상을 시설 담당자가 확인하도록 하세요."),
    "metal_corrosion": (3, "금속 부식 의심", "부식 위치를 기록하고 시설 담당자가 두께·단면 손실과 보수 필요성을 확인하도록 하세요."),
}
# Historical training key combines Crack and Alligator Crack; it is not a material classifier.
RISK["concrete_crack"] = RISK["surface_crack"]


def canonical_label(label: str) -> str:
    return "surface_crack" if label == "concrete_crack" else label


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
    description = "사진에서 " + ", ".join(unique_categories) + " 항목이 탐지되었습니다. 점검자의 현장 확인이 필요합니다."
    if any(item.evidence_scope == "photo_presence" for item in hazards):
        description += " 사진 전체 분류의 추가 의견은 결함 위치를 확정하지 않으므로 사진과 현장을 직접 확인하세요."
    return {
        "category": category,
        "risk_level": risk_level,
        "hazards": unique_categories,
        "description": description,
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
