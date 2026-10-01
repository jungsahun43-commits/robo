from __future__ import annotations

import base64
import binascii
import io
import os
from dataclasses import asdict
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".config" / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".config" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config" / "matplotlib"))
for variable in ("YOLO_CONFIG_DIR", "TORCH_HOME", "MPLCONFIGDIR"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .policy import Detection, analyze_detections, likely_resolved

MODEL_ENVIRONMENTS = (
    ("SAFELOG_MODEL_PATH", "runs/ppe-baseline/weights/best.pt"),
    ("SAFELOG_FIRE_MODEL_PATH", ""),
    ("SAFELOG_AUX_MODEL_PATH", ""),
)
_configured_paths = [value for key, default in MODEL_ENVIRONMENTS if (value := os.environ.get(key, default))]
_configured_paths += [value for value in os.environ.get("SAFELOG_AUX_MODEL_PATHS", "").split(os.pathsep) if value]
MODEL_PATHS = list(dict.fromkeys((Path(value) if Path(value).is_absolute() else ROOT / value).resolve() for value in _configured_paths))


def model_id(path: Path) -> str:
    return path.parents[1].name if path.parent.name == "weights" else path.stem


MODEL_NAME = os.environ.get(
    "SAFELOG_MODEL_NAME",
    "+".join(model_id(path) for path in MODEL_PATHS),
)
CONFIDENCE = float(os.environ.get("SAFELOG_MODEL_CONFIDENCE", "0.25"))
MAX_IMAGE_BYTES = 8 * 1024 * 1024

app = FastAPI(title="SafeLog trained AI", version="1.0.0")
_models: list[YOLO] | None = None
_model_lock = RLock()


class HazardRequest(BaseModel):
    image: str
    userMemo: str = ""
    promptVersion: str = Field(min_length=1)


class ComparisonRequest(BaseModel):
    beforeImage: str
    afterImage: str
    actionNote: str = ""
    promptVersion: str = Field(min_length=1)


class SummaryRequest(BaseModel):
    inspectionContext: str = Field(min_length=1)
    promptVersion: str = Field(min_length=1)


def models() -> list[YOLO]:
    global _models
    with _model_lock:
        if _models is None:
            missing = [path for path in MODEL_PATHS if not path.exists()]
            if missing:
                raise HTTPException(503, "학습 모델이 준비되지 않았습니다.")
            _models = [YOLO(str(path)) for path in MODEL_PATHS]
        return _models


def decode_image(value: str) -> tuple[bytes, str]:
    if not value.startswith("data:image/") or ";base64," not in value:
        raise HTTPException(400, "image data URL이 필요합니다.")
    header, encoded = value.split(",", 1)
    extension = ".png" if "png" in header else ".jpg"
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(400, "이미지 base64가 올바르지 않습니다.") from error
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "이미지 크기는 0보다 크고 8MB 이하여야 합니다.")
    return data, extension


def detect(value: str) -> list[Detection]:
    data, _ = decode_image(value)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 32_000_000:
                raise HTTPException(413, "이미지를 3200만 픽셀 이하로 줄여 주세요.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = np.asarray(image)[:, :, ::-1].copy()  # YOLO expects BGR numpy input.
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "이미지를 읽을 수 없습니다.") from error
    detections: list[Detection] = []
    with _model_lock:
        for model_path, detector in zip(MODEL_PATHS, models()):
            result = detector.predict(pixels, conf=CONFIDENCE, verbose=False)[0]
            names = result.names
            detections.extend(
                Detection(str(names[int(class_id)]), float(confidence), model_id(model_path), tuple(map(float, box)))
                for class_id, confidence, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
            )
    return detections


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if all(path.exists() for path in MODEL_PATHS) else "unavailable",
        "model": MODEL_NAME,
        "models": [str(path) for path in MODEL_PATHS],
        "modelExists": all(path.exists() for path in MODEL_PATHS),
    }


@app.post("/v1/analyze-hazard")
def analyze_hazard(request: HazardRequest) -> dict[str, object]:
    detections = detect(request.image)
    result = analyze_detections(detections)
    return {
        "hazardCategory": result["category"],
        "riskLevel": result["risk_level"],
        "detectedHazards": result["hazards"],
        "suggestedDescription": result["description"],
        "suggestedAction": result["action"],
        "confidence": result["confidence"],
        "modelName": MODEL_NAME,
        "promptVersion": request.promptVersion,
        "detections": [asdict(item) for item in detections],
        "requiresHumanReview": True,
    }


@app.post("/v1/compare-action")
def compare_action(request: ComparisonRequest) -> dict[str, object]:
    resolved, remaining, confidence = likely_resolved(
        detect(request.beforeImage), detect(request.afterImage)
    )
    assessment = (
        "학습 모델에서 기존 위험 객체가 더 이상 탐지되지 않았습니다. 점검자의 현장 확인이 필요합니다."
        if resolved else "조치 후 사진에 위험이 남아 있거나 조치 전 위험을 확인하지 못했습니다."
    )
    return {
        "likelyResolved": resolved,
        "remainingRisks": remaining,
        "assessment": assessment,
        "confidence": confidence,
        "modelName": MODEL_NAME,
        "promptVersion": request.promptVersion,
    }


@app.post("/v1/summarize")
def summarize(request: SummaryRequest) -> dict[str, object]:
    lines = [line.strip() for line in request.inspectionContext.splitlines() if line.strip()]
    return {
        "summary": f"총 {len(lines)}건의 점검 기록을 검토했습니다. 각 조치 결과는 점검자가 최종 확인해야 합니다.",
        "keyRisks": lines[:5] or ["기록 없음"],
        "modelName": "safelog-summary-template-v1",
        "promptVersion": request.promptVersion,
    }
