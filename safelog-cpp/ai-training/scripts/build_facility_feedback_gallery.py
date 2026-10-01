"""Validation-only error examples; source labels are never changed."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import textwrap

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scripts.optimize_facility_presence import baseline_presence


def main():
    profile = json.loads((ROOT / "reports/facility-inference-profile-round1.json").read_text())
    dacl = profile["models"]["facility-dacl-optimized"]
    score = json.loads((ROOT / "runs/presence-scores-val.json").read_text())
    detector = json.loads((ROOT / f"runs/proposals-{dacl['source_run']}-val-{dacl['imgsz']}.json").read_text())
    by_name = {item["image"]: item for item in detector["images"]}
    targets, probabilities = np.array(score["targets"], dtype=bool), np.array(score["probabilities"])
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    manifest = []
    for row, label in enumerate(("wet_surface", "surface_cavity", "efflorescence")):
        i = score["classes"].index(label)
        base = baseline_presence(detector, i, dacl["thresholds"][label], score["images"])
        cutoff = profile["photo_classifier"]["thresholds"][label]
        prediction = base | ((probabilities[:, i] >= cutoff) & (cutoff < 1))
        for col, mode in enumerate(("FP", "FN")):
            indices = np.where((~targets[:, i] & prediction) if mode == "FP" else (targets[:, i] & ~prediction))[0]
            indices = sorted(indices, key=lambda n: probabilities[n, i], reverse=True)[:3]
            for slot in range(3):
                axis = axes[row, col * 3 + slot]
                axis.axis("off")
                if slot >= len(indices): continue
                n = indices[slot]; name = score["images"][n]
                with Image.open(ROOT / "data/dacl10k-yolo/images/val" / name) as image:
                    axis.imshow(image)
                item = by_name[name]
                for cls, box in zip(item["target_classes"], item["target_boxes"]):
                    if cls == i:
                        x1, y1, x2, y2 = box
                        axis.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor="red", linewidth=1))
                labels = [score["classes"][c] for c in np.where(targets[n])[0]]
                caption = textwrap.fill("GT: " + ", ".join(labels), width=36)
                axis.set_title(f"{label} {mode} | 분류 점수 {probabilities[n,i]:.2f}\n{caption}", fontsize=7)
                manifest.append({"label": label, "error": mode, "image": name, "classifier_score": float(probabilities[n,i]), "source_labels": labels})
    fig.suptitle("1차 모델 검증 오류 예시 | 빨간 박스는 해당 항목의 정답 영역", fontsize=16)
    fig.text(.01,.01,"항목별 FP는 원본 라벨에 해당 항목이 없는 사진의 제안입니다. 다른 결함이 있을 수 있습니다. 라벨은 변경하지 않았습니다.",fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.95))
    fig.savefig(ROOT / "artifacts/facility-validation-errors.png", dpi=120)
    (ROOT / "runs/facility-validation-errors.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
