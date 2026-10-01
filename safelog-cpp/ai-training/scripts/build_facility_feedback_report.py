from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_facility_report import LABELS_KO, percentage


def main():
    validation = json.loads((ROOT / "reports/facility-feedback-validation.json").read_text())
    test = json.loads((ROOT / "reports/facility-feedback-test.json").read_text())
    training = json.loads((ROOT / "runs/facility-presence-refined/TRAINING.json").read_text())
    for source, target in (("TRAINING.json", "facility-feedback-training.json"), ("history.json", "facility-feedback-training-history.json")):
        shutil.copyfile(ROOT / "runs/facility-presence-refined" / source, ROOT / "reports" / target)
    lines = ["# SafeLog 시설 AI — 2차 피드백 보강", "",
        "## 목적과 방법", "",
        "1차에서 늘어난 오탐과 개선되지 않은 항목을 확인하고, 같은 학습 사진 6,225장으로 사진 분류 모델을 추가 학습했다.",
        "1차 best.pt에서 시작해 입력을 384→512px로 키웠다. 양성 가중치 BCE 대신, 잘못 높은 점수를 주는 음성 항목의 BCE 손실을 최대 3배 강조했다.",
        "검증/시험 사진은 optimizer에 입력하지 않았다. 원본 라벨도 바꾸지 않았다. 새 데이터가 추가된 실험은 아니다.",
        f"학습 {training['actual_epochs']} epoch, 가장 좋은 검증 사진 분류 macro AP {percentage(training['best_val_macro_average_precision'])}. 박스 mAP·현장 안전 정확도가 아니다.",
        "항목별 후보 임계값을 검증 사진 710장에서 탐색했다. 1차 대비 사진 미탐률·오탐률이 모두 증가하지 않는 후보 중 FNR+FPR이 최소인 모델을 선택했다.",
        "동률이거나 나아진 후보가 없으면 기존 항목을 유지한다. 선택되지 않은 모델의 항목은 기준 1.00으로 비활성화한다.",
        "선택된 모델과 항목별 기준을 고정한 뒤 기존 시험 사진 975장을 재평가했다. 시험 결과가 나빠진 항목도 공개하며 시험 결과로 설정을 다시 고르지 않았다.", "",
        "## 검증에서의 선택", "",
        "| 항목 | 선택 모델 | 기준 | 1차 검증 미탐 | 2차 검증 미탐 | 1차 검증 오탐 | 2차 검증 오탐 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for label, item in validation["per_class"].items():
        old, selected = item["round1"], item["selected"]
        new = selected["metrics"]
        lines.append(f"| {LABELS_KO[label]} | {selected['source']} | {selected['threshold']:.2f} | {percentage(old['miss_fraction'])} | {percentage(new['miss_fraction'])} | {percentage(old['false_positive_rate'])} | {percentage(new['false_positive_rate'])} |")
    lines += ["", "## 같은 시험 사진에서의 비교", "",
        "항목의 존재 여부만 평가한다. 박스 위치와 현장 전체 안전/불안전 정확도는 평가하지 않는다.",
        "미탐률=FN/(TP+FN), 오탐률=FP/(FP+TN). 항목별 양성/음성 사진을 분모로 사용한다.", "",
        "| 항목 | 1차 미탐 | 2차 미탐 | 1차 오탐 | 2차 오탐 | 추가 회수/놓친 양성 | 제거/추가 오탐 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for label, item in test["per_class"].items():
        old, new = item["round1"], item["round2"]
        lines.append(f"| {LABELS_KO[label]} | {percentage(old['miss_fraction'])} | {percentage(new['miss_fraction'])} | {percentage(old['false_positive_rate'])} | {percentage(new['false_positive_rate'])} | {item['recovered_positives']}/{item['lost_positives']} | {item['removed_false_positives']}/{item['new_false_positives']} |")
    release_path = ROOT / "reports/facility-feedback-release.json"
    if release_path.exists():
        decision = json.loads(release_path.read_text())
        counts = decision["label_photo_counts"]
        state = "기존 1차 모델을 기본 설정으로 유지했다. 2차 후보는 기본 모델로 채택하지 않았다." if decision["status"] == "kept_round1" else "2차 후보를 기본 설정으로 채택했다."
        lines[2:2] = ["## 이번 출시 판단", "", state,
            f"항목-사진 쌍 합계에서 미탐은 {counts['round1_fn']}→{counts['round2_fn']}, 오탐은 {counts['round1_fp']}→{counts['round2_fp']}였다. 같은 사진에 여러 항목이 있을 수 있어 고유 사진 수가 아니다.",
            "후보의 항목·기준은 검증으로 고정했다. 시험 오류를 보고 기준을 다시 조정하지 않고 전체 후보의 교체 여부를 판단했다.",
            "시험 분할을 출시 판단에도 사용했으므로 독립적인 성능 보증 자료는 아니다. 실제 현장 성능 주장을 위해서는 새로운 외부 평가가 필요하다.", ""]
    lines += ["", "## 해석 및 다음 피드백", "",
        "- 검증에서 지킨 오차 제한은 시험·실제 현장에서 보장되지 않는다. 검증의 작은 차이가 시험에서 반대로 나올 수 있다.",
        "- 동일 검증·시험 분할을 반복 확인하고 있어 독립적인 외부 검증으로 해석할 수 없다. 장면 ID가 없어 사진 간 장면 독립성도 보장하지 못한다.",
        "- 검증 오류 예시에서 물기·백화·녹 얼룩의 외관이 비슷해 보이는 사진과 작은 정답 영역이 보였다. 라벨에 없는 항목을 오탐으로 집계하지만 원본 주석의 완전성을 별도로 확인한 것은 아니다.",
        "- 모델 출력으로 원본 라벨을 임의 수정하지 않았다. 현장 점검자가 판단한 정상/손상 사례를 확보해 다음 외부 평가 자료로 쓰는 것이 필요하다.",
        "- 기존 시설 박스 모델·화재·PPE는 변경하지 않았다. 금속 부식은 정상 시험 사진이 없는 한계를 유지한다.",
        "- `box=null`, `evidence_scope=photo_presence`인 의견은 위치를 찾은 결과가 아니다. 사람의 최종 확인이 필요하다.", "",
        "## 팀원 실행", "", "최신 feature/ai-engine 코드와 모델 ZIP을 함께 받은 뒤 기존처럼 실행한다.", "",
        "```powershell", "./start_ai_server.ps1 -FacilitiesOnly", "```", "",
        "`/health.photoClassifiers`에서 활성 모델·해상도·항목별 기준을 확인한다. 기본 설정은 facility-inference-profile.json, 1차 설정은 facility-inference-profile-round1.json, 실험 후보는 facility-inference-profile-round2-candidate.json이다.",
        "실험 후보를 기본 설정으로 복사해 사용하지 않는다. 기본 서버는 출시 판단을 통과한 설정을 사용한다.", "",
        "## 보강 재현", "", "```powershell",
        "./.venv/Scripts/python.exe scripts/train_facility_presence.py --name facility-presence-refined --initial runs/facility-presence/best.pt --hard-negatives --imgsz 512 --batch 12 --epochs 18 --patience 5",
        "./.venv/Scripts/python.exe scripts/refine_facility_feedback.py select",
        "./.venv/Scripts/python.exe scripts/refine_facility_feedback.py test",
        "./.venv/Scripts/python.exe scripts/refine_facility_feedback.py export",
        "./.venv/Scripts/python.exe scripts/refine_facility_feedback.py release",
        "./.venv/Scripts/python.exe scripts/build_facility_feedback_report.py",
        "./.venv/Scripts/python.exe scripts/package_models.py --require-facilities --require-optimization", "```", "",
        "이미 존재하는 학습 실험은 덮어쓰지 않는다. 순서대로 검증 선택 → 고정 설정의 시험 평가 → 모델 전달을 수행한다.", ""]
    (ROOT / "reports/FACILITY_FEEDBACK_KO.md").write_text("\n".join(lines), encoding="utf-8")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    labels = list(test["per_class"]); x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for axis, key, title in zip(axes, ("miss_fraction", "false_positive_rate"), ("사진 미탐률", "사진 오탐률")):
        old = [test["per_class"][label]["round1"][key] * 100 for label in labels]
        new = [test["per_class"][label]["round2"][key] * 100 for label in labels]
        b1 = axis.bar(x-.18,old,.36,label="1차 보강",color="#8795a5")
        b2 = axis.bar(x+.18,new,.36,label="2차 피드백 보강",color="#2878b5")
        axis.bar_label(b1, fmt="%.1f",padding=3,fontsize=9); axis.bar_label(b2,fmt="%.1f",padding=3,fontsize=9)
        axis.set_xticks(x,[LABELS_KO[label] for label in labels]); axis.set_ylabel("비율 (%)")
        axis.set_title(title + " — 낮을수록 좋음",loc="left",pad=14)
        axis.set_ylim(0,max(old+new)*1.25+3); axis.legend(frameon=False)
        axis.grid(axis="y",alpha=.2); axis.set_axisbelow(True); axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("시설 AI 2차 피드백 비교 | 기존 시험 사진 975장",fontsize=17)
    fig.text(.02,.01,"같은 원본 시험 분할의 재평가입니다. 사진의 항목 존재 여부만 평가하며 결함 위치·현장 안전 정확도는 아닙니다.\n검증에서 선택한 뒤 시험에 적용했습니다. 검증의 오차 제한을 실제 현장에 보장하지 않습니다.",fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95)); fig.savefig(ROOT / "artifacts/facility-feedback-comparison.png",dpi=150)


if __name__ == "__main__": main()
