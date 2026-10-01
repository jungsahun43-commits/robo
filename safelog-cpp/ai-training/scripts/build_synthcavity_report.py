"""Report the actual third experiment and its acceptance result, including failures."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_facility_report import LABELS_KO, percentage


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def main():
    reports = ROOT / "reports"
    run = ROOT / "runs/facility-presence-synthetic"
    training, audit = read(run / "TRAINING.json"), read(reports / "facility-round3-data-audit.json")
    validation, decision = read(reports / "facility-round3-validation.json"), read(reports / "facility-round3-release.json")
    adopted = decision["status"] == "adopted_round3"
    lines = ["# SafeLog 시설 AI — 3차 합성 공동 보강", "", "## 적용 결과", "",
             "3차 후보를 기본 모델에 반영했다." if adopted else "3차 후보는 기본 모델로 채택하지 않았다. 기존 1차 facility-validation-v2를 유지한다.",
             f"판단: {decision['reason']}", "",
             "실험 프로필은 facility-inference-profile-round3-candidate.json이다. 기본 프로필과 분리해 시험했고 교체 판단 전에는 기본 서버에 적용하지 않았다.",
             "## 사용한 데이터", "",
             "[공식 synth-dacl 자료](https://doi.org/10.60776/9D6E4M)의 synthcavity 두 원본 ZIP을 발행처 MD5로 확인했다. 이용 조건은 CC BY-NC 4.0이다.",
             f"실제 학습 사진 {training['train_images']:,}장 + 합성 학습 사진 {audit['supplemental_images']:,}장. 실제 검증은 {training['val_images']:,}장이다.",
             f"합성 공동 양성 {audit['positive_cavity_photos']:,}장, 음성 {audit['negative_cavity_photos']:,}장. 중복 픽셀 제외 {audit['duplicate_pixels_removed']}장, 실제 원본 분할과 일치 {audit['overlap_with_dacl_train_val_test']}장.",
             f"절차 생성 장면은 {audit['procedural_scenes']:,}개다. render/render_noise 변형은 같은 장면에 속하며 독립적인 촬영 현장이 아니다.",
             "합성 자료는 별도의 실제 현장 사진이나 독립적인 평가 자료가 아니다. 실제 자료와의 중복 검사는 원본 픽셀 SHA256 기준이다.",
             "render/render_noise 사진과 gt_cavity OR gt_render_cavities 마스크를 대응시켰다. 여섯 항목은 정답이 없는 -1로 보관하고 손실에서 제외했다.",
             "물기 신규 현장 데이터는 이번 학습에 추가하지 않았다. 합성 공동 자료로 누수 인식 성능이 입증됐다고 해석하지 않는다.", "",
             "## 학습", "",
             f"기존 1차 분류기로 초기화해 512px, 배치 12, seed 42, 최대 {training['requested_epochs']} epoch에서 실제 {training['actual_epochs']} epoch 수행했다.",
             f"합성 사진 손실 가중치는 {training['supplemental_weight']}다. 실제 자료는 기존 양성 가중치 BCE, 합성 자료는 공동 항목의 BCE만 사용했다.",
             f"최고 실제 검증 사진 분류 macro AP는 {training['best_val_macro_average_precision'] * 100:.1f}%다. 박스 mAP나 현장 안전 정확도가 아니다.",
             "검증/시험 사진은 optimizer에 넣지 않았고 원본 라벨은 수정하지 않았다. 체크포인트 선택에는 실제 검증 macro AP를 사용했다.", "",
             "## 검증 선택", "",
             "교체 기준은 학습·시험 전에 FACILITY_ROUND3_PLAN_KO.md로 고정했다. 물기/공동만 후보로 검토했고 다른 다섯 항목은 유지했다.",
             "오탐률이 증가하지 않고 미탐률이 최소 2%p 줄어든 기준만 자격을 부여했다. 동률에는 높은 기준을 선택했다.", "",
             "| 항목 | 후보 자격 | 적용 기준 | 1차 미탐 | 선택 후 미탐 | 1차 오탐 | 선택 후 오탐 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, item in validation["per_class"].items():
        old, new = item["round1"], item["selected"]
        lines.append(f"| {LABELS_KO[label]} | {'통과' if item['eligible'] else '미통과·기존 유지'} | {item['threshold']:.2f} | {percentage(old['miss_fraction'])} | {percentage(new['miss_fraction'])} | {percentage(old['false_positive_rate'])} | {percentage(new['false_positive_rate'])} |")
    lines += ["", "후보 자체의 진단값(기존 오탐률 이내에서 FNR+FPR이 가장 작은 기준):", ""]
    for label, item in validation["per_class"].items():
        diagnostic = item["candidate_under_fpr_cap"]
        if diagnostic:
            metric = diagnostic["metrics"]
            lines.append(f"- {LABELS_KO[label]}: 기준 {diagnostic['threshold']:.2f}, 검증 미탐 {percentage(metric['miss_fraction'])}, 오탐 {percentage(metric['false_positive_rate'])}. 후보 자격과 실제 적용 여부는 위 표를 따른다.")
        else:
            lines.append(f"- {LABELS_KO[label]}: 탐색한 기준 중 기존 오탐률 제한을 만족하는 새 후보가 없었다.")
    if validation["eligible_heads"]:
        test = read(reports / "facility-round3-test.json")
        lines += ["", "## 고정 후보의 시험 비교", "", f"기존 시험 사진 {test['images']}장을 재평가했다. 같은 사진에서 여러 항목을 평가하므로 집계 건수는 항목-사진 쌍이다.",
                  "미탐률=FN/(TP+FN), 오탐률=FP/(FP+TN). 항목 존재만 평가했고 박스 위치나 사진 전체 안전을 평가하지 않았다.", "",
                  "| 항목 | 1차 미탐 | 3차 미탐 | 1차 오탐 | 3차 오탐 | 회수/놓친 양성 | 제거/추가 오탐 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label, item in test["per_class"].items():
            old, new = item["round1"], item["round3"]
            lines.append(f"| {LABELS_KO[label]} | {percentage(old['miss_fraction'])} | {percentage(new['miss_fraction'])} | {percentage(old['false_positive_rate'])} | {percentage(new['false_positive_rate'])} | {item['recovered_positives']}/{item['lost_positives']} | {item['removed_false_positives']}/{item['new_false_positives']} |")
        counts = decision["label_photo_counts"]
        lines += ["", f"합계: 미탐 {counts['round1_fn']}→{counts['round3_fn']}, 오탐 {counts['round1_fp']}→{counts['round3_fp']}."]
    else:
        lines += ["", "검증에서 자격을 얻은 새 항목이 없어 새 후보의 시험 비교는 수행하지 않았다. 975장의 성능 개선을 완료했다고 기록하지 않는다."]
    lines += ["", "## 한계와 다음 작업", "",
              "- 합성 사진의 공동 크기·질감이 실제 현장과 다르다. 양성/음성 분포도 실제 촬영 분포를 대표하지 않는다.",
              "- 같은 검증/시험을 반복 확인하고 시험을 교체 판단에 사용하므로 독립적인 성능 보증이 아니다. 새 현장/촬영 회차별 외부 평가가 필요하다.",
              "- 검증이나 시험에서 오류가 늘어난 후보는 기본 모델로 채택하지 않는다. 시험 결과를 보고 임계값·항목을 다시 선택하지 않았다.",
              "- 젖은 표면과 공동의 실제 정상/손상 사례를 함께 모으고 사람이 정답을 확인해야 다음 보강의 근거가 된다.",
              "- 사진 분류의 box=null 의견은 위치를 확정하지 않는다. 최종 판단과 완료 확인은 점검자가 한다.", "",
              "## 재현", "", "```powershell",
              "./.venv/Scripts/python.exe scripts/download_synthcavity.py",
              "./.venv/Scripts/python.exe scripts/prepare_synthcavity.py",
              "./.venv/Scripts/python.exe scripts/train_facility_presence.py --name facility-presence-synthetic --initial runs/facility-presence/best.pt --extra-manifest data/synthcavity-training/manifest.json --extra-weight 0.2 --imgsz 512 --batch 12 --epochs 16 --patience 5",
              "./.venv/Scripts/python.exe scripts/refine_synthcavity.py select",
              "./.venv/Scripts/python.exe scripts/refine_synthcavity.py test",
              "./.venv/Scripts/python.exe scripts/refine_synthcavity.py export",
              "./.venv/Scripts/python.exe scripts/refine_synthcavity.py release",
              "./.venv/Scripts/python.exe scripts/build_synthcavity_report.py", "```", "",
              "채택하지 않은 이번 후보의 CPU ONNX 비교와 기본 설정 보존 확인은 `python scripts/verify_synthcavity.py`로 재현한다.",
              "이 변환 비교는 모델 정확도나 Android 동작 검증이 아니다.", "",
              "기존 완료 실험은 덮어쓰지 않는다. 원본 자료·학습 가중치·가상환경은 Git에서 제외한다."]
    (reports / "FACILITY_ROUND3_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for source, target in (("TRAINING.json", "facility-round3-training.json"), ("history.json", "facility-round3-training-history.json")):
        shutil.copyfile(run / source, reports / target)
    print("Round 3 report written", flush=True)


if __name__ == "__main__": main()
