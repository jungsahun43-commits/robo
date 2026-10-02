"""Archive the unreleased model for research without overriding app defaults."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "reports/facility-round4-verification.json").read_text(encoding="utf-8"))
    if report["status"] != "passed": raise ValueError("Verify the actual model first")
    files = {}
    for suffix, key in (("pt", "model_sha256"), ("onnx", "onnx_sha256")):
        source = ROOT / f"runs/facility-presence-damsegment/best.{suffix}"
        with source.open("rb") as handle: digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != report[key]: raise ValueError("Verified model changed")
        files[source] = f"experiments/facility-round4/best.{suffix}"
    for name in ("EXPORT.json", "TRAINING.json", "history.json"):
        files[ROOT / "runs/facility-presence-damsegment" / name] = f"experiments/facility-round4/{name}"
    files[ROOT / "datasets/damsegment_source.json"] = "experiments/facility-round4/damsegment_source.json"
    for path in (ROOT / "reports").glob("facility-round4-*.json"):
        if path.name == "facility-round4-research-bundle.json": continue
        files[path] = f"experiments/facility-round4/reports/{path.name}"
    for name in ("FACILITY_ROUND4_KO.md", "FACILITY_ROUND4_PLAN_KO.md", "facility-inference-profile-round4-candidate.json"):
        files[ROOT / "reports" / name] = f"experiments/facility-round4/reports/{name}"
    target = ROOT / "artifacts/safelog-facility-round4-research.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files.items(): archive.write(path, name)
        archive.writestr("experiments/facility-round4/README_KO.md", """# 4차 연구용 분류기

이 파일은 앱 기본 모델 전달 ZIP과 다르며 새 모델은 교체 기준을 통과하지 못했다.
기본 safelog-trained-models.zip과 facility-validation-v2를 계속 사용한다.
이 ZIP은 experiments/facility-round4 안에만 풀리며 models/와 reports/ 기본 설정을 덮어쓰지 않는다.
feature/ai-engine의 최신 코드 및 reports/FACILITY_ROUND4_KO.md를 함께 읽는다.

best.pt는 실제 추가 사진으로 학습한 7항목 EfficientNet-B0이며 best.onnx는 같은 가중치의 변환본이다.
EXPORT.json의 RGB 전처리와 출력 순서를 따라야 한다. 7 logits에 sigmoid를 적용하며 위치 박스는 출력하지 않는다.
새 추가 정답은 균열/박락 두 항목뿐이다. API의 물기/공동 정확도가 개선되었다는 의미가 아니다.
보류 패치 진단의 큰 균열 개선은 단일 댐 자료 기준이며 촬영 장면의 완전한 독립성이 확인되지 않았다.
후보 프로필에는 채택되지 않은 새 모델 항목이 활성화되지 않았다. 시험 결과로 임계값을 다시 고르지 않는다.

출처: Gharehbaghi et al. (2025), DamSegment V1, https://doi.org/10.17632/z5z6gtt5t4.1 (CC BY 4.0).
수정: 공개 패치 JPEG 재인코딩, 부분 정답 변환, 유사 사진 그룹 분리, 분류기 파인튜닝.
기반 DACL10k 자료/가중치의 CC BY-NC 4.0 조건을 유지한다. 이 모델은 비상업 연구/교육용이다.
DACL 출처: Flotzinger, Rosch and Braml (2023), dacl10k, https://arxiv.org/abs/2309.00460.
원본 데이터는 포함하지 않았다. 사진 안전 판단은 사람이 검토한다.
""")
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None: raise ValueError("Corrupted research ZIP")
        for suffix, key in (("pt", "model_sha256"), ("onnx", "onnx_sha256")):
            with archive.open(f"experiments/facility-round4/best.{suffix}") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != report[key]: raise ValueError("Archived model mismatch")
    with target.open("rb") as handle: digest = hashlib.file_digest(handle, "sha256").hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    record = {"status": "passed", "bundle": target.name, "sha256": digest, "bytes": target.stat().st_size,
              "scope": "Unreleased research model and reports only; app default paths untouched"}
    (ROOT / "reports/facility-round4-research-bundle.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__": main()
