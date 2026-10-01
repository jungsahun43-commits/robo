from __future__ import annotations

import importlib.util
import json
import platform
import sys


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("torch", "ultralytics", "fastapi", "onnx", "onnxruntime"):
        report[package] = importlib.util.find_spec(package) is not None

    if report["torch"]:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    print(json.dumps(report, ensure_ascii=False, indent=2))
    required = ("torch", "ultralytics", "fastapi")
    return 0 if all(report[name] for name in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
