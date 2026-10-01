$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "가상환경을 찾을 수 없습니다. 먼저 ./setup_windows.ps1을 실행하세요."
}

function Train-Evaluate-Export {
    param(
        [string]$Data,
        [string]$Name,
        [int]$Epochs = 100
    )
    $ResumeArgs = @()
    if (Test-Path "runs/$Name/weights/last.pt") { $ResumeArgs = @("--resume") }
    & $Python scripts/train.py --data $Data --epochs $Epochs --batch 16 --device 0 --name $Name @ResumeArgs
    if ($LASTEXITCODE -ne 0) { throw "$Name 학습 실패" }

    $Weights = "runs/$Name/weights/best.pt"
    & $Python scripts/evaluate.py $Weights $Data --device 0
    if ($LASTEXITCODE -ne 0) { throw "$Name 평가 실패" }
    & $Python scripts/export_onnx.py $Weights
    if ($LASTEXITCODE -ne 0) { throw "$Name ONNX 변환 실패" }
}

Push-Location $PSScriptRoot
try {
    Train-Evaluate-Export "data/construction-ppe/data.yaml" "ppe-baseline"
    Train-Evaluate-Export "data/indoor-fire-smoke/data.yaml" "fire-smoke"
    Train-Evaluate-Export "data/chvg-yolo/data.yaml" "chvg-ppe"
    & $Python scripts/prepare_sh17.py
    if ($LASTEXITCODE -ne 0) { throw "SH17 설정 실패" }
    & $Python scripts/prepare_sh17_training_copy.py
    if ($LASTEXITCODE -ne 0) { throw "SH17 학습용 사본 생성 실패" }
    $ResumeArgs = @()
    if (Test-Path "runs/sh17-ppe/weights/last.pt") { $ResumeArgs = @("--resume") }
    & $Python scripts/train.py --data data/sh17-1280/data.yaml --model yolo11s.pt --epochs 100 --batch 16 --device 0 --name sh17-ppe --cache disk @ResumeArgs
    if ($LASTEXITCODE -ne 0) { throw "sh17-ppe 학습 실패" }
    & $Python scripts/evaluate.py runs/sh17-ppe/weights/best.pt data/sh17-1280/data.yaml --device 0 --split val
    if ($LASTEXITCODE -ne 0) { throw "sh17-ppe 평가 실패" }
    & $Python scripts/export_onnx.py runs/sh17-ppe/weights/best.pt
    if ($LASTEXITCODE -ne 0) { throw "sh17-ppe ONNX 변환 실패" }
    & $Python scripts/summarize_results.py
    if ($LASTEXITCODE -ne 0) { throw "결과 비교표 생성 실패" }
    & $Python scripts/build_training_report.py
    if ($LASTEXITCODE -ne 0) { throw "학습 보고서 생성 실패" }
    & $Python scripts/package_models.py
    if ($LASTEXITCODE -ne 0) { throw "모델 공유 ZIP 생성 실패" }
}
finally {
    Pop-Location
}
