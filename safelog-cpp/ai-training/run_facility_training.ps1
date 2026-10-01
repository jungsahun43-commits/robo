param([switch]$SkipDownload, [switch]$PrepareOnly, [string]$Device = "0")
$ErrorActionPreference = "Stop"
$TaskPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $TaskPython)) { throw "먼저 ./setup_windows.ps1을 실행하세요." }
$env:PYTHONIOENCODING = "utf-8"
function Invoke-TrainingPython([string[]]$Arguments) {
    & $TaskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python 단계 실패: $($Arguments -join ' ')" }
}
Push-Location $PSScriptRoot
try {
    foreach ($Dataset in @("ostrava-corrosion", "dacl10k")) {
        if (-not $SkipDownload) { Invoke-TrainingPython @("scripts/download_facility_data.py", $Dataset) }
        Invoke-TrainingPython @("scripts/prepare_facility_data.py", $Dataset)
        $DataYaml = "data/$Dataset-yolo/data.yaml"
        Invoke-TrainingPython @("scripts/audit_yolo_dataset.py", $DataYaml)
        if ($PrepareOnly) { continue }
        $RunName = if ($Dataset -eq "dacl10k") { "facility-dacl" } else { "facility-corrosion" }
        $Epochs = if ($Dataset -eq "dacl10k") { "100" } else { "120" }
        $Patience = if ($Dataset -eq "dacl10k") { "30" } else { "35" }
        $TrainingArguments = @("scripts/train.py", "--data", $DataYaml, "--model", "yolo11s.pt", "--epochs", $Epochs,
            "--batch", "8", "--imgsz", "960", "--name", $RunName, "--device", $Device,
            "--cache", "disk", "--patience", $Patience, "--workers", "4")
        if (Test-Path "runs/$RunName/weights/last.pt") { $TrainingArguments += "--resume" }
        Invoke-TrainingPython $TrainingArguments
        $Weights = "runs/$RunName/weights/best.pt"
        Invoke-TrainingPython @("scripts/evaluate.py", $Weights, $DataYaml, "--imgsz", "960", "--device", $Device)
        Invoke-TrainingPython @("scripts/evaluate_fixed_threshold.py", $Weights, $DataYaml, "--imgsz", "960", "--device", $Device)
        Invoke-TrainingPython @("scripts/export_onnx.py", $Weights, "--imgsz", "960")
    }
    if (-not $PrepareOnly) { Invoke-TrainingPython @("scripts/build_facility_report.py") }
}
finally { Pop-Location }
