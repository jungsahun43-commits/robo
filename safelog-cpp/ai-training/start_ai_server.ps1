param(
    [int]$Port = 8080,
    [switch]$AllModels,
    [switch]$LocalOnly
)
$ErrorActionPreference = "Stop"
$TaskPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $TaskPython)) { throw "먼저 ./setup_windows.ps1을 실행하세요." }

function Get-ModelPath([string]$Name) {
    $Packaged = Join-Path $PSScriptRoot "models/$Name.pt"
    if (Test-Path $Packaged) { return $Packaged }
    return Join-Path $PSScriptRoot "runs/$Name/weights/best.pt"
}

$env:SAFELOG_MODEL_PATH = Get-ModelPath "ppe-baseline"
$env:SAFELOG_FIRE_MODEL_PATH = Get-ModelPath "fire-smoke"
$env:SAFELOG_AUX_MODEL_PATH = ""
$env:SAFELOG_AUX_MODEL_PATHS = ""
$Additional = @()
if ($AllModels) {
    $Additional = @((Get-ModelPath "chvg-ppe"), (Get-ModelPath "sh17-ppe"))
    $env:SAFELOG_AUX_MODEL_PATHS = $Additional -join [IO.Path]::PathSeparator
}
foreach ($TaskModel in @($env:SAFELOG_MODEL_PATH, $env:SAFELOG_FIRE_MODEL_PATH) + $Additional) {
    if ($TaskModel -and -not (Test-Path $TaskModel)) { throw "모델 파일이 없습니다: $TaskModel" }
}
$BindHost = if ($LocalOnly) { "127.0.0.1" } else { "0.0.0.0" }
Push-Location $PSScriptRoot
try { & $TaskPython -m uvicorn safelog_ai.server:app --host $BindHost --port $Port }
finally { Pop-Location }
