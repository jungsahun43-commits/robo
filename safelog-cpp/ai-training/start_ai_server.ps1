param(
    [int]$Port = 8080,
    [switch]$AllModels,
    [switch]$Facilities,
    [switch]$FacilitiesOnly,
    [switch]$LocalOnly,
    [switch]$BaselineFacilities,
    [switch]$Round1Facilities
)
$ErrorActionPreference = "Stop"
if ($FacilitiesOnly -and $AllModels) { throw "FacilitiesOnly와 AllModels 중 하나를 선택하세요." }
$TaskPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $TaskPython)) { throw "먼저 ./setup_windows.ps1을 실행하세요." }

function Get-ModelPath([string]$Name) {
    $Packaged = Join-Path $PSScriptRoot "models/$Name.pt"
    if (Test-Path $Packaged) { return $Packaged }
    if ($Name.StartsWith("facility-presence")) { return Join-Path $PSScriptRoot "runs/$Name/best.pt" }
    return Join-Path $PSScriptRoot "runs/$Name/weights/best.pt"
}

$PrimaryPpe = "ppe-baseline"
$SelectionPath = Join-Path $PSScriptRoot "reports/selected-models.json"
if (Test-Path $SelectionPath) {
    $Selection = Get-Content -LiteralPath $SelectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $PrimaryPpe = $Selection.primary_ppe
}
$env:SAFELOG_MODEL_PATH = Get-ModelPath $PrimaryPpe
if ($FacilitiesOnly) { $env:SAFELOG_MODEL_PATH = "" }
$env:SAFELOG_FIRE_MODEL_PATH = Get-ModelPath "fire-smoke"
$env:SAFELOG_AUX_MODEL_PATH = ""
$env:SAFELOG_AUX_MODEL_PATHS = ""
$env:SAFELOG_FACILITY_MODEL_PATHS = ""
$env:SAFELOG_PRESENCE_MODEL_PATH = ""
$env:SAFELOG_PRESENCE_MODEL_PATHS = ""
$PresenceModels = @()
$FacilityProfilePath = Join-Path $PSScriptRoot "reports/facility-inference-profile.json"
if ($Round1Facilities) { $FacilityProfilePath = Join-Path $PSScriptRoot "reports/facility-inference-profile-round1.json" }
$env:SAFELOG_FACILITY_PROFILE_PATH = $FacilityProfilePath
$Additional = @()
if ($AllModels) {
    $Additional = @((Get-ModelPath "chvg-ppe"), (Get-ModelPath "sh17-ppe"))
    $env:SAFELOG_AUX_MODEL_PATHS = $Additional -join [IO.Path]::PathSeparator
}
$FacilityModels = @()
if ($Facilities -or $FacilitiesOnly -or $AllModels) {
    $FacilityNames = @("facility-dacl", "facility-corrosion")
    if ((Test-Path $FacilityProfilePath) -and -not $BaselineFacilities) {
        $FacilityProfile = Get-Content -LiteralPath $FacilityProfilePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $FacilityNames = @($FacilityProfile.models.PSObject.Properties.Name)
        if ($FacilityProfile.PSObject.Properties.Name -contains "photo_classifiers") {
            $PresenceModels = @($FacilityProfile.photo_classifiers | ForEach-Object { Get-ModelPath $_.model })
            $env:SAFELOG_PRESENCE_MODEL_PATHS = $PresenceModels -join [IO.Path]::PathSeparator
        } elseif ($FacilityProfile.photo_classifier) {
            $env:SAFELOG_PRESENCE_MODEL_PATH = Get-ModelPath $FacilityProfile.photo_classifier.model
        }
        if ($FacilityNames.Count -ne 2) { throw "시설 검증 설정에 모델 2개가 필요합니다." }
    }
    $FacilityModels = @($FacilityNames | ForEach-Object { Get-ModelPath $_ })
    $env:SAFELOG_FACILITY_MODEL_PATHS = $FacilityModels -join [IO.Path]::PathSeparator
}
foreach ($TaskModel in @($env:SAFELOG_MODEL_PATH, $env:SAFELOG_FIRE_MODEL_PATH, $env:SAFELOG_PRESENCE_MODEL_PATH) + $Additional + $FacilityModels + $PresenceModels) {
    if ($TaskModel -and -not (Test-Path $TaskModel)) { throw "모델 파일이 없습니다: $TaskModel" }
}
$BindHost = if ($LocalOnly) { "127.0.0.1" } else { "0.0.0.0" }
Push-Location $PSScriptRoot
try { & $TaskPython -m uvicorn safelog_ai.server:app --host $BindHost --port $Port }
finally { Pop-Location }
