param([switch]$CpuOnly)
$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python 3.11 또는 3.12를 먼저 설치하고 'Add Python to PATH'를 선택하세요."
}

Push-Location $PSScriptRoot
try {
    python -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), 'Python 3.11 or 3.12 required'"
    if ($LASTEXITCODE -ne 0) { throw "Python 버전을 확인하세요." }
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "가상환경 생성 실패" }
    $TaskPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    & $TaskPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip 설치 실패" }
    if (-not $CpuOnly -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        & $TaskPython -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
    } else {
        & $TaskPython -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) { throw "PyTorch 설치 실패" }
    & $TaskPython -m pip install -r requirements-tested.txt
    if ($LASTEXITCODE -ne 0) { throw "의존성 설치 실패" }
    & $TaskPython -m pip install httpx==0.28.1
    if ($LASTEXITCODE -ne 0) { throw "테스트 의존성 설치 실패" }
    & $TaskPython scripts/check_environment.py
    if ($LASTEXITCODE -ne 0) { throw "환경 검사 실패" }
    Write-Host "설치 완료. start_ai_server.ps1 또는 run_training_suite.ps1을 실행하세요."
} finally { Pop-Location }
