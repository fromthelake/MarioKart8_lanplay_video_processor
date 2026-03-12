param(
    [switch]$CreateVenv
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

function Resolve-PythonBootstrap {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }
    throw "No Python launcher was found. Install Python 3.10+ first."
}

if ($CreateVenv -or -not (Test-Path ".venv\Scripts\python.exe")) {
    $bootstrap = Resolve-PythonBootstrap
    if ($bootstrap.Length -gt 1) {
        & $bootstrap[0] $bootstrap[1] -m venv .venv
    } else {
        & $bootstrap[0] -m venv .venv
    }
}

$python = ".\.venv\Scripts\python.exe"
$playExe = ".\.venv\Scripts\mk8-local-play.exe"
$resultsExe = ".\.venv\Scripts\mk8-local-results.exe"
Write-Host "Using Python interpreter: $python"
& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install -e .

if (-not (Test-Path $playExe) -or -not (Test-Path $resultsExe)) {
    Write-Host "Console launchers missing after editable install, retrying with forced reinstall..."
    & $python -m pip install --force-reinstall -e .
}

if (-not (Test-Path $playExe) -or -not (Test-Path $resultsExe)) {
    throw "Setup completed dependency install, but mk8-local-play.exe / mk8-local-results.exe were not created in .venv\Scripts."
}

if (-not (Test-Path "app_config.json")) {
    Copy-Item "app_config.example.json" "app_config.json"
    Write-Host "Created app_config.json from app_config.example.json"
}

& $playExe --check

Write-Host ""
Write-Host "Setup finished."
Write-Host "Next steps:"
Write-Host "1. Install Tesseract if --check reports it missing."
Write-Host "2. Put videos into Input_Videos."
Write-Host "3. Run .\.venv\Scripts\mk8-local-play.exe --all"
Write-Host "   or run .\.venv\Scripts\python.exe main.py --all"
