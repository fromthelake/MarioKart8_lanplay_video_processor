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
    throw "No Python launcher was found. Install Python 3.12 first."
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
Write-Host "Using Python interpreter: $python"
& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install -e .

if (-not (Test-Path $playExe)) {
    Write-Host "Console launchers missing after editable install, retrying with forced reinstall..."
    & $python -m pip install --force-reinstall -e .
}

if (-not (Test-Path $playExe)) {
    throw "Setup completed dependency install, but mk8-local-play.exe was not created in .venv\Scripts."
}

if (-not (Test-Path "config\app_config.json")) {
    throw "Missing config\app_config.json. Restore it from git before running setup."
}

& $playExe --check

Write-Host ""
Write-Host "Setup finished."
Write-Host "This app runs from the local .venv in this project folder."
Write-Host "No global Python package install or PATH change is required for mk8-local-play."
Write-Host "Next steps:"
Write-Host "1. Install Tesseract if --check reports it missing."
Write-Host "2. Put videos into Input_Videos."
Write-Host "3. Run .\.venv\Scripts\mk8-local-play.exe --all"
Write-Host "   or run .\.venv\Scripts\python.exe -m mk8_local_play.main --all"
