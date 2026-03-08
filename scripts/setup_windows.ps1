param(
    [switch]$CreateVenv
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

if ($CreateVenv -or -not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

if (-not (Test-Path "app_config.json")) {
    Copy-Item "app_config.example.json" "app_config.json"
    Write-Host "Created app_config.json from app_config.example.json"
}

& $python Main_RunMe.py --check

Write-Host ""
Write-Host "Setup finished."
Write-Host "Next steps:"
Write-Host "1. Install Tesseract if --check reports it missing."
Write-Host "2. Put videos into Input_Videos."
Write-Host "3. Run .\.venv\Scripts\python.exe Main_RunMe.py --all"
