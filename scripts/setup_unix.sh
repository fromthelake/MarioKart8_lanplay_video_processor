#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

if [[ ! -f "app_config.json" ]]; then
  cp app_config.example.json app_config.json
  echo "Created app_config.json from app_config.example.json"
fi

".venv/bin/python" Main_RunMe.py --check

echo
echo "Setup finished."
echo "Next steps:"
echo "1. Install Tesseract if --check reports it missing."
echo "2. Put videos into Input_Videos."
echo "3. Run .venv/bin/python Main_RunMe.py --all"
