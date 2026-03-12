#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to a Python 3.12 interpreter and rerun this script." >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

echo "Using Python interpreter: .venv/bin/python"
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -e .

if [[ ! -f "config/app_config.json" ]]; then
  echo "Missing config/app_config.json. Restore it from git before running setup." >&2
  exit 1
fi

".venv/bin/mk8-local-play" --check

echo
echo "Setup finished."
echo "This app runs from the local .venv in this project folder."
echo "No global Python package install or PATH change is required for mk8-local-play."
echo "Next steps:"
echo "1. Install Tesseract if --check reports it missing."
echo "2. Put videos into Input_Videos."
echo "3. Run .venv/bin/mk8-local-play --all"
echo "   or run .venv/bin/python -m mk8_local_play.main --all"
