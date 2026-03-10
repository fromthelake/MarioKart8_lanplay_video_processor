#!/usr/bin/env bash
set -euo pipefail

VIDEO_NAME="${1:-Test_3_Races.mkv}"
BASELINE_DIR="${2:-benchmarks/baselines/quick/Test_3_Races}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/Scripts/python.exe"
else
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

HOLD_DIR="$PROJECT_ROOT/Input_Videos_Hold"

restore_inputs() {
  if [[ -d "$HOLD_DIR" ]]; then
    find "$HOLD_DIR" -maxdepth 1 -type f -exec mv -f {} "$PROJECT_ROOT/Input_Videos/" \;
    rmdir "$HOLD_DIR" 2>/dev/null || true
  fi
}

trap restore_inputs EXIT
restore_inputs
mkdir -p "$HOLD_DIR"

find "$PROJECT_ROOT/Input_Videos" -maxdepth 1 -type f ! -name "$VIDEO_NAME" -exec mv -f {} "$HOLD_DIR/" \;
find "$PROJECT_ROOT/Output_Results/Frames" -maxdepth 1 -type f -delete 2>/dev/null || true
find "$PROJECT_ROOT/Output_Results/Debug/Score_Frames" -maxdepth 1 -type f -delete 2>/dev/null || true
rm -f "$PROJECT_ROOT/Output_Results/Debug/debug_max_val.csv"
rm -f "$PROJECT_ROOT"/Output_Results/*_Tournament_Results.xlsx
rm -f "$PROJECT_ROOT/Output_Results/Tournament_Results.xlsx"
rm -f "$PROJECT_ROOT/Output_Results/~\$Tournament_Results.xlsx"

"$PYTHON_BIN" main.py --check

extract_start="$(date +%s.%N)"
"$PYTHON_BIN" main.py --extract --video "$VIDEO_NAME"
extract_end="$(date +%s.%N)"

ocr_start="$(date +%s.%N)"
"$PYTHON_BIN" main.py --ocr --video "$VIDEO_NAME"
ocr_end="$(date +%s.%N)"

"$PYTHON_BIN" tools/validate_outputs.py --baseline-dir "$BASELINE_DIR"

"$PYTHON_BIN" - "$extract_start" "$extract_end" "$ocr_start" "$ocr_end" <<'PY'
import sys
extract_start = float(sys.argv[1])
extract_end = float(sys.argv[2])
ocr_start = float(sys.argv[3])
ocr_end = float(sys.argv[4])
print(f"extract_seconds={extract_end - extract_start:.2f}")
print(f"ocr_seconds={ocr_end - ocr_start:.2f}")
PY
