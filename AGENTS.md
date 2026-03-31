# Repository Guidance

## What This Project Is

This repository contains a Python 3.12 desktop and CLI application that processes Mario Kart 8 LAN-play tournament recordings.

The pipeline:
- scans input videos for track, race-number, race-score, and total-score screens
- exports normalized frame bundles under `Output_Results/Frames`
- OCRs names, scores, positions, tracks, and characters
- validates tournament/session state
- writes timestamped Excel and CSV outputs under `Output_Results/`

The current codebase is the behavioral baseline. Existing working behavior must be preserved unless a change is explicitly approved or a verified bug fix requires a contained change.

## Safe Working Rules

- Treat current behavior as intentional unless tests, reproducible runs, or logs prove otherwise.
- Prefer minimal, contained fixes over refactors.
- Do not perform broad cleanup, dependency changes, architectural rewrites, or pattern migrations without approval.
- Do not touch unrelated user state in `Input_Videos/` or historical generated outputs unless the task requires it.
- If output differences are ambiguous but the final business result may still be correct, escalate for human evaluation before reverting.
- For bug fixes, change the smallest possible surface area and preserve surrounding behavior.

## Key Commands

Use the repo-local virtualenv. Do not rely on a global install of this app.

Windows baseline checks:

```powershell
.\.venv\Scripts\python.exe -m compileall mk8_local_play
.\.venv\Scripts\python.exe -m mk8_local_play.main --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Useful scoped runs:

```powershell
.\.venv\Scripts\python.exe -m mk8_local_play.main --extract --video <video-name>
.\.venv\Scripts\python.exe -m mk8_local_play.main --ocr --selection --video <video-name>
.\.venv\Scripts\python.exe -m mk8_local_play.main --all --selection --video <video-name>
.\.venv\Scripts\python.exe -m mk8_local_play.main --selection --subfolders --videos "2026-03-28/VideoA.mp4" "2026-03-28/VideoB.mp4"
```

Variant-family debug probe for saved character crops:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_character_variant_families.py --crop-dir Output_Results\Debug\character_probe_20260328
```

Packaged entrypoint:

```powershell
.\.venv\Scripts\mk8-local-play.exe --check
```

Baseline comparison when a saved benchmark exists:

```powershell
.\.venv\Scripts\python.exe tools\validate_outputs.py --baseline-dir benchmarks\baselines\<name>
```

Curated demo baseline command:

```powershell
.\.venv\Scripts\python.exe -m mk8_local_play.main --selection --video Demo_CaptureCard_Race.mp4
.\.venv\Scripts\python.exe tools\validate_outputs.py --baseline-dir benchmarks\baselines\demo_capturecard_race --prefix Demo_CaptureCard_Race --race-class Demo_CaptureCard_Race
```

## Conventions And Patterns

- Runtime paths are project-root relative and centered around `PROJECT_ROOT`.
- `main.py` is the top-level CLI/GUI entrypoint and orchestrates extract/OCR runs.
- Extraction and OCR are intentionally separated into distinct modules and phases.
- Layout-specific ROI logic is explicit; preserve the current LAN 1 vs LAN 2 behavior unless the task targets it directly.
- Runtime config lives in `config/app_config.json` with environment variable overrides.
- `config/app_config.json:export_image_format` controls exported screenshots globally. Accepted values are `jpg`/`jpeg` and `png`. `MK8_EXPORT_IMAGE_FORMAT` can still override it for scoped experiments.
- `Output_Results/Frames/` now uses per-video per-race bundle folders. Score-screen OCR inputs are persisted as `anchor_<frame>` and `consensus_<frame>` images so `--selection` and `--ocr` consume the same saved bundles.
- Character OCR now includes a family-level variant refinement pass for roster families with many color variants such as `Shy Guy`, `Yoshi`, and `Birdo`. The pass compares only family members, includes the default/base roster member, and runs before the conservative `Mii` fallback so stable color-family identities are not erased by near-tied variant noise.
- `--videos` accepts multiple explicit file paths. With `--subfolders`, exact relative paths are resolved before filename fallback so scoped runs do not accidentally include same-named files from sibling folders such as `backup/`.
- Generated outputs belong under `Output_Results/`; curated baselines belong under `benchmarks/baselines/`.
- Comments should explain intent, assumptions, or tradeoffs, not restate syntax.

## Areas Requiring Extra Care

- `mk8_local_play/extract_frames.py`
  Extraction orchestration, frame export, and end-to-end extraction summaries.
- `mk8_local_play/extract_initial_scan.py`
  Detection heuristics and segment-based scan behavior.
- `mk8_local_play/extract_score_screen_selection.py`
  RaceScore and TotalScore frame timing, row-count recovery, and late-frame logic.
- `mk8_local_play/extract_text.py`
  OCR orchestration, batching, frame grouping, and low-res dispatch.
- `mk8_local_play/ocr_scoreboard_consensus.py`
  Core OCR geometry, consensus rules, row presence, score digit reading, and character matching.
- `mk8_local_play/low_res_identity.py`
  Low-resolution fallback identity pipeline and row restoration behavior.
- `mk8_local_play/ocr_session_validation.py`
  Session rebases, reset handling, and score-validation rules.
- `mk8_local_play/ocr_export.py`
  User-facing workbook schema and export formatting.

These modules encode the core behavior contract. Avoid opportunistic cleanup here.

## What Good Changes Look Like Here

- The change is narrow, reversible, and justified by a verified issue or approved request.
- Baseline commands are run before and after the change.
- Existing behavior outside the task scope remains unchanged.
- If the touched area lacks coverage, the change adds the smallest useful regression or characterization test.
- Documentation is updated when workflow, commands, config, output shape, or architecture meaningfully changes.

## Required Evaluation Escalation Format

Use this exact block when internal values drift but final correctness is unclear:

```text
EVALUATION NEEDED
- Task/change:
- What changed:
- Where it changed:
- Baseline result:
- New result:
- What is still correct:
- What is different:
- Whether final business output is still correct:
- Risk if we keep it:
- Risk if we revert it:
- My recommended judgment:
```
