# Changelog

All notable changes to this project will be documented in this file.

The format is intentionally simple and human-readable.

## [Unreleased]

### Added
- `pyproject.toml` for editable installs and console-script entrypoints
- New primary entrypoints:
  - `main.py`
  - `extract_frames.py`
  - `extract_text.py`
- New extraction modules:
  - `extract_initial_scan.py`
  - `extract_score_screen_selection.py`
  - `extract_video_io.py`
  - `extract_common.py`
- New OCR modules:
  - `ocr_name_matching.py`
  - `ocr_scoreboard_consensus.py`
  - `ocr_session_validation.py`
  - `ocr_export.py`
  - `ocr_common.py`
- New documentation:
  - `README.md`
  - `README.md`
  - `docs/PROJECT_STRUCTURE.md`
  - `docs/CONTRIBUTING.md`
- New release history file:
  - `docs/CHANGELOG.md`
- Repo placeholder files for cleaner fresh clones:
  - `Input_Videos/.gitkeep`
  - `benchmarks/baselines/.gitkeep`
  - `Output_Results/Debug/Score_Frames/.gitkeep`

### Changed
- Setup scripts now install the project in editable mode.
- New console command names:
  - `mk8-local-play`
- The application source code now lives in the `mk8_local_play/` package.
- The package entrypoints now route through `mk8_local_play.main`.
- Maintainer documentation is now grouped under `docs/`.
- Benchmark baselines are now grouped under `benchmarks/baselines/`.
- Track metadata and reference images are now grouped under `reference_data/`.
- Runtime path resolution, package data, benchmark scripts, and documentation were updated to match the grouped layout.
- Tracked IDE project files were removed from the repository.
- GUI startup now degrades more safely on systems without Tk support.
- Child scripts launched from `main.py` now prefer the repo-local `.venv`.
- `python -m mk8_local_play.main --check` reports both the parent Python and the child-script Python.
- Unix benchmark and setup scripts now use Unix-native virtualenv paths first.
- Assets are grouped under `assets/`.
- Naming moved toward clearer, human-readable module and function names.
- Internal extraction naming now prefers descriptive terms like `initial scan` instead of vague phase labels.
- OCR output writes timestamped workbooks to `Output_Results/`
- Track, cup, and character metadata now derive from a compact `reference_data/game_catalog.json` built from `database/firestore-export.json`.
- User workbooks now include `Character` and `Position After Race`.
- Debug workbooks now include explicit session rebase/reset flags, RaceScore recovery fields, identity labels, and character match details.
- Workbook output now keeps only timestamped files, with debug workbooks grouped under `Output_Results/Debug/`.
- Character icon matching now uses alpha-aware full-color template matching and `48x48` resized templates inside the fixed icon ROI.
- `Position After Race` is now recalculated from the final validated totals and allows shared placements such as `1,1,1,4,...`.
- RaceScore consensus is now signal-specific:
  - player names use all 7 nearby frames
  - race points use the first 3 frames
  - character matching uses the last 3 frames
  - left-side position matching uses the last 3 frames
  - RaceScore player count uses the first 3 frames
- TotalScore consensus now loads and uses only 3 center frames instead of loading a wider bundle first.
- Debug exports now expose explicit score-read sources for RacePoints, old totals, and new totals (`7-segment` vs `ocr_fallback`).
- RacePoints runtime seven-segment detection now uses the tuned fixed ROI layout with `white_threshold=180` and `active_ratio_threshold=0.45`.
- Session rebases remain visible in validation/debug output as an attention point, but no longer count as OCR review failures by themselves.
- Initial scan phase summaries now report confirmed track/race detection counts consistently during parallel scanning.

### Fixed
- Headless CLI runs no longer depend on GUI-only image imports.
- OCR/export no longer failed when `main.py` was started from the wrong Python interpreter while `.venv` existed.
- Several cross-platform path and setup issues affecting Windows, Linux, and macOS were cleaned up.
- The first visible race in a partially recorded video can now rebase the running totals instead of causing mismatch cascades.
- Later connection-reset races now keep the authoritative tournament totals running while clearly flagging the OCR reset rows.
- RaceScore player counts now recover from later frames when the black results banner hides the last row.
- Duplicate exact player names can now be split with character-aware identity tracking, producing stable names such as `Name_1` and `Name_2`.
- Review-reason parsing no longer turns values like `15.0` into incorrect messages such as `150`.
- Position-guided player count detection now treats rows with `Coeff < 0.60` as empty, preventing false extra rows in stable 10-player cases.
- `Digit confidence is low` and race-points mismatches caused by late RaceScore frames drifting downward are now eliminated on validated multi-video OCR runs.
- False RacePoints and TotalScore regressions caused by padded digit rows no longer trigger unnecessary OCR fallback on validated races.

### Documentation
- Setup instructions were rewritten for hobbyist-friendly use from a Git clone.
- Project structure is now documented in plain language.
- Technical pipeline docs now explain that AAC `decode_pce` console warnings are usually noisy audio-stream warnings, not immediate video-decoding failures.
- Contributing expectations now explicitly cover naming, comments, cross-platform behavior, and validation.
