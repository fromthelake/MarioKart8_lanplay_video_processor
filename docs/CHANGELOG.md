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
  - `ReadMe.txt`
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
  - `mk8-local-results`
- The application source code now lives in the `mk8_local_play/` package.
- `main.py`, `extract_frames.py`, and `extract_text.py` now act as thin root launchers.
- Maintainer documentation is now grouped under `docs/`.
- Benchmark baselines are now grouped under `benchmarks/baselines/`.
- Track metadata and reference images are now grouped under `reference_data/`.
- Runtime path resolution, package data, benchmark scripts, and documentation were updated to match the grouped layout.
- Tracked IDE project files were removed from the repository.
- GUI startup now degrades more safely on systems without Tk support.
- Child scripts launched from `main.py` now prefer the repo-local `.venv`.
- `main.py --check` now reports both the parent Python and the child-script Python.
- Unix benchmark and setup scripts now use Unix-native virtualenv paths first.
- Assets are grouped under `assets/`.
- Naming moved toward clearer, human-readable module and function names.
- Internal extraction naming now prefers descriptive terms like `initial scan` instead of vague phase labels.
- OCR output writes timestamped workbooks to `Output_Results/`

### Fixed
- Headless CLI runs no longer depend on GUI-only image imports.
- OCR/export no longer failed when `main.py` was started from the wrong Python interpreter while `.venv` existed.
- Several cross-platform path and setup issues affecting Windows, Linux, and macOS were cleaned up.

### Documentation
- Setup instructions were rewritten for hobbyist-friendly use from a Git clone.
- Project structure is now documented in plain language.
- Contributing expectations now explicitly cover naming, comments, cross-platform behavior, and validation.
