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
- Runtime GPU settings now default extraction (`execution_mode`) to `cpu` and EasyOCR (`easyocr_gpu_mode`) to `auto`, with `gpu` and `cpu` override modes still available from config, env vars, and the GUI.
- Overlap OCR now defaults to `auto` mode with `2` consumers. When EasyOCR CUDA is available, full multi-video runs use the streamed per-race overlap path by default; when CUDA is unavailable, the overlap default resolves back to the existing sequential behavior. Explicit `video` / `race` mode overrides and custom consumer counts remain supported for experiments.
- Initial scan now supports a multi-video shared-process path. It defaults to `2` workers for multi-video runs, and `MK8_PARALLEL_VIDEO_SCAN_WORKERS` can still override it manually. On the current 7-video benchmark set, `2` workers reduced extraction-only runtime from `06:59` to `03:58`, while `3` and `4` workers were slower.
- Extraction worker defaults are now tuned from the full 7-video benchmark set: `pass1_scan_workers=4`, `score_analysis_workers=4`, and cross-video total-score workers resolve to `2` on `16+` logical CPU threads and `1` otherwise.
- Selection and scoped extract runs now clear only the selected videos' exported artifacts internally before extraction starts, so repeated scoped runs start cleanly without external shell cleanup.
- Console reporting is now clearer and more consistent during long runs: selected videos get stable neon accents, video-owned values are colored without coloring whole lines, the final performance summary is table-based, and `Pipeline time avoided` shows the wall-clock savings gained from overlap and parallelism.
- Console workflow ordering is now consistent across the input summary, frame-count preflight, scan/start labels, and per-video summaries. Scan progress now uses time-based `HH:MM:SS / HH:MM:SS` output, shared scan status centralizes CPU/RAM reporting, total-score progress is shorter, and overlap OCR now logs explicit `Finalizing OCR ...` / `Finalize: ...` timing for videos whose post-processing outlasts the last OCR race update.
- Score-screen extraction now supports both LAN 2 two-player split-screen and LAN 1 one-player full-screen layouts for `2RaceScore` / `3TotalScore`.
- Initial score-screen detection now checks both supported score-anchor ROIs in one pass and tags the winning layout on each score candidate.
- Initial scan ignore detection now supports multiple gallery/review templates so Nintendo Switch Album / Gallery control bars can be rejected before score candidates are queued.
- Exported `2RaceScore` / `3TotalScore` frame filenames and metadata now carry the detected score layout id so OCR can select the correct ROI set without guessing.
- Score-frame debug output now also writes annotated ROI demo images under `Output_Results/Debug/Score_Layout_Demos/` for both `2RaceScore` and `3TotalScore`.
- Low-resolution and ultra-low-resolution score-row geometry now follows the detected LAN 1 vs LAN 2 score layout instead of assuming one fixed scoreboard placement.
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
- OCR output now also writes a timestamped `*_Final_Standings.csv` alongside the workbook and results CSV.
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
- Static gallery-opened RaceScore screens are now filtered using first-frame-vs-rest similarity checks with conservative thresholds, so obvious Nintendo Switch Album stills can be rejected without dropping known low-motion real races.
- Debug exports now expose explicit score-read sources for RacePoints, old totals, and new totals (`7-segment` vs `ocr_fallback`).
- RacePoints runtime seven-segment detection now uses the tuned fixed ROI layout with `white_threshold=180` and `active_ratio_threshold=0.45`.
- Session rebases remain visible in validation/debug output as an attention point, but no longer count as OCR review failures by themselves.
- Initial scan phase summaries now report confirmed track/race detection counts consistently during parallel scanning.
- Parallel initial scan now streams live detection counts during the scan instead of appearing idle until segment workers finish.
- The GUI now includes a `Clear Output Results` action with an `Are you sure?` confirmation before deleting generated output files.

### Fixed
- Session-level character relabeling now supports Mii fallback when one player repeatedly produces weak, near-tied, unstable non-Mii character matches across the saved `2RaceScore` frames.
- Mii fallback rows now keep a short explicit review note: `mii_fallback_unstable_character_match`.
- Headless CLI runs no longer depend on GUI-only image imports.
- OCR/export no longer failed when `main.py` was started from the wrong Python interpreter while `.venv` existed.
- Several cross-platform path and setup issues affecting Windows, Linux, and macOS were cleaned up.
- The first visible race in a partially recorded video can now rebase the running totals instead of causing mismatch cascades.
- Later connection-reset races now keep the authoritative tournament totals running while clearly flagging the OCR reset rows.
- RaceScore player counts now recover from later frames when the black results banner hides the last row.
- Duplicate exact player names can now be split with character-aware identity tracking, producing stable names such as `Name_1` and `Name_2`.
- Review-reason parsing no longer turns values like `15.0` into incorrect messages such as `150`.
- Position-guided player count detection now treats rows with `Coeff < 0.60` as empty, preventing false extra rows in stable 10-player cases.
- Position-guided OCR row counts now allow a guarded row-1 exception at `Coeff >= 0.40` when first place is visually occupied but partly covered by the Nintendo `Capture taken.` overlay.
- RaceScore frame selection now scales the old post-12th timing by FPS and can search slightly later frames for a valid 12th-place row when a 12-player race would otherwise be exported one frame too early.
- `Digit confidence is low` and race-points mismatches caused by late RaceScore frames drifting downward are now eliminated on validated multi-video OCR runs.
- False RacePoints and TotalScore regressions caused by padded digit rows no longer trigger unnecessary OCR fallback on validated races.
- Low-resolution videos now use a dedicated player-identity path with `PlayerNameMissing_X` placeholders, fixed name/character ROI matching, and computed race points / totals instead of OCR score digits.
- Low-resolution score consensus now keeps position-confirmed rows even when OCR is too weak to fill the row reliably.
- Low-resolution character matching now uses the tuned fixed ROI and `51x52` template sizing validated on multiple `640x360 -> 1280x720` captures.
- Low-resolution `11 vs 12` last-row misses can now recover row 12 from the combined `character + player-name` blob when the rest of the video clearly indicates a 12-player race.
- Placeholder identities no longer resolve to other `PlayerNameMissing_X` labels, preventing duplicate placeholder names inside one race.
- Placeholder identity rescue now has a conservative forced-choice fallback for strong unresolved candidates. Forced promotions are marked as `placeholder_name_forced_choice` and keep their candidate/support/score trail in `ReviewReason`.
- Low-resolution ROI/template sizing and blob fallback thresholds are now exposed through `config/app_config.json` for future tuning without code edits.
- `Output_Results` can now be cleared safely from both CLI and GUI without breaking the expected folder structure, because the app recreates `Frames/`, `Debug/`, and `Debug/Score_Frames/` immediately after cleanup.
- Position-guided player counts now use the highest convincing row index instead of collapsing at the first failed middle row, which fixes `12 -> 5` count failures and allows row `12` to count when any convincing position template is present there, even if template `11` visually wins.
- Review reasons are now shorter, deduplicated, capped for export, and no longer repeat connection-reset messages across later races after a detected reset.
- Session validation now preserves the original OCR total-score row position, preventing false `Scoreboard total order is not descending.` warnings after tournament-only `Position After Race` recomputation.
- Post-reset local `TotalScore` values are now validated against the reset-local session totals instead of continuing to trigger false tournament-total mismatches in later races.
- Score-frame debug annotations now save per-frame OCR bundle diagnostics under `Debug/Score_Frames/<Video>/Race_###/<2RaceScore|3TotalScore>/`, with native-resolution overlays, 1px segment boxes, and the legacy center-frame file preserved alongside the per-frame images.
- The seven-segment reader now uses one canonical explicit segment layout for RacePoints / OldTotalScore and scales that same layout into the larger TotalScore digit boxes.
- OCR junk punctuation on the edges of player names no longer creates false new identities when the underlying name, character, and totals clearly match an existing player.
- RacePoints can now reconcile to the `OldTotalScore -> TotalScore` delta when the bundle evidence supports the implied score better than the initially selected point read.

### Documentation
- Setup instructions were rewritten for hobbyist-friendly use from a Git clone.
- Project structure is now documented in plain language.
- Technical pipeline docs now explain that AAC `decode_pce` console warnings are usually noisy audio-stream warnings, not immediate video-decoding failures.
- Contributing expectations now explicitly cover naming, comments, cross-platform behavior, and validation.
