# Codebase Map

## Summary

This project processes Mario Kart 8 LAN-play recordings into tournament workbooks.

Primary flow:
1. Read input videos from `Input_Videos/`
2. Normalize gameplay frames to a fixed `1280x720` working canvas
3. Detect track/race/score screens
4. Export frame bundles into `Output_Results/Frames/`
5. OCR and reconcile names, positions, scores, tracks, cups, and characters
6. Validate running totals and session boundaries
7. Export timestamped workbook files into `Output_Results/`

Runtime baseline:
- Python 3.12 only
- repo-local `.venv`
- EasyOCR used for text OCR
- FFmpeg required for repair and merge flows

## Architecture And Runtime Flow

### Entry points

- `mk8_local_play/main.py`
  Main CLI and optional Tk GUI entrypoint. Also performs runtime checks, output cleanup, selection scoping, per-run logger reset, runtime-setting persistence, overlap full-run orchestration, a scoped `--debug` override for headless runs, and end-to-end orchestration.
- `mk8_local_play/console_logging.py`
  Shared console logger and resource monitor for CLI/GUI output. Owns elapsed-time formatting, summary blocks, resource peaks, and the per-run timer reset behavior.
- `pyproject.toml`
  Defines the `mk8-local-play` console script pointing to `mk8_local_play.main:main`.

### Extraction phase

- `mk8_local_play/extract_frames.py`
  Extraction orchestrator. Loads videos, determines crop/upscale geometry, runs initial scan, runs score-screen selection, exports frames, and builds extraction summaries.
- `mk8_local_play/extract_initial_scan.py`
  Fast scan for track-name, race-number, and score-screen anchors. Uses fixed ROIs, segment-based scanning, row-box score detection, a row `2..6` confirmation prefix for score candidates, and multiple race-number template/ROI variants.
- `mk8_local_play/extract_score_screen_selection.py`
  Second pass over score candidates to choose RaceScore and TotalScore frames. Contains coarse-to-fine search, transition-centered RaceScore bundle export, FPS-scaled timing logic, 12th-place/template recovery, and tie-aware sustained-drop logic for TotalScore timing.
- `mk8_local_play/extract_video_io.py`
  Shared seek/read/grab helpers, corrupt-video sampling, and FFmpeg repair flow.
- `mk8_local_play/extract_common.py`
  Shared extraction constants, scaling, video discovery, and normalization helpers.

### OCR and identity phase

- `mk8_local_play/extract_text.py`
  OCR orchestrator. Groups exported frames, runs EasyOCR-based text extraction, coordinates consensus building, low-res handling, validation, export, and overlap-mode consumption of finalized per-video or per-race OCR jobs.
- `mk8_local_play/ocr_scoreboard_consensus.py`
  Core score-screen OCR logic: ROIs, row presence detection, position-template matching, score digit reading, character matching, and multi-frame consensus.
- `mk8_local_play/low_res_identity.py`
  Dedicated low-resolution identity path. Rebuilds identities from fixed ROIs, character matching, and blob fallback when OCR is too weak.
- `mk8_local_play/ocr_name_matching.py`
  Fuzzy matching and canonicalization for noisy player names across races, including duplicate-name chain resolution and targeted final-race ambiguity notes.
- `mk8_local_play/ocr_common.py`
  Shared OCR metadata and frame-loading helpers.

### Validation and export phase

- `mk8_local_play/ocr_session_validation.py`
  Validates totals, identifies session rebases/resets, attaches review reasons, and recomputes final post-race ordering from validated totals.
- `mk8_local_play/ocr_export.py`
  Builds user/debug export dataframes and writes timestamped workbook files.

### Metadata and runtime support

- `mk8_local_play/app_runtime.py`
  Loads `config/app_config.json`, persists simple runtime settings, checks FFmpeg, and reports GPU/OpenCL runtime status.
- `mk8_local_play/game_catalog.py`
  Loads the compact game catalog used for cups, tracks, and characters.
- `mk8_local_play/track_metadata.py`
  Compatibility wrapper around track metadata consumers.
- `mk8_local_play/project_paths.py`
  Defines `PACKAGE_ROOT` and `PROJECT_ROOT`.
- `mk8_local_play/data_paths.py`
  Resolves packaged asset/data paths.

## Module Map And Data

### Inputs

- `Input_Videos/`
  Source videos. The app can process the root only or recurse with `--subfolders`.
- `config/app_config.json`
  Runtime settings for worker counts, export image format, EasyOCR GPU/overlap modes, consensus frames, debug-output toggles, and low-res thresholds.
- `reference_data/game_catalog.json`
  Runtime metadata source for tracks, cups, and characters.
- `assets/templates/`
  Detection templates used during extraction and some OCR support logic.
- `assets/character/`, `assets/cup/`, `assets/gui/`
  Character templates, cup assets, and GUI art.

### Generated outputs

- `Output_Results/Frames/`
  Per-video race bundles used by OCR. Score-screen folders now persist the OCR bundle as `anchor_<frame>`, `consensus_<frame>`, and transition-centered `2RaceScore` context frames so `--selection` and `--ocr` reuse the same saved bundle intent.
- `Output_Results/*.xlsx` and `Output_Results/*.csv`
  Timestamped tournament outputs.
- `Output_Results/Debug/`
  Debug workbooks, CSVs, score-frame annotations, and score-layout demo images when debug output is enabled.
  Per-race debug images now mirror the `Frames/` video/race/bundle folder structure under `Debug/Score_Frames/<Video>/Race_###/<2RaceScore|3TotalScore>/`.

### Helper tools and scripts

- `scripts/setup_windows.ps1`, `scripts/setup_unix.sh`
  Supported setup paths.
- `scripts/quick_benchmark.*`, `scripts/release_benchmark.*`
  Benchmark helpers.
- `tools/validate_outputs.py`
  Compares current outputs against a stored baseline using PNG hashes and workbook row comparisons.
- `tools/run_with_perf_guard.py`
  Performance-guard helper for controlled runs.
- `tools/position_template_diagnostics.py`
  Diagnostics for left-side position template behavior.
- `tools/generate_name_ocr_debug_html.py`, `tools/evaluate_batch_name_consensus.py`
  OCR diagnostics and consensus analysis helpers.

## Run, Build, And Verification

Baseline verification:

```powershell
.\.venv\Scripts\python.exe -m compileall mk8_local_play
.\.venv\Scripts\python.exe -m mk8_local_play.main --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Common scoped commands:

```powershell
.\.venv\Scripts\python.exe -m mk8_local_play.main --extract --video <video-name>
.\.venv\Scripts\python.exe -m mk8_local_play.main --ocr --selection --video <video-name>
.\.venv\Scripts\python.exe -m mk8_local_play.main --all --selection --video <video-name>
.\.venv\Scripts\python.exe -m mk8_local_play.main --selection --subfolders --videos "2026-03-28/VideoA.mp4" "2026-03-28/VideoB.mp4"
```

Notes:
- `--all` is broader than `--selection`; it can include historical frame groups already present in `Output_Results/Frames`.
- `--selection` is the safer baseline for scoped verification because OCR stays limited to the selected video classes.
- `--videos` is the scoped multi-file variant. When combined with `--subfolders`, explicit relative paths are matched exactly before basename/stem fallback.
- Child scripts are expected to run through the repo-local `.venv`.
- The first curated baseline is `benchmarks/baselines/demo_capturecard_race/` and must be validated with both `--prefix Demo_CaptureCard_Race` and `--race-class Demo_CaptureCard_Race`.
- With EasyOCR CUDA enabled and more than one selected input video, overlap `auto` now defaults to streamed per-race OCR with two consumers. Explicit `video` / `race` mode overrides and higher consumer counts remain available for experiments.
- For headless debugging, `mk8-local-play.exe --selection --debug --video <video-name>` enables debug workbook/CSV and score-layout image output without changing normal CLI defaults.

## Major Dependencies

Python packages in `pyproject.toml`:
- `opencv-python`
- `numpy`
- `pandas`
- `openpyxl`
- `pillow`
- `easyocr`
- `jellyfish`
- `textdistance`
- `psutil`

External tools:
- FFmpeg

## Verified Facts, Strong Inferences, Unknowns

Verified:
- The repo is packaged via `pyproject.toml` and exposes `mk8-local-play`.
- `.\.venv\Scripts\python.exe -m compileall mk8_local_play` succeeds.
- `.\.venv\Scripts\python.exe -m mk8_local_play.main --check` succeeds in the current environment.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"` succeeds.
- The current environment reports EasyOCR importable and FFmpeg available.
- The repo now has a small stdlib `unittest` regression suite for config loading, output validation helpers, export formatting, session validation, and selection scoping.
- Scoring recomputation now resets running totals per video/race class, preventing repeated names in separate source videos from leaking tournament totals across exports.

Strong inferences:
- The practical regression strategy today is reproducible CLI runs plus output comparison, not unit-test-first development.
- The highest-risk logic is concentrated in score-frame selection, OCR consensus, low-res identity recovery, and session validation.

Unknowns:
- Only one curated benchmark baseline is currently committed under `benchmarks/baselines/demo_capturecard_race/`.
- The current large `Output_Results/` corpus is still useful for manual and ad hoc regression work, but it is not a controlled automated benchmark set.

## Risks And Technical Debt Hotspots

- Large monolithic modules:
  `extract_frames.py`, `extract_text.py`, and `ocr_scoreboard_consensus.py` carry a lot of behavior and are expensive to change safely.
- Heuristic sensitivity:
  score-screen timing, row visibility thresholds, position-template gates, and low-res fallback thresholds are behaviorally critical.
- Validation complexity:
  session rebases and reset handling can create subtle false positives or hidden regressions if changed casually.
- Verification gap:
  there is no repo-native automated test suite covering the core pipeline.
- Output sprawl:
  `Output_Results/` contains substantial historical artifacts; useful for reference, but easy to misuse as an uncontrolled baseline.

## Priority Improvement Opportunities

- Add small, stable regression tests around pure or mostly pure logic in:
  - session validation
  - export formatting
  - workbook comparison helpers
  - video identity and selection scoping
- Create one or more curated benchmark baselines under `benchmarks/baselines/` for repeatable output validation.
- Keep technical docs aligned whenever ROI geometry, score-layout behavior, or runtime commands change.

## Change-Control Defaults

- Existing behavior is the contract unless a verified bug or approved change says otherwise.
- Fixes should be minimal and contained.
- Do not auto-revert on intermediate metric drift alone.
- When final correctness is ambiguous, escalate for human evaluation before rollback.
