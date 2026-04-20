# Technical Pipeline Reference

This document explains the current extraction and OCR pipeline at a technical level.

It is intended for contributors who want to:
- understand how the project finds Mario Kart result screens
- reproduce the current pipeline in another codebase
- adjust ROIs, templates, or metadata safely

## 1. Pipeline Overview

The processing flow is:

1. Open each input video with OpenCV.
2. Normalize each sampled frame onto a fixed `1280x720` working canvas.
3. Run a fast corruption preflight using sampled OpenCV seek/read probes.
4. If the file is suspect, repair it with `ffmpeg` into a working `.mp4`.
5. Run the initial scan to find:
   - track-name screens
   - race-number screens
   - score-screen candidates
6. Run the second pass to lock in race-score and total-score exports.
7. OCR the exported images.
8. Resolve track / cup / character metadata from the game catalog.
9. Export the final workbook.

Current runtime overlap behavior:
- default CPU runs remain sequential: extraction phase then OCR phase
- `overlap_ocr_mode=auto` is now the default
- when EasyOCR is resolved to CUDA and more than one input video is selected, full runs default to streamed per-race overlap
- in streamed race overlap, extraction remains the producer and OCR race jobs are queued as soon as each finalized race bundle is saved
- if CUDA-backed EasyOCR is unavailable, overlap `auto` resolves back to the existing sequential extraction-then-OCR behavior
- `overlap_ocr_consumers` defaults to `2`, while explicit `video` / `race` mode overrides and higher consumer counts remain available for experiments

Current extraction parallelism notes:
- a single video initial scan still uses overlapped parallel segments internally
- an additional multi-video initial-scan path is available through `MK8_PARALLEL_VIDEO_SCAN_WORKERS`
- when enabled with more than one selected video, the extractor prepares each video, runs the initial scan for multiple videos in one shared process pool, then keeps the later score-window selection and export path serial
- the default is `2` workers for multi-video runs
- on the current Windows benchmark laptop, `MK8_PARALLEL_VIDEO_SCAN_WORKERS=2` is the best verified setting so far
- higher values such as `3` and `4` oversubscribed CPU and memory bandwidth in local testing and were slower than `2`

## 2. Working Image Size

The extraction and OCR pipeline use a fixed working image size:

- width: `1280`
- height: `720`

Source of truth:
- [extract_common.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_common.py)
- [ocr_common.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_common.py)

The active gameplay region is detected by scanning for non-black borders, then cropped and upscaled onto this fixed canvas.

This means:
- black bars and capture padding are removed first
- all later ROI coordinates assume the normalized `1280x720` image

## 3. Initial Scan Templates

Initial detection templates live in:
- `assets/templates/Score_template_white.png`
- `assets/templates/Score_template_black.png`
- `assets/templates/Trackname_template.png`
- `assets/templates/Race_template.png`
- `assets/templates/Race_template_NL_final.png`
- `assets/templates/12th_pos_template.png`
- `assets/templates/12th_pos_templateNL.png`
- `assets/templates/ignore.png`
- `assets/templates/albumgallery_ignore.png`
- `assets/templates/ignore_2.png`

They are loaded in:
- [extract_frames.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_frames.py)

The initial scan target definitions live in:
- [extract_initial_scan.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_initial_scan.py)

Current initial-scan ROIs on the normalized `1280x720` image:

- score anchor:
  - LAN 2 ROI: `(315, 57, 52, 610)`
  - LAN 1 ROI: `(565, 57, 52, 610)`
  - threshold: `0.3`
  - skip after hit: `20s`
- track-name anchor:
  - ROI: `(141, 607, 183, 101)`
  - threshold: `0.6`
  - skip after hit: `0s`
- race-number anchor:
  - primary ROI: `(540, 590, 144, 48)`
  - alternate ROI: `(640, 590, 144, 48)`
  - Dutch alternate ROI: `(694, 594, 130, 40)`
  - threshold: `0.6`
  - skip after hit: `60s`
- ignore review bar:
  - ROI: `(413, 667, 808, 36)`
  - threshold: `0.75`
  - skip after hit: `5s`
- ignore album gallery:
  - ROI: `(660, 667, 561, 33)`
  - threshold: `0.62`
  - skip after hit: `5s`
- ignore album gallery alt:
  - ROI: `(556, 667, 664, 34)`
  - threshold: `0.62`
  - skip after hit: `5s`

The matcher expands each ROI slightly before template matching to tolerate small capture shifts.

Current score-screen detection behavior:
- the initial scan checks both supported score layouts on each score-target frame
- score detection is now driven by the left-side row-box position signal, not by a plain score-template coefficient
- that left-side signal is evaluated against the masked `Score_template_white.png` / `Score_template_black.png` row tiles
- the required prefix length comes from `POSITION_SCAN_MIN_PLAYERS`, currently defaulting to `6`
- the scan confirmation prefix now starts at row `2`, so rows `2..6` must match their own rank and the prefix average must clear the configured average floor
- row `1` is intentionally excluded from this confirmation prefix because Nintendo `Capture taken.` overlays can obscure first place while the rest of the scoreboard is still valid
- the stronger passing layout becomes the score candidate layout tag
- track-name detection still uses grayscale ROI preprocessing plus template matching
- race-number detection now supports both the legacy and Dutch template/ROI variants
- ignore-template matching runs on the same normalized `1280x720` post-crop frame, so the ROIs inherit the same black-border correction as the score templates
- any ignore hit above its threshold is treated as a hard veto and skips ahead before score candidates are queued
- parallel initial scan now also streams live score/track/race detection counts back to the parent process so long-running segment scans remain visible in the console

## 4. Corrupt-Video Preflight

Corrupt preflight now uses sampled OpenCV reads instead of a full `ffprobe -count_frames` walk.

Implementation:
- [extract_video_io.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_video_io.py)

Current strategy:
- `60` sampled frames per video
- `10` samples from the first `2%`
- `10` samples from the last `2%`
- remaining samples distributed across the middle
- duplicate frame numbers are removed automatically
- each read uses a timeout

A file is marked suspect if a probe:
- times out
- returns `read=False`
- lands materially away from the requested frame

If suspect, the file is repaired with `ffmpeg` to a working `.mp4`.

Console note:
- some OpenCV/FFmpeg builds print AAC warnings such as `decode_pce: Input buffer exhausted before END element found`
- in this project those messages usually come from the audio stream, not the video frames
- if preflight, scan, and OCR continue normally, treat them as noisy warnings rather than proof of a video-decoding failure

## 5. Repair Pipeline

Repair is handled in:
- [extract_video_io.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_video_io.py)

Current behavior:
- original is archived into `Input_Videos/corrupt/corrupt_<original-name>`
- repaired file is written back into `Input_Videos/<stem>.mp4`

Current repair command characteristics:
- `libx264`
- `yuv420p`
- `+genpts+discardcorrupt`
- `ignore_err`
- `-an`
- `-fps_mode cfr`

Repair progress is logged using ffmpeg progress output.

## 6. Score-Screen Selection ROIs

The second pass uses fixed ROIs on the normalized `1280x720` score image.

Defined in:
- [extract_frames.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_frames.py)

Current constants:
- LAN 2 scoreboard points ROI: `(290, 32, 102, 660)`
- LAN 1 scoreboard points ROI: `(540, 32, 102, 660)`
- LAN 2 12th-place validation ROI: `(313, 632, 651, 88)`
- LAN 1 12th-place validation ROI: `(313, 632, 651, 88)`
- Dutch 12th-place validation ROI: `(306, 658, 670, 41)`

These are used by:
- [extract_score_screen_selection.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_score_screen_selection.py)

Current RaceScore selection behavior:
- the first score-screen hit still seeds a provisional RaceScore export time
- before the first confirmed score hit, second-pass search now advances in coarse `+10` frame steps; once a hit is found it rewinds `10` frames and resumes fine `+1` scanning
- the second pass now uses the same row-box score helper as the initial scan instead of a standalone score-template coefficient
- when a supported 12th-place template is seen, the later RaceScore offset is FPS-scaled instead of using a raw fixed-frame jump
- both `12th_pos_template.png` and `12th_pos_templateNL.png` are checked; either one can trigger 12th-row logic
- a true 12th-place hit also expands the RaceScore consensus window so OCR can use a wider early/late bundle around the anchor
- saved `2RaceScore` context frames are now centered on the detected roll-up transition instead of a midpoint split, so OCR consumes the intended pre-rollup and post-transition bundle directly

Current TotalScore selection behavior:
- TotalScore is still timed from the end of the RaceScore phase rather than by a separate positive template detector
- a one-frame score drop is no longer enough to end RaceScore
- the selector now tracks the start of a continuous score-signal drop and confirms it only after `5.0 * fps` worth of uninterrupted absence
- that absence check uses a tie-aware row-prefix rule on rows `1..6`, so tied totals do not create a false drop
- once the drop is confirmed, the existing `-2.7s` timing offset is applied from the `drop_start_frame`, not from the later confirmation frame
- the current TotalScore baseline uses a learned timing fast-path:
  - after `race_score_frame`, transition detection first probes a very small primary window around the measured tipping moment
  - if that fails, it falls back to the older broad transition search
  - after the transition, stable-total detection first probes an early and a late cluster that were learned from the trace study
  - if neither cluster matches, it falls back to the older broad stable-total search
  - the old broad search path therefore still exists as a safety net, but most races avoid the unnecessary decoder work

## 7. OCR Regions

Main OCR logic lives in:
- [extract_text.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_text.py)
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Current OCR runtime notes:
- EasyOCR now defaults to `auto` mode through `config/app_config.json:easyocr_gpu_mode` or the GUI
- `auto` uses CUDA when PyTorch can see it and otherwise falls back to CPU
- `gpu` requests CUDA explicitly and falls back to CPU with a clear runtime message if CUDA is unavailable
- `cpu` disables GPU OCR even when CUDA is present
- when GPU OCR is active, the pipeline forces effective OCR workers to `1`
- this is intentional; local benchmarks showed per-process EasyOCR worker counts above `1` scale poorly, while the meaningful throughput gain comes from overlap scheduling across multiple OCR consumers
- extraction `auto` prefers CUDA and otherwise falls back to CPU
- extraction OpenCL remains available only through explicit `execution_mode=gpu`; it is no longer selected automatically

Important OCR geometry on the normalized `1280x720` frame:

### Player-name rows
Defined in:
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Current player name row boxes:
- LAN 2 x-range roughly `428..620`
- LAN 1 x-range roughly `678..870`
- 12 stacked rows from top to bottom

### Position strip
Defined in:
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Current base ROI:
- LAN 2: `((315, 57), (367, 667))`
- LAN 1: `((565, 57), (617, 667))`

This is refined using:
- offset X/Y
- strip padding
- per-row padding

The default OCR position-template matcher now uses masked black/white tiles from:
- `assets/templates/Score_template_white.png`
- `assets/templates/Score_template_black.png`

Current tile windows:
- tile size: `52 x 52`
- LAN 2 tile ROI starts at about `(313, 46)`
- LAN 1 tile ROI starts at about `(563, 46)`
- row step: `52`

Current row-count gating notes:
- the normal position-template presence threshold stays at `Coeff >= 0.50`
- row 1 uses the dedicated row-1 presence threshold `Coeff >= 0.30`
- player count now uses the highest row with convincing position-strip presence instead of stopping at the first failed middle row
- player-count support is position-template based (no occupancy-score requirement in the final gate)
- for counting, the row index decides the player count; any convincing position template `1..12` may satisfy that row
- the winning template label on that row is debug signal only
- this protects both top-row screenshot overlays and late-row tie / neighbour confusion such as `11` winning visually on row `12`
- for TotalScore drop confirmation, the code also supports a tie-aware prefix where a row may accept any non-decreasing rank up to its own row number
- transition debounce keeps a fixed confirm-hit target (`p5` by default), while false-gap tolerance is FPS-scaled so timing-equivalent gap tolerance is preserved across 30fps/60fps videos
- the black/white matcher checks the white tile first, then the black tile, and keeps the stronger masked score for that row

### Character icons
Defined in:
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)
- [low_res_identity.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/low_res_identity.py)

Current character icon settings:
- LAN 2 left edge: `377`
- LAN 1 left edge: `627`
- template size: `48x48`
- row step: `52`

Character template images are loaded from the asset folder using the character metadata catalog.

Current character matching behavior:
- the main score-row character matcher compares roster templates with aligned alpha-cutout scoring across calibrated local offsets
- family refinement runs after OCR/identity standardization and before the conservative Mii fallback
- family refinement uses aligned alpha-cutout color scoring on saved `2RaceScore` anchor crops, then applies a player-level dominance gate before changing exported characters
- catalog-backed family groups currently include `Birdo`, `Yoshi`, `Shy Guy`, and `Inkling`
- explicit close-cutout groups currently include `Peach`, `Cat Peach`, `Baby Peach`, `Pink Gold Peach`, `Peachette`, and `Mario`, `Metal Mario`, `Gold Mario`

Low-resolution identity fallback notes:
- low-resolution runs now keep position-confirmed visible rows even when OCR is too weak to supply a usable name or score
- low-resolution character matching uses a fixed net ROI per row and `51x52` resized character templates
- race 1 uses full character search; later races reuse the previous race as the shortlist basis
- when a low-resolution race score screen falls to `11` rows but the video context still implies `12`, a combined `character + player-name` blob ROI can restore row 12 as a fallback
- the low-resolution ROI/template tuning values and blob thresholds are runtime-configurable through `config/app_config.json`
- the same LAN 1 vs LAN 2 score-layout selection also applies to the low-res and ultra-low-res score-row geometry

Current score digit origins:
- LAN 2 race points start: `[(830, 71), (843, 71)]`
- LAN 1 race points start: `[(1080, 71), (1093, 71)]`
- LAN 2 total points start: `[(916, 66), (933, 66), (950, 66)]`
- LAN 1 total points start: `[(1166, 66), (1183, 66), (1200, 66)]`

### Track name OCR
Defined in:
- [extract_text.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_text.py)

Current track-name OCR ROI:
- `((319, 633), (925, 685))`

## 8. Metadata JSON / Catalog

The runtime metadata source is:
- `reference_data/game_catalog.json`

Loaded by:
- [game_catalog.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/game_catalog.py)

The catalog currently contains:
- cups
- tracks
- characters

It provides the canonical identifiers and names used during export.

The compact catalog is derived from a larger local source export mentioned elsewhere in the repo, but the runtime pipeline reads the compact `game_catalog.json` file.

## 9. Main Config Inputs

Runtime config comes from:
- `config/app_config.json`
- `config/app_config.json`

Loaded by:
- [app_runtime.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/app_runtime.py)

Important settings include:
- extraction GPU mode
- EasyOCR GPU mode
- overlap OCR mode
- overlap OCR consumers
- OCR worker count
- score-analysis worker count
- pass-1 scan worker count
- RaceScore consensus frame count

Current GPU-related settings:
- `execution_mode`
  - default: `cpu`
  - values: `auto`, `gpu`, `cpu`
  - controls the OpenCV extraction/runtime GPU path
  - `cpu` is the current default because it benchmarked better than OpenCL on the active laptop profile
  - `auto` prefers CUDA and otherwise uses CPU
  - `gpu` allows OpenCL fallback when CUDA is unavailable
- `easyocr_gpu_mode`
  - default: `auto`
  - values: `auto`, `gpu`, `cpu`
  - controls EasyOCR name/track OCR GPU use
  - the OCR phase forces effective OCR workers to `1` in GPU mode, because that was the best measured configuration on the current test machine
- `overlap_ocr_mode`
  - default: `auto`
  - values: `auto`, `video`, `race`
  - `auto` resolves to streamed per-race overlap only when EasyOCR CUDA is available
  - `video` keeps the older per-video overlap scheduler
  - `race` forces the experimental per-race overlap scheduler
- `overlap_ocr_consumers`
  - default: `2`
  - controls the number of spawned OCR consumers used by overlap mode
  - higher values remain experimental and should be benchmarked on representative multi-video inputs

## 10. Output Maintenance

The application now supports clearing generated output safely without leaving the runtime in a broken state.

Implemented in:
- [main.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/main.py)

Available entrypoints:
- CLI: `python -m mk8_local_play.main --clear-output-results`
- GUI: `Clear Output Results`

Both paths require an explicit `Are you sure?` confirmation before deleting anything.

Current cleanup behavior:
- deletes all files and subfolders under `Output_Results/`
- immediately recreates:
  - `Output_Results/Frames/`
  - `Output_Results/Debug/`
  - `Output_Results/Debug/Score_Frames/`

This keeps extraction, OCR, profiling, and debug-export code paths working after a full cleanup.
- debug output toggles

Current score-screen bundle usage:
- `RaceScore` loads `APP_CONFIG.ocr_consensus_frames` frames, currently `7`
- `TotalScore` loads and uses `3` center frames
- OCR reads the chosen score layout from exported frame metadata first and from the score-frame filename as fallback

Current digit-read strategy:
- the primary score reader is a fixed 7-segment-style pixel signature check
- OCR fallback is only used when the recognized digits contain a real gap inside the digit block or the parsed value is invalid
- normal left or right padding in a digit row no longer triggers OCR fallback by itself
- this applies to `DetectedRacePoints`, `DetectedOldTotalScore`, and `DetectedNewTotalScore`
- `DetectedRacePoints` and `DetectedOldTotalScore` use the canonical explicit segment layout directly
- `DetectedNewTotalScore` / `DetectedTotalScore` now uses the same canonical layout scaled into the larger TotalScore digit box
- optional annotated score-frame debug output saves both the center `2RaceScore` frame and the center `3TotalScore` frame as 5x images with:
  - cyan row boxes
  - yellow digit boxes
  - green/red segment boxes for on/off
  - the detected digit in the top-left corner of each digit box

Current `RacePoints` runtime segment settings:
- `white_threshold = 180`
- `active_ratio_threshold = 0.45`
- `top_middle: x=28, y=8, width=17, height=9`
- `left_middle: x=9, y=17, width=16, height=23`
- `middle_middle: x=32, y=21, width=9, height=18`
- `right_middle: x=51, y=17, width=15, height=23`
- `left_bottom: x=9, y=57, width=16, height=23`
- `middle_bottom: x=32, y=59, width=9, height=18`
- `right_bottom: x=51, y=57, width=16, height=23`
- `middle_bottom_edge: x=28, y=82, width=17, height=9`
- `center: x=24, y=44, width=25, height=9`

Low-resolution behavior:
- videos with source height `<= 479` use a dedicated low-res name pipeline
- low-res keeps the normal frame selection and player-count detection
- low-res replaces only player identity / name resolution with fixed name ROI + character ROI + global assignment
- low-res does not use OCR race points or OCR total scores
- low-res computes race points from position/player count and rebuilds totals cumulatively
- unresolved low-res identities remain `PlayerNameMissing_X`
- after score OCR and identity standardization, a session-level Mii fallback can relabel a player to `Mii` when the saved `2RaceScore` frames show repeatedly weak, near-tied, unstable non-Mii character winners
- those rows receive review reason `mii_fallback_unstable_character_match`
- before the Mii fallback, character-family refinement can now stabilize variant families from saved `2RaceScore` anchors:
  - catalog-backed color families such as `Birdo`, `Yoshi`, `Shy Guy`, and `Inkling`
  - the explicit `Peach` family: `Peach`, `Cat Peach`, `Baby Peach`, `Pink Gold Peach`, `Peachette`
  - the explicit `Mario` family: `Mario`, `Metal Mario`, `Gold Mario`
- family refinement uses aligned alpha-cutout color scoring rather than the older unaligned diagnostic HSV scoring in production
- debug exports now expose the family comparison directly through `Character Family`, `Character Family Best`, `Character Family Best Coeff`, `Character Family Second`, `Character Family Second Coeff`, and `Character Family Margin`

Validation / review behavior:
- session rebases remain visible in the debug export as an attention point
- session rebases no longer count as OCR review failures by themselves
- actual score mismatches, out-of-range values, and non-monotonic total-score rows still produce review reasons
- final-race duplicate-name ambiguity notes are now only attached to the rows that remain truly interchangeable, and the note names the conflicting identity label(s)

## 10. Output Files

Primary generated outputs:
- `Output_Results/Frames/`
  - per-video race bundles with extracted track / race / score images and persisted OCR consensus frames
- `Output_Results/*.xlsx`
  - final result workbook
- `Output_Results/Debug/`
  - optional debug and profiling artifacts

Frame bundle layout now mirrors the OCR input structure more directly:
- `Output_Results/Frames/<VideoLabel>/Race_001/0TrackName.<ext>`
- `Output_Results/Frames/<VideoLabel>/Race_001/1RaceNumber.<ext>`
- `Output_Results/Frames/<VideoLabel>/Race_001/2RaceScore/anchor_<frame>.<ext>`
- `Output_Results/Frames/<VideoLabel>/Race_001/2RaceScore/consensus_<frame>.<ext>`
- `Output_Results/Frames/<VideoLabel>/Race_001/3TotalScore/anchor_<frame>.<ext>`
- `Output_Results/Frames/<VideoLabel>/Race_001/3TotalScore/consensus_<frame>.<ext>`

`<ext>` comes from `config/app_config.json` -> `export_image_format`.
The current default is `jpg`; `png` remains available as a lossless fallback.
The numeric suffix is the actual decoded source-video frame number used for that image.

Annotated score-layout demo images are also written under:
- `Output_Results/Debug/Score_Layout_Demos/`

These show the active score ROIs on exported `2RaceScore` and `3TotalScore` frames for human review.

Other debug artifacts are grouped by video so the debug tree sorts like `Output_Results/Frames/`:
- `Output_Results/Debug/Score_Frames/<VideoLabel>/Race_001/2RaceScore/annotated_2RaceScore.<ext>`
- `Output_Results/Debug/Score_Frames/<VideoLabel>/Race_001/2RaceScore/annotated_2RaceScore_frame_<frame>.<ext>`
- `Output_Results/Debug/Score_Frames/<VideoLabel>/Race_001/3TotalScore/annotated_3TotalScore.<ext>`
- `Output_Results/Debug/Score_Frames/<VideoLabel>/Race_001/3TotalScore/annotated_3TotalScore_frame_<frame>.<ext>`
- `Output_Results/Debug/Identity_Linking/<VideoLabel>/identity_linking.xlsx`
- `Output_Results/Debug/Low_Res/<VideoLabel>/identity_assignment.csv`
- `Output_Results/Debug/Low_Res/<VideoLabel>/identity_resolution.csv`

## 10A. Console Reporting Baseline

The console and GUI share the same runtime logger. Each new top-level run resets the timer and resource peaks, so elapsed timestamps always start at `00:00:00` for a fresh `Run`, `Selection`, or profiled run, even if the GUI has been open for a long time.

Current reporting conventions:
- elapsed timestamps shown in log prefixes are wall-clock time since the current run started
- live progress rows use aligned `Comp` / `Done` fields and include CPU/RAM/GPU where useful for stall detection
- RAM in progress and phase summaries is reported as percentage
- confirmed scan detections are emitted in frame order as `Race ### | Track/Race/Score | Source HH:MM:SS | Frame #######`
- `Run - Performance Summary` reports wall-clock run timing and split phase timings
- OCR profiler sections explicitly label cumulative timing so they are not confused with wall-clock runtime
- OCR progress lines show completed groups plus `Active N` while race bundles are still running
- OCR finalization is split into OCR race reading, validation/export, validation logic, and workbook/CSV export timings
- per-video run summaries use an aligned table for source length, wall time, stage-sum time, scan time, score time, OCR time, finalization time, processing rate, race count, player-count status, and review status
- each selected video is assigned a stable neon console accent for the current run
- labels remain neutral while video-owned values use that video's color across input summary, live scan, score selection, overlap OCR, and per-video summary rows
- `Time saved by overlap` is reported as the cumulative per-video elapsed wall-clock minus total run wall-clock, so users can see the benefit from overlap and parallelism directly

Current OCR profiling sections:
- `OCR Call Matrix`
- `Totals By Field`
- `Totals By Bundle`
- `OCR engine profile (cumulative across all OCR calls)`
- `Observation stage profile (cumulative across all observations)`

When changing console output, preserve the distinction between:
- wall-clock run timing
- cumulative OCR engine timing
- cumulative observation-stage timing
- validation/export finalization timing

## 10B. OCR Performance Guardrails

The current OCR defaults are based on measured large-run benchmarks and should be treated as the performance baseline unless a new benchmark proves a better alternative.

Current intended defaults:
- `MK8_PLAYER_NAME_BATCH_RAW_MODE=weak`
- `MK8_PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE=50`
- `MK8_TOTAL_SCORE_NAME_ROW_FALLBACK_ENABLED=0`
- `MK8_TOTAL_SCORE_RACE_POINTS_ENABLED=0`
- `MK8_DIGIT_OCR_FALLBACK_ENABLED=0`
- `easyocr_gpu_mode=auto`

These defaults were chosen after profiling showed that the old path spent most of its time on work that did not improve final exported results:
- reading `RacePoints` on `3TotalScore` frames
- per-row digit OCR fallback after seven-segment parsing
- unconditional `batch_raw` name OCR on every observation
- `3TotalScore` row-level name fallback

When changing OCR process flow, preserve these rules unless a benchmark on representative multi-race inputs shows a better result:
- `3TotalScore` does not carry race points and should not trigger race-point OCR work.
- Seven-segment digit parsing is the primary digit reader. Re-enable digit OCR fallback only with measured proof that it improves final business output.
- `inv_otsu` is the primary batch name OCR path. Additional raw OCR should stay conditional unless profiling proves otherwise.
- Any change to fallback thresholds or OCR pass counts should be validated on larger extracted race classes, not only on the single-race demo.
- In GPU OCR mode, keep effective OCR workers at `1` unless a new benchmark proves a better per-process OCR worker count on representative source videos.
- Overlap defaults now use `overlap_ocr_mode=auto` and `overlap_ocr_consumers=2`; keep higher overlap consumer counts experimental unless a new benchmark shows a stable throughput win.
 
## 11. Files To Read First

If you want to continue development, start here:
- [main.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/main.py)
- [extract_frames.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_frames.py)
- [extract_initial_scan.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_initial_scan.py)
- [extract_score_screen_selection.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_score_screen_selection.py)
- [extract_text.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_text.py)
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)
- [game_catalog.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/game_catalog.py)

## 12. Maintenance Note

If you change any ROI, template, or working-canvas assumption, update this document at the same time.
