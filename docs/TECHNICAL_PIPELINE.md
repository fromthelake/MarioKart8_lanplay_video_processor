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
- `assets/templates/Score_template.png`
- `assets/templates/Trackname_template.png`
- `assets/templates/Race_template.png`
- `assets/templates/12th_pos_template.png`

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
  - ROI: `(640, 590, 144, 48)`
  - threshold: `0.6`
  - skip after hit: `60s`

The matcher expands each ROI slightly before template matching to tolerate small capture shifts.

Current score-screen detection behavior:
- the initial scan checks both supported score layouts on each score-target frame
- the stronger passing match becomes the score candidate layout tag
- track-name and race-number detection are unchanged

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

These are used by:
- [extract_score_screen_selection.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_score_screen_selection.py)

Current RaceScore selection behavior:
- the first score-screen hit still seeds a provisional RaceScore export time
- when the 12th-place validation template is seen, the later RaceScore offset is now FPS-scaled instead of using a raw fixed-frame jump
- for races after race 1, if the previous TotalScore implies a 12-player field, the selector may walk a few frames later to find a valid 12th row before finalizing RaceScore
- for race 1, that same late-frame search is only attempted when the fixed-offset RaceScore frame looks like a plausible 11-of-12 case
- TotalScore timing itself is unchanged by this refinement
- `12th_pos_template.png` remains the shared template asset; only the search ROI changes by score layout

## 7. OCR Regions

Main OCR logic lives in:
- [extract_text.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_text.py)
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

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

Position templates come from:
- `assets/templates/Score_template.png`

The row windows currently use fixed starts:
- `0, 50, 102, 154, 206, 258, 310, 362, 414, 466, 518, 570`

Current row-count gating notes:
- the normal position-template presence threshold stays at `Coeff >= 0.60`
- row 1 now has a guarded exception at `Coeff >= 0.40` when the row is strongly occupied
- player count now uses the highest row with convincing position-strip presence instead of stopping at the first failed middle row
- for counting, the row index decides the player count; any convincing position template `1..12` may satisfy that row
- the winning template label on that row is debug signal only
- this protects both top-row screenshot overlays and late-row tie / neighbour confusion such as `11` winning visually on row `12`

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
- OCR worker count
- score-analysis worker count
- pass-1 scan worker count
- RaceScore consensus frame count

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

Validation / review behavior:
- session rebases remain visible in the debug export as an attention point
- session rebases no longer count as OCR review failures by themselves
- actual score mismatches, out-of-range values, and non-monotonic total-score rows still produce review reasons

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
- `Output_Results/Debug/Score_Frames/<VideoLabel>/Race_001/annotated_2RaceScore.<ext>`
- `Output_Results/Debug/Score_Frames/<VideoLabel>/Race_001/annotated_3TotalScore.<ext>`
- `Output_Results/Debug/Identity_Linking/<VideoLabel>/identity_linking.xlsx`
- `Output_Results/Debug/Low_Res/<VideoLabel>/identity_assignment.csv`
- `Output_Results/Debug/Low_Res/<VideoLabel>/identity_resolution.csv`

## 10A. Console Reporting Baseline

The console and GUI share the same runtime logger. Each new top-level run resets the timer and resource peaks, so elapsed timestamps always start at `00:00` for a fresh `Run`, `Selection`, or profiled run, even if the GUI has been open for a long time.

Current reporting conventions:
- elapsed timestamps shown in log prefixes are wall-clock time since the current run started
- `Run - Performance Summary` reports wall-clock phase durations
- OCR profiler sections explicitly label cumulative timing so they are not confused with wall-clock runtime
- OCR progress lines show completed groups plus `In flight: N` while race bundles are still running
- per-video run summaries use an aligned table for source length, processing time, race count, and player-count status

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

## 10B. OCR Performance Guardrails

The current OCR defaults are based on measured large-run benchmarks and should be treated as the performance baseline unless a new benchmark proves a better alternative.

Current intended defaults:
- `MK8_PLAYER_NAME_BATCH_RAW_MODE=weak`
- `MK8_PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE=50`
- `MK8_TOTAL_SCORE_NAME_ROW_FALLBACK_ENABLED=0`
- `MK8_TOTAL_SCORE_RACE_POINTS_ENABLED=0`
- `MK8_DIGIT_OCR_FALLBACK_ENABLED=0`

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
