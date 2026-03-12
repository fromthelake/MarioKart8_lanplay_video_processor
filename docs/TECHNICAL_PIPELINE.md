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
  - ROI: `(315, 57, 52, 610)`
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
- scoreboard points ROI: `(290, 32, 102, 660)`
- 12th-place validation ROI: `(313, 632, 651, 88)`

These are used by:
- [extract_score_screen_selection.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_score_screen_selection.py)

## 7. OCR Regions

Main OCR logic lives in:
- [extract_text.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/extract_text.py)
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Important OCR geometry on the normalized `1280x720` frame:

### Player-name rows
Defined in:
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Current player name row boxes:
- x-range roughly `428..620`
- 12 stacked rows from top to bottom

### Position strip
Defined in:
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Current base ROI:
- `((315, 57), (367, 667))`

This is refined using:
- offset X/Y
- strip padding
- per-row padding

Position templates come from:
- `assets/templates/Score_template_fix.png`

The row windows currently use fixed starts:
- `0, 50, 102, 154, 206, 258, 310, 362, 414, 466, 518, 570`

### Character icons
Defined in:
- [ocr_scoreboard_consensus.py](/C:/Ai/MarioKart8_lanplay_video_processor/mk8_local_play/ocr_scoreboard_consensus.py)

Current character icon settings:
- left edge: `377`
- template size: `48x48`
- row step: `52`

Character template images are loaded from the asset folder using the character metadata catalog.

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
- debug output toggles
- optional Tesseract path override

Current score-screen bundle usage:
- `RaceScore` loads `APP_CONFIG.ocr_consensus_frames` frames, currently `7`
- `TotalScore` loads and uses `3` center frames

## 10. Output Files

Primary generated outputs:
- `Output_Results/Frames/`
  - extracted track / race / score images
- `Output_Results/*.xlsx`
  - final result workbook
- `Output_Results/Debug/`
  - optional debug and profiling artifacts

Frame naming convention uses the source video stem plus race number and frame kind, for example:
- `<VideoStem>+Race_001+0TrackName.png`
- `<VideoStem>+Race_001+1RaceNumber.png`
- `<VideoStem>+Race_001+2RaceScore.png`
- `<VideoStem>+Race_001+3TotalScore.png`

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
