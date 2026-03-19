# Project Structure

This document explains the code layout in human terms.

## Big Picture

The project has two main phases:

1. Find the right screenshots inside the video
2. Read those screenshots and turn them into structured results

## Entry Points

- `mk8_local_play/`
  - real application package
  - contains the implementation modules that power the packaged CLI
- `pyproject.toml`
  - defines the packaged `mk8-local-play` command

## Extraction Modules

- `mk8_local_play/extract_frames.py`
  - extraction orchestrator
  - loads videos, scales the image, coordinates detection, and writes extracted frames
- `mk8_local_play/extract_initial_scan.py`
  - fast first scan over the video
  - looks for three anchor screens:
    - score screen
    - track-name screen
    - race-number screen
- `mk8_local_play/extract_score_screen_selection.py`
  - takes rough score detections and chooses the best race-score and total-score frames
- `mk8_local_play/extract_video_io.py`
  - shared helpers for frame reads, seeks, and export metadata
- `mk8_local_play/extract_common.py`
  - shared extraction utilities such as scaling, cropping, template matching, and GPU/runtime helpers

## OCR Modules

- `mk8_local_play/extract_text.py`
  - OCR/export orchestrator
  - groups screenshots into races and coordinates the OCR pipeline
- `mk8_local_play/ocr_scoreboard_consensus.py`
  - reads several nearby score frames
  - combines them into one best guess
  - maps race-score rows to total-score rows
- `mk8_local_play/ocr_name_matching.py`
  - fuzzy matching for noisy OCR player names across races
  - chooses a canonical player spelling for each row history
- `mk8_local_play/ocr_session_validation.py`
  - computes running totals
  - detects likely new sessions inside one source video
  - flags rows that need manual review
- `mk8_local_play/ocr_export.py`
  - writes the final workbook
  - builds the user-facing OCR completion summary
- `mk8_local_play/ocr_common.py`
  - shared OCR frame and metadata helpers

## Runtime And Configuration

- `mk8_local_play/app_runtime.py`
  - loads `config/app_config.json`
  - checks runtime dependencies
  - detects OpenCV GPU/OpenCL availability
- `config/app_config.json`
  - tracked runtime config used by setup and local runs
- `mk8_local_play/console_logging.py`
  - consistent operator-style logging and resource reporting

## Game Catalog

- `database/firestore-export.json`
  - local source export used to derive the compact catalog
- `reference_data/game_catalog.json`
  - single source of truth for cups, tracks, and characters
- `tools/build_game_catalog.py`
  - rebuilds the compact catalog from the Firestore export
- `mk8_local_play/game_catalog.py`
  - loader around the compact catalog
- `mk8_local_play/track_metadata.py`
  - compatibility wrapper for existing track tuple consumers
- `reference_data/track_reference_images/`
  - reference assets kept in-repo for manual checking and hobby use

## Assets And User Data

- `assets/templates/`
  - detection templates used during extraction
- `assets/gui/`
  - GUI background image
- `Input_Videos/`
  - user-provided source videos
- `Output_Results/Frames/`
  - extracted screenshots
- `Output_Results/Debug/`
  - optional debug output
- `Output_Results/*_Tournament_Results.xlsx`
  - timestamped Excel outputs
- `Output_Results/Debug/*_Tournament_Results_Debug.xlsx`
  - timestamped debug Excel outputs

## Scripts And Tools

- `scripts/setup_windows.ps1`
  - first-time Windows setup
- `scripts/setup_unix.sh`
  - first-time Linux/macOS setup
- `docs/LINUX_MACOS_SETUP.md`
  - short Linux/macOS setup guide
- `scripts/quick_benchmark.*`
  - lightweight benchmarking helpers
- `scripts/release_benchmark.*`
  - fuller benchmark flow for optimization passes
- `tools/validate_outputs.py`
  - compare a current run against a saved baseline

## Naming Rules Used In The Codebase

The project now tries to follow these rules:

- file names describe what the file does
- module names describe a domain, not an implementation accident
- terms like `initial scan` are preferred over vague labels like `pass1`
- comments explain intent and tradeoffs, not obvious syntax

That means a junior developer should be able to answer:
- what phase is this file responsible for
- what inputs it reads
- what outputs it produces
- why the logic exists
