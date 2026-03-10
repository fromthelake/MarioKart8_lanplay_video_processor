# Mario Kart 8 Local Play Video Processor

This project turns Mario Kart 8 local-play capture videos into a clean Excel results sheet.

In plain language:
- you place one or more recorded match videos in `Input_Videos/`
- the tool finds the important result screens
- the tool reads track names, player names, race points, and total scores
- the tool writes timestamped Excel files to `Output_Results/`

The project is designed for hobby use from a Git clone and supports:
- Windows
- Linux
- macOS

If you want a full beginner walkthrough with terminal hotkeys, install checks, and step-by-step platform instructions, read:
- [BEGINNER_SETUP.md](./BEGINNER_SETUP.md)

## Start Here

### I just want to run it

- Windows:
  - `.\scripts\setup_windows.ps1`
  - `.\.venv\Scripts\mk8-local-play.exe --all`
- Linux/macOS:
  - `chmod +x ./scripts/setup_unix.sh`
  - `./scripts/setup_unix.sh`
  - `.venv/bin/mk8-local-play --all`

### I want to tweak settings

- open `app_config.example.json`
- copy it to `app_config.json`
- change worker counts, debug flags, or `tesseract_cmd`

### I want to benchmark changes

- use the scripts in `scripts/`
- compare against a curated baseline with `tools/validate_outputs.py`

## Who This Is For

This repo is for people who:
- record Mario Kart 8 local-play sessions
- want a faster way to convert result screens into tournament standings
- want a tool they can inspect, tune, and benchmark themselves

You do not need to understand the code to use it. If you can open a terminal and follow a short setup guide, you can run it.

## What The Tool Actually Does

The processing pipeline has two major stages.

1. Frame extraction
- scans each input video
- finds the useful Mario Kart result screens
- exports those screens into `Output_Results/Frames/`

2. OCR and validation
- reads the exported screenshots
- combines nearby frames to reduce OCR mistakes
- matches fuzzy player names across races
- checks race points and running totals
- resolves tracks, cups, and future character metadata from one compact game catalog
- writes an Excel workbook

Metadata note:
- the repo runtime uses `reference_data/game_catalog.json` as the single source of truth
- that compact catalog is derived locally from `database/firestore-export.json`

## Quick Start

### Windows

```powershell
.\scripts\setup_windows.ps1
.\.venv\Scripts\mk8-local-play.exe --all
```

### Linux or macOS

```bash
chmod +x ./scripts/setup_unix.sh
./scripts/setup_unix.sh
.venv/bin/mk8-local-play --all
```

## What You Need

Required:
- Python 3.10 or newer
- Tesseract OCR

Optional:
- FFmpeg
  - only needed for the merge-video feature

Recommended:
- Python 3.12
- a local virtual environment in `.venv`

## First-Time Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd MarioKart8_lanplay_video_processor
```

### 2. Install Python dependencies

Use one of the setup scripts:
- Windows: `.\scripts\setup_windows.ps1`
- Linux/macOS: `./scripts/setup_unix.sh`

These scripts:
- create `.venv` if needed
- install the project in editable mode with its dependencies
- create `app_config.json` from `app_config.example.json` if needed
- run the installed command with `--check`

### 3. Install Tesseract OCR

Windows:
- install Tesseract OCR
- default location usually works automatically
- if not, set `tesseract_cmd` in `app_config.json`

Linux:
- usually package name `tesseract-ocr`

macOS:
- easiest route is Homebrew:

```bash
brew install tesseract
```

### 4. Run the built-in environment check

```bash
python main.py --check
```

Important detail:
- `main.py` now prefers the repo-local `.venv` automatically when it exists
- that means `python main.py ...` still uses the project environment even if your shell started from a different Python
- after setup, you also get these commands inside the virtual environment:
  - `mk8-local-play`
  - `mk8-local-results`
- on Windows, run them as `.venv\Scripts\mk8-local-play.exe`
- on Linux/macOS, run them as `.venv/bin/mk8-local-play`

## Normal Usage

### Put videos in the input folder

Place your recordings in:
- `Input_Videos/`

### Run the full pipeline

Windows:

```powershell
.\.venv\Scripts\mk8-local-play.exe --all
```

Linux/macOS:

```bash
.venv/bin/mk8-local-play --all
```

### Run only extraction

Windows:

```powershell
.\.venv\Scripts\mk8-local-play.exe --extract
```

Linux/macOS:

```bash
.venv/bin/mk8-local-play --extract
```

### Run only OCR/export

Windows:

```powershell
.\.venv\Scripts\mk8-local-play.exe --ocr
```

Linux/macOS:

```bash
.venv/bin/mk8-local-play --ocr
```

### Run only one video

Windows:

```powershell
.\.venv\Scripts\mk8-local-play.exe --all --video Demo_CaptureCard_Race.mp4
```

Linux/macOS:

```bash
.venv/bin/mk8-local-play --all --video Demo_CaptureCard_Race.mp4
```

## GUI

GUI support:
- Windows: supported
- Linux: supported when Tk is available in the Python build
- macOS: supported when Tk is available in the Python build

Start the GUI with:

Windows:

```powershell
.\.venv\Scripts\mk8-local-play.exe
```

Linux/macOS:

```bash
.venv/bin/mk8-local-play
```

If Tk is not available, the CLI still works.

## Expected Input

Best results come from:
- Mario Kart 8 local play
- vertical split-screen layout
- clear capture-card footage
- no heavy stream overlays
- no cropped game image

Less reliable inputs:
- heavily compressed videos
- webcam overlays
- unusual layouts
- aggressive cropping

## Known Limitations

- best results come from vertical split-screen local-play captures
- stream overlays and webcam boxes can hide OCR targets
- strong compression can reduce name and score accuracy
- this project is tuned for the Mario Kart 8 result layout it knows about, not arbitrary video layouts

## Output

Main output:
- the newest timestamped workbook in `Output_Results/*_Tournament_Results.xlsx`

Additional outputs:
- timestamped debug workbooks in `Output_Results/Debug/*_Tournament_Results_Debug.xlsx`
- extracted frame screenshots in `Output_Results/Frames/`
- optional debug artifacts in `Output_Results/Debug/`

## Repository Layout

- `main.py`
  - thin root launcher for the main CLI and GUI
- `extract_frames.py`
  - thin root launcher for extraction-only runs
- `extract_text.py`
  - thin root launcher for OCR-only runs
- `mk8_local_play/`
  - real application package
  - contains the extraction, OCR, runtime, and orchestration modules
- `assets/`
  - GUI images and detection templates
- `scripts/`
  - setup and benchmark helpers
- `benchmarks/baselines/`
  - curated comparison baselines for performance and regression checks
- `reference_data/`
  - track metadata and manual reference images kept with the repo
- `tools/validate_outputs.py`
  - compare current outputs against a baseline
- `docs/`
  - maintainer-facing project documentation

## Configuration

Copy `app_config.example.json` to `app_config.json` if you want to tune behavior manually.

Main settings:
- `tesseract_cmd`
- `execution_mode`
- `ocr_workers`
- `score_analysis_workers`
- `pass1_scan_workers`
- `ocr_consensus_frames`
- `write_debug_csv`
- `write_debug_score_images`
- `write_debug_linking_excel`

Environment variables can also override these settings.

## Why The Project Also Supports `pip install -e .`

This repo still works fine as plain Python scripts.

The editable install mainly improves the hobbyist experience:
- installs dependencies in one standard step
- exposes human-readable commands
- makes Windows, Linux, and macOS usage more consistent
- keeps the code linked to your Git checkout, which is ideal for local tuning and benchmarking
- makes updates easy after a `git pull`

Recommended update flow from a Git checkout:

Windows:

```powershell
git pull
.\scripts\setup_windows.ps1
```

Linux/macOS:

```bash
git pull
./scripts/setup_unix.sh
```

If you prefer to reinstall manually instead of rerunning the setup script:

- Windows: `.\.venv\Scripts\python.exe -m pip install -e .`
- Linux/macOS: `.venv/bin/python -m pip install -e .`

The primary command names are:
- `mk8-local-play`
- `mk8-local-results`

Both commands run the same CLI as `main.py`.

## Benchmarking And Regression Checks

Benchmark baselines stay in the repo on purpose.

They are useful when:
- testing OCR changes
- measuring speedups
- checking whether frame exports changed

Useful scripts:
- `scripts/quick_benchmark.ps1`
- `scripts/quick_benchmark.sh`
- `scripts/release_benchmark.ps1`
- `scripts/release_benchmark.sh`

Validation tool:

```bash
python tools/validate_outputs.py --baseline-dir benchmarks/baselines/<your-baseline>
```

## Troubleshooting

### `Tesseract: MISSING`

Install Tesseract or set `tesseract_cmd` in `app_config.json`.

### GUI does not start

Use the CLI instead:

```bash
python main.py --all
```

On Linux and macOS, GUI support depends on Tk being present in the Python build.

### OCR works from `.venv` but not from `python`

This should now be handled automatically by `main.py`, which prefers the repo-local `.venv`.

### Output looks wrong

Check:
- the video layout matches the expected local-play format
- overlays are not covering names or points
- the extracted screenshots in `Output_Results/Frames/` look correct

## Extra Documentation

- [ReadMe.txt](./ReadMe.txt): plain-text pointer to the main documentation
- [BEGINNER_SETUP.md](./BEGINNER_SETUP.md): step-by-step beginner installation guide for Windows, Linux, and macOS
- [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md): development and repo hygiene notes
- [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md): what each module does
- [docs/RELEASE_CHECKLIST.md](./docs/RELEASE_CHECKLIST.md): pre-release sanity checklist
- [docs/CHANGELOG.md](./docs/CHANGELOG.md): notable project changes
- [LICENSE](./LICENSE): hobby/private-use license and liability disclaimer

## Legal Note

This repository only provides the program code around the processing workflow.

- It does not claim ownership of Nintendo intellectual property.
- It is meant for hobby and private-use scenarios.
- Users remain responsible for respecting all third-party rights when using it.

## Primary Entry Points

Use these files:
- `main.py`
- `extract_frames.py`
- `extract_text.py`
