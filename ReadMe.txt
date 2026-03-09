Mario Kart 8 Local Play Video Processor

This project scans Mario Kart 8 local play vertical split-screen videos, extracts the important race result screenshots, reads the names and scores from those screenshots, and exports the results to an Excel workbook.

Final output:
- `Output_Results/Tournament_Results.xlsx`

In simple terms:
1. you put one or more videos into `Input_Videos`
2. the tool finds the race result screens
3. the tool reads the text and points
4. the tool writes the final tournament results to Excel

Index

1. What This Tool Does
2. What You Need Before You Start
3. Video Requirements
4. Windows Guide
5. Linux Guide
6. macOS Guide
7. Command Reference
8. What Output To Expect
9. Configuration
10. Benchmark And Validation Scripts
11. Troubleshooting
12. Notes About Git And Generated Files

1. What This Tool Does

The tool is built for Mario Kart 8 local play videos with a vertical split-screen layout.

It creates:
- extracted race screenshots in `Output_Results/Frames`
- optional debug files in `Output_Results/Debug`
- a final Excel workbook in `Output_Results/Tournament_Results.xlsx`

The extracted screenshots usually include these types:
- `0TrackName`
- `1RaceNumber`
- `2RaceScore`
- `3TotalScore`

Those screenshots are the key screens used to build the Excel output.

2. What You Need Before You Start

Before running any command, install these:

1. Python
- Python `3.12` is recommended
- Python `3.10+` should usually work
- using a virtual environment is recommended

2. Tesseract OCR
- required
- used to read player names and track text

3. FFmpeg
- optional
- only needed if you want to use the video merge feature

If you only remember one thing from this section, remember this:
- install Python first
- then install dependencies
- then run `Main_RunMe.py --check`

3. Video Requirements

Best supported input:
- Mario Kart 8 local play
- vertical split-screen
- clear capture-card footage
- no large overlays
- full game image visible

Less reliable input:
- heavy compression
- cropped videos
- stream overlays
- webcam boxes
- unusual layouts

If the tool struggles, the first thing to verify is whether the video format matches the expected layout.

4. Windows Guide

Windows is the easiest platform for this project because it supports both:
- GUI mode
- command-line mode

Windows Setup

1. Install Python 3.12
2. Install Tesseract OCR
3. Optionally install FFmpeg if you want video merging
4. Open PowerShell in the project folder

Fastest Windows Setup

Run:

```powershell
.\scripts\setup_windows.ps1
```

What this command does:
- creates `.venv` if needed
- installs Python packages from `requirements.txt`
- creates `app_config.json` from `app_config.example.json` if needed
- runs the built-in environment check

What you should expect:
- Python package installation output
- a runtime summary from `Main_RunMe.py --check`
- ideally:
  - `Tesseract: OK`
  - `FFmpeg: OK` or `FFmpeg: MISSING` if you did not install FFmpeg

If Tesseract says `MISSING`, install it first or set the path in `app_config.json`.

Windows First Run

After setup:

```powershell
.\.venv\Scripts\python.exe Main_RunMe.py --check
```

What this command does:
- checks whether Python, Tesseract, FFmpeg, and config values are available

What you should expect:
- a printed summary showing paths and status
- for a normal OCR-capable setup:
  - `Tesseract: OK`

Then place your videos in:
- `Input_Videos`

Then run:

```powershell
.\.venv\Scripts\python.exe Main_RunMe.py --all
```

What this command does:
- runs frame extraction
- runs OCR/export

What you should expect:
- operator-style console output showing live phase progress, resource usage, and detected races
- screenshots created in `Output_Results/Frames`
- final Excel created at `Output_Results/Tournament_Results.xlsx`

To test a single video without moving files around:

```powershell
.\.venv\Scripts\python.exe Main_RunMe.py --all --video Test_3_Races.mkv
```

Windows GUI Mode

If you prefer the GUI:

```powershell
.\.venv\Scripts\python.exe Main_RunMe.py
```

What this does:
- opens the Windows GUI

What you should expect:
- buttons for opening folders, extracting races, exporting to Excel, and merging videos

5. Linux Guide

Linux works best in command-line mode.

Linux Setup

Install system dependencies first:

```bash
sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip tesseract-ocr
```

If you want the merge feature:

```bash
sudo apt-get install ffmpeg
```

Fastest Linux Setup

Run:

```bash
chmod +x ./scripts/setup_unix.sh
./scripts/setup_unix.sh
```

What this command does:
- creates `.venv` if needed
- installs Python packages
- creates `app_config.json` if needed
- runs the environment check

What you should expect:
- package installation output
- a final check summary
- ideally:
  - `Tesseract: OK`

Linux First Run

Run:

```bash
.venv/bin/python Main_RunMe.py --check
```

What this command does:
- verifies the runtime environment

Expected result:
- shows folder paths
- shows `Tesseract: OK`

Then place your videos in:
- `Input_Videos`

Then run:

```bash
.venv/bin/python Main_RunMe.py --all
```

What this command does:
- runs extraction
- runs OCR/export

Expected result:
- screenshots in `Output_Results/Frames`
- Excel output in `Output_Results/Tournament_Results.xlsx`

To test one video only:

```bash
.venv/bin/python Main_RunMe.py --all --video Test_3_Races.mkv
```

Linux Note

If Tkinter is unavailable, that is fine. Use the CLI. The CLI is the preferred Linux path anyway.

6. macOS Guide

macOS works similarly to Linux.

macOS Setup

Recommended:
- install Python 3.12
- install Homebrew if you do not already have it

Install dependencies:

```bash
brew install python@3.12 tesseract
```

Optional for merge support:

```bash
brew install ffmpeg
```

Then run:

```bash
chmod +x ./scripts/setup_unix.sh
./scripts/setup_unix.sh
```

What this command does:
- creates `.venv`
- installs Python packages
- creates `app_config.json` if needed
- runs `--check`

Expected result:
- setup completes
- `Tesseract: OK`

macOS First Run

Run:

```bash
.venv/bin/python Main_RunMe.py --all
```

Expected result:
- screenshots written to `Output_Results/Frames`
- Excel file written to `Output_Results/Tournament_Results.xlsx`

To test one video only:

```bash
.venv/bin/python Main_RunMe.py --all --video Test_3_Races.mkv
```

7. Command Reference

This section explains what each command does and what you should expect from it.

`Main_RunMe.py --check`

Example:

```powershell
python Main_RunMe.py --check
```

What it does:
- checks runtime dependencies and config

What you should expect:
- Python executable path
- input/output folder paths
- Tesseract status
- FFmpeg status
- worker counts
- debug settings

Good result:
- `Tesseract: OK`

`Main_RunMe.py --extract`

Example:

```powershell
python Main_RunMe.py --extract
```

What it does:
- scans videos in `Input_Videos`
- finds race result screens
- writes screenshots into `Output_Results/Frames`

What you should expect:
- printed race detections
- files like:
  - `...+0TrackName.png`
  - `...+1RaceNumber.png`
  - `...+2RaceScore.png`
  - `...+3TotalScore.png`

Optional single-video example:

```powershell
python Main_RunMe.py --extract --video Divisie_1.mkv
```

`Main_RunMe.py --ocr`

Example:

```powershell
python Main_RunMe.py --ocr
```

What it does:
- reads the screenshots from `Output_Results/Frames`
- OCRs names and track information
- writes the final Excel workbook

What you should expect:
- printed OCR progress
- `Output_Results/Tournament_Results.xlsx`

Optional single-video example:

```powershell
python Main_RunMe.py --ocr --video Test_3_Races.mkv
```

`Main_RunMe.py --all`

Example:

```powershell
python Main_RunMe.py --all
```

What it does:
- runs both `--extract` and `--ocr`

What you should expect:
- extracted screenshots
- final Excel workbook

Optional single-video example:

```powershell
python Main_RunMe.py --all --video Test_3_Races.mkv
```

`Main_RunMe.py --all --video <filename>`

What it does:
- limits extraction and OCR/export to one specific input video
- useful for quick testing and benchmark iterations

What you should expect:
- only that video's race groups are processed
- a shorter run with the same output format

`Main_RunMe.py`

Example:

```powershell
python Main_RunMe.py
```

What it does:
- launches the GUI

What you should expect:
- a window with buttons for extraction, export, folder opening, and optional merge

8. What Output To Expect

After `--extract`:
- files appear in `Output_Results/Frames`
- these are the key screenshots the tool selected

After `--ocr` or `--all`:
- `Output_Results/Tournament_Results.xlsx` appears
- the workbook includes detected score validation fields, review flags, and session-aware score columns

Optional debug output:
- `Output_Results/Debug/debug_max_val.csv`
- `Output_Results/Debug/Score_Frames`
- `Output_Results/Debug/linking_*.xlsx`

What success usually looks like:
- several `png` files per race in `Output_Results/Frames`
- one final Excel workbook
- no missing Tesseract error
- console output with:
  - `[mm:ss]` runtime timestamps
  - phase headers
  - progress every 5%
  - CPU/RAM and GPU/VRAM when available
  - per-race detection events

9. Configuration

You can configure the tool with:
- `app_config.json`
- environment variables

Start from:
- `app_config.example.json`

Main config keys:
- `tesseract_cmd`
- `execution_mode`
- `ocr_workers`
- `ocr_consensus_frames`
- `score_analysis_workers`
- `pass1_scan_workers`
- `pass1_segment_overlap_frames`
- `pass1_min_segment_frames`
- `write_debug_csv`
- `write_debug_score_images`
- `write_debug_linking_excel`

Example:

```json
{
  "tesseract_cmd": "/usr/bin/tesseract",
  "execution_mode": "cpu",
  "ocr_workers": 16,
  "ocr_consensus_frames": 7,
  "score_analysis_workers": 4,
  "pass1_scan_workers": 4,
  "pass1_segment_overlap_frames": 2100,
  "pass1_min_segment_frames": 30000,
  "write_debug_csv": true,
  "write_debug_score_images": true,
  "write_debug_linking_excel": true
}
```

Windows Tesseract example:

```json
{
  "tesseract_cmd": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
}
```

Environment variables:
- `MK8_TESSERACT_CMD`
- `MK8_EXECUTION_MODE`
- `MK8_OCR_WORKERS`
- `MK8_OCR_CONSENSUS_FRAMES`
- `MK8_SCORE_ANALYSIS_WORKERS`
- `MK8_PASS1_SCAN_WORKERS`
- `MK8_PASS1_SEGMENT_OVERLAP_FRAMES`
- `MK8_PASS1_MIN_SEGMENT_FRAMES`
- `MK8_WRITE_DEBUG_CSV`
- `MK8_WRITE_DEBUG_SCORE_IMAGES`
- `MK8_WRITE_DEBUG_LINKING_EXCEL`

Current performance defaults and decisions:
- `execution_mode: cpu` is the practical default for this project
- on the current validated baseline machine, OpenCL did not improve throughput versus CPU mode
- CPU mode produced the same Excel output and visually identical screenshots, but the screenshot PNG files were not byte-for-byte identical because the image backend produced tiny per-pixel differences
- treat OpenCL as an optional fallback or experiment path, not as the primary performance path unless new benchmarks show a real gain
- keep `ocr_consensus_frames: 7` as the safety default
- `ocr_consensus_frames: 4` was faster, but it changed OCR confidence and mapping metadata values
- `ocr_consensus_frames: 3` was rejected because it introduced a real scoring validation difference on `Divisie_1`

10. Benchmark And Validation Scripts

These scripts are mainly for regression testing and performance checking.

Windows PowerShell:

```powershell
.\scripts\quick_benchmark.ps1
.\scripts\release_benchmark.ps1
```

Linux/macOS:

```bash
./scripts/quick_benchmark.sh
./scripts/release_benchmark.sh
```

What they do:
- isolate the selected benchmark video
- clear generated outputs
- run `--check`
- run extraction
- run OCR/export
- validate output against a stored baseline
- print timing information

Default benchmark videos:
- quick benchmark: `Test_3_Races.mkv`
- release benchmark: `Divisie_1.mkv`

Expected result:
- `validation_passed: True`
- timing lines like:
  - `extract_seconds=...`
  - `ocr_seconds=...`

Runtime logging:
- the normal console output is now operator-focused rather than raw profiler output
- it shows:
  - run-time timestamps like `[00:23]`
  - phase starts and completions
  - progress every 5%
  - CPU/RAM and GPU/VRAM when available
  - per-race detection events
  - final performance summary with per-video durations
- GPU/VRAM metrics are best-effort:
  - NVIDIA systems usually report through `nvidia-smi`
  - other platforms may omit GPU metrics while still running normally

Validation policy during performance work:
- default target is exact output parity with the stored baseline
- if workbook output changes, or files are missing/unexpected, treat that as a hard failure
- if only a very small number of exported frame images differ while workbook output stays identical, treat that as a manual review case instead of an automatic rejection
- use `Test_3_Races.mkv` first for quick verification, then `Divisie_1.mkv` for release verification

Performance findings already tested:
- ROI-only preprocessing was faster but changed output, so it was rejected
- sequential consensus frame reading was kept because it improved performance without changing validated output
- pass-2 worker-owned race output was kept because it improved performance without changing validated output
- pass-2 to OCR streaming per race was tested and rejected because it was not stable enough versus the baseline
- OpenCL was tested and did not beat CPU mode on the validated machine
- do not re-run these same experiments unless hardware, OpenCV build, or acceptance criteria have changed

Pass-one scan workers:
- pass-one segment scanning is only used for longer videos
- overlapping segment scans can improve extraction time on multi-core CPUs
- a higher worker count can change which auxiliary screenshots are exported even when workbook output is unchanged
- practical starting points:
  - 1 worker for small or older systems
  - 2 workers for 8 to 15 logical cores
  - 3 workers for 16 to 23 logical cores
  - 4 workers for 24+ logical cores
- for this repository's current implementation, 4 workers is a good default on high-end systems and 6 should be treated as an upper bound for experimentation
- on a tested 24-core Windows laptop, 4 workers performed better than 6 and 8 for `Divisie_1.mkv`, so the current default of 4 is also a good tuned setting for similar high-end hardware

11. Troubleshooting

Problem: `Tesseract was not found`
- install Tesseract
- run `python Main_RunMe.py --check`
- set `tesseract_cmd` in `app_config.json` if needed

Problem: `FFmpeg was not found`
- install FFmpeg
- FFmpeg is only required for merging videos

Problem: no races were found
- verify the video is Mario Kart 8 local play vertical split-screen
- check whether the full game image is visible
- inspect `Output_Results/Frames`

Problem: OCR output looks wrong
- inspect `Output_Results/Frames`
- inspect `Output_Results/Debug/Score_Frames`
- low-quality or unusual input can reduce OCR quality

Problem: you want maximum validated performance
- use CPU mode first
- keep `ocr_consensus_frames` at `7`
- run the release benchmark before changing performance settings:

```powershell
.\scripts\release_benchmark.ps1
```

Problem: you only want to test one video
- use the `--video` option:

```powershell
.\.venv\Scripts\python.exe Main_RunMe.py --all --video Test_3_Races.mkv
```

Problem: GUI does not start on Linux or macOS
- use the CLI:

```bash
.venv/bin/python Main_RunMe.py --all
```

Problem: too many debug files are generated
- disable them in `app_config.json`:

```json
{
  "write_debug_csv": false,
  "write_debug_score_images": false,
  "write_debug_linking_excel": false
}
```

12. Notes About Git And Generated Files

Generated outputs are intentionally not tracked in git.

Examples:
- extracted screenshots
- OCR debug images
- `Tournament_Results.xlsx`
- local benchmark outputs
- local large input videos

This keeps the repository clean and makes commits focus on code and documentation rather than generated artifacts.
